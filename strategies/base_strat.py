# Base class for vol strategies. Delta hedging, position tracking, PnL decomp.
# All four strategies inherit from this. Don't put anything execution-specific here.

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.pricer import batch_greeks, bsm_price
from core.chain import OptionChain

logger = logging.getLogger(__name__)


@dataclass
class OptionLeg:
    strike: float
    expiry: float    # years
    is_call: bool
    qty: float       # positive = long, negative = short
    entry_price: float
    entry_time: float = field(default_factory=time.time)

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short"


@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float  = 0.0
    theta: float = 0.0
    rho: float   = 0.0

    def __add__(self, other: "PortfolioGreeks") -> "PortfolioGreeks":
        return PortfolioGreeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega   + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho     + other.rho,
        )


@dataclass
class PnLDecomposition:
    spread_capture: float    = 0.0
    delta_pnl: float         = 0.0   # from delta hedges
    gamma_pnl: float         = 0.0
    theta_decay: float       = 0.0
    vega_pnl: float          = 0.0
    transaction_costs: float = 0.0
    total: float             = 0.0

    def recompute_total(self) -> None:
        self.total = (
            self.spread_capture + self.delta_pnl + self.gamma_pnl
            + self.theta_decay + self.vega_pnl + self.transaction_costs
        )


class BaseVolStrategy(ABC):
    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        taker_fee: float = 0.0006,    # 6bps, Deribit taker as of last check
        maker_fee: float = -0.0001,   # -1bp rebate
        max_vega_notional: float = 1e6,
        max_delta_notional: float = 5e5,
        chain: Optional[OptionChain] = None,   # real listed strikes, falls back to a
                                                # synthetic grid heuristic if not given
    ):
        self.spot = spot
        self.rate = rate
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.max_vega_notional = max_vega_notional
        self.max_delta_notional = max_delta_notional
        self.chain = chain

        self.legs: list[OptionLeg] = []
        self.hedge_qty: float = 0.0
        self.realized_pnl: float = 0.0
        self.pnl = PnLDecomposition()
        self._last_sigma: Optional[float] = None

    def update_spot(self, new_spot: float, sigma: float) -> None:
        old_spot = self.spot
        self.spot = new_spot
        self._last_sigma = sigma
        if abs(old_spot - new_spot) > 1e-10:
            self.pnl.delta_pnl += self.hedge_qty * (new_spot - old_spot)

    def portfolio_greeks(self, sigma: Optional[float] = None) -> PortfolioGreeks:
        if not self.legs:
            return PortfolioGreeks()

        sig = sigma or self._last_sigma or 0.3  # 0.3 fallback is ugly but beats crashing

        n = len(self.legs)
        S_arr   = np.full(n, self.spot)
        K_arr   = np.array([l.strike   for l in self.legs])
        T_arr   = np.array([l.expiry   for l in self.legs])
        r_arr   = np.full(n, self.rate)
        sig_arr = np.full(n, sig)
        ic_arr  = np.array([l.is_call  for l in self.legs])
        qty_arr = np.array([l.qty      for l in self.legs])

        d, g, v, t, r = batch_greeks(S_arr, K_arr, T_arr, r_arr, sig_arr, ic_arr)

        return PortfolioGreeks(
            delta=float(np.dot(d, qty_arr)) + self.hedge_qty,
            gamma=float(np.dot(g, qty_arr)),
            vega=float(np.dot(v,  qty_arr)),
            theta=float(np.dot(t, qty_arr)),
            rho=float(np.dot(r,   qty_arr)),
        )

    def hedge_delta(self, sigma: Optional[float] = None, use_maker: bool = True) -> float:
        greeks = self.portfolio_greeks(sigma)
        option_delta = greeks.delta - self.hedge_qty
        target_hedge = -option_delta
        trade_size   = target_hedge - self.hedge_qty

        if abs(trade_size) < 1e-6:
            return 0.0

        fee = self.maker_fee if use_maker else self.taker_fee
        cost = abs(trade_size) * self.spot * abs(fee)

        self.pnl.transaction_costs -= cost
        self.hedge_qty = target_hedge

        logger.debug("delta_hedge", extra={"trade": trade_size, "hedge": self.hedge_qty, "cost": cost})
        return trade_size

    def should_rebalance(self, sigma: float, tol_delta: float = 0.05) -> bool:
        # WARNING: this default is naive. 5% of spot is way too wide near ATM with high gamma.
        # subclasses should override with proper W-W bands. this is just so nothing explodes
        # if someone forgets to override
        greeks = self.portfolio_greeks(sigma)
        return abs(greeks.delta) > tol_delta * self.spot

    def add_leg(self, leg: OptionLeg) -> None:
        self.legs.append(leg)
        fee = abs(leg.qty) * leg.entry_price * self.taker_fee
        self.pnl.transaction_costs -= fee
        logger.info("leg_added", extra={"K": leg.strike, "T": leg.expiry, "qty": leg.qty})

    def remove_leg(self, idx: int, exit_price: float, use_maker: bool = False) -> float:
        if idx >= len(self.legs):
            raise IndexError(f"leg index {idx} out of range")

        leg = self.legs.pop(idx)
        fee_rate = self.maker_fee if use_maker else self.taker_fee
        fee = abs(leg.qty) * exit_price * abs(fee_rate)
        pnl = leg.qty * (exit_price - leg.entry_price) - fee

        self.realized_pnl += pnl
        self.pnl.transaction_costs -= fee
        logger.info("leg_closed", extra={"pnl": pnl, "K": leg.strike})
        return pnl

    def partial_close_leg(self, idx: int, close_qty: float, exit_price: float, use_maker: bool = False) -> float:
        # trims a leg's size instead of closing it outright, entry_price is untouched
        # since it's a cost basis, only the closed slice realizes PnL
        if idx >= len(self.legs):
            raise IndexError(f"leg index {idx} out of range")

        leg = self.legs[idx]
        close_qty = min(abs(close_qty), abs(leg.qty))
        if close_qty < 1e-10:
            return 0.0
        if close_qty >= abs(leg.qty) - 1e-10:
            return self.remove_leg(idx, exit_price, use_maker)

        sign   = 1.0 if leg.qty > 0 else -1.0
        closed = sign * close_qty
        fee_rate = self.maker_fee if use_maker else self.taker_fee
        fee = close_qty * exit_price * abs(fee_rate)
        pnl = closed * (exit_price - leg.entry_price) - fee

        leg.qty -= closed
        self.realized_pnl += pnl
        self.pnl.transaction_costs -= fee
        logger.info("leg_partial_closed", extra={"pnl": pnl, "K": leg.strike, "closed_qty": closed})
        return pnl

    def mark_to_market(self, sigma: float) -> float:
        # NOTE: spread_capture, gamma_pnl, theta_decay, vega_pnl are never populated
        # anywhere in this codebase, only delta_pnl and transaction_costs are tracked.
        # Full greeks-based attribution is a separate piece of work, not covered here.
        unrealized_option_pnl = sum(
            leg.qty * (bsm_price(self.spot, leg.strike, leg.expiry, self.rate, sigma, leg.is_call) - leg.entry_price)
            for leg in self.legs
        )
        self.pnl.recompute_total()
        return unrealized_option_pnl + self.realized_pnl + self.pnl.total

    @abstractmethod
    def generate_signals(self, market_data: dict) -> list:
        ...

    def risk_check(self, sigma: float) -> dict:
        greeks = self.portfolio_greeks(sigma)
        vega_notional = abs(greeks.vega) * self.spot
        return {
            "net_delta":          greeks.delta,
            "vega_notional":      vega_notional,
            "vega_limit_breach":  vega_notional > self.max_vega_notional,
            "delta_limit_breach": abs(greeks.delta * self.spot) > self.max_delta_notional,
        }


class UnderlyingBook(BaseVolStrategy):
    # concrete BaseVolStrategy with no signal logic of its own. used as a per-underlying
    # leg/greeks/PnL container by strategies that trade more than one underlying at once
    # (dispersion: index + N components), composed rather than inherited so the
    # multi-underlying case doesn't leak into the single-spot assumption everything
    # else in this base class is built on.
    def generate_signals(self, market_data: dict) -> list:
        return []

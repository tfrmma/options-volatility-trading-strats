# Discrete-event backtest engine.
#
# Execute against bid/ask. Not mid. Not mark. Bid/ask.
# Options have 20-50bps spreads minimum, if you fill at mid your backtest
# is fiction and you'll find out the hard way in production.

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    timestamp: float
    spot: float
    sigma: float
    bid: float
    ask: float
    expiry: float
    strike: float
    is_call: bool
    option_bid: float
    option_ask: float


@dataclass
class Fill:
    timestamp: float
    side: str
    qty: float
    price: float
    is_option: bool
    strike: Optional[float] = None
    expiry: Optional[float] = None
    is_call: Optional[bool] = None
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.price


@dataclass
class BacktestResult:
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    pnl_series: list[float] = field(default_factory=list)

    def total_pnl(self) -> float:
        return self.pnl_series[-1] if self.pnl_series else 0.0

    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        equity = np.array([e for _, e in self.equity_curve])
        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / np.where(peak > 0, peak, 1.0)
        return float(np.min(dd))

    def sharpe(self, ann_factor: float = 252.0) -> float:
        if len(self.pnl_series) < 2:
            return 0.0
        daily = np.diff(self.pnl_series)
        std   = np.std(daily, ddof=1)
        return float(np.mean(daily) / std * np.sqrt(ann_factor)) if std > 1e-10 else 0.0

    def summary(self) -> dict:
        return {
            "total_pnl":   self.total_pnl(),
            "max_drawdown": self.max_drawdown(),
            "sharpe":       self.sharpe(),
            "n_fills":      len(self.fills),
            "total_fees":   sum(f.fee for f in self.fills),
        }


class BacktestEngine:

    def __init__(
        self,
        taker_fee: float = 0.0006,
        maker_fee: float = -0.0001,
        slippage_bps: float = 1.0,
        initial_capital: float = 1_000_000.0,
        slippage_fn: Optional[Callable] = None,
    ):
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.initial_capital = initial_capital
        self.slippage_fn = slippage_fn or (lambda price, qty: price * slippage_bps * 1e-4)

        self.result = BacktestResult()
        self._cash = initial_capital
        self._option_positions: dict[tuple, float] = {}
        self._spot_position: float = 0.0

    def run(self, data: pl.DataFrame, strategy_fn: Callable, verbose: bool = False) -> BacktestResult:
        self.result = BacktestResult()
        self._cash = self.initial_capital
        self._option_positions = {}
        self._spot_position = 0.0

        for row in data.iter_rows(named=True):
            snap = self._parse_row(row)
            if snap is None:
                continue

            for order in (strategy_fn(snap, self) or []):
                self._fill(order, snap)

            equity = self._cash + self._mtm(snap)
            self.result.equity_curve.append((snap.timestamp, equity))
            self.result.pnl_series.append(equity - self.initial_capital)

            if verbose:
                logger.debug("tick", extra={"ts": snap.timestamp, "equity": equity})

        return self.result

    def _fill(self, order: dict, snap: MarketSnapshot) -> Optional[Fill]:
        is_option = order["type"] == "option"
        side      = order["side"]
        qty       = abs(order["qty"])

        if is_option:
            raw   = snap.option_ask if side == "buy" else snap.option_bid
            slip  = self.slippage_fn(raw, qty) * (1 if side == "buy" else -1)
            price = max(raw + slip, 0.0)
            fee   = qty * price * self.taker_fee

            key   = (order["strike"], order["expiry"], order["is_call"])
            signed = qty if side == "buy" else -qty
            self._option_positions[key] = self._option_positions.get(key, 0.0) + signed
            self._cash -= signed * price + fee

            fill = Fill(snap.timestamp, side, signed, price, True,
                        order["strike"], order["expiry"], order["is_call"], fee)
        else:
            raw   = snap.ask if side == "buy" else snap.bid
            sign  = 1 if side == "buy" else -1
            price = raw + sign * self.slippage_fn(raw, qty)
            fee   = qty * price * self.taker_fee
            signed = sign * qty

            self._spot_position += signed
            self._cash -= signed * price + fee
            fill = Fill(snap.timestamp, side, signed, price, False, fee=fee)

        self.result.fills.append(fill)
        return fill

    def _mtm(self, snap: MarketSnapshot) -> float:
        # mark options at mid, good enough for daily PnL, not for intraday
        # TODO: this marks every open leg at the current snapshot's bid/ask, which is
        # only correct if the feed emits one row per live instrument. multi-leg books
        # (straddles, dispersion) need per-leg quotes here, not one snapshot's price
        # applied across the board.
        option_mid = 0.5 * (snap.option_bid + snap.option_ask)
        mtm = self._spot_position * snap.spot
        for key, qty in self._option_positions.items():
            if abs(qty) > 1e-10:
                mtm += qty * option_mid
        return mtm

    @staticmethod
    def _parse_row(row: dict) -> Optional[MarketSnapshot]:
        try:
            return MarketSnapshot(**{k: row[k] for k in MarketSnapshot.__dataclass_fields__})
        except (KeyError, TypeError) as e:
            logger.warning(f"bad row, skipping: {e}")
            return None

    def current_position(self) -> dict:
        return {"spot": self._spot_position, "options": dict(self._option_positions), "cash": self._cash}

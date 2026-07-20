# VRP harvesting. Sell variance when implied is rich vs realized.
#
# The VRP is persistent but not constant, it compresses, flips sign occasionally,
# and goes haywire around macro events. Track the z-score, don't just sell vol
# mechanically every time IV > RV.

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from strategies.base_strat import BaseVolStrategy, OptionLeg
from core.pricer import bsm_price, bsm_greeks
from core.estimators import yang_zhang, ewma_vol
from core.chain import target_strike

logger = logging.getLogger(__name__)

@dataclass
class VRPMetrics:
    implied_vol: float
    realized_vol: float
    vrp: float
    vrp_zscore: float
    vrp_percentile: float


@dataclass
class VRPSignal:
    action: str          # "short_var", "long_var", "flatten", "hold"
    expiry: float
    target_vega: float
    metrics: VRPMetrics
    confidence: float    # scale position by this, [0, 1]


class VolArbStrategy(BaseVolStrategy):
    """
    Short variance when IV > RV (normal regime), long when unusually cheap.
    Position size scales with VRP z-score, don't go full size on a 1-sigma signal.

    Exit when VRP has mean-reverted, not on a timer.
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        rv_window: int = 20,
        vrp_entry_zscore: float = 1.0,
        vrp_exit_zscore: float = 0.0,
        vrp_history_window: int = 60,
        max_vega_target: float = 10000.0,
        vega_rebalance_tol: float = 0.15,   # don't touch the position for less than 15% drift
        **kwargs,
    ):
        super().__init__(spot=spot, rate=rate, **kwargs)
        self.rv_window = rv_window
        self.vrp_entry_zscore  = vrp_entry_zscore
        self.vrp_exit_zscore   = vrp_exit_zscore
        self.vrp_history_window = vrp_history_window
        self.max_vega_target   = max_vega_target
        self.vega_rebalance_tol = vega_rebalance_tol

        self._vrp_history: list[float] = []
        self._in_position = False

    def compute_vrp(self, iv: float, rv: float) -> VRPMetrics:
        vrp = iv - rv
        self._vrp_history.append(vrp)
        if len(self._vrp_history) > self.vrp_history_window:
            self._vrp_history.pop(0)

        history = np.array(self._vrp_history)
        if len(history) < 5:
            return VRPMetrics(iv, rv, vrp, 0.0, 0.5)

        mu, std = np.mean(history), np.std(history, ddof=1)
        zscore = (vrp - mu) / std if std > 1e-8 else 0.0
        pctile = float(np.mean(history <= vrp))
        return VRPMetrics(iv, rv, vrp, zscore, pctile)

    def generate_signals(self, market_data: dict) -> list[VRPSignal]:
        # market_data: implied_vol, expiry, + either ohlcv_df or log_returns or realized_vol_fallback
        iv     = market_data["implied_vol"]
        expiry = market_data["expiry"]
        ohlcv  = market_data.get("ohlcv_df")

        if ohlcv is not None and len(ohlcv) >= self.rv_window:
            rv = yang_zhang(
                ohlcv["open"].to_numpy()[-self.rv_window:],
                ohlcv["high"].to_numpy()[-self.rv_window:],
                ohlcv["low"].to_numpy()[-self.rv_window:],
                ohlcv["close"].to_numpy()[-self.rv_window:],
            )
        elif "log_returns" in market_data:
            rv = ewma_vol(market_data["log_returns"])
        else:
            rv = market_data.get("realized_vol_fallback", iv * 0.9)
            logger.warning("using fallback RV, no OHLCV or returns in market_data")

        metrics = self.compute_vrp(iv, rv)

        if self._in_position:
            if metrics.vrp_zscore < self.vrp_exit_zscore:
                return [VRPSignal("flatten", expiry, 0.0, metrics, 1.0)]
            return [VRPSignal("hold", expiry, self._scaled_vega(metrics.vrp_zscore), metrics,
                              min(metrics.vrp_zscore / self.vrp_entry_zscore, 1.0))]

        if metrics.vrp_zscore > self.vrp_entry_zscore:
            return [VRPSignal("short_var", expiry, self._scaled_vega(metrics.vrp_zscore), metrics,
                              min(metrics.vrp_zscore / (self.vrp_entry_zscore * 2), 1.0))]

        if metrics.vrp_zscore < -self.vrp_entry_zscore:
            return [VRPSignal("long_var", expiry, self._scaled_vega(abs(metrics.vrp_zscore)), metrics,
                              min(abs(metrics.vrp_zscore) / (self.vrp_entry_zscore * 2), 1.0))]

        return []

    def execute_signal(self, signal: VRPSignal, sigma: float) -> None:
        if signal.action == "flatten":
            self._flatten(sigma)
            self._in_position = False
            return

        if signal.action == "hold":
            self._rebalance_vega(signal, sigma)
            if self.should_rebalance(sigma):
                self.hedge_delta(sigma)
            return

        sign   = -1.0 if signal.action == "short_var" else 1.0
        strike = target_strike(self.spot, signal.expiry, self.chain)
        unit_v = self._straddle_unit_vega(strike, signal.expiry, sigma)

        if unit_v < 1e-8:
            logger.warning("zero unit vega, skipping entry")
            return

        qty = sign * signal.target_vega * signal.confidence / (unit_v * self.spot * 2.0)

        self.add_leg(OptionLeg(strike=strike, expiry=signal.expiry, is_call=True,  qty=qty,
                               entry_price=bsm_price(self.spot, strike, signal.expiry, self.rate, sigma, True)))
        self.add_leg(OptionLeg(strike=strike, expiry=signal.expiry, is_call=False, qty=qty,
                               entry_price=bsm_price(self.spot, strike, signal.expiry, self.rate, sigma, False)))

        self.hedge_delta(sigma)
        self._in_position = True
        logger.info("var_trade", extra={
            "action": signal.action, "vrp": signal.metrics.vrp,
            "z": signal.metrics.vrp_zscore, "qty": qty,
        })

    def _scaled_vega(self, zscore: float) -> float:
        return min(zscore / (self.vrp_entry_zscore * 2.0), 1.0) * self.max_vega_target

    def _rebalance_vega(self, signal: VRPSignal, sigma: float) -> None:
        # scale the existing straddle toward the new target as conviction (z-score)
        # drifts, instead of freezing size at entry and only ever touching delta after.
        # partial close on the way down, add at the same strikes on the way up.
        if not self.legs:
            return

        current_vega_notional = abs(self.portfolio_greeks(sigma).vega) * self.spot
        target_vega_notional  = signal.target_vega * signal.confidence
        if current_vega_notional < 1e-8:
            return

        ratio = target_vega_notional / current_vega_notional
        if abs(ratio - 1.0) < self.vega_rebalance_tol:
            return  # not worth paying the spread over a small wobble in conviction

        for idx in reversed(range(len(self.legs))):
            leg   = self.legs[idx]
            price = bsm_price(self.spot, leg.strike, leg.expiry, self.rate, sigma, leg.is_call)
            if ratio < 1.0:
                self.partial_close_leg(idx, abs(leg.qty) * (1.0 - ratio), price)
            else:
                add_qty = leg.qty * (ratio - 1.0)
                if abs(add_qty) > 1e-8:
                    self.add_leg(OptionLeg(strike=leg.strike, expiry=leg.expiry, is_call=leg.is_call,
                                           qty=add_qty, entry_price=price))

        logger.info("vega_rebalanced", extra={"ratio": ratio, "target_vega_notional": target_vega_notional})

    def _flatten(self, sigma: float) -> None:
        self.close_all(sigma)

    def _straddle_unit_vega(self, strike: float, expiry: float, sigma: float) -> float:
        _, _, vega, _, _ = bsm_greeks(self.spot, strike, expiry, self.rate, sigma, True)
        return vega  # call and put vega are the same in BSM

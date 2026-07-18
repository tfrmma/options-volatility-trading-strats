# Vol surface RV. Calendar spreads and skew arb.
#
# Trade mispricings in the surface itself rather than outright vol level.
# Calendar: sell rich near-term, buy cheap back-month.
# Skew: sell overpriced wing, buy cheap wing.
#
# All positions should be vega/theta neutral at entry.
# Residual exposure is vanna, volga — that's fine, that's the trade.

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from strategies.base_strat import BaseVolStrategy, OptionLeg
from core.surface import VolSurface
from core.pricer import bsm_price, bsm_greeks

logger = logging.getLogger(__name__)


@dataclass
class SkewMetrics:
    rr_25: float      # 25-delta risk reversal: IV(25C) - IV(25P)
    bf_25: float      # 25-delta butterfly: 0.5*(IV(25C)+IV(25P)) - ATM
    atm_iv: float
    expiry: float

    @property
    def put_skew_rich(self) -> bool:
        return self.rr_25 < -0.05  # puts bid by 5+ vol pts

    @property
    def call_skew_rich(self) -> bool:
        return self.rr_25 > 0.05


@dataclass
class CalendarMetrics:
    near_iv: float
    far_iv: float
    near_expiry: float
    far_expiry: float
    var_ratio: float          # near_TV / far_TV, normalized by T
    theoretical_ratio: float  # from flat vol assumption

    @property
    def near_rich(self) -> bool:
        return self.var_ratio > self.theoretical_ratio * 1.05

    @property
    def near_cheap(self) -> bool:
        return self.var_ratio < self.theoretical_ratio * 0.95


class VolSurfaceTrading(BaseVolStrategy):

    def __init__(
        self,
        spot: float,
        surface: VolSurface,
        rate: float = 0.0,
        calendar_threshold: float = 0.05,   # 5% deviation from fair ratio
        skew_threshold: float = 0.03,        # 3 vol pts dislocation
        target_vega_per_trade: float = 5000.0,
        **kwargs,
    ):
        super().__init__(spot=spot, rate=rate, **kwargs)
        self.surface = surface
        self.calendar_threshold = calendar_threshold
        self.skew_threshold = skew_threshold
        self.target_vega_per_trade = target_vega_per_trade

    def compute_skew_metrics(self, expiry: float) -> SkewMetrics:
        atm_iv = self.surface.implied_vol(0.0, expiry)
        call_K  = self._find_delta_strike(0.25, expiry, atm_iv, is_call=True)
        put_K   = self._find_delta_strike(0.25, expiry, atm_iv, is_call=False)

        call_iv = self.surface.implied_vol(np.log(call_K / self.spot), expiry)
        put_iv  = self.surface.implied_vol(np.log(put_K  / self.spot), expiry)

        return SkewMetrics(
            rr_25=call_iv - put_iv,
            bf_25=0.5 * (call_iv + put_iv) - atm_iv,
            atm_iv=atm_iv,
            expiry=expiry,
        )

    def compute_calendar_metrics(
        self, near_expiry: float, far_expiry: float, log_moneyness: float = 0.0,
    ) -> CalendarMetrics:
        near_iv = self.surface.implied_vol(log_moneyness, near_expiry)
        far_iv  = self.surface.implied_vol(log_moneyness, far_expiry)

        near_tv = near_iv**2 * near_expiry
        far_tv  = far_iv**2  * far_expiry
        var_ratio = near_tv / far_tv if far_tv > 1e-10 else 1.0

        return CalendarMetrics(
            near_iv=near_iv, far_iv=far_iv,
            near_expiry=near_expiry, far_expiry=far_expiry,
            var_ratio=var_ratio,
            theoretical_ratio=near_expiry / far_expiry,  # flat vol baseline
        )

    def generate_signals(self, market_data: dict) -> list:
        expiries = market_data.get("expiries", [])
        if len(expiries) < 2:
            return []

        signals = []

        for T in expiries:
            skew = self.compute_skew_metrics(T)
            market_skew = market_data.get("market_skews", {}).get(T)
            if market_skew:
                sig = self._skew_signal(skew, market_skew, T)
                if sig:
                    signals.append(sig)

        for i in range(len(expiries) - 1):
            cal = self.compute_calendar_metrics(expiries[i], expiries[i + 1])
            sig = self._calendar_signal(cal)
            if sig:
                signals.append(sig)

        return signals

    def enter_calendar(
        self, metrics: CalendarMetrics, sigma_near: float, sigma_far: float, sell_near: bool = True,
    ) -> None:
        # sell near (theta collect), buy back (vega hedge). vega-neutral at entry
        strike    = self._round_strike(self.spot)
        near_sign = -1.0 if sell_near else 1.0

        near_v = self._straddle_vega(strike, metrics.near_expiry, sigma_near)
        far_v  = self._straddle_vega(strike, metrics.far_expiry,  sigma_far)

        if near_v < 1e-8 or far_v < 1e-8:
            return

        near_qty = near_sign * self.target_vega_per_trade / (near_v * self.spot * 2.0)
        far_qty  = -near_sign * near_qty * near_v / far_v  # vega-neutral

        for is_call in [True, False]:
            self.add_leg(OptionLeg(
                strike=strike, expiry=metrics.near_expiry, is_call=is_call, qty=near_qty,
                entry_price=bsm_price(self.spot, strike, metrics.near_expiry, self.rate, sigma_near, is_call),
            ))
            self.add_leg(OptionLeg(
                strike=strike, expiry=metrics.far_expiry,  is_call=is_call, qty=far_qty,
                entry_price=bsm_price(self.spot, strike, metrics.far_expiry,  self.rate, sigma_far,  is_call),
            ))

        self.hedge_delta(sigma_near)
        logger.info("calendar_entered", extra={
            "sell_near": sell_near, "near_iv": metrics.near_iv,
            "far_iv": metrics.far_iv, "ratio": metrics.var_ratio,
        })

    def enter_risk_reversal(
        self,
        expiry: float,
        call_strike: float, put_strike: float,
        sigma_call: float,  sigma_put: float,
        sell_call: bool = True,
    ) -> None:
        c_sign = -1.0 if sell_call else 1.0
        p_sign = -c_sign

        call_v = self._unit_vega(call_strike, expiry, sigma_call, True)
        put_v  = self._unit_vega(put_strike,  expiry, sigma_put,  False)

        if call_v < 1e-8 or put_v < 1e-8:
            return

        call_qty = c_sign * self.target_vega_per_trade / (call_v * self.spot)
        put_qty  = p_sign * abs(call_qty) * call_v / put_v  # vega match

        self.add_leg(OptionLeg(
            strike=call_strike, expiry=expiry, is_call=True, qty=call_qty,
            entry_price=bsm_price(self.spot, call_strike, expiry, self.rate, sigma_call, True),
        ))
        self.add_leg(OptionLeg(
            strike=put_strike, expiry=expiry, is_call=False, qty=put_qty,
            entry_price=bsm_price(self.spot, put_strike,  expiry, self.rate, sigma_put,  False),
        ))

        self.hedge_delta(sigma_call)
        logger.info("rr_entered", extra={"sell_call": sell_call, "cK": call_strike, "pK": put_strike})

    def _skew_signal(self, model_skew: SkewMetrics, market_skew: SkewMetrics, expiry: float) -> Optional[dict]:
        rr_diff = market_skew.rr_25 - model_skew.rr_25
        if abs(rr_diff) > self.skew_threshold:
            return {"action": "risk_reversal", "expiry": expiry,
                    "sell_puts": rr_diff < 0, "rr_diff": rr_diff}
        return None

    def _calendar_signal(self, metrics: CalendarMetrics) -> Optional[dict]:
        deviation = abs(metrics.var_ratio - metrics.theoretical_ratio) / metrics.theoretical_ratio
        if deviation > self.calendar_threshold:
            return {"action": "calendar", "metrics": metrics, "sell_near": metrics.near_rich}
        return None

    def _find_delta_strike(self, target_delta: float, expiry: float, atm_iv: float, is_call: bool) -> float:
        # binary search — slow but called infrequently. don't optimize until it's actually hot
        lo, hi = self.spot * 0.5, self.spot * 2.0
        for _ in range(50):
            mid   = 0.5 * (lo + hi)
            iv_k  = self.surface.implied_vol(np.log(mid / self.spot), expiry)
            delta, _, _, _, _ = bsm_greeks(self.spot, mid, expiry, self.rate, iv_k, is_call)
            d = delta if is_call else abs(delta)
            if abs(d - target_delta) < 1e-5:
                return mid
            if d > target_delta:
                lo = mid if is_call else hi
            else:
                hi = mid if is_call else lo
        return 0.5 * (lo + hi)

    def _straddle_vega(self, strike: float, expiry: float, sigma: float) -> float:
        _, _, vega, _, _ = bsm_greeks(self.spot, strike, expiry, self.rate, sigma, True)
        return vega

    def _unit_vega(self, strike: float, expiry: float, sigma: float, is_call: bool) -> float:
        _, _, vega, _, _ = bsm_greeks(self.spot, strike, expiry, self.rate, sigma, is_call)
        return vega

    @staticmethod
    def _round_strike(spot: float) -> float:
        if spot < 1000:   return round(spot / 5)   * 5.0
        if spot < 10000:  return round(spot / 50)  * 50.0
        if spot < 100000: return round(spot / 500) * 500.0
        return            round(spot / 1000) * 1000.0

# Delta-neutral straddles/strangles. Trade realized vs implied vol.
#
# The thesis is trivial. The execution isn't.
# Time-based delta rebalancing will eat your edge — use W-W bands.

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from strategies.base_strat import BaseVolStrategy, OptionLeg
from core.pricer import bsm_greeks, bsm_price

logger = logging.getLogger(__name__)


@dataclass
class StraddleSignal:
    action: str        # "enter_long", "enter_short", "close"
    atm_strike: float
    expiry: float
    qty: float
    expected_vega_notional: float
    vol_edge: float    # IV - RV, vol points


class DeltaNeutralStrategy(BaseVolStrategy):

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        vol_edge_threshold: float = 0.03,   # 3 vol pts minimum — below this the fees eat you
        vega_target: float = 5000.0,
        strangle_width: float = 0.0,        # 0 = straddle. 0.1 = 10% OTM wings
        ww_band_multiplier: float = 1.5,
        stop_loss_vega_pct: float = 0.30,
        **kwargs,
    ):
        super().__init__(spot=spot, rate=rate, **kwargs)
        self.vol_edge_threshold = vol_edge_threshold
        self.vega_target = vega_target
        self.strangle_width = strangle_width
        self.ww_band_multiplier = ww_band_multiplier
        self.stop_loss_vega_pct = stop_loss_vega_pct

        self._entry_vega: float = 0.0
        self._entry_iv: float = 0.0

    def generate_signals(self, market_data: dict) -> list[StraddleSignal]:
        # market_data: spot, implied_vol, realized_vol_30d, expiry
        S      = market_data["spot"]
        iv     = market_data["implied_vol"]
        rv     = market_data["realized_vol_30d"]
        expiry = market_data["expiry"]
        vol_edge = iv - rv

        if self.legs:
            if self._check_stop(iv):
                return [StraddleSignal("close", S, expiry, 0, 0, vol_edge)]
            return []

        if abs(vol_edge) < self.vol_edge_threshold:
            return []

        atm_strike = self._round_strike(S)
        qty = self._size_for_vega(atm_strike, expiry, iv)
        if qty < 1e-6:
            return []

        action = "enter_short" if vol_edge > 0 else "enter_long"
        return [StraddleSignal(
            action=action,
            atm_strike=atm_strike,
            expiry=expiry,
            qty=qty,
            expected_vega_notional=abs(qty) * self._unit_vega(atm_strike, expiry, iv) * S,
            vol_edge=vol_edge,
        )]


    def execute_signal(self, signal: StraddleSignal, sigma: float) -> None:
        if signal.action == "close":
            self._close_all(sigma)
            return

        sign = -1.0 if signal.action == "enter_short" else 1.0
        call_K = signal.atm_strike * (1.0 + self.strangle_width)
        put_K  = signal.atm_strike * (1.0 - self.strangle_width)

        self.add_leg(OptionLeg(
            strike=call_K, expiry=signal.expiry, is_call=True,
            qty=sign * signal.qty,
            entry_price=bsm_price(self.spot, call_K, signal.expiry, self.rate, sigma, True),
        ))
        self.add_leg(OptionLeg(
            strike=put_K, expiry=signal.expiry, is_call=False,
            qty=sign * signal.qty,
            entry_price=bsm_price(self.spot, put_K, signal.expiry, self.rate, sigma, False),
        ))

        greeks = self.portfolio_greeks(sigma)
        self._entry_vega = abs(greeks.vega)
        self._entry_iv   = sigma

        self.hedge_delta(sigma)
        logger.info("straddle_entered", extra={
            "action": signal.action, "K": signal.atm_strike, "vega": self._entry_vega
        })

    def should_rebalance(self, sigma: float, tol_delta: float = 0.05) -> bool:
        # Whalley-Wilmott: band = (3/2 * lambda * gamma * S^2 * sigma^2)^(1/3)
        # NOT time-based. if you hedge on a timer you're paying fees for nothing
        greeks = self.portfolio_greeks(sigma)
        gamma  = abs(greeks.gamma)

        if gamma < 1e-10:
            return False

        lam  = abs(self.taker_fee)
        band = self.ww_band_multiplier * (1.5 * lam * gamma * self.spot**2 * sigma**2) ** (1.0 / 3.0)
        return abs(greeks.delta) > band

    def _check_stop(self, current_iv: float) -> bool:
        if self._entry_vega == 0:
            return False
        greeks  = self.portfolio_greeks(current_iv)
        vega_pnl = greeks.vega * (current_iv - self._entry_iv)
        max_loss = -self.stop_loss_vega_pct * self._entry_vega * self.spot
        return vega_pnl < max_loss

    def _close_all(self, sigma: float) -> None:
        while self.legs:
            leg = self.legs[0]
            self.remove_leg(0, bsm_price(self.spot, leg.strike, leg.expiry, self.rate, sigma, leg.is_call))
        self.realized_pnl += self.hedge_qty * self.spot
        self.hedge_qty = 0.0

    def _size_for_vega(self, strike: float, expiry: float, sigma: float) -> float:
        unit_vega = self._unit_vega(strike, expiry, sigma)
        if unit_vega < 1e-8:
            return 0.0
        return self.vega_target / (2.0 * unit_vega * self.spot)  # x2 for call + put

    def _unit_vega(self, strike: float, expiry: float, sigma: float) -> float:
        _, _, vega, _, _ = bsm_greeks(self.spot, strike, expiry, self.rate, sigma, True)
        return vega

    @staticmethod
    def _round_strike(spot: float) -> float:
        # TODO: pull listed strikes from chain instead of rounding. this is embarrassing
        if spot < 1000:      return round(spot / 5)    * 5.0
        if spot < 10000:     return round(spot / 50)   * 50.0
        if spot < 100000:    return round(spot / 500)  * 500.0
        return               round(spot / 1000) * 1000.0

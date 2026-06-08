# Dispersion trading. Short index vol, long component vol.
#
# Thesis: implied correlation is systematically too high because people overpay
# for index puts as macro hedges. You short the correlation risk premium.
#
# Known risks: correlation spikes hard in stress events (2008, covid march).
# This trade does NOT like tail events. Size accordingly.

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from strategies.base_strat import BaseVolStrategy, OptionLeg
from core.pricer import bsm_price, bsm_greeks

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    symbol: str
    weight: float       # index weight, should sum to 1
    spot: float
    implied_vol: float
    realized_vol: float


@dataclass
class DispersionMetrics:
    implied_correlation: float
    realized_correlation: float
    correlation_premium: float   # IC - RC — this is what you're trading
    index_iv: float
    basket_iv: float
    fair_index_iv: float         # what index IV should be given component IVs + RC


@dataclass
class DispersionPosition:
    index_legs: list[OptionLeg] = field(default_factory=list)
    component_legs: dict[str, list[OptionLeg]] = field(default_factory=dict)
    entry_metrics: Optional[DispersionMetrics] = None
    is_active: bool = False


class DispersionStrategy(BaseVolStrategy):

    def __init__(
        self,
        index_spot: float,
        components: list[ComponentSpec],
        rate: float = 0.0,
        corr_premium_threshold: float = 0.05,   # 5 correlation points minimum
        vega_notional_index: float = 50000.0,
        **kwargs,
    ):
        super().__init__(spot=index_spot, rate=rate, **kwargs)
        self.components = {c.symbol: c for c in components}
        self.corr_premium_threshold = corr_premium_threshold
        self.vega_notional_index = vega_notional_index
        self.position = DispersionPosition()

    def compute_implied_correlation(
        self, index_iv: float, components: list[ComponentSpec],
    ) -> DispersionMetrics:
        # back out implied correlation from index IV vs component IVs
        # assumes uniform pairwise correlation (simplification, but standard)
        weights = np.array([c.weight for c in components])
        ivs     = np.array([c.implied_vol for c in components])
        rvs     = np.array([c.realized_vol for c in components])

        cov_outer = np.outer(weights * ivs, weights * ivs)
        diag_term    = np.sum((weights * ivs) ** 2)
        off_diag_term = np.sum(cov_outer) - diag_term

        index_var = index_iv ** 2

        if abs(off_diag_term) < 1e-12:
            impl_corr = 0.0
        else:
            impl_corr = float(np.clip((index_var - diag_term) / off_diag_term, 0.0, 1.0))

        # TODO: realized correlation should come from actual pairwise return correlations
        # this proxy is a rough estimate — fine for signal generation, wrong for sizing
        realized_corr = float(np.clip(np.mean(rvs) / max(index_iv, 1e-8), 0.0, 1.0))

        fair_var = realized_corr * off_diag_term + diag_term
        fair_iv  = float(np.sqrt(max(fair_var, 0.0)))
        basket_iv = float(np.dot(weights, ivs))

        return DispersionMetrics(
            implied_correlation=impl_corr,
            realized_correlation=realized_corr,
            correlation_premium=impl_corr - realized_corr,
            index_iv=index_iv,
            basket_iv=basket_iv,
            fair_index_iv=fair_iv,
        )

    def generate_signals(self, market_data: dict) -> list:
        index_iv       = market_data["index_iv"]
        expiry         = market_data["expiry"]
        component_data = market_data.get("components", list(self.components.values()))

        if self.position.is_active:
            metrics = self.compute_implied_correlation(index_iv, component_data)
            if metrics.correlation_premium < self.corr_premium_threshold * 0.3:
                return [{"action": "close_dispersion", "metrics": metrics}]
            return []

        metrics = self.compute_implied_correlation(index_iv, component_data)
        if metrics.correlation_premium < self.corr_premium_threshold:
            return []

        logger.info("dispersion_signal", extra={
            "impl_corr": metrics.implied_correlation,
            "rc": metrics.realized_correlation,
            "premium": metrics.correlation_premium,
        })
        return [{"action": "enter_dispersion", "expiry": expiry, "metrics": metrics}]

    def enter_dispersion(self, expiry: float, metrics: DispersionMetrics, sigma_index: float) -> None:
        # sell index straddle, buy component straddles vega-weighted
        index_strike = self._round_strike(self.spot)
        index_unit_v = self._straddle_vega(self.spot, index_strike, expiry, sigma_index)

        if index_unit_v < 1e-8:
            logger.error("zero index vega — aborting")
            return

        index_qty = -self.vega_notional_index / (index_unit_v * self.spot * 2.0)

        for is_call in [True, False]:
            p = bsm_price(self.spot, index_strike, expiry, self.rate, sigma_index, is_call)
            self.add_leg(OptionLeg(strike=index_strike, expiry=expiry, is_call=is_call,
                                   qty=index_qty, entry_price=p))

        self.position.index_legs = self.legs[-2:]

        for comp in self.components.values():
            comp_K = self._round_strike(comp.spot)
            unit_v = self._straddle_vega(comp.spot, comp_K, expiry, comp.implied_vol)
            if unit_v < 1e-8:
                logger.warning(f"zero vega for {comp.symbol}, skipping component leg")
                continue

            # size each component so its vega_notional = index_vega_notional * weight
            comp_qty = (self.vega_notional_index * comp.weight) / (unit_v * comp.spot * 2.0)

            for is_call in [True, False]:
                p = bsm_price(comp.spot, comp_K, expiry, self.rate, comp.implied_vol, is_call)
                self.add_leg(OptionLeg(strike=comp_K, expiry=expiry, is_call=is_call,
                                       qty=comp_qty, entry_price=p))

            self.position.component_legs.setdefault(comp.symbol, []).extend(self.legs[-2:])

        self.position.entry_metrics = metrics
        self.position.is_active = True
        self.hedge_delta(sigma_index)
        logger.info("dispersion_entered", extra={"corr_premium": metrics.correlation_premium})

    @staticmethod
    def _straddle_vega(spot: float, strike: float, expiry: float, sigma: float) -> float:
        _, _, vega, _, _ = bsm_greeks(spot, strike, expiry, 0.0, sigma, True)
        return vega

    @staticmethod
    def _round_strike(spot: float) -> float:
        if spot < 1000:   return round(spot / 5)   * 5.0
        if spot < 10000:  return round(spot / 50)  * 50.0
        if spot < 100000: return round(spot / 500) * 500.0
        return            round(spot / 1000) * 1000.0

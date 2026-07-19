# Dispersion trading. Short index vol, long component vol.
#
# Thesis: implied correlation is systematically too high because people overpay
# for index puts as macro hedges. You short the correlation risk premium.
#
# Known risks: correlation spikes hard in stress events (2008, covid march).
# This trade does NOT like tail events. Size accordingly.
#
# Does NOT inherit BaseVolStrategy. Index and components each sit on a different
# underlying with their own spot, and BaseVolStrategy is built around a single self.spot,
# baking multi-underlying support into it would leak into the other three strategies that
# don't need it. Instead this holds one UnderlyingBook per underlying and aggregates.

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from strategies.base_strat import UnderlyingBook, OptionLeg
from core.pricer import bsm_price, bsm_greeks
from core.chain import OptionChain, target_strike

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
    correlation_premium: float   # IC - RC, this is what you're trading
    index_iv: float
    basket_iv: float
    fair_index_iv: float         # what index IV should be given component IVs + RC


@dataclass
class DispersionPosition:
    index_legs: list[OptionLeg] = field(default_factory=list)
    component_legs: dict[str, list[OptionLeg]] = field(default_factory=dict)
    entry_metrics: Optional[DispersionMetrics] = None
    is_active: bool = False


class DispersionStrategy:

    def __init__(
        self,
        index_spot: float,
        components: list[ComponentSpec],
        rate: float = 0.0,
        corr_premium_threshold: float = 0.05,   # 5 correlation points minimum
        vega_notional_index: float = 50000.0,
        index_chain: Optional[OptionChain] = None,
        component_chains: Optional[dict[str, OptionChain]] = None,
        **book_kwargs,
    ):
        self.rate = rate
        self.components_spec = {c.symbol: c for c in components}
        self.corr_premium_threshold = corr_premium_threshold
        self.vega_notional_index = vega_notional_index
        component_chains = component_chains or {}

        self.index_book = UnderlyingBook(spot=index_spot, rate=rate, chain=index_chain, **book_kwargs)
        self.component_books: dict[str, UnderlyingBook] = {
            c.symbol: UnderlyingBook(spot=c.spot, rate=rate, chain=component_chains.get(c.symbol), **book_kwargs)
            for c in components
        }
        self.position = DispersionPosition()

    @property
    def spot(self) -> float:
        # index spot, kept as a property since a lot of the sizing/strike math below
        # reads naturally as "the" spot, callers that need a component's spot go
        # through component_books[symbol].spot instead
        return self.index_book.spot

    def update_spot(
        self,
        index_spot: float,
        sigma_index: float,
        component_spots: dict[str, float],
        component_sigmas: Optional[dict[str, float]] = None,
    ) -> None:
        component_sigmas = component_sigmas or {}
        self.index_book.update_spot(index_spot, sigma_index)
        for symbol, book in self.component_books.items():
            if symbol in component_spots:
                book.update_spot(component_spots[symbol], component_sigmas.get(symbol, sigma_index))

    def mark_to_market(self, sigma_index: float, component_sigmas: Optional[dict[str, float]] = None) -> float:
        component_sigmas = component_sigmas or {}
        total = self.index_book.mark_to_market(sigma_index)
        for symbol, book in self.component_books.items():
            total += book.mark_to_market(component_sigmas.get(symbol, sigma_index))
        return total

    def hedge_delta(
        self,
        sigma_index: Optional[float] = None,
        component_sigmas: Optional[dict[str, float]] = None,
        use_maker: bool = True,
    ) -> dict[str, float]:
        # each book hedges against its own underlying, you can't hedge BTC delta with
        # ETH spot, so there's no cross-book aggregation here, just fan-out
        component_sigmas = component_sigmas or {}
        trades = {"index": self.index_book.hedge_delta(sigma_index, use_maker)}
        for symbol, book in self.component_books.items():
            trades[symbol] = book.hedge_delta(component_sigmas.get(symbol), use_maker)
        return trades

    def risk_check(self, sigma_index: float, component_sigmas: Optional[dict[str, float]] = None) -> dict:
        # same delta_usd = delta * spot / vega_notional = |vega| * spot convention as
        # BaseVolStrategy.risk_check, just summed across underlyings since raw delta/vega
        # from different underlyings aren't directly comparable
        component_sigmas = component_sigmas or {}
        books = [("index", self.index_book, sigma_index)]
        books += [(sym, book, component_sigmas.get(sym, sigma_index)) for sym, book in self.component_books.items()]

        per_book = {}
        total_delta_usd = 0.0
        total_vega_notional = 0.0
        for name, book, sig in books:
            greeks = book.portfolio_greeks(sig)
            delta_usd = greeks.delta * book.spot
            vega_notional = abs(greeks.vega) * book.spot
            per_book[name] = {"delta_usd": delta_usd, "vega_notional": vega_notional}
            total_delta_usd += delta_usd
            total_vega_notional += vega_notional

        return {
            "net_delta_usd":      total_delta_usd,
            "vega_notional":      total_vega_notional,
            "vega_limit_breach":  total_vega_notional > self.index_book.max_vega_notional,
            "delta_limit_breach": abs(total_delta_usd) > self.index_book.max_delta_notional,
            "per_book":           per_book,
        }

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
        # this proxy is a rough estimate, fine for signal generation, wrong for sizing
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
        component_data = market_data.get("components", list(self.components_spec.values()))

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
        index_spot = self.index_book.spot
        index_strike = target_strike(index_spot, expiry, self.index_book.chain)
        index_unit_v = self._straddle_vega(index_spot, index_strike, expiry, sigma_index)

        if index_unit_v < 1e-8:
            logger.error("zero index vega, aborting")
            return

        index_qty = -self.vega_notional_index / (index_unit_v * index_spot * 2.0)

        for is_call in [True, False]:
            p = bsm_price(index_spot, index_strike, expiry, self.rate, sigma_index, is_call)
            self.index_book.add_leg(OptionLeg(strike=index_strike, expiry=expiry, is_call=is_call,
                                              qty=index_qty, entry_price=p))

        self.position.index_legs = list(self.index_book.legs[-2:])

        for comp in self.components_spec.values():
            book = self.component_books[comp.symbol]
            comp_K = target_strike(book.spot, expiry, book.chain)
            unit_v = self._straddle_vega(book.spot, comp_K, expiry, comp.implied_vol)
            if unit_v < 1e-8:
                logger.warning(f"zero vega for {comp.symbol}, skipping component leg")
                continue

            # size each component so its vega_notional = index_vega_notional * weight
            comp_qty = (self.vega_notional_index * comp.weight) / (unit_v * book.spot * 2.0)

            for is_call in [True, False]:
                p = bsm_price(book.spot, comp_K, expiry, self.rate, comp.implied_vol, is_call)
                book.add_leg(OptionLeg(strike=comp_K, expiry=expiry, is_call=is_call,
                                       qty=comp_qty, entry_price=p))

            self.position.component_legs.setdefault(comp.symbol, []).extend(book.legs[-2:])

        self.position.entry_metrics = metrics
        self.position.is_active = True
        self.hedge_delta(sigma_index)
        logger.info("dispersion_entered", extra={"corr_premium": metrics.correlation_premium})

    def exit_dispersion(self, sigma_index: float, component_sigmas: Optional[dict[str, float]] = None) -> float:
        # generate_signals emits "close_dispersion" but nothing closed it, added for
        # symmetry with enter_dispersion so the strategy is actually closeable
        component_sigmas = component_sigmas or {}

        realized = 0.0
        while self.index_book.legs:
            leg = self.index_book.legs[0]
            realized += self.index_book.remove_leg(
                0, bsm_price(self.index_book.spot, leg.strike, leg.expiry, self.rate, sigma_index, leg.is_call))
        self.index_book.hedge_qty = 0.0

        for symbol, book in self.component_books.items():
            sig = component_sigmas.get(symbol, sigma_index)
            while book.legs:
                leg = book.legs[0]
                realized += book.remove_leg(
                    0, bsm_price(book.spot, leg.strike, leg.expiry, self.rate, sig, leg.is_call))
            book.hedge_qty = 0.0

        self.position = DispersionPosition()
        logger.info("dispersion_exited", extra={"realized_pnl": realized})
        return realized

    @staticmethod
    def _straddle_vega(spot: float, strike: float, expiry: float, sigma: float) -> float:
        _, _, vega, _, _ = bsm_greeks(spot, strike, expiry, 0.0, sigma, True)
        return vega

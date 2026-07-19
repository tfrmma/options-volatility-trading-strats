# Regression tests for the #7 dispersion redesign: composition instead of inheritance,
# one UnderlyingBook per underlying so component legs price off their own spot, not the
# index spot. Previously untested (and previously wrong).

import pytest

from strategies.dispersion import DispersionStrategy, ComponentSpec


def make_strategy() -> DispersionStrategy:
    components = [
        ComponentSpec(symbol="ETH", weight=0.6, spot=3000.0, implied_vol=0.7, realized_vol=0.5),
        ComponentSpec(symbol="SOL", weight=0.4, spot=150.0, implied_vol=0.9, realized_vol=0.6),
    ]
    return DispersionStrategy(
        index_spot=50000.0, components=components, rate=0.0,
        corr_premium_threshold=0.05, vega_notional_index=10000.0,
        taker_fee=0.0, maker_fee=0.0,
    )


def enter(strat: DispersionStrategy, sigma_index: float = 0.55) -> None:
    metrics = strat.compute_implied_correlation(sigma_index, list(strat.components_spec.values()))
    strat.enter_dispersion(expiry=0.25, metrics=metrics, sigma_index=sigma_index)


class TestComponentSpotIndependence:

    def test_books_keep_their_own_spot(self):
        strat = make_strategy()
        enter(strat)
        assert strat.index_book.spot == 50000.0
        assert strat.component_books["ETH"].spot == 3000.0
        assert strat.component_books["SOL"].spot == 150.0

    def test_strikes_scale_to_each_underlying_not_the_index(self):
        strat = make_strategy()
        enter(strat)
        for leg in strat.component_books["ETH"].legs:
            assert 2000 < leg.strike < 4000
        for leg in strat.component_books["SOL"].legs:
            assert 100 < leg.strike < 200
        for leg in strat.index_book.legs:
            assert 40000 < leg.strike < 60000


class TestIndependentHedging:

    def test_each_book_hedges_flat_against_its_own_spot(self):
        strat = make_strategy()
        enter(strat)
        for book in [strat.index_book, *strat.component_books.values()]:
            assert abs(book.portfolio_greeks().delta) < 1e-3


class TestMarkToMarketAggregation:

    def test_mtm_equals_sum_of_books(self):
        strat = make_strategy()
        enter(strat)
        component_sigmas = {"ETH": 0.7, "SOL": 0.9}
        expected = (
            strat.index_book.mark_to_market(0.55)
            + strat.component_books["ETH"].mark_to_market(0.7)
            + strat.component_books["SOL"].mark_to_market(0.9)
        )
        assert strat.mark_to_market(0.55, component_sigmas) == pytest.approx(expected)


class TestRiskCheckAggregation:

    def test_delta_usd_near_zero_after_hedging(self):
        strat = make_strategy()
        enter(strat)
        check = strat.risk_check(0.55, {"ETH": 0.7, "SOL": 0.9})
        assert abs(check["net_delta_usd"]) < 50.0  # near-flat in dollar terms across all 3 books
        assert check["vega_notional"] > 0.0


class TestExitDispersion:

    def test_exit_flattens_every_book(self):
        strat = make_strategy()
        enter(strat)
        strat.exit_dispersion(sigma_index=0.55, component_sigmas={"ETH": 0.7, "SOL": 0.9})

        assert strat.index_book.legs == []
        assert strat.index_book.hedge_qty == 0.0
        for book in strat.component_books.values():
            assert book.legs == []
            assert book.hedge_qty == 0.0
        assert strat.position.is_active is False

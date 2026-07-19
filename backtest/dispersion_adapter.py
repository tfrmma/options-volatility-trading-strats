# Bridges DispersionStrategy (signal generation + its own greeks/PnL bookkeeping) to
# BacktestEngine (real fills, cash, equity curve). Nothing else in this repo has an
# adapter like this yet, dispersion is the first strategy actually wired end to end
# through the engine rather than just unit-tested in isolation.
#
# Split: the strategy computes WHAT it wants (target legs, sizing, hedge target), this
# adapter submits that to the engine and gets back what ACTUALLY happened (real fill
# price, real fee), then feeds that back into the strategy's own books via
# record_fill() so its greeks/PnL are driven by real fills instead of assuming its own
# theoretical bsm_price. Mirrors how a real strategy/OMS split works.
#
# Scope: option legs go through real engine fills. The delta hedge stays on
# BaseVolStrategy's existing theoretical-spot hedge_delta(), spot execution nuance
# isn't the thing being proven here, the options wiring is. See CODE_REVIEW_STATUS.md.

import logging
from typing import Optional

from backtest.engine import BacktestEngine, MarketSnapshot
from strategies.base_strat import OptionLeg
from strategies.dispersion import DispersionStrategy, ComponentSpec, DispersionPosition

logger = logging.getLogger(__name__)


class DispersionBacktestAdapter:

    def __init__(self, strategy: DispersionStrategy, component_symbols: list[str]):
        self.strategy = strategy
        self.component_symbols = component_symbols
        self._last_snap: dict[str, MarketSnapshot] = {}

    def __call__(self, snap: MarketSnapshot, engine: BacktestEngine) -> list:
        strat = self.strategy
        book = strat.index_book if snap.symbol == strat.index_symbol else strat.component_books.get(snap.symbol)
        if book is not None:
            book.update_spot(snap.spot, snap.sigma)
        self._last_snap[snap.symbol] = snap

        # decision point: the index's put quote, always last in a timestep per
        # simulate_dispersion_feed's ordering convention, by then every component for
        # this timestep is already cached
        is_decision_point = (
            snap.symbol == strat.index_symbol and not snap.is_call
            and all(sym in self._last_snap for sym in self.component_symbols)
        )
        if is_decision_point:
            self._act(engine)
        return []   # this adapter fills directly via engine._fill, nothing to return

    def _act(self, engine: BacktestEngine) -> None:
        strat = self.strategy
        index_snap = self._last_snap[strat.index_symbol]

        components = []
        for sym in self.component_symbols:
            comp_snap = self._last_snap[sym]
            spec = strat.components_spec[sym]
            # simplified realized-vol proxy for this synthetic feed (scaled instantaneous
            # GARCH vol, not a real trailing estimator), tuned so the correlation premium
            # actually varies through the run instead of saturating at the realized_corr
            # clip. proving the wiring, not backtesting a real edge, see dispersion.py's
            # own TODO on this same proxy for the production version.
            components.append(ComponentSpec(
                symbol=sym, weight=spec.weight, spot=comp_snap.spot,
                implied_vol=comp_snap.sigma, realized_vol=comp_snap.sigma * 0.3,
            ))

        metrics = strat.compute_implied_correlation(index_snap.sigma, components)

        if not strat.position.is_active:
            if metrics.correlation_premium >= strat.corr_premium_threshold:
                self._enter(engine, index_snap, components, metrics)
            return

        if metrics.correlation_premium < strat.corr_premium_threshold * 0.3:
            self._exit(engine)
        else:
            component_sigmas = {c.symbol: c.implied_vol for c in components}
            strat.hedge_delta(index_snap.sigma, component_sigmas)

    def _enter(self, engine, index_snap, components, metrics) -> None:
        strat = self.strategy
        strat.enter_dispersion(expiry=index_snap.expiry, metrics=metrics, sigma_index=index_snap.sigma)
        # enter_dispersion just added legs at theoretical bsm prices to the strategy's
        # own books, replace those with what the engine actually fills them at
        self._reconcile(engine, strat.index_book, strat.index_symbol)
        for sym in self.component_symbols:
            self._reconcile(engine, strat.component_books[sym], sym)

        component_sigmas = {c.symbol: c.implied_vol for c in components}
        strat.hedge_delta(index_snap.sigma, component_sigmas)

    def _reconcile(self, engine: BacktestEngine, book, symbol: str) -> None:
        theoretical_legs = list(book.legs)
        book.legs = []
        for leg in theoretical_legs:
            order = {"type": "option", "side": "buy" if leg.qty > 0 else "sell",
                     "qty": abs(leg.qty), "strike": leg.strike, "expiry": leg.expiry,
                     "is_call": leg.is_call, "symbol": symbol}
            fill = engine._fill(order, self._last_snap[symbol])
            if fill is None:
                logger.warning(f"dispersion leg failed to fill: {symbol} K={leg.strike}")
                continue
            book.add_leg(OptionLeg(strike=leg.strike, expiry=leg.expiry, is_call=leg.is_call,
                                    qty=leg.qty, entry_price=fill.price))

    def _exit(self, engine: BacktestEngine) -> None:
        strat = self.strategy
        for symbol, book in [(strat.index_symbol, strat.index_book)] + \
                             [(s, strat.component_books[s]) for s in self.component_symbols]:
            while book.legs:
                leg = book.legs[0]
                order = {"type": "option", "side": "sell" if leg.qty > 0 else "buy",
                         "qty": abs(leg.qty), "strike": leg.strike, "expiry": leg.expiry,
                         "is_call": leg.is_call, "symbol": symbol}
                fill = engine._fill(order, self._last_snap[symbol])
                book.legs.pop(0)   # remove regardless, don't want to loop forever on a bad quote
                if fill is None:
                    logger.warning(f"dispersion exit leg failed to fill: {symbol} K={leg.strike}")
                    continue
                book.realized_pnl += leg.qty * (fill.price - leg.entry_price)
            book.hedge_qty = 0.0

        strat.position = DispersionPosition()

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
# Realized vol comes from core.estimators.close_to_close on each component's own
# trailing spot history, not a proxy, once there's enough history, see
# _MIN_RV_HISTORY. The delta hedge is routed through real engine spot fills too, not
# BaseVolStrategy's theoretical-spot hedge_delta().

import logging
from typing import Optional

import numpy as np

from backtest.engine import BacktestEngine, MarketSnapshot
from strategies.base_strat import OptionLeg
from strategies.dispersion import DispersionStrategy, ComponentSpec, DispersionPosition
from core.estimators import close_to_close

logger = logging.getLogger(__name__)

_MIN_RV_HISTORY = 10   # close_to_close needs a handful of returns before the estimate means anything


class DispersionBacktestAdapter:

    def __init__(self, strategy: DispersionStrategy, component_symbols: list[str],
                 rv_window: int = 20, rv_ann_factor: float = 252.0):
        self.strategy = strategy
        self.component_symbols = component_symbols
        self.rv_window = rv_window
        self.rv_ann_factor = rv_ann_factor   # matches the feed's step size, daily by default
        self._last_snap: dict[str, MarketSnapshot] = {}
        self._spot_history: dict[str, list[float]] = {sym: [] for sym in component_symbols}

    def __call__(self, snap: MarketSnapshot, engine: BacktestEngine) -> list:
        strat = self.strategy
        book = strat.index_book if snap.symbol == strat.index_symbol else strat.component_books.get(snap.symbol)
        if book is not None:
            book.update_spot(snap.spot, snap.sigma)
        self._last_snap[snap.symbol] = snap

        if snap.symbol in self._spot_history:
            hist = self._spot_history[snap.symbol]
            if not hist or hist[-1] != snap.spot:   # multiple rows per symbol per tick (call+put), don't double-append
                hist.append(snap.spot)
                if len(hist) > self.rv_window * 3:
                    del hist[: len(hist) - self.rv_window * 3]

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

    def _realized_vol(self, symbol: str, fallback_sigma: float) -> float:
        hist = self._spot_history.get(symbol, [])
        if len(hist) < _MIN_RV_HISTORY:
            # warm-up only: not enough trailing history yet for close_to_close to mean
            # anything, fall back to a scaled implied vol until the window fills
            return fallback_sigma * 0.3
        window = hist[-self.rv_window:]
        rv = close_to_close(np.array(window), ann_factor=self.rv_ann_factor)
        return float(rv) if np.isfinite(rv) else fallback_sigma * 0.3

    def _act(self, engine: BacktestEngine) -> None:
        strat = self.strategy
        index_snap = self._last_snap[strat.index_symbol]

        components = []
        for sym in self.component_symbols:
            comp_snap = self._last_snap[sym]
            spec = strat.components_spec[sym]
            components.append(ComponentSpec(
                symbol=sym, weight=spec.weight, spot=comp_snap.spot,
                implied_vol=comp_snap.sigma, realized_vol=self._realized_vol(sym, comp_snap.sigma),
            ))

        metrics = strat.compute_implied_correlation(index_snap.sigma, components)

        if not strat.position.is_active:
            if metrics.correlation_premium >= strat.corr_premium_threshold:
                self._enter(engine, index_snap, components, metrics)
            return

        if metrics.correlation_premium < strat.corr_premium_threshold * 0.3:
            self._exit(engine)
        else:
            self._hedge_all(engine, index_snap, components)

    def _enter(self, engine, index_snap, components, metrics) -> None:
        strat = self.strategy
        strat.enter_dispersion(expiry=index_snap.expiry, metrics=metrics, sigma_index=index_snap.sigma)
        # enter_dispersion just added legs at theoretical bsm prices to the strategy's
        # own books, replace those with what the engine actually fills them at
        self._reconcile(engine, strat.index_book, strat.index_symbol)
        for sym in self.component_symbols:
            self._reconcile(engine, strat.component_books[sym], sym)

        self._hedge_all(engine, index_snap, components)

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

    def _hedge_all(self, engine: BacktestEngine, index_snap, components) -> None:
        strat = self.strategy
        self._hedge_one(engine, strat.index_book, strat.index_symbol, index_snap.sigma)
        for c in components:
            self._hedge_one(engine, strat.component_books[c.symbol], c.symbol, c.implied_vol)

    def _hedge_one(self, engine: BacktestEngine, book, symbol: str, sigma: float) -> None:
        # same target-hedge math as BaseVolStrategy.hedge_delta, but the trade goes
        # through the engine's real spot bid/ask and real fee instead of a theoretical
        # self.spot fill, this is what makes the hedge, not just the option legs,
        # actually execution-realistic end to end
        greeks = book.portfolio_greeks(sigma)
        option_delta = greeks.delta - book.hedge_qty
        target_hedge = -option_delta
        trade_size   = target_hedge - book.hedge_qty
        if abs(trade_size) < 1e-6:
            return

        order = {"type": "spot", "side": "buy" if trade_size > 0 else "sell",
                 "qty": abs(trade_size), "symbol": symbol}
        fill = engine._fill(order, self._last_snap[symbol])
        if fill is None:
            logger.warning(f"{symbol} hedge failed to fill: trade_size={trade_size}")
            return

        book.hedge_qty = target_hedge
        book.pnl.transaction_costs -= fill.fee

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

            if book.hedge_qty != 0.0:
                order = {"type": "spot", "side": "sell" if book.hedge_qty > 0 else "buy",
                         "qty": abs(book.hedge_qty), "symbol": symbol}
                fill = engine._fill(order, self._last_snap[symbol])
                if fill is not None:
                    book.pnl.transaction_costs -= fill.fee
                book.hedge_qty = 0.0

        strat.position = DispersionPosition()

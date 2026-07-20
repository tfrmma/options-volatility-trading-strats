# Bridges single-underlying BaseVolStrategy subclasses (delta_neutral, vol_arb,
# surface_trading) to BacktestEngine. Lighter-weight than DispersionBacktestAdapter:
# instead of rewriting the strategy's own legs with real fill prices, this diffs the
# strategy's desired position (by strike/expiry/is_call) before and after its own
# decision call, and makes sure the ENGINE actually trades that delta at real bid/ask.
#
# The strategy's own legs/realized_pnl stay theoretical, exactly as already tested,
# they're "what does this strategy want to do next" state, not the backtest's
# authoritative PnL. engine.result is authoritative, same as it is for anything else
# running through BacktestEngine. This is a different (simpler) reconciliation
# philosophy than dispersion_adapter.py uses, dispersion's composed multi-book design
# made rewriting each book's legs with real prices the natural fit there; a single book
# here doesn't need that.
#
# One instance per strategy. All three (delta_neutral, vol_arb, surface_trading) fit
# this because they all: track everything through BaseVolStrategy's self.legs, trade a
# single underlying, and expose their own decision-making as a plain method the adapter
# calls without needing to know what's inside it.

import logging
from typing import Callable, Optional

from backtest.engine import BacktestEngine, MarketSnapshot

logger = logging.getLogger(__name__)


class SingleUnderlyingBacktestAdapter:

    def __init__(
        self,
        strategy,
        symbol: str,
        market_data_fn: Callable[[MarketSnapshot], Optional[dict]],
        decide_fn: Callable[[object, dict, float], None],
    ):
        # market_data_fn(snap) -> market_data dict for the strategy's generate_signals,
        # or None if this tick doesn't carry enough to decide anything (e.g. a rolling
        # RV window that isn't full yet). decide_fn(strategy, market_data, sigma) runs
        # whatever the concrete strategy's decision path is, strategy-specific, the
        # adapter doesn't guess at it.
        self.strategy = strategy
        self.symbol = symbol
        self.market_data_fn = market_data_fn
        self.decide_fn = decide_fn

    def __call__(self, snap: MarketSnapshot, engine: BacktestEngine) -> list:
        if snap.symbol != self.symbol:
            return []

        self.strategy.update_spot(snap.spot, snap.sigma)
        market_data = self.market_data_fn(snap)
        if market_data is None:
            return []

        before = self._leg_qty_by_key()
        self.decide_fn(self.strategy, market_data, snap.sigma)
        after = self._leg_qty_by_key()

        self._reconcile(engine, snap, before, after)
        return []

    def _leg_qty_by_key(self) -> dict:
        out = {}
        for leg in self.strategy.legs:
            key = (leg.strike, leg.expiry, leg.is_call)
            out[key] = out.get(key, 0.0) + leg.qty
        return out

    def _reconcile(self, engine: BacktestEngine, snap: MarketSnapshot, before: dict, after: dict) -> None:
        for key in set(before) | set(after):
            delta = after.get(key, 0.0) - before.get(key, 0.0)
            if abs(delta) < 1e-8:
                continue

            strike, expiry, is_call = key
            order = {
                "type": "option", "side": "buy" if delta > 0 else "sell",
                "qty": abs(delta), "strike": strike, "expiry": expiry,
                "is_call": is_call, "symbol": self.symbol,
            }
            fill = engine._fill(order, snap)
            if fill is None:
                logger.warning(f"{self.symbol} leg failed to fill: {key} delta={delta}")


def delta_neutral_decide(strategy, market_data: dict, sigma: float) -> None:
    for signal in strategy.generate_signals(market_data):
        strategy.execute_signal(signal, sigma)


def vol_arb_decide(strategy, market_data: dict, sigma: float) -> None:
    for signal in strategy.generate_signals(market_data):
        strategy.execute_signal(signal, sigma)


def surface_trading_decide(strategy, market_data: dict, sigma: float) -> None:
    # generate_signals returns generic action dicts, not a uniform Signal type, and
    # dispatches to enter_calendar/enter_risk_reversal directly, there's no
    # execute_signal to call polymorphically like the other two. surface_trading also
    # has no exit path at all, not even a manual one beyond the inherited close_all(),
    # generate_signals never emits anything like "close", so there's nothing for this
    # decide_fn to call for an exit, that's a gap in the strategy, not something this
    # adapter can paper over.
    for signal in strategy.generate_signals(market_data):
        if signal["action"] == "calendar":
            metrics = signal["metrics"]
            sigma_far = market_data.get("sigma_by_expiry", {}).get(metrics.far_expiry, sigma)
            strategy.enter_calendar(metrics, sigma_near=sigma, sigma_far=sigma_far,
                                     sell_near=signal["sell_near"])
        elif signal["action"] == "risk_reversal":
            expiry = signal["expiry"]
            atm_iv = strategy.surface.implied_vol(0.0, expiry)
            call_K = strategy._find_delta_strike(0.25, expiry, atm_iv, is_call=True)
            put_K  = strategy._find_delta_strike(0.25, expiry, atm_iv, is_call=False)
            strategy.enter_risk_reversal(expiry, call_K, put_K, sigma_call=sigma, sigma_put=sigma,
                                          sell_call=not signal["sell_puts"])

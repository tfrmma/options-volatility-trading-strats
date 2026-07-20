# End-to-end tests: delta_neutral, vol_arb, and surface_trading actually running
# through BacktestEngine via SingleUnderlyingBacktestAdapter, not just unit-tested
# against their own theoretical fills. Mirrors what test_dispersion_adapter.py already
# does for dispersion, this is the same thing for the other three.

import pytest
import polars as pl

from backtest.engine import BacktestEngine
from backtest.single_underlying_adapter import (
    SingleUnderlyingBacktestAdapter, delta_neutral_decide, vol_arb_decide, surface_trading_decide,
)
from strategies.delta_neutral import DeltaNeutralStrategy
from strategies.vol_arb import VolArbStrategy
from strategies.surface_trading import VolSurfaceTrading
from core.chain import OptionChain
from core.surface import VolSurface, SVIParams


def make_row(ts, spot, sigma, bid, ask, expiry, strike, is_call, opt_bid, opt_ask, symbol):
    return {
        "timestamp": ts, "spot": spot, "sigma": sigma, "bid": bid, "ask": ask,
        "expiry": expiry, "strike": strike, "is_call": is_call,
        "option_bid": opt_bid, "option_ask": opt_ask, "symbol": symbol,
    }


class TestDeltaNeutralEndToEnd:

    def test_entry_fills_real_orders_through_the_engine(self):
        strike, expiry = 100.0, 0.25
        chain = OptionChain(strikes_by_expiry={expiry: [strike]})
        strat = DeltaNeutralStrategy(spot=100.0, taker_fee=0.0006, maker_fee=-0.0001, chain=chain,
                                      vol_edge_threshold=0.03)

        rows = []
        for t in range(5):
            rows.append(make_row(t, 100.0, 0.8, 99.9, 100.1, expiry, strike, True, 8.0, 8.2, "BTC"))
            rows.append(make_row(t, 100.0, 0.8, 99.9, 100.1, expiry, strike, False, 7.5, 7.7, "BTC"))
        feed = pl.DataFrame(rows)

        def market_data_fn(snap):
            # implied 0.8 vs a fixed "realized" 0.5: vol_edge=0.3, comfortably over the
            # 0.03 threshold, entry fires on the very first decision tick
            return {"spot": snap.spot, "implied_vol": snap.sigma, "realized_vol_30d": 0.5,
                     "expiry": snap.expiry}

        adapter = SingleUnderlyingBacktestAdapter(strat, "BTC", market_data_fn, delta_neutral_decide)
        engine = BacktestEngine(taker_fee=0.0006, maker_fee=-0.0001, slippage_bps=1.0,
                                 initial_capital=1_000_000.0)
        result = engine.run(feed, adapter)

        assert len(engine.result.fills) > 0
        assert all(f.symbol == "BTC" for f in engine.result.fills)
        assert len(strat.legs) == 2   # one straddle: call + put


class TestVolArbEndToEnd:

    def test_entry_fills_real_orders_through_the_engine(self):
        strike, expiry = 100.0, 0.25
        chain = OptionChain(strikes_by_expiry={expiry: [strike]})
        strat = VolArbStrategy(spot=100.0, taker_fee=0.0006, maker_fee=-0.0001, chain=chain,
                                vrp_entry_zscore=1.0)
        strat._vrp_history = [0.0] * 10   # tight trailing history so the real vrp spikes the z-score

        rows = [make_row(0, 100.0, 0.9, 99.9, 100.1, expiry, strike, True, 9.0, 9.2, "BTC"),
                make_row(0, 100.0, 0.9, 99.9, 100.1, expiry, strike, False, 8.5, 8.7, "BTC")]
        feed = pl.DataFrame(rows)

        def market_data_fn(snap):
            return {"implied_vol": snap.sigma, "expiry": snap.expiry, "realized_vol_fallback": 0.3}

        adapter = SingleUnderlyingBacktestAdapter(strat, "BTC", market_data_fn, vol_arb_decide)
        engine = BacktestEngine(taker_fee=0.0006, maker_fee=-0.0001, slippage_bps=1.0,
                                 initial_capital=1_000_000.0)
        engine.run(feed, adapter)

        assert len(engine.result.fills) > 0
        assert all(f.symbol == "BTC" for f in engine.result.fills)
        assert strat._in_position is True


class TestSurfaceTradingEndToEnd:

    def test_calendar_entry_fills_real_orders_through_the_engine(self):
        near_T, far_T = 0.05, 0.5
        surface = VolSurface()
        # deliberately steep term structure so the near/far variance ratio deviates
        # hard from the flat-vol baseline and the calendar signal fires
        surface.add_slice(SVIParams(a=0.15, b=0.05, rho=0.0, m=0.0, sigma=0.1, expiry=near_T))
        surface.add_slice(SVIParams(a=0.03, b=0.05, rho=0.0, m=0.0, sigma=0.1, expiry=far_T))

        strike = 100.0
        chain = OptionChain(strikes_by_expiry={near_T: [strike], far_T: [strike]})
        strat = VolSurfaceTrading(spot=100.0, surface=surface, taker_fee=0.0006, maker_fee=-0.0001,
                                   chain=chain, calendar_threshold=0.05)

        rows = [
            make_row(0, 100.0, 0.5, 99.9, 100.1, near_T, strike, True,  6.0, 6.2, "BTC"),
            make_row(0, 100.0, 0.5, 99.9, 100.1, near_T, strike, False, 5.5, 5.7, "BTC"),
            make_row(0, 100.0, 0.3, 99.9, 100.1, far_T,  strike, True,  9.0, 9.2, "BTC"),
            make_row(0, 100.0, 0.3, 99.9, 100.1, far_T,  strike, False, 8.5, 8.7, "BTC"),
        ]
        feed = pl.DataFrame(rows)

        # surface_trading's calendar signal needs BOTH expiries' quotes cached before it
        # can safely enter (it's a 2-expiry trade), and generate_signals has no "already
        # in position" guard the way delta_neutral/vol_arb do, so deciding on every tick
        # both fires too early (missing quotes) and re-enters repeatedly. gate on the
        # last row, same convention test_dispersion_adapter.py uses for the same reason.
        calls = {"n": 0}

        def market_data_fn(snap):
            calls["n"] += 1
            if calls["n"] < len(rows):
                return None
            return {"expiries": [near_T, far_T], "sigma_by_expiry": {near_T: 0.5, far_T: 0.3}}

        adapter = SingleUnderlyingBacktestAdapter(strat, "BTC", market_data_fn, surface_trading_decide)
        engine = BacktestEngine(taker_fee=0.0006, maker_fee=-0.0001, slippage_bps=1.0,
                                 initial_capital=1_000_000.0)
        engine.run(feed, adapter)

        assert len(engine.result.fills) > 0
        assert all(f.symbol == "BTC" for f in engine.result.fills)
        assert len(strat.legs) == 4   # near call+put, far call+put

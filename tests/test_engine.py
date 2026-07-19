import pytest
import polars as pl

from backtest.engine import BacktestEngine


def make_row(ts, spot, sigma, bid, ask, expiry, strike, is_call, opt_bid, opt_ask):
    return {
        "timestamp": ts, "spot": spot, "sigma": sigma, "bid": bid, "ask": ask,
        "expiry": expiry, "strike": strike, "is_call": is_call,
        "option_bid": opt_bid, "option_ask": opt_ask,
    }


class TestMultiLegMarking:

    def test_each_leg_marked_at_its_own_last_quote(self):
        # straddle: tick 0 quotes the call, tick 1 quotes the put. before the fix, _mtm
        # would've marked BOTH legs using whichever quote came in on the current tick
        engine = BacktestEngine(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0, initial_capital=0.0)
        rows = [
            make_row(0, 100.0, 0.5, 99.9, 100.1, 0.25, 100.0, True,  5.0, 5.2),
            make_row(1, 100.0, 0.5, 99.9, 100.1, 0.25, 100.0, False, 4.0, 4.2),
        ]
        data = pl.DataFrame(rows)

        def strategy_fn(snap, eng):
            return [{"type": "option", "side": "buy", "qty": 1.0, "strike": snap.strike,
                     "expiry": snap.expiry, "is_call": snap.is_call}]

        result = engine.run(data, strategy_fn)
        final_equity = result.equity_curve[-1][1]

        cash_spent = -5.2 - 4.2                       # bought both legs at ask
        call_mtm   = 0.5 * (5.0 + 5.2)                 # marked at its OWN quote from tick 0
        put_mtm    = 0.5 * (4.0 + 4.2)                 # marked at its own quote from tick 1
        assert final_equity == pytest.approx(cash_spent + call_mtm + put_mtm)

    def test_fill_skipped_for_instrument_never_quoted(self):
        engine = BacktestEngine(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0, initial_capital=0.0)
        rows = [make_row(0, 100.0, 0.5, 99.9, 100.1, 0.25, 100.0, True, 5.0, 5.2)]
        data = pl.DataFrame(rows)

        def strategy_fn(snap, eng):
            # order for a strike that's never been quoted on this feed
            return [{"type": "option", "side": "buy", "qty": 1.0, "strike": 999.0,
                     "expiry": snap.expiry, "is_call": True}]

        result = engine.run(data, strategy_fn)
        assert result.fills == []
        assert engine.current_position()["options"] == {}


class TestSharpeAnnualization:

    def test_tick_data_gets_much_higher_ann_factor_than_daily(self):
        tick_engine  = BacktestEngine(initial_capital=0.0)
        daily_engine = BacktestEngine(initial_capital=0.0)

        # same pnl shape, different timestamp spacing: 1 second vs 1 day
        for i in range(10):
            tick_engine.result.equity_curve.append((i * 1.0, float(i)))
            tick_engine.result.pnl_series.append(float(i))
            daily_engine.result.equity_curve.append((i * 86400.0, float(i)))
            daily_engine.result.pnl_series.append(float(i))

        assert tick_engine.result._infer_ann_factor() > daily_engine.result._infer_ann_factor() * 1000

    def test_daily_spacing_infers_roughly_252ish_scale(self):
        engine = BacktestEngine(initial_capital=0.0)
        for i in range(10):
            engine.result.equity_curve.append((i * 86400.0, float(i)))
            engine.result.pnl_series.append(float(i))
        # 365.25 calendar days/year, not 252 trading days, close enough to the old
        # hardcoded default that nothing silently jumps by 10x for daily data
        assert 300 < engine.result._infer_ann_factor() < 370

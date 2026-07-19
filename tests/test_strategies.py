# Regression tests for the strategy-layer bugs found in code review:
#   - mark_to_market() silently dropping hedge PnL
#   - _close_all/_flatten double-booking hedge notional into realized_pnl
#   - _find_delta_strike() never converging for puts
#
# This layer had zero coverage before, these are targeted, not exhaustive.

import numpy as np
import pytest

from core.pricer import bsm_greeks
from core.surface import VolSurface, SVIParams
from strategies.delta_neutral import DeltaNeutralStrategy
from strategies.vol_arb import VolArbStrategy
from strategies.surface_trading import VolSurfaceTrading


class TestMarkToMarketHedgePnl:

    def test_mtm_reflects_hedge_pnl_with_no_option_legs(self):
        strat = DeltaNeutralStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0)
        strat.hedge_qty = 2.0
        strat.update_spot(110.0, sigma=0.5)  # 2 * (110 - 100) = 20

        assert strat.pnl.delta_pnl == pytest.approx(20.0)
        assert strat.mark_to_market(sigma=0.5) == pytest.approx(20.0)

    def test_mtm_matches_manual_sum_with_legs(self):
        from strategies.base_strat import OptionLeg
        strat = DeltaNeutralStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0)
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=True, qty=1.0, entry_price=5.0))
        strat.hedge_qty = -0.5
        strat.update_spot(105.0, sigma=0.4)

        from core.pricer import bsm_price
        option_val = bsm_price(105.0, 100.0, 0.25, 0.0, 0.4, True) - 5.0
        expected = option_val + strat.realized_pnl + strat.pnl.delta_pnl + strat.pnl.transaction_costs
        assert strat.mark_to_market(sigma=0.4) == pytest.approx(expected)


class TestCloseDoesNotDoubleCount:

    def test_delta_neutral_close_all_no_phantom_pnl(self):
        strat = DeltaNeutralStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0)
        strat.hedge_qty = 5.0
        strat.update_spot(105.0, sigma=0.5)  # delta_pnl = 25, no legs to close

        strat._close_all(sigma=0.5)

        assert strat.hedge_qty == 0.0
        assert strat.realized_pnl == pytest.approx(0.0)
        assert strat.mark_to_market(sigma=0.5) == pytest.approx(25.0)

    def test_vol_arb_flatten_no_phantom_pnl(self):
        strat = VolArbStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0)
        strat.hedge_qty = -3.0
        strat.update_spot(90.0, sigma=0.5)  # delta_pnl = -3 * (90 - 100) = 30

        strat._flatten(sigma=0.5)

        assert strat.hedge_qty == 0.0
        assert strat.realized_pnl == pytest.approx(0.0)
        assert strat.mark_to_market(sigma=0.5) == pytest.approx(30.0)


class TestFindDeltaStrikeConverges:

    @pytest.fixture
    def surface(self):
        s = VolSurface()
        s.add_slice(SVIParams(a=0.05, b=0.10, rho=-0.2, m=0.0, sigma=0.10, expiry=0.25))
        return s

    def test_put_strike_converges_to_target_delta(self, surface):
        strat = VolSurfaceTrading(spot=100.0, surface=surface, rate=0.0)
        put_K = strat._find_delta_strike(0.25, 0.25, atm_iv=0.3, is_call=False)

        # must not be pinned at the search bracket edge (the bug returned spot*2.0)
        assert strat.spot * 0.5 < put_K < strat.spot * 1.5

        iv_at_k = surface.implied_vol(np.log(put_K / strat.spot), 0.25)
        delta, *_ = bsm_greeks(strat.spot, put_K, 0.25, 0.0, iv_at_k, False)
        assert abs(delta) == pytest.approx(0.25, abs=1e-3)

    def test_call_and_put_strikes_bracket_spot(self, surface):
        strat = VolSurfaceTrading(spot=100.0, surface=surface, rate=0.0)
        call_K = strat._find_delta_strike(0.25, 0.25, atm_iv=0.3, is_call=True)
        put_K  = strat._find_delta_strike(0.25, 0.25, atm_iv=0.3, is_call=False)

        assert put_K < strat.spot < call_K


class TestVegaRebalancing:

    def test_hold_trims_position_when_target_shrinks(self):
        strat = VolArbStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0, vega_rebalance_tol=0.10)
        from strategies.base_strat import OptionLeg
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=True,  qty=-10.0, entry_price=5.0))
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=False, qty=-10.0, entry_price=5.0))
        strat._in_position = True

        from strategies.vol_arb import VRPSignal, VRPMetrics
        metrics = VRPMetrics(implied_vol=0.5, realized_vol=0.4, vrp=0.1, vrp_zscore=0.3, vrp_percentile=0.6)
        # target vega notional deliberately far below current, should trigger a trim
        current_vega_notional = abs(strat.portfolio_greeks(0.5).vega) * strat.spot
        signal = VRPSignal("hold", 0.25, target_vega=current_vega_notional * 0.5, metrics=metrics, confidence=1.0)

        strat.execute_signal(signal, sigma=0.5)

        new_vega_notional = abs(strat.portfolio_greeks(0.5).vega) * strat.spot
        assert new_vega_notional < current_vega_notional
        assert new_vega_notional == pytest.approx(current_vega_notional * 0.5, rel=0.05)
        assert strat.realized_pnl != 0.0  # trimming realizes PnL on the closed slice

    def test_hold_adds_to_position_when_target_grows(self):
        strat = VolArbStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0, vega_rebalance_tol=0.10)
        from strategies.base_strat import OptionLeg
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=True,  qty=-10.0, entry_price=5.0))
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=False, qty=-10.0, entry_price=5.0))
        strat._in_position = True

        from strategies.vol_arb import VRPSignal, VRPMetrics
        metrics = VRPMetrics(implied_vol=0.5, realized_vol=0.4, vrp=0.1, vrp_zscore=0.9, vrp_percentile=0.9)
        current_vega_notional = abs(strat.portfolio_greeks(0.5).vega) * strat.spot
        signal = VRPSignal("hold", 0.25, target_vega=current_vega_notional * 1.5, metrics=metrics, confidence=1.0)

        n_legs_before = len(strat.legs)
        strat.execute_signal(signal, sigma=0.5)

        new_vega_notional = abs(strat.portfolio_greeks(0.5).vega) * strat.spot
        assert new_vega_notional > current_vega_notional
        assert len(strat.legs) > n_legs_before  # added new legs at the existing strikes

    def test_small_drift_within_tolerance_does_not_trade(self):
        strat = VolArbStrategy(spot=100.0, taker_fee=0.0, maker_fee=0.0, vega_rebalance_tol=0.50)
        from strategies.base_strat import OptionLeg
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=True,  qty=-10.0, entry_price=5.0))
        strat.legs.append(OptionLeg(strike=100.0, expiry=0.25, is_call=False, qty=-10.0, entry_price=5.0))
        strat._in_position = True

        from strategies.vol_arb import VRPSignal, VRPMetrics
        metrics = VRPMetrics(implied_vol=0.5, realized_vol=0.4, vrp=0.1, vrp_zscore=0.3, vrp_percentile=0.6)
        current_vega_notional = abs(strat.portfolio_greeks(0.5).vega) * strat.spot
        # only a 10% drift, tolerance is 50%, should be a no-op
        signal = VRPSignal("hold", 0.25, target_vega=current_vega_notional * 1.1, metrics=metrics, confidence=1.0)

        n_legs_before = len(strat.legs)
        strat.execute_signal(signal, sigma=0.5)
        assert len(strat.legs) == n_legs_before
        assert strat.realized_pnl == 0.0

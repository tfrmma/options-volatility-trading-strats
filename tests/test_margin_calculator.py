import pytest

from backtest.margin_calculator import (
    compute_portfolio_margin, liquidation_check, max_position_size, OptionPosition,
)


class TestPortfolioMargin:

    def test_naked_short_call_has_positive_worst_case_loss(self):
        # naked short call, unlimited upside risk, worst case must be a large upward
        # spot shock and a nonzero loss
        positions = [OptionPosition(strike=100.0, expiry=0.25, is_call=True, qty=-1.0, current_iv=0.6)]
        margin = compute_portfolio_margin(positions, spot_position=0.0, spot=100.0)

        assert margin.worst_case_loss > 0.0
        assert margin.worst_scenario[0] > 0.0  # worst case is an up-move for a short call
        assert margin.initial_margin > margin.maintenance_margin  # im_multiplier > mm_multiplier by default

    def test_delta_hedged_position_has_lower_margin_than_naked(self):
        # same short call, but delta hedged with long spot, should stress-test to a
        # meaningfully smaller worst-case loss than the naked version
        naked = [OptionPosition(strike=100.0, expiry=0.25, is_call=True, qty=-1.0, current_iv=0.6)]
        naked_margin = compute_portfolio_margin(naked, spot_position=0.0, spot=100.0)

        # roughly delta-hedge with 0.5 units long spot (near-ATM call delta is ~0.5-0.6)
        hedged_margin = compute_portfolio_margin(naked, spot_position=0.55, spot=100.0)

        assert hedged_margin.worst_case_loss < naked_margin.worst_case_loss

    def test_flat_book_has_zero_margin(self):
        margin = compute_portfolio_margin([], spot_position=0.0, spot=100.0)
        assert margin.worst_case_loss == 0.0
        assert margin.worst_scenario == (0.0, 0.0)


class TestLiquidationCheck:

    def test_healthy_account_no_margin_call(self):
        positions = [OptionPosition(strike=100.0, expiry=0.25, is_call=True, qty=-1.0, current_iv=0.6)]
        margin = compute_portfolio_margin(positions, spot_position=0.0, spot=100.0)
        check = liquidation_check(equity=margin.maintenance_margin * 2.0, margin=margin)

        assert check["margin_call"] == False
        assert check["liquidation"] == False

    def test_below_maintenance_triggers_margin_call_not_liquidation(self):
        positions = [OptionPosition(strike=100.0, expiry=0.25, is_call=True, qty=-1.0, current_iv=0.6)]
        margin = compute_portfolio_margin(positions, spot_position=0.0, spot=100.0)
        check = liquidation_check(equity=margin.maintenance_margin * 0.9, margin=margin)

        assert check["margin_call"] == True
        assert check["liquidation"] == False

    def test_below_80pct_maintenance_triggers_liquidation(self):
        positions = [OptionPosition(strike=100.0, expiry=0.25, is_call=True, qty=-1.0, current_iv=0.6)]
        margin = compute_portfolio_margin(positions, spot_position=0.0, spot=100.0)
        check = liquidation_check(equity=margin.maintenance_margin * 0.5, margin=margin)

        assert check["liquidation"] == True


class TestMaxPositionSize:

    def test_basic_sizing(self):
        size = max_position_size(equity=100_000.0, target_margin_util=0.5, unit_margin=1000.0)
        assert size == pytest.approx(50.0)

    def test_zero_unit_margin_returns_zero_not_divide_by_zero(self):
        assert max_position_size(equity=100_000.0, target_margin_util=0.5, unit_margin=0.0) == 0.0

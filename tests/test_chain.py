import pytest

from core.chain import OptionChain, target_strike, round_to_synthetic_grid
from strategies.delta_neutral import DeltaNeutralStrategy


class TestOptionChain:

    def test_nearest_strike_picks_closest(self):
        chain = OptionChain(strikes_by_expiry={0.25: [95.0, 100.0, 105.0, 110.0]})
        assert chain.nearest_strike(0.25, 101.0) == 100.0
        assert chain.nearest_strike(0.25, 104.0) == 105.0

    def test_missing_expiry_raises(self):
        chain = OptionChain(strikes_by_expiry={0.25: [100.0]})
        with pytest.raises(ValueError):
            chain.nearest_strike(0.5, 100.0)

    def test_from_records_groups_by_expiry(self):
        records = [
            {"strike": 100.0, "expiry": 0.25}, {"strike": 105.0, "expiry": 0.25},
            {"strike": 200.0, "expiry": 0.5},
        ]
        chain = OptionChain.from_records(records)
        assert chain.strikes_by_expiry[0.25] == [100.0, 105.0]
        assert chain.strikes_by_expiry[0.5] == [200.0]

    def test_target_strike_uses_chain_when_given(self):
        chain = OptionChain(strikes_by_expiry={0.25: [95.0, 103.0]})
        assert target_strike(100.0, 0.25, chain) == 103.0

    def test_target_strike_falls_back_to_synthetic_grid_without_a_chain(self):
        assert target_strike(50123.0, 0.25, None) == round_to_synthetic_grid(50123.0)


class TestDeltaNeutralUsesRealChain:

    def test_entry_snaps_to_listed_strike_not_synthetic_grid(self):
        # spot is 50123, the synthetic grid would round to 50000 (nearest 1000 above 10k),
        # but the "real" chain only lists 50500, that's what should actually get traded
        chain = OptionChain(strikes_by_expiry={0.25: [49500.0, 50500.0, 51500.0]})
        strat = DeltaNeutralStrategy(spot=50123.0, taker_fee=0.0, maker_fee=0.0, chain=chain)

        signals = strat.generate_signals({
            "spot": 50123.0, "implied_vol": 0.8, "realized_vol_30d": 0.5, "expiry": 0.25,
        })
        assert len(signals) == 1
        assert signals[0].atm_strike == 50500.0

    def test_no_chain_falls_back_to_synthetic_grid(self):
        strat = DeltaNeutralStrategy(spot=50123.0, taker_fee=0.0, maker_fee=0.0)
        signals = strat.generate_signals({
            "spot": 50123.0, "implied_vol": 0.8, "realized_vol_30d": 0.5, "expiry": 0.25,
        })
        assert signals[0].atm_strike == round_to_synthetic_grid(50123.0)

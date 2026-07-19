import numpy as np
import pytest

from backtest.market_sim import SimConfig, simulate_market, to_ohlcv, generate_option_chain
from core.pricer import bsm_price


class TestSimulateMarket:

    def test_seed_is_reproducible(self):
        cfg = SimConfig(n_steps=50)
        steps_a = simulate_market(cfg, seed=42)
        steps_b = simulate_market(cfg, seed=42)
        assert [s.spot for s in steps_a] == [s.spot for s in steps_b]

    def test_different_seeds_diverge(self):
        cfg = SimConfig(n_steps=50)
        steps_a = simulate_market(cfg, seed=1)
        steps_b = simulate_market(cfg, seed=2)
        assert [s.spot for s in steps_a] != [s.spot for s in steps_b]

    def test_step_count_matches_config(self):
        cfg = SimConfig(n_steps=100)
        steps = simulate_market(cfg, seed=1)
        assert len(steps) == 100

    def test_vol_never_goes_negative_or_nan(self):
        # GARCH variance is floored at 1e-8, sqrt of that must always be real and positive
        cfg = SimConfig(n_steps=500, garch_alpha=0.3, garch_beta=0.69)  # high persistence, stress it
        steps = simulate_market(cfg, seed=7)
        vols = np.array([s.vol for s in steps])
        assert np.all(vols > 0)
        assert not np.any(np.isnan(vols))

    def test_bid_below_spot_below_ask(self):
        cfg = SimConfig(n_steps=200)
        steps = simulate_market(cfg, seed=3)
        for s in steps:
            assert s.bid < s.spot < s.ask

    def test_zero_toxic_prob_means_no_toxic_prints(self):
        cfg = SimConfig(n_steps=200, toxic_prob=0.0)
        steps = simulate_market(cfg, seed=5)
        assert not any(s.is_toxic for s in steps)

    def test_zero_jump_intensity_means_no_jumps(self):
        cfg = SimConfig(n_steps=200, jump_intensity=0.0)
        steps = simulate_market(cfg, seed=5)
        assert not any(s.jump_occurred for s in steps)

    def test_first_step_spread_matches_base_spread_formula(self):
        # spread_mult is 1.0 on the very first step (vol_t starts at exactly base_vol,
        # before any GARCH update), so the spread there should match the base formula
        # with no vol-widening applied, easy to check against a closed form
        cfg = SimConfig(n_steps=1, base_spread_bps=5.0, spread_vol_sensitivity=2.0, toxic_prob=0.0)
        step = simulate_market(cfg, seed=1)[0]
        expected_half_spread = step.spot * cfg.base_spread_bps * 1e-4
        assert (step.ask - step.spot) == pytest.approx(expected_half_spread, rel=1e-9)
        assert (step.spot - step.bid) == pytest.approx(expected_half_spread, rel=1e-9)


class TestToOhlcv:

    def test_bar_size_one_is_identity(self):
        cfg = SimConfig(n_steps=20)
        steps = simulate_market(cfg, seed=1)
        bars = to_ohlcv(steps, bar_size=1)
        spots = np.array([s.spot for s in steps])
        assert np.allclose(bars["open"], spots)
        assert np.allclose(bars["close"], spots)
        assert np.allclose(bars["high"], spots)
        assert np.allclose(bars["low"], spots)

    def test_bar_aggregation_is_correct(self):
        cfg = SimConfig(n_steps=20)
        steps = simulate_market(cfg, seed=1)
        bar_size = 4
        bars = to_ohlcv(steps, bar_size=bar_size)

        assert len(bars["open"]) == 20 // bar_size
        for i in range(len(bars["open"])):
            bar = steps[i * bar_size:(i + 1) * bar_size]
            assert bars["open"][i]  == bar[0].spot
            assert bars["close"][i] == bar[-1].spot
            assert bars["high"][i]  == max(s.spot for s in bar)
            assert bars["low"][i]   == min(s.spot for s in bar)

    def test_remainder_steps_are_dropped_not_padded(self):
        steps = simulate_market(SimConfig(n_steps=10), seed=1)
        bars = to_ohlcv(steps, bar_size=3)
        assert len(bars["open"]) == 3  # 10 // 3, the trailing partial bar is dropped


class TestGenerateOptionChain:

    def test_record_count_matches_expiries_strikes_and_both_sides(self):
        chain = generate_option_chain(
            spot=100.0, expiries=[0.1, 0.25], vol_surface_fn=lambda lk, T: 0.5, n_strikes=11,
        )
        assert len(chain) == 2 * 11 * 2  # expiries * strikes * (call, put)

    def test_atm_strike_is_close_to_spot(self):
        chain = generate_option_chain(
            spot=100.0, expiries=[0.25], vol_surface_fn=lambda lk, T: 0.5, n_strikes=11,
        )
        strikes = sorted({r["strike"] for r in chain})
        atm = min(strikes, key=lambda k: abs(k - 100.0))
        assert atm == pytest.approx(100.0, rel=1e-6)  # linspace(-w, w, odd n) includes 0 exactly

    def test_bid_never_exceeds_ask_and_never_negative(self):
        chain = generate_option_chain(
            spot=100.0, expiries=[0.05, 0.5], vol_surface_fn=lambda lk, T: 0.8, n_strikes=15,
        )
        for r in chain:
            assert r["bid"] <= r["ask"]
            assert r["bid"] >= 0.0

    def test_prices_match_bsm_price_at_the_quoted_iv(self):
        vol_fn = lambda lk, T: 0.4
        chain = generate_option_chain(
            spot=100.0, expiries=[0.25], vol_surface_fn=vol_fn, n_strikes=5, spread_bps=20.0,
        )
        for r in chain:
            mid = bsm_price(100.0, r["strike"], r["expiry"], 0.0, 0.4, r["is_call"])
            quoted_mid = 0.5 * (r["bid"] + r["ask"])
            assert quoted_mid == pytest.approx(mid, rel=1e-6)

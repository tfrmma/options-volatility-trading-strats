# Math layer tests. If the greeks are wrong everything downstream is wrong.
# pytest tests/ -v

import pytest
import numpy as np
from core.pricer import bsm_price, bsm_greeks, implied_vol, batch_price, batch_greeks
from core.estimators import yang_zhang, garman_klass, close_to_close, realized_kernel
from core.surface import SVIParams, fit_svi_slice


# ── BSM ───────────────────────────────────────────────────────────────────────

class TestBSMBoundaries:

    def test_atm_straddle_nonnegative(self):
        call = bsm_price(100, 100, 1/12, 0.0, 0.80, True)
        put  = bsm_price(100, 100, 1/12, 0.0, 0.80, False)
        assert call >= 0 and put >= 0

    def test_put_call_parity(self):
        S, K, T, r, sigma = 50_000, 48_000, 30/365, 0.0, 0.65
        call = bsm_price(S, K, T, r, sigma, True)
        put  = bsm_price(S, K, T, r, sigma, False)
        assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-6

    def test_deep_itm_call_intrinsic(self):
        price = bsm_price(100, 1, 0.001, 0.0, 0.01, True)
        assert abs(price - 99.0) < 0.01

    def test_expiry_zero_intrinsic(self):
        assert bsm_price(100, 90, 0.0, 0.0, 0.5, True)  == pytest.approx(10.0, abs=1e-6)
        assert bsm_price(100, 90, 0.0, 0.0, 0.5, False) == pytest.approx(0.0,  abs=1e-6)

    def test_zero_vol_intrinsic(self):
        assert bsm_price(100, 90,  1.0, 0.0, 0.0, True)  == pytest.approx(10.0, abs=1e-6)
        assert bsm_price(100, 110, 1.0, 0.0, 0.0, True)  == pytest.approx(0.0,  abs=1e-6)


class TestGreeks:

    def test_call_delta_in_01(self):
        d, _, _, _, _ = bsm_greeks(100, 100, 0.25, 0.0, 0.5, True)
        assert 0 < d < 1

    def test_put_delta_in_neg1_0(self):
        d, _, _, _, _ = bsm_greeks(100, 100, 0.25, 0.0, 0.5, False)
        assert -1 < d < 0

    def test_gamma_positive(self):
        for is_call in [True, False]:
            _, g, _, _, _ = bsm_greeks(100, 100, 0.25, 0.0, 0.5, is_call)
            assert g >= 0

    def test_vega_positive(self):
        for is_call in [True, False]:
            _, _, v, _, _ = bsm_greeks(100, 100, 0.25, 0.0, 0.5, is_call)
            assert v >= 0

    def test_theta_negative_long(self):
        _, _, _, t, _ = bsm_greeks(100, 100, 0.25, 0.0, 0.5, True)
        assert t < 0

    def test_call_put_delta_parity(self):
        # call_delta - put_delta = 1 with r=0 (put-call delta parity)
        dc, _, _, _, _ = bsm_greeks(100, 100, 0.5, 0.0, 0.4, True)
        dp, _, _, _, _ = bsm_greeks(100, 100, 0.5, 0.0, 0.4, False)
        assert abs(dc - dp - 1.0) < 1e-8

    def test_greeks_at_expiry_edge(self):
        d, g, v, t, r = bsm_greeks(100, 100, 1e-15, 0.0, 0.5, True)
        assert all(np.isfinite(x) for x in [d, g, v, t, r])


class TestImpliedVol:

    def test_round_trip(self):
        S, K, T, r = 100, 95, 0.5, 0.02
        for sigma_true in [0.20, 0.50, 0.80, 1.50]:
            price = bsm_price(S, K, T, r, sigma_true, True)
            sigma_rec = implied_vol(price, S, K, T, r, True)
            assert abs(sigma_rec - sigma_true) < 1e-5, f"round-trip failed at sigma={sigma_true}"

    def test_arb_price_nan(self):
        assert np.isnan(implied_vol(-1.0, 100, 100, 0.5, 0.0, True))

    def test_zero_time_nan(self):
        assert np.isnan(implied_vol(5.0, 100, 100, 0.0, 0.0, True))


class TestBatchPricer:

    def test_batch_matches_scalar(self):
        n = 20
        S     = np.full(n, 50_000.0)
        K     = np.linspace(40_000, 60_000, n)
        T     = np.full(n, 30/365)
        r     = np.zeros(n)
        sigma = np.full(n, 0.70)
        ic    = np.ones(n, dtype=bool)

        batch  = batch_price(S, K, T, r, sigma, ic)
        scalar = np.array([bsm_price(S[i], K[i], T[i], r[i], sigma[i], True) for i in range(n)])
        np.testing.assert_allclose(batch, scalar, rtol=1e-6)


# ── Estimators ────────────────────────────────────────────────────────────────

class TestVolEstimators:

    @pytest.fixture
    def ohlcv(self):
        rng = np.random.default_rng(42)
        n = 100
        prices = 50000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        opens  = prices
        closes = prices * np.exp(rng.normal(0, 0.005, n))
        highs  = np.maximum(opens, closes) * (1 + abs(rng.normal(0, 0.005, n)))
        lows   = np.minimum(opens, closes) * (1 - abs(rng.normal(0, 0.005, n)))
        return opens, highs, lows, closes

    def test_yz_in_plausible_range(self, ohlcv):
        opens, highs, lows, closes = ohlcv
        yz = yang_zhang(opens, highs, lows, closes)
        assert 0.1 < yz < 1.0, f"YZ looks wrong: {yz:.4f}"

    def test_gk_positive(self, ohlcv):
        opens, highs, lows, closes = ohlcv
        assert garman_klass(opens, highs, lows, closes) > 0

    def test_yz_c2c_in_same_ballpark(self, ohlcv):
        opens, highs, lows, closes = ohlcv
        yz  = yang_zhang(opens, highs, lows, closes)
        c2c = close_to_close(closes)
        assert abs(yz - c2c) / c2c < 0.5   # shouldn't be wildly different on same data

    def test_rk_nonnegative(self):
        rk = realized_kernel(np.random.default_rng(0).normal(0, 0.001, 500))
        assert rk >= 0


# ── SVI ───────────────────────────────────────────────────────────────────────

class TestSVI:

    def test_round_trip(self):
        true_p = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.0, sigma=0.1, expiry=0.25)
        k      = np.linspace(-0.5, 0.5, 15)
        w_true = true_p.total_variance(k)
        w_noisy = w_true + np.random.default_rng(7).normal(0, 1e-5, len(k))

        fitted = fit_svi_slice(k, w_noisy, 0.25)
        rmse   = np.sqrt(np.mean((fitted.total_variance(k) - w_true)**2))
        assert rmse < 1e-3, f"SVI fit RMSE too high: {rmse:.6f}"

    def test_validity_check(self):
        assert SVIParams(a=0.04, b=0.10, rho=-0.2, m=0.0, sigma=0.05, expiry=0.5).is_valid()
        assert not SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.05).is_valid()

    def test_total_var_nonneg(self):
        p = SVIParams(a=0.04, b=0.10, rho=-0.3, m=0.0, sigma=0.08, expiry=0.5)
        w = p.total_variance(np.linspace(-2.0, 2.0, 100))
        assert np.all(w >= 0)

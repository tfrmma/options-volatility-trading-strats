import numpy as np
import pytest

from core.surface import SVIParams, fit_svi_slice, calibrate_surface, VolSurface


class TestCalendarArbConstraint:

    def test_unconstrained_fit_can_violate_calendar_but_constrained_fit_cannot(self):
        # this is the actual bug the TODO was about: an independent per-slice fit has
        # no idea the previous tenor exists, so nothing stops it from producing total
        # variance BELOW the near tenor's at some strikes, a real calendar arb.
        # constructs adversarial "market" data for the far tenor that's deliberately
        # lower than the near tenor almost everywhere, then checks the constrained fit
        # refuses to reproduce that violation while the unconstrained one does.
        k = np.linspace(-0.5, 0.5, 15)

        near_true = SVIParams(a=0.02, b=0.6, rho=-0.3, m=0.0, sigma=0.15, expiry=0.05)
        w_near = near_true.total_variance(k)
        near_fit = fit_svi_slice(k, w_near, expiry=0.05, n_restarts=5)

        w_far_market = w_near.copy() * 0.7        # scaled down everywhere: guaranteed violation
        mid = len(k) // 2
        w_far_market[mid] = w_near[mid] * 1.05     # except right at the money

        unconstrained_far = fit_svi_slice(k, w_far_market, expiry=0.25, n_restarts=5)
        constrained_far   = fit_svi_slice(k, w_far_market, expiry=0.25, n_restarts=5, prev_slice=near_fit)

        w_near_check = near_fit.total_variance(k)
        unconstrained_violations = np.sum(unconstrained_far.total_variance(k) < w_near_check - 1e-8)
        constrained_violations   = np.sum(constrained_far.total_variance(k) < w_near_check - 1e-8)

        assert unconstrained_violations > 0   # confirms this scenario actually tests something
        assert constrained_violations == 0

    def test_constrained_fit_still_fits_reasonably_on_clean_data(self):
        # the constraint shouldn't distort a calibration that never needed it, calendar
        # arb-free market data should fit about as well with or without prev_slice
        k = np.linspace(-0.5, 0.5, 15)

        near_true = SVIParams(a=0.01, b=0.3, rho=-0.2, m=0.0, sigma=0.15, expiry=0.05)
        far_true  = SVIParams(a=0.04, b=0.3, rho=-0.2, m=0.0, sigma=0.15, expiry=0.25)

        near_fit = fit_svi_slice(k, near_true.total_variance(k), expiry=0.05, n_restarts=5)
        far_fit  = fit_svi_slice(k, far_true.total_variance(k), expiry=0.25, n_restarts=5, prev_slice=near_fit)

        fit_error = np.mean((far_fit.total_variance(k) - far_true.total_variance(k)) ** 2)
        assert fit_error < 1e-4

    def test_flat_fallback_respects_the_calendar_floor_too(self):
        # even the degraded fallback path (calibration totally fails) shouldn't
        # trivially violate the constraint it was trying to satisfy. force the
        # "calibration totally failed" branch with a prev_slice floor high enough that
        # no feasible SVI params can clear it for this data
        k = np.linspace(-0.5, 0.5, 5)
        huge_prev = SVIParams(a=50.0, b=0.0, rho=0.0, m=0.0, sigma=1.0, expiry=0.05)
        low_market_w = np.full_like(k, 0.01)

        with pytest.warns(UserWarning):
            result = fit_svi_slice(k, low_market_w, expiry=0.25, n_restarts=5, prev_slice=huge_prev)

        assert np.all(result.total_variance(k) >= huge_prev.total_variance(k) - 1e-6)


class TestCalibrateSurfaceOrdering:

    def test_slices_end_up_sorted_regardless_of_groupby_order(self):
        import polars as pl
        from core.pricer import bsm_price

        # build a tiny synthetic chain with expiries in a deliberately scrambled order
        rows = []
        for T in [0.5, 0.05, 0.25]:
            for k in np.linspace(-0.3, 0.3, 7):
                K = 100.0 * np.exp(k)
                iv = 0.5 + 0.1 * T   # mild upward term structure, no calendar arb by construction
                mid = bsm_price(100.0, K, T, 0.0, iv, True)
                rows.append({"strike": K, "expiry": T, "bid": mid * 0.99, "ask": mid * 1.01,
                             "is_call": True, "spot": 100.0, "rate": 0.0})

        chain_df = pl.DataFrame(rows)
        surface = calibrate_surface(chain_df)

        expiries = [s.expiry for s in surface.slices]
        assert expiries == sorted(expiries)

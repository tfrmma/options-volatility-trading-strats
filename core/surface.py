# SVI surface calibration. Gatheral (2004) parametrization per slice.
# Static arb checks: butterfly per slice, calendar across slices.
#
# Don't interpolate in vol space — interpolate total variance.
# Variance is additive. Vol isn't. This is not a subtle point.

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize

from core.pricer import implied_vol_chain


@dataclass
class SVIParams:
    # w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    a: float      # vertical shift
    b: float      # wing steepness, b >= 0
    rho: float    # skew, |rho| < 1
    m: float      # ATM shift
    sigma: float  # curvature, sigma > 0
    expiry: float = 0.0

    def total_variance(self, log_moneyness: np.ndarray) -> np.ndarray:
        k = log_moneyness
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + self.sigma**2))

    def implied_vol(self, log_moneyness: np.ndarray) -> np.ndarray:
        w = np.maximum(self.total_variance(log_moneyness), 1e-10)
        return np.sqrt(w / max(self.expiry, 1e-10))

    def is_valid(self) -> bool:
        # necessary conditions for no butterfly arb within slice
        # not sufficient across slices — that needs the calendar check
        if self.b < 0 or self.sigma <= 0 or abs(self.rho) >= 1:
            return False
        if self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) < 0:
            return False
        return True


@dataclass
class VolSurface:
    slices: list[SVIParams] = field(default_factory=list)
    expiries: np.ndarray = field(default_factory=lambda: np.array([]))

    def add_slice(self, params: SVIParams) -> None:
        self.slices.append(params)
        self.expiries = np.array([s.expiry for s in self.slices])

    def implied_vol(self, log_moneyness: float, expiry: float) -> float:
        if not self.slices:
            return np.nan

        idx = np.searchsorted(self.expiries, expiry)

        # flat extrapolation beyond boundaries — crude but beats exploding
        if idx == 0:
            return float(self.slices[0].implied_vol(np.array([log_moneyness]))[0])
        if idx >= len(self.slices):
            return float(self.slices[-1].implied_vol(np.array([log_moneyness]))[0])

        # interpolate in total variance space, not vol space
        t0, t1 = self.expiries[idx - 1], self.expiries[idx]
        w0 = self.slices[idx - 1].total_variance(np.array([log_moneyness]))[0]
        w1 = self.slices[idx].total_variance(np.array([log_moneyness]))[0]

        alpha = (expiry - t0) / (t1 - t0)
        w_interp = (1 - alpha) * w0 + alpha * w1

        return float(np.sqrt(max(w_interp, 0.0) / max(expiry, 1e-10)))


def _svi_objective(params: np.ndarray, k: np.ndarray, market_w: np.ndarray) -> float:
    a, b, rho, m, sigma = params
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    return float(np.mean((w - market_w)**2))


def _svi_constraints(params: np.ndarray) -> list:
    # Gatheral no-butterfly conditions. necessary, not jointly sufficient across slices
    return [
        {'type': 'ineq', 'fun': lambda p: p[1]},
        {'type': 'ineq', 'fun': lambda p: p[4] - 1e-6},
        {'type': 'ineq', 'fun': lambda p: 1.0 - abs(p[2]) - 1e-6},
        {'type': 'ineq', 'fun': lambda p: p[0] + p[1] * p[4] * np.sqrt(1 - p[2]**2)},
    ]


def fit_svi_slice(
    log_moneyness: np.ndarray,
    market_total_var: np.ndarray,
    expiry: float,
    n_restarts: int = 5,
) -> SVIParams:
    # multiple restarts because this landscape will absolutely eat a single-start solver
    k = log_moneyness
    w = market_total_var
    best_result, best_loss = None, np.inf

    init_candidates = [
        [np.mean(w), 0.1,  0.0,  0.0,  0.05],
        [np.mean(w), 0.1, -0.3,  0.0,  0.05],
        [np.mean(w), 0.1,  0.3,  0.0,  0.05],
        [np.mean(w), 0.05, -0.5, -0.1, 0.10],
        [np.mean(w), 0.2,  0.0,  0.1,  0.08],
    ]

    bounds = [(-1.0, 2.0), (0.0, 2.0), (-0.999, 0.999), (-2.0, 2.0), (1e-5, 2.0)]

    for x0 in init_candidates[:n_restarts]:
        try:
            res = minimize(
                _svi_objective, x0, args=(k, w),
                method='SLSQP', bounds=bounds,
                constraints=_svi_constraints(np.array(x0)),
                options={'ftol': 1e-12, 'maxiter': 1000},
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_result = res.x
        except Exception:
            continue

    if best_result is None:
        # calibration totally failed — return flat surface and scream about it
        warnings.warn(f"SVI calibration failed for T={expiry:.4f}, falling back to flat")
        atm_vol = float(np.mean(np.sqrt(w / max(expiry, 1e-10))))
        return SVIParams(a=atm_vol**2 * expiry, b=0.01, rho=0.0, m=0.0, sigma=0.1, expiry=expiry)

    a, b, rho, m, sigma = best_result
    params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=expiry)

    if not params.is_valid():
        warnings.warn(f"SVI params invalid at T={expiry:.4f} — check input data quality")

    return params


def calibrate_surface(chain_df) -> VolSurface:
    # chain_df columns: strike, expiry, bid, ask, is_call, spot, rate
    import polars as pl

    surface = VolSurface()
    S = chain_df["spot"][0]
    r = chain_df["rate"][0]

    for expiry, slice_df in chain_df.group_by("expiry"):
        T = float(expiry)
        if T <= 0:
            continue

        K_arr   = slice_df["strike"].to_numpy().astype(float)
        bid_arr = slice_df["bid"].to_numpy().astype(float)
        ask_arr = slice_df["ask"].to_numpy().astype(float)
        ic_arr  = slice_df["is_call"].to_numpy().astype(bool)

        mid_arr = 0.5 * (bid_arr + ask_arr)
        ivs = implied_vol_chain(mid_arr, S, K_arr, np.full(len(K_arr), T), r, ic_arr)

        valid = ~np.isnan(ivs) & (ivs > 0.01) & (ivs < 5.0)
        if valid.sum() < 3:
            warnings.warn(f"Not enough valid IV points for T={T:.4f}, skipping")
            continue

        log_k     = np.log(K_arr[valid] / S)
        total_var = (ivs[valid]**2) * T

        surface.add_slice(fit_svi_slice(log_k, total_var, T))

    surface.slices.sort(key=lambda s: s.expiry)
    surface.expiries = np.array([s.expiry for s in surface.slices])

    _check_calendar_arb(surface)
    return surface


def _check_calendar_arb(surface: VolSurface) -> None:
    # TODO: enforce this properly during joint calibration instead of just warning
    # right now it's basically a loud post-hoc sanity check
    test_ks = np.linspace(-1.0, 1.0, 21)

    for i in range(len(surface.slices) - 1):
        s1, s2 = surface.slices[i], surface.slices[i + 1]
        w1 = s1.total_variance(test_ks)
        w2 = s2.total_variance(test_ks)
        n_violations = np.sum(w2 < w1 - 1e-8)
        if n_violations > 0:
            warnings.warn(
                f"Calendar arb: T={s1.expiry:.3f} -> T={s2.expiry:.3f}, "
                f"{n_violations} violations"
            )

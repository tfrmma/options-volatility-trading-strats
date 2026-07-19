# BSM pricer + greeks. Numba JIT throughout.
# Keep this file pure, no I/O, no state, no imports that aren't math.

import numpy as np
from numba import njit, prange
import warnings


# can't use scipy.norm inside njit, so inline the CDF.
# Abramowitz & Stegun 26.2.17, ~6 decimal places, good enough
@njit(cache=True)
def _norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x) * poly
    return cdf if x >= 0.0 else 1.0 - cdf


@njit(cache=True)
def _norm_pdf(x: float) -> float:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit(cache=True)
def bsm_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@njit(cache=True)
def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool):
    # returns (delta, gamma, vega, theta, rho), vega/theta are per-unit, not per pct point
    if T <= 1e-10 or sigma <= 1e-10:
        delta = 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
        return delta, 0.0, 0.0, 0.0, 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    nd1 = _norm_pdf(d1)
    disc = np.exp(-r * T)

    gamma = nd1 / (S * sigma * sqrt_T)
    vega  = S * nd1 * sqrt_T

    if is_call:
        delta = Nd1
        theta = (-(S * nd1 * sigma) / (2.0 * sqrt_T) - r * K * disc * Nd2) / 365.0
        rho   = K * T * disc * Nd2 / 100.0
    else:
        delta = Nd1 - 1.0
        theta = (-(S * nd1 * sigma) / (2.0 * sqrt_T) + r * K * disc * _norm_cdf(-d2)) / 365.0
        rho   = -K * T * disc * _norm_cdf(-d2) / 100.0

    return delta, gamma, vega, theta, rho


@njit(parallel=True, cache=True)
def batch_price(S_arr, K_arr, T_arr, r_arr, sigma_arr, is_call_arr):
    n = len(S_arr)
    prices = np.empty(n)
    for i in prange(n):
        prices[i] = bsm_price(S_arr[i], K_arr[i], T_arr[i], r_arr[i], sigma_arr[i], is_call_arr[i])
    return prices


@njit(parallel=True, cache=True)
def batch_greeks(S_arr, K_arr, T_arr, r_arr, sigma_arr, is_call_arr):
    n = len(S_arr)
    delta = np.empty(n)
    gamma = np.empty(n)
    vega  = np.empty(n)
    theta = np.empty(n)
    rho   = np.empty(n)

    for i in prange(n):
        d, g, v, t, r = bsm_greeks(
            S_arr[i], K_arr[i], T_arr[i], r_arr[i], sigma_arr[i], is_call_arr[i]
        )
        delta[i] = d
        gamma[i] = g
        vega[i]  = v
        theta[i] = t
        rho[i]   = r

    return delta, gamma, vega, theta, rho


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    if T <= 0 or market_price <= 0:
        return np.nan

    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if market_price < intrinsic - 1e-8:
        return np.nan  # arb violation in input, garbage in nan out

    sigma = 0.3  # decent starting guess for crypto, tune if you're doing rates
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, is_call)
        _, _, vega, _, _ = bsm_greeks(S, K, T, r, sigma, is_call)

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        if abs(vega) < 1e-12:
            break  # vega collapsed, NR is useless here, fall through to bisection

        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4

    # bisection fallback, ugly but guaranteed to converge
    lo, hi = 1e-4, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bsm_price(S, K, T, r, mid, is_call) > market_price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return mid

    warnings.warn(f"IV solver didn't converge: S={S:.2f} K={K:.2f} T={T:.4f}")
    return np.nan


@njit(parallel=True, cache=True)
def _implied_vol_batch_kernel(market_prices, S_arr, K_arr, T_arr, r_arr, is_call_arr, tol, max_iter):
    n = len(market_prices)
    out = np.empty(n)

    for i in prange(n):
        mp, S, K, T, r, is_call = market_prices[i], S_arr[i], K_arr[i], T_arr[i], r_arr[i], is_call_arr[i]

        if T <= 0.0 or mp <= 0.0:
            out[i] = np.nan
            continue

        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        if mp < intrinsic - 1e-8:
            out[i] = np.nan
            continue

        sigma = 0.3
        converged = False
        for _ in range(max_iter):
            price = bsm_price(S, K, T, r, sigma, is_call)
            _, _, vega, _, _ = bsm_greeks(S, K, T, r, sigma, is_call)
            diff = price - mp
            if abs(diff) < tol:
                converged = True
                break
            if abs(vega) < 1e-12:
                break  # vega collapsed, fall through to bisection below
            sigma -= diff / vega
            if sigma <= 0.0:
                sigma = 1e-4

        if not converged:
            lo, hi = 1e-4, 10.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if bsm_price(S, K, T, r, mid, is_call) > mp:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < tol:
                    sigma = mid
                    converged = True
                    break

        out[i] = sigma if converged else np.nan

    return out


def implied_vol_chain(
    market_prices: np.ndarray,
    S: float,
    K_arr: np.ndarray,
    T_arr: np.ndarray,
    r: float,
    is_call_arr: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    # numba parallel batch solve instead of a python-level loop calling the scalar
    # solver once per strike. numba over cython here too, same call as the rest of this
    # file, one JIT toolchain beats maintaining a second compiled-extension build path
    # for a chain that's maybe a few hundred points wide.
    n = len(market_prices)
    result = _implied_vol_batch_kernel(
        np.asarray(market_prices, dtype=np.float64),
        np.full(n, S, dtype=np.float64),
        np.asarray(K_arr, dtype=np.float64),
        np.asarray(T_arr, dtype=np.float64),
        np.full(n, r, dtype=np.float64),
        np.asarray(is_call_arr, dtype=np.bool_),
        tol, max_iter,
    )

    n_failed = int(np.sum(np.isnan(result)))
    if n_failed > 0:
        warnings.warn(f"IV solver didn't converge for {n_failed}/{n} chain points")
    return result

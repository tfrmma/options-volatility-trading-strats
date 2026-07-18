# Vol estimators. Yang-Zhang is the default, don't use close-to-close
# unless you enjoy throwing away half your signal.
#
# Yang & Zhang (2000), handles overnight gaps, minimum variance, drift-independent.
# That's three things close-to-close gets wrong simultaneously.

import numpy as np
import polars as pl
from typing import Optional


def close_to_close(closes: np.ndarray, ann_factor: float = 252.0) -> float:
    # baseline only. if this is your main estimator, reconsider your choices
    log_rets = np.diff(np.log(closes))
    return np.std(log_rets, ddof=1) * np.sqrt(ann_factor)


def garman_klass(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ann_factor: float = 252.0,
) -> float:
    # better than C2C, biased with drift. fine for daily, sketchy intraday
    log_hl = np.log(highs / lows)
    log_co = np.log(closes / opens)
    gk = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    return np.sqrt(np.mean(gk) * ann_factor)


def yang_zhang(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ann_factor: float = 252.0,
) -> float:
    n = len(closes)
    if n < 2:
        return np.nan

    log_co = np.log(opens[1:] / closes[:-1])   # overnight
    log_oc = np.log(closes[1:] / opens[1:])    # open-to-close

    # Rogers-Satchell intraday component
    log_hc = np.log(highs[1:] / closes[1:])
    log_ho = np.log(highs[1:] / opens[1:])
    log_lc = np.log(lows[1:]  / closes[1:])
    log_lo = np.log(lows[1:]  / opens[1:])
    rs = log_hc * log_ho + log_lc * log_lo

    sigma_oc_sq = np.var(log_co, ddof=1)
    sigma_co_sq = np.var(log_oc, ddof=1)
    sigma_rs_sq = np.mean(rs)

    # finite-sample optimal k from the paper
    k_adj = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz_var = sigma_oc_sq + k_adj * sigma_co_sq + (1.0 - k_adj) * sigma_rs_sq

    return np.sqrt(max(yz_var, 0.0) * ann_factor)


def rolling_yang_zhang(
    df: pl.DataFrame,
    window: int = 30,
    ann_factor: float = 252.0,
) -> np.ndarray:
    # polars because pandas chokes on anything over ~5M rows of options data
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    n = len(closes)
    result = np.full(n, np.nan)

    for i in range(window, n + 1):
        result[i - 1] = yang_zhang(
            opens[i - window:i],
            highs[i - window:i],
            lows[i - window:i],
            closes[i - window:i],
            ann_factor,
        )

    return result


def realized_variance_hf(log_returns: np.ndarray, subsampling: int = 1) -> float:
    # subsampling > 1 averages over multiple grids to reduce microstructure noise.
    # use 5-min bars from tick data, raw ticks are just bid-ask bounce
    if subsampling == 1:
        return float(np.sum(log_returns**2))

    rv_estimates = [
        np.sum(log_returns[offset::subsampling]**2)
        for offset in range(subsampling)
    ]
    return float(np.mean(rv_estimates))


def realized_kernel(log_returns: np.ndarray, bandwidth: Optional[int] = None) -> float:
    # Barndorff-Nielsen et al. (2008). Parzen kernel, flat-top variant.
    # slower than plain RV but correct on raw tick data. worth it.
    n = len(log_returns)
    if bandwidth is None:
        bandwidth = max(1, int(np.ceil(n ** 0.6)))  # rule-of-thumb from Iqbal (2014)

    H = bandwidth
    gamma_0 = np.sum(log_returns**2)

    def parzen(x: float) -> float:
        ax = abs(x)
        if ax <= 0.5:
            return 1.0 - 6.0 * ax**2 + 6.0 * ax**3
        elif ax <= 1.0:
            return 2.0 * (1.0 - ax) ** 3
        return 0.0

    rk = gamma_0
    for h in range(1, H + 1):
        gamma_h = np.sum(log_returns[h:] * log_returns[:-h])
        rk += 2.0 * parzen(h / H) * gamma_h

    return max(float(rk), 0.0)


def vol_ratio(realized: float, implied: float) -> float:
    # > 1: implied rich, sell vol. < 1: implied cheap, buy vol. that's the whole trade
    if implied <= 0:
        return np.nan
    return implied / realized


def ewma_vol(log_returns: np.ndarray, lam: float = 0.94, ann_factor: float = 252.0) -> float:
    # RiskMetrics. lam=0.94 daily, 0.97 monthly. fast and good enough for hedging
    var = log_returns[0] ** 2
    for r in log_returns[1:]:
        var = lam * var + (1.0 - lam) * r**2
    return np.sqrt(var * ann_factor)

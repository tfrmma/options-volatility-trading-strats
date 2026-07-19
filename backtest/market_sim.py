# Synthetic market generator. GARCH(1,1) vol clustering + Merton jumps.
#
# Good enough for strategy development. Not a substitute for real data
# when it comes to regime changes, liquidity gaps, or exchange-specific quirks.

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    spot0: float       = 50_000.0
    base_vol: float    = 0.60
    dt: float          = 1.0 / 252.0
    n_steps: int       = 252

    # GARCH(1,1), keep alpha + beta < 1 or variance explodes
    garch_omega: float = 0.01
    garch_alpha: float = 0.10
    garch_beta: float  = 0.85

    # Merton jumps
    jump_intensity: float = 5.0     # per year
    jump_mean: float      = 0.0
    jump_std: float       = 0.05

    # bid/ask widens with vol, rough but directionally correct
    base_spread_bps: float       = 5.0
    spread_vol_sensitivity: float = 2.0

    # occasional toxic prints, price gaps that aren't explainable by diffusion
    toxic_prob: float   = 0.02
    toxic_impact: float = 0.015


@dataclass
class SimStep:
    t: float
    spot: float
    vol: float
    bid: float
    ask: float
    log_return: float
    is_toxic: bool
    jump_occurred: bool


def simulate_market(cfg: SimConfig, seed: Optional[int] = None) -> list[SimStep]:
    rng   = np.random.default_rng(seed)
    steps = []
    spot  = cfg.spot0
    var_t = cfg.base_vol ** 2

    for i in range(cfg.n_steps):
        vol_t = np.sqrt(var_t)

        diffusion = vol_t * np.sqrt(cfg.dt) * rng.standard_normal()

        # Merton jump component
        n_jumps = rng.poisson(cfg.jump_intensity * cfg.dt)
        jump    = float(np.sum(rng.normal(cfg.jump_mean, cfg.jump_std, n_jumps))) if n_jumps > 0 else 0.0
        jumped  = n_jumps > 0

        log_ret  = diffusion + jump
        new_spot = spot * np.exp(log_ret)

        # GARCH update, work in return scale, not annualized
        var_t = cfg.garch_omega + cfg.garch_alpha * log_ret**2 + cfg.garch_beta * var_t
        var_t = max(var_t, 1e-8)

        is_toxic = rng.random() < cfg.toxic_prob
        if is_toxic:
            new_spot *= np.exp(rng.choice([-1, 1]) * cfg.toxic_impact)

        spread_mult = 1.0 + cfg.spread_vol_sensitivity * (vol_t / cfg.base_vol - 1.0)
        half_spread = new_spot * cfg.base_spread_bps * 1e-4 * max(spread_mult, 0.5)

        steps.append(SimStep(
            t=i * cfg.dt, spot=new_spot, vol=vol_t,
            bid=new_spot - half_spread, ask=new_spot + half_spread,
            log_return=log_ret, is_toxic=is_toxic, jump_occurred=jumped,
        ))
        spot = new_spot

    return steps


def to_ohlcv(steps: list[SimStep], bar_size: int = 1) -> dict:
    n_bars = len(steps) // bar_size
    opens  = np.empty(n_bars)
    highs  = np.empty(n_bars)
    lows   = np.empty(n_bars)
    closes = np.empty(n_bars)

    for i in range(n_bars):
        bar      = steps[i * bar_size:(i + 1) * bar_size]
        opens[i]  = bar[0].spot
        closes[i] = bar[-1].spot
        highs[i]  = max(s.spot for s in bar)
        lows[i]   = min(s.spot for s in bar)

    return {"open": opens, "high": highs, "low": lows, "close": closes}


def generate_option_chain(
    spot: float,
    expiries: list[float],
    vol_surface_fn,     # callable(log_moneyness, expiry) -> iv
    n_strikes: int = 11,
    wing_width: float = 0.30,
    spread_bps: float = 20.0,
) -> list[dict]:
    from core.pricer import bsm_price
    records  = []
    log_ks   = np.linspace(-wing_width, wing_width, n_strikes)

    for T in expiries:
        for lk in log_ks:
            K  = spot * np.exp(lk)
            iv = vol_surface_fn(lk, T)
            for is_call in [True, False]:
                mid  = bsm_price(spot, K, T, 0.0, iv, is_call)
                half = mid * spread_bps * 1e-4
                records.append({
                    "strike": K, "expiry": T, "is_call": is_call,
                    "bid": max(mid - half, 0.0), "ask": mid + half,
                    "spot": spot, "rate": 0.0,
                })

    return records


def simulate_dispersion_feed(
    index_symbol: str,
    weights: dict[str, float],
    component_configs: dict[str, SimConfig],
    expiry: float,
    seed: int = 0,
    option_spread_bps: float = 80.0,
    index_vol_diversification: float = 0.85,
):
    # Synthetic multi-symbol feed for backtesting dispersion end to end, not for
    # anything else. Two simplifications that keep this tractable:
    #
    # 1. Each component gets its own independent GARCH path (simulate_market), the
    #    index is built as the weighted SPOT sum of components, that's literally what
    #    an index is, and it produces realistic (imperfect) index/component
    #    correlation for free instead of needing a full correlated multivariate GARCH
    #    model. Index vol is the weighted average of component vols scaled down by
    #    `index_vol_diversification`, real index vol sits below that average whenever
    #    correlation < 1, which is the entire dynamic dispersion trades on. The scaling
    #    factor is an approximation, not fit to anything.
    #
    # 2. Only a single fixed-strike (ATM at t=0) straddle is quoted per symbol per
    #    step, not a full chain, that's the specific instrument set a dispersion trade
    #    actually holds for one trade's lifetime.
    #
    # Feed ordering per timestep: every component's call+put, THEN the index's
    # call+put last. DispersionBacktestAdapter relies on that order to know a
    # timestep's quotes are all in before it makes a decision, this is a convention
    # specific to this generator + that adapter, not a general engine requirement.
    import polars as pl
    from core.pricer import bsm_price

    paths = {sym: simulate_market(cfg, seed=seed + i) for i, (sym, cfg) in enumerate(component_configs.items())}
    n = min(len(p) for p in paths.values())

    strikes = {sym: round(paths[sym][0].spot) for sym in paths}
    strikes[index_symbol] = round(sum(weights[sym] * paths[sym][0].spot for sym in paths))

    rows = []
    for t in range(n):
        index_spot = 0.0
        index_vol_weighted = 0.0

        for sym, path in paths.items():
            step = path[t]
            index_spot += weights[sym] * step.spot
            index_vol_weighted += weights[sym] * step.vol

            for is_call in (True, False):
                mid  = bsm_price(step.spot, strikes[sym], expiry, 0.0, step.vol, is_call)
                half = mid * option_spread_bps * 1e-4
                rows.append({
                    "timestamp": float(t), "spot": step.spot, "sigma": step.vol,
                    "bid": step.bid, "ask": step.ask, "expiry": expiry,
                    "strike": float(strikes[sym]), "is_call": is_call,
                    "option_bid": max(mid - half, 0.0), "option_ask": mid + half,
                    "symbol": sym,
                })

        index_vol = index_vol_weighted * index_vol_diversification
        index_bid = index_spot * (1.0 - 0.0005)
        index_ask = index_spot * (1.0 + 0.0005)

        for is_call in (True, False):   # call first, put last: put is the decision trigger
            mid  = bsm_price(index_spot, strikes[index_symbol], expiry, 0.0, index_vol, is_call)
            half = mid * option_spread_bps * 1e-4
            rows.append({
                "timestamp": float(t), "spot": index_spot, "sigma": index_vol,
                "bid": index_bid, "ask": index_ask, "expiry": expiry,
                "strike": float(strikes[index_symbol]), "is_call": is_call,
                "option_bid": max(mid - half, 0.0), "option_ask": mid + half,
                "symbol": index_symbol,
            })

    return pl.DataFrame(rows), strikes

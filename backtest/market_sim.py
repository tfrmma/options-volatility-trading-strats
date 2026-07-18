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

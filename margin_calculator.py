"""
@file margin_calculator.py
@author Taha - Algorithmic Trader
@brief Institutional-grade options-volatility-trading-strats.

@note This is a public structural showcase. For full production-grade 
      deployment, architecture consulting, or recruitment inquiries:
      Contact: email: fadilrezokt@gmail.com / linkedin.com/in/tahaotc
"""


# Deribit-style portfolio margin. Scenario-based, not flat rate.
#
# The exchange stress-tests your portfolio across a grid of spot/vol moves.
# Your margin is the worst-case loss across that grid. This is the right way
# to think about tail risk — don't use Reg-T style fixed margin for options.

import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.pricer import bsm_price


# approximate Deribit shock grid
_SPOT_SHOCKS = np.array([-0.15, -0.10, -0.07, -0.03, 0.0, 0.03, 0.07, 0.10, 0.15])
_VOL_SHOCKS  = np.array([-0.25, 0.0, 0.25])


@dataclass
class MarginRequirement:
    initial_margin: float
    maintenance_margin: float
    margin_ratio: float        # equity / IM
    worst_case_loss: float
    worst_scenario: tuple      # (spot_shock, vol_shock)


@dataclass
class OptionPosition:
    strike: float
    expiry: float
    is_call: bool
    qty: float
    current_iv: float


def compute_portfolio_margin(
    positions: list[OptionPosition],
    spot_position: float,
    spot: float,
    rate: float = 0.0,
    im_multiplier: float = 1.10,
    mm_multiplier: float = 1.00,
) -> MarginRequirement:
    worst_loss    = 0.0
    worst_scenario = (0.0, 0.0)

    for ds in _SPOT_SHOCKS:
        stressed_spot = spot * (1.0 + ds)
        for dv in _VOL_SHOCKS:
            loss = -_scenario_pnl(positions, spot_position, spot, stressed_spot, dv, rate)
            if loss > worst_loss:
                worst_loss     = loss
                worst_scenario = (ds, dv)

    return MarginRequirement(
        initial_margin=worst_loss * im_multiplier,
        maintenance_margin=worst_loss * mm_multiplier,
        margin_ratio=0.0,  # caller fills this in with actual equity
        worst_case_loss=worst_loss,
        worst_scenario=worst_scenario,
    )


def _scenario_pnl(
    positions: list[OptionPosition],
    spot_position: float,
    spot_now: float,
    stressed_spot: float,
    vol_shock: float,
    rate: float,
) -> float:
    pnl = spot_position * (stressed_spot - spot_now)
    for pos in positions:
        stressed_iv  = max(pos.current_iv * (1.0 + vol_shock), 0.01)
        current_val  = bsm_price(spot_now,     pos.strike, pos.expiry, rate, pos.current_iv, pos.is_call)
        stressed_val = bsm_price(stressed_spot, pos.strike, pos.expiry, rate, stressed_iv,   pos.is_call)
        pnl += pos.qty * (stressed_val - current_val)
    return pnl


def liquidation_check(equity: float, margin: MarginRequirement) -> dict:
    mm = max(margin.maintenance_margin, 1.0)
    ratio = equity / mm
    return {
        "margin_ratio":       ratio,
        "margin_call":        ratio < 1.0,
        "liquidation":        ratio < 0.8,   # Deribit auto-liq below 80% MM
        "equity":             equity,
        "maintenance_margin": margin.maintenance_margin,
        "initial_margin":     margin.initial_margin,
    }


def max_position_size(equity: float, target_margin_util: float, unit_margin: float) -> float:
    # how large can I go given target utilization and per-unit margin cost?
    if unit_margin < 1e-8:
        return 0.0
    return (equity * target_margin_util) / unit_margin

"""
@file margin_calculator.py
@author Taha - Algorithmic Trader
@brief Institutional-grade options-volatility-trading-strats.

@note This is a public structural showcase. For full production-grade 
      deployment, architecture consulting, or recruitment inquiries:
      Contact: email: fadilrezokt@gmail.com / linkedin.com/in/tahaotc
"""

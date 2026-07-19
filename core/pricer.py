# Real listed strikes aren't a continuous grid, exchanges list whatever they list,
# usually tighter spacing near spot on weeklies than on LEAPS. The strategies used to
# each carry their own copy of a "round to a fake grid" heuristic, this replaces that
# with an actual chain lookup when one's available, and keeps the heuristic only as a
# fallback for when there isn't one (quick sizing checks, synthetic backtests).

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptionChain:
    strikes_by_expiry: dict[float, list[float]]

    def nearest_strike(self, expiry: float, target: float) -> float:
        strikes = self.strikes_by_expiry.get(expiry)
        if not strikes:
            raise ValueError(f"no listed strikes for expiry {expiry}")
        return min(strikes, key=lambda k: abs(k - target))

    def expiries(self) -> list[float]:
        return sorted(self.strikes_by_expiry.keys())

    @classmethod
    def from_records(cls, records: list[dict]) -> "OptionChain":
        # records shaped like market_sim.generate_option_chain()'s output, or a real
        # chain pulled from an exchange with the same {strike, expiry, ...} shape
        by_expiry: dict[float, set] = {}
        for r in records:
            by_expiry.setdefault(r["expiry"], set()).add(r["strike"])
        return cls(strikes_by_expiry={T: sorted(ks) for T, ks in by_expiry.items()})


def round_to_synthetic_grid(spot: float) -> float:
    # NOT what you want against a live book, this is a last resort for when there's no
    # real chain to snap to at all
    if spot < 1000:   return round(spot / 5) * 5.0
    if spot < 10000:  return round(spot / 50) * 50.0
    if spot < 100000: return round(spot / 500) * 500.0
    return round(spot / 1000) * 1000.0


def target_strike(target: float, expiry: float, chain: Optional[OptionChain]) -> float:
    if chain is not None:
        return chain.nearest_strike(expiry, target)
    return round_to_synthetic_grid(target)

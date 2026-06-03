# options-volatility-trading-strats

Crypto options volatility strategies. Built for Deribit/Binance.

## Strategies

- **delta_neutral** — straddles/strangles with Whalley-Wilmott rebalancing bands
- **vol_arb** — VRP harvesting (IV vs RV via Yang-Zhang)
- **dispersion** — implied correlation arb (sell index vol, buy component vol)
- **surface_trading** — skew and calendar spread relative value

## Structure

```
core/       pricer (BSM + greeks, Numba JIT), vol estimators, SVI surface calibration
strategies/ base class + four strategy implementations
backtest/   discrete-event engine, market simulator (GARCH + Merton jumps), margin calc
data/       Deribit WebSocket feed, Parquet storage
tests/      math layer unit tests (PCP, greek boundaries, IV round-trip, SVI fit)
```

## Install

```bash
pip install -r requirements.txt
```

## Test

```bash
pytest tests/ -v
```

## Notes

- Execute against bid/ask. Never mid. Options spreads are wide.
- Portfolio margin is scenario-based — don't use flat margin rates.
- Delta hedging: W-W bands, not time-based. Time-based rebalancing destroys edge.
- Yang-Zhang is the default RV estimator. Close-to-close throws away intraday info.
- SVI surface calibration uses multiple restarts — landscape has local minima.

## TODO

- Joint SVI calibration with calendar arbitrage enforcement
- Partial close / vega rebalancing in vol_arb
- Pull actual listed strikes from chain instead of rounding
- Cython IV solver for chain-level batch computation

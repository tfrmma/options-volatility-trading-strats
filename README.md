# options-volatility-trading-strats

Crypto options volatility strategies built against Deribit/Binance-style option chains. Four
strategy implementations sharing a common greeks/PnL engine, a discrete-event backtester with
realistic bid/ask execution, and a scenario-based margin model.

This is a research and backtesting codebase, not an execution system. There's no exchange
connectivity for order placement, no OMS, no risk gateway. It exists to answer one question:
*is the vol edge real once you account for spread, fees, and delta-hedging slippage*, before
any of it touches a live venue.

## Why this exists

Most public "options vol" repos price a Black-Scholes call, plot an IV smile, and call it a
day. None of that tells you whether a strategy survives contact with a real order book. This
repo is built around the parts that actually determine whether a vol strategy makes money:

- **Execution realism.** Every fill in the backtester happens at bid/ask, never mid. Crypto
  options routinely trade 20-50bps wide even on BTC/ETH front-month. A strategy that looks
  profitable at mid and unprofitable at touch isn't a strategy, it's a rounding error.
- **Rebalancing discipline.** Delta hedging is done on Whalley-Wilmott no-transaction-cost
  bands, not on a timer. Time-based rebalancing is the single most common way retail vol
  strategies bleed their edge to fees, because gamma near expiry demands rebalancing that a
  fixed clock either does too often (paying spread for nothing) or too rarely (letting delta
  run).
- **Estimator choice matters more than people think.** Yang-Zhang is the default realized vol
  estimator because it handles overnight gaps and is drift-independent, close-to-close throws
  away most of the intraday information content for a marginal reduction in code complexity.
- **Margin is scenario-based, not flat.** Deribit-style portfolio margin stress-tests the book
  across a spot/vol shock grid and margins to the worst case. A flat percentage-of-notional
  model will misprice risk badly for anything with convex payoff, which is the entire point of
  holding options.

## Architecture

```
core/       Pure math: BSM pricer + greeks (Numba JIT), realized vol estimators, SVI surface
            calibration. No I/O, no state, no strategy-specific logic. This layer is imported
            by everything else and tested in isolation because if the greeks are wrong,
            everything built on top of them is wrong in a way that's hard to detect later.

strategies/ BaseVolStrategy (leg tracking, portfolio greeks, PnL decomposition, delta hedging)
            plus four concrete strategies. Strategies own signal generation and position
            sizing; they do not own execution, that's the backtest engine's job.

backtest/   Discrete-event engine that replays a data feed tick by tick, fills strategy orders
            against bid/ask, and tracks equity/PnL/drawdown. Also holds the synthetic market
            generator (GARCH(1,1) vol clustering + Merton jumps, for strategy development
            without real data) and the portfolio margin calculator.

data/       Deribit WebSocket client for live/paper feeds, and a partitioned Parquet store for
            historical tick data (partitioned by currency/date, because querying an
            unpartitioned multi-hundred-GB options tick dataset is a mistake you only make
            once).

tests/      Math layer unit tests: put-call parity, greek boundary conditions, IV solver
            round-trip, SVI fit convergence. This is the layer with the highest cost of being
            wrong, so it's the most thoroughly tested. Strategy-level and backtest-level tests
            are not part of this pass; see Known Limitations.
```

## Strategies

### `delta_neutral` — straddles / strangles
Sells (or buys) ATM straddles when implied vol diverges from realized vol by more than a
threshold, then delta-hedges on W-W bands rather than a clock. Thesis is trivial (sell rich
vol, buy cheap vol), execution is not: fee-aware band width, vega-based position sizing, and a
vega-loss stop are what separate this from "sell strangles and hope."

### `vol_arb` — VRP harvesting
Systematic variance risk premium harvesting. Implied vol is persistently above realized on
average, but the premium compresses, occasionally inverts, and goes haywire around macro
events, so this trades the VRP z-score against its own trailing history rather than selling
vol mechanically every time IV > RV. On a "hold" signal it doesn't just sit there re-hedging
delta, it scales the existing position toward the new conviction-weighted vega target, trimming
(partial close, realizing PnL on the closed slice) or adding at the same strikes, whichever the
drift calls for, subject to a tolerance band so it isn't paying the spread on every small wobble.

### `dispersion` — implied correlation arb
Sells index vol, buys vega-weighted component vol, on the thesis that implied correlation is
structurally too high because index puts get bid up as macro hedges. This does not like tail
events, correlation goes to 1 exactly when you don't want it to, so it's sized conservatively
relative to the other strategies. Unlike the other three, `DispersionStrategy` does not inherit
`BaseVolStrategy`, it composes one `UnderlyingBook` per underlying (index + each component) so
every leg prices, greeks, and hedges off its own spot instead of a single shared one. It's the
one strategy actually wired end to end through `BacktestEngine`, via
`backtest/dispersion_adapter.py` and `backtest/market_sim.py::simulate_dispersion_feed()`; the
other three are unit-tested against their own theoretical fills but don't have an engine adapter
yet.

### `surface_trading` — skew and calendar RV
Trades dislocations in the surface itself: calendar spreads (sell rich near-term variance, buy
cheap back-month) and risk reversals (sell the overpriced wing). Positions are vega/theta
neutral at entry by construction; residual exposure is vanna/volga, which is the trade, not a
side effect.

## Design decisions and trade-offs

**Numba over Cython for the pricer, including the batch IV solver.** JIT compilation gets most
of the speedup with a fraction of the build complexity, no separate compiled-extension toolchain
to maintain alongside the rest of the JIT'd math in this file. `implied_vol_chain()` is a
`@njit(parallel=True)` batch Newton-Raphson-with-bisection-fallback solver for the same reason
`batch_price`/`batch_greeks` are: chain sizes here are at most a few hundred points, not the
kind of scale where Cython's extra control over memory layout would pay for its build cost. The
scalar `implied_vol()` stays plain Python, it's not the hot path, `implied_vol_chain` is what
actually gets called against a full chain.

**Polars over pandas for tick data.** Pandas' row-oriented model and copy-heavy semantics start
to hurt once you're past a few million rows of options ticks. Polars' lazy evaluation and
Arrow-backed columnar storage handle that better, at the cost of a less familiar API if you're
coming from pandas.

**Interpolating in total variance, not vol space.** Variance is additive across time, vol
isn't. Interpolating IV directly between expiries is a common shortcut that produces a subtly
wrong term structure. `VolSurface.implied_vol()` interpolates `w(k) = sigma^2 * T` and converts
back at the end.

**SVI with multiple random restarts.** The Gatheral SVI objective is non-convex and a
single-start SLSQP solver gets stuck in local minima often enough to matter. Five restarts with
different initial skew/curvature guesses is a pragmatic middle ground between calibration
quality and fit time; it is not a guarantee of the global optimum.

**Butterfly arb enforced per-slice, calendar arb enforced sequentially across slices.**
`SVIParams.is_valid()` enforces the Gatheral no-butterfly condition within a single expiry
slice during optimization. `calibrate_surface()` fits slices in increasing expiry order and
passes each already-fitted slice into the next one's `fit_svi_slice()` call as a hard
`w_this(k) >= w_prev(k)` inequality constraint over a strike grid, not a post-hoc check. This is
sequential-constrained, not a single joint optimization over every slice's parameters at once,
that's a bigger and messier optimization problem and this gets the actual guarantee (no
calendar arb between consecutive tenors) without it. If a fit still fails outright (the
constraint plus butterfly validity leaves no feasible region for genuinely arb-laden input
data), it falls back to a flat slice pinned at or above the previous tenor's ATM level, not a
silent violation.

**Real listed strikes when a chain is available, a synthetic grid when it isn't.** Every
strategy takes an optional `chain: OptionChain` (or `index_chain`/`component_chains` for
dispersion). `core/chain.py` snaps a target strike to the nearest one actually listed for that
expiry; without a chain, it falls back to the old round-to-a-fake-grid heuristic, which is fine
for synthetic backtests but not for real order placement.

## Install

```bash
pip install -e .
```

This installs `core`, `strategies`, `backtest`, and `data` as a proper editable package, so
imports resolve regardless of where you run scripts from. For test-only dependencies:

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest tests/ -v
```

`tests/test_core.py` covers the math layer (`core/`): BSM pricing/greeks boundary conditions,
put-call parity, IV solver round-trip accuracy, batch pricer consistency with the scalar path,
and SVI fit convergence against synthetic noisy data.

`tests/test_strategies.py` covers the mark_to_market/close-double-count/put-strike-search bugs
fixed in the strategy layer, plus `vol_arb`'s vega rebalancing (trim/add/no-op-within-tolerance).
`tests/test_dispersion.py` covers the composition redesign (components price off their own
spot, hedge independently, aggregate correctly). `tests/test_engine.py` covers the
per-instrument quote cache, the sharpe annualization fix, and per-symbol position
tracking/marking for multiple underlyings ticking through the same feed.
`tests/test_margin_calculator.py` covers the portfolio margin scenario grid and liquidation
thresholds. `tests/test_market_sim.py` covers the GARCH/jump price generator and the synthetic
option chain builder. `tests/test_chain.py` covers real-chain strike snapping vs the synthetic
grid fallback. `tests/test_svi_calendar.py` covers the calendar-arb constraint actually
rejecting violations an unconstrained fit would allow, and the sequential fitting order.
`tests/test_dispersion_adapter.py` runs `DispersionStrategy` through `BacktestEngine` end to
end via `DispersionBacktestAdapter`: real fills tagged by symbol, an equity curve driven by
those fills, and confirmation that a strategy that never wants to enter never trades.

## Known limitations

This is a research and backtesting codebase, not an execution system, and that's by design, not
a gap. `market_sim.py` and `data/websocket_client.py` cover feed simulation and read-only
market data. There's no order placement, no OMS, no risk gateway, no exchange connectivity.
Turning this into something that trades live is a different project.

## License

MIT

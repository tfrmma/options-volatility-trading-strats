# Discrete-event backtest engine.
#
# Execute against bid/ask. Not mid. Not mark. Bid/ask.
# Options have 20-50bps spreads minimum, if you fill at mid your backtest
# is fiction and you'll find out the hard way in production.

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_DEFAULT_SYMBOL = "underlying"  # single-underlying feeds don't need to tag every row


@dataclass
class MarketSnapshot:
    timestamp: float
    spot: float
    sigma: float
    bid: float
    ask: float
    expiry: float
    strike: float
    is_call: bool
    option_bid: float
    option_ask: float
    symbol: str = _DEFAULT_SYMBOL  # which underlying this tick belongs to


@dataclass
class Fill:
    timestamp: float
    side: str
    qty: float
    price: float
    is_option: bool
    symbol: str = _DEFAULT_SYMBOL
    strike: Optional[float] = None
    expiry: Optional[float] = None
    is_call: Optional[bool] = None
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.price


@dataclass
class BacktestResult:
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    pnl_series: list[float] = field(default_factory=list)

    def total_pnl(self) -> float:
        return self.pnl_series[-1] if self.pnl_series else 0.0

    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        equity = np.array([e for _, e in self.equity_curve])
        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / np.where(peak > 0, peak, 1.0)
        return float(np.min(dd))

    def sharpe(self, ann_factor: Optional[float] = None) -> float:
        if len(self.pnl_series) < 2:
            return 0.0
        pnl_diffs = np.diff(self.pnl_series)
        std = np.std(pnl_diffs, ddof=1)
        if std <= 1e-10:
            return 0.0
        if ann_factor is None:
            ann_factor = self._infer_ann_factor()
        return float(np.mean(pnl_diffs) / std * np.sqrt(ann_factor))

    def _infer_ann_factor(self) -> float:
        # periods/year from the actual timestamp spacing instead of assuming daily bars,
        # this engine runs on anything from tick data to daily closes, a hardcoded 252
        # is wrong for anything that isn't literally daily
        if len(self.equity_curve) < 2:
            return 252.0
        timestamps = np.array([t for t, _ in self.equity_curve])
        dt = float(np.median(np.diff(timestamps)))
        if dt <= 0:
            return 252.0
        seconds_per_year = 365.25 * 24 * 3600
        return seconds_per_year / dt

    def summary(self) -> dict:
        return {
            "total_pnl":   self.total_pnl(),
            "max_drawdown": self.max_drawdown(),
            "sharpe":       self.sharpe(),
            "n_fills":      len(self.fills),
            "total_fees":   sum(f.fee for f in self.fills),
        }


class BacktestEngine:
    # positions and quote caches are keyed by symbol (spot side) or (symbol, strike,
    # expiry, is_call) (option side), not just by instrument. a feed that never sets
    # `symbol` on its rows behaves exactly as before, everything defaults to one shared
    # "underlying" bucket. a feed that does set it (index + N components ticking through
    # the same DataFrame) gets each one tracked and marked independently, which is what
    # dispersion needs to be backtested end to end instead of just unit-tested in isolation.

    def __init__(
        self,
        taker_fee: float = 0.0006,
        maker_fee: float = -0.0001,
        slippage_bps: float = 1.0,
        initial_capital: float = 1_000_000.0,
        slippage_fn: Optional[Callable] = None,
    ):
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.initial_capital = initial_capital
        self.slippage_fn = slippage_fn or (lambda price, qty: price * slippage_bps * 1e-4)

        self.result = BacktestResult()
        self._cash = initial_capital
        self._option_positions: dict[tuple, float] = {}
        self._last_option_quotes: dict[tuple, tuple[float, float]] = {}
        self._spot_positions: dict[str, float] = {}
        self._last_spot_mark: dict[str, float] = {}
        self._last_underlying_quotes: dict[str, tuple[float, float]] = {}

    def run(self, data: pl.DataFrame, strategy_fn: Callable, verbose: bool = False) -> BacktestResult:
        self.result = BacktestResult()
        self._cash = self.initial_capital
        self._option_positions = {}
        self._last_option_quotes = {}
        self._spot_positions = {}
        self._last_spot_mark = {}
        self._last_underlying_quotes = {}

        for row in data.iter_rows(named=True):
            snap = self._parse_row(row)
            if snap is None:
                continue

            # cache this tick before filling/marking, _fill and _mtm both read from here
            # so every symbol/instrument gets marked at its own last known quote, not
            # whatever happens to be ticking on the current row
            self._last_option_quotes[(snap.symbol, snap.strike, snap.expiry, snap.is_call)] = \
                (snap.option_bid, snap.option_ask)
            self._last_underlying_quotes[snap.symbol] = (snap.bid, snap.ask)
            self._last_spot_mark[snap.symbol] = snap.spot

            for order in (strategy_fn(snap, self) or []):
                self._fill(order, snap)

            equity = self._cash + self._mtm()
            self.result.equity_curve.append((snap.timestamp, equity))
            self.result.pnl_series.append(equity - self.initial_capital)

            if verbose:
                logger.debug("tick", extra={"ts": snap.timestamp, "equity": equity})

        return self.result

    def _fill(self, order: dict, snap: MarketSnapshot) -> Optional[Fill]:
        is_option = order["type"] == "option"
        side      = order["side"]
        qty       = abs(order["qty"])
        symbol    = order.get("symbol", _DEFAULT_SYMBOL)

        if is_option:
            key = (symbol, order["strike"], order["expiry"], order["is_call"])
            quote = self._last_option_quotes.get(key)
            if quote is None:
                # can't fill an instrument we've never seen a quote for, shouldn't happen
                # if strategies only order what they can see, but don't fabricate a price
                logger.warning(f"no quote cached for {key}, skipping fill")
                return None

            bid, ask = quote
            raw   = ask if side == "buy" else bid
            slip  = self.slippage_fn(raw, qty) * (1 if side == "buy" else -1)
            price = max(raw + slip, 0.0)
            fee   = qty * price * self.taker_fee

            signed = qty if side == "buy" else -qty
            self._option_positions[key] = self._option_positions.get(key, 0.0) + signed
            self._cash -= signed * price + fee

            fill = Fill(snap.timestamp, side, signed, price, True, symbol,
                        order["strike"], order["expiry"], order["is_call"], fee)
        else:
            quote = self._last_underlying_quotes.get(symbol)
            if quote is None:
                logger.warning(f"no underlying quote cached for {symbol}, skipping fill")
                return None

            bid, ask = quote
            raw   = ask if side == "buy" else bid
            sign  = 1 if side == "buy" else -1
            price = raw + sign * self.slippage_fn(raw, qty)
            fee   = qty * price * self.taker_fee
            signed = sign * qty

            self._spot_positions[symbol] = self._spot_positions.get(symbol, 0.0) + signed
            self._cash -= signed * price + fee
            fill = Fill(snap.timestamp, side, signed, price, False, symbol, fee=fee)

        self.result.fills.append(fill)
        return fill

    def _mtm(self) -> float:
        # every open position, spot or option, on every symbol touched so far, marked at
        # its own last observed quote. no snap argument needed, run() already refreshed
        # the caches for the current tick's symbol before calling this.
        mtm = 0.0
        for symbol, qty in self._spot_positions.items():
            if abs(qty) < 1e-10:
                continue
            mark = self._last_spot_mark.get(symbol)
            if mark is None:
                logger.warning(f"no spot mark cached for {symbol}, marking at 0")
                continue
            mtm += qty * mark

        for key, qty in self._option_positions.items():
            if abs(qty) < 1e-10:
                continue
            quote = self._last_option_quotes.get(key)
            if quote is None:
                logger.warning(f"no quote cached for open position {key}, marking at 0")
                continue
            bid, ask = quote
            mtm += qty * 0.5 * (bid + ask)
        return mtm

    @staticmethod
    def _parse_row(row: dict) -> Optional[MarketSnapshot]:
        try:
            # only pull fields that are actually present, missing optional ones (symbol,
            # on a feed that predates it) fall back to their dataclass default instead
            # of blowing up the whole row
            kwargs = {k: row[k] for k in MarketSnapshot.__dataclass_fields__ if k in row}
            return MarketSnapshot(**kwargs)
        except TypeError as e:
            logger.warning(f"bad row, skipping: {e}")
            return None

    def current_position(self) -> dict:
        return {
            "spot":    dict(self._spot_positions),
            "options": dict(self._option_positions),
            "cash":    self._cash,
        }

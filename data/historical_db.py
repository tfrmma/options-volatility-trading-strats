# Partitioned Parquet storage for options tick data.
# Partition by (dataset/currency=X/date=Y), avoids full scans when you
# only need one currency for a specific date range.
# Learned this after trying to query 200GB of unpartitioned options data. Never again.

import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_OPTION_SCHEMA = pa.schema([
    ("timestamp",     pa.float64()),
    ("symbol",        pa.string()),
    ("currency",      pa.string()),
    ("expiry_ts",     pa.float64()),
    ("strike",        pa.float64()),
    ("is_call",       pa.bool_()),
    ("bid",           pa.float64()),
    ("ask",           pa.float64()),
    ("mark_price",    pa.float64()),
    ("mark_iv",       pa.float64()),
    ("index_price",   pa.float64()),
    ("delta",         pa.float64()),
    ("gamma",         pa.float64()),
    ("vega",          pa.float64()),
    ("theta",         pa.float64()),
    ("open_interest", pa.float64()),
])

_INDEX_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("currency",  pa.string()),
    ("price",     pa.float64()),
])


class ParquetStore:

    def __init__(self, base_path: str, flush_interval: int = 1000):
        self.base = Path(base_path)
        (self.base / "options").mkdir(parents=True, exist_ok=True)
        (self.base / "index").mkdir(parents=True, exist_ok=True)

        self._opt_buf:  list[dict] = []
        self._idx_buf:  list[dict] = []
        self._flush_n = flush_interval

    def write_tick(self, tick: dict) -> None:
        self._opt_buf.append(tick)
        if len(self._opt_buf) >= self._flush_n:
            self.flush_options()

    def write_index(self, currency: str, timestamp: float, price: float) -> None:
        self._idx_buf.append({"timestamp": timestamp, "currency": currency, "price": price})
        if len(self._idx_buf) >= self._flush_n:
            self.flush_index()

    def flush_options(self) -> None:
        if not self._opt_buf:
            return
        self._flush_buf(self._opt_buf, "options", _OPTION_SCHEMA)
        logger.debug(f"flushed {len(self._opt_buf)} option ticks")
        self._opt_buf.clear()

    def flush_index(self) -> None:
        if not self._idx_buf:
            return
        self._flush_buf(self._idx_buf, "index", _INDEX_SCHEMA)
        self._idx_buf.clear()

    def _flush_buf(self, buf: list[dict], dataset: str, schema: pa.Schema) -> None:
        currencies = {r.get("currency", "UNKNOWN") for r in buf}
        for currency in currencies:
            rows = [r for r in buf if r.get("currency") == currency]
            if not rows:
                continue

            day = datetime.utcfromtimestamp(rows[0]["timestamp"]).date()
            part_dir = self.base / dataset / f"currency={currency}" / f"date={day}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_file = part_dir / "data.parquet"

            table = pa.Table.from_pylist(rows, schema=schema)
            if part_file.exists():
                table = pa.concat_tables([pq.read_table(part_file), table])
            pq.write_table(table, part_file, compression="zstd")

    def read_options(
        self,
        currency: str,
        start_date: date,
        end_date: date,
        columns: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        parts = []
        d = start_date
        while d <= end_date:
            f = self.base / "options" / f"currency={currency}" / f"date={d}" / "data.parquet"
            if f.exists():
                df = pl.read_parquet(str(f), columns=columns)
                parts.append(df.with_columns(pl.lit(currency).alias("currency")))
            d += timedelta(days=1)

        if not parts:
            logger.warning(f"no option data: {currency} [{start_date} - {end_date}]")
            return pl.DataFrame()

        return pl.concat(parts).sort("timestamp")

    def read_index(self, currency: str, start_date: date, end_date: date) -> pl.DataFrame:
        parts = []
        d = start_date
        while d <= end_date:
            f = self.base / "index" / f"currency={currency}" / f"date={d}" / "data.parquet"
            if f.exists():
                parts.append(pl.read_parquet(str(f)))
            d += timedelta(days=1)
        return pl.concat(parts).sort("timestamp") if parts else pl.DataFrame()

    def get_option_chain(
        self,
        currency: str,
        as_of: float,
        expiry: float,
        tolerance_sec: float = 30.0,
    ) -> pl.DataFrame:
        day = datetime.utcfromtimestamp(as_of).date()
        df  = self.read_options(currency, day, day)
        if df.is_empty():
            return df

        df = df.filter(
            (pl.col("expiry_ts") == expiry) &
            pl.col("timestamp").is_between(as_of - tolerance_sec, as_of + tolerance_sec)
        )
        if df.is_empty():
            return df

        latest_ts = df["timestamp"].max()
        return df.filter(pl.col("timestamp") == latest_ts)

"""
Load M5-format retail data from local CSV or Parquet files.

The M5 competition data (https://www.kaggle.com/c/m5-forecasting-accuracy) comes
as three files:

- ``sales_train_evaluation.csv``: one row per item-store series, one column per
  day (``d_1`` ... ``d_1941``).
- ``calendar.csv``: one row per day, with events and SNAP flags.
- ``sell_prices.csv``: weekly prices per item and store.

This module reads those files (or Parquet equivalents), turns sales into long
format and joins everything into one frame with one row per series per day.
It reads local files only. It never falls back to synthetic data: if a file is
missing you get an error, not a model silently trained on made-up numbers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from src.utils.config import Config

logger = logging.getLogger(__name__)

SERIES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _scan(path: Path) -> pl.LazyFrame:
    """Lazily scan a CSV or Parquet file, chosen by suffix."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the M5 files from Kaggle into this directory, "
            "or run `python -m src.train --smoke` to use synthetic data instead."
        )
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    if path.suffix == ".csv":
        return pl.scan_csv(path)
    raise ValueError(f"Unsupported file type for {path}: expected .csv or .parquet")


def _ensure_date(calendar: pl.DataFrame) -> pl.DataFrame:
    """Make sure the calendar ``date`` column is a Date, not a string."""
    if calendar.schema["date"] == pl.String:
        return calendar.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    if calendar.schema["date"] != pl.Date:
        return calendar.with_columns(pl.col("date").cast(pl.Date))
    return calendar


def sales_to_long(sales: pl.DataFrame) -> pl.DataFrame:
    """Convert wide M5 sales (one column per day) to long format.

    Long-format input (already has a ``d`` column) is returned unchanged apart
    from casting ``sales`` to float, so the target can later hold nulls.
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if day_cols:
        id_cols = [c for c in sales.columns if not c.startswith("d_")]
        sales = sales.unpivot(index=id_cols, on=day_cols, variable_name="d", value_name="sales")
    return sales.with_columns(pl.col("sales").cast(pl.Float64))


def sales_to_wide(sales_long: pl.DataFrame) -> pl.DataFrame:
    """Convert long sales back to the wide M5 layout (inverse of ``sales_to_long``)."""
    day_order = (
        sales_long.select("d")
        .unique()
        .with_columns(pl.col("d").str.slice(2).cast(pl.Int64).alias("_n"))
        .sort("_n")["d"]
        .to_list()
    )
    id_cols = [c for c in SERIES_ID_COLUMNS if c in sales_long.columns]
    wide = sales_long.pivot(on="d", index=id_cols, values="sales")
    return wide.select([*id_cols, *day_order])


def merge_m5_frames(
    sales: pl.DataFrame, calendar: pl.DataFrame, prices: pl.DataFrame
) -> pl.DataFrame:
    """Join sales, calendar and prices into one long frame, sorted by series and date.

    Adds a ``snap`` column holding the SNAP flag for each row's own state and
    drops the per-state ``snap_XX`` columns, which would otherwise hand the model
    two irrelevant flags on every row.
    """
    sales = sales_to_long(sales)
    calendar = _ensure_date(calendar)

    merged = sales.join(calendar, on="d", how="left").join(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )

    snap_cols = [c for c in merged.columns if c.startswith("snap_")]
    if snap_cols and "state_id" in merged.columns:
        snap_expr: pl.Expr = pl.lit(0, dtype=pl.Int8)
        for col in snap_cols:
            state = col.removeprefix("snap_")
            snap_expr = (
                pl.when(pl.col("state_id") == state)
                .then(pl.col(col).cast(pl.Int8))
                .otherwise(snap_expr)
            )
        merged = merged.with_columns(snap_expr.alias("snap")).drop(snap_cols)

    return merged.sort(["id", "date"])


def write_m5_files(
    sales_long: pl.DataFrame,
    calendar: pl.DataFrame,
    prices: pl.DataFrame,
    out_dir: str | Path,
    config: Config | None = None,
) -> dict[str, Path]:
    """Write frames to disk in the M5 CSV layout, using the file names in ``config``."""
    data_config = (config or Config()).data
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "sales": out / data_config.sales_file,
        "calendar": out / data_config.calendar_file,
        "prices": out / data_config.prices_file,
    }
    sales_to_wide(sales_long).with_columns(pl.col(pl.Float64).cast(pl.Int64)).write_csv(
        paths["sales"]
    )
    calendar.write_csv(paths["calendar"])
    prices.write_csv(paths["prices"])
    return paths


class DataLoader:
    """Load and merge M5-format files from a local directory.

    Example:
        >>> loader = DataLoader(data_dir="data/raw")
        >>> df = loader.load_all(stores=["CA_1"])
    """

    def __init__(self, config: Config | None = None, data_dir: str | Path | None = None) -> None:
        data_config = (config or Config()).data
        root = Path(data_dir) if data_dir is not None else Path(data_config.raw_path)
        self.paths: dict[str, Path] = {
            "sales": root / data_config.sales_file,
            "calendar": root / data_config.calendar_file,
            "prices": root / data_config.prices_file,
        }

    def load_sales(self, stores: Sequence[str] | None = None) -> pl.DataFrame:
        """Load sales in long format, optionally restricted to some stores."""
        scan = _scan(self.paths["sales"])
        if stores:
            scan = scan.filter(pl.col("store_id").is_in(list(stores)))
        return sales_to_long(scan.collect())

    def load_calendar(self) -> pl.DataFrame:
        """Load the calendar with ``date`` parsed as a Date."""
        return _ensure_date(_scan(self.paths["calendar"]).collect())

    def load_prices(self, stores: Sequence[str] | None = None) -> pl.DataFrame:
        """Load weekly prices, optionally restricted to some stores."""
        scan = _scan(self.paths["prices"])
        if stores:
            scan = scan.filter(pl.col("store_id").is_in(list(stores)))
        return scan.collect()

    def load_all(self, stores: Sequence[str] | None = None) -> pl.DataFrame:
        """Load all three files and merge them into one long frame."""
        sales = self.load_sales(stores)
        if sales.is_empty():
            raise ValueError(f"No sales rows found for stores={list(stores or [])}")
        merged = merge_m5_frames(sales, self.load_calendar(), self.load_prices(stores))
        logger.info(
            "Loaded %s rows, %s series, %s columns",
            f"{len(merged):,}",
            f"{merged['id'].n_unique():,}",
            len(merged.columns),
        )
        return merged

"""
Hierarchical features: how each series' wider group (item, department,
category, store, state) has been selling.

For every hierarchy level the module builds that level's daily total, shifts it
by ``min_lag`` days and takes rolling statistics, then joins the result back to
each row. It also adds each series' share of its level's volume, again from
lagged values only.

Earlier versions of this module joined same-day level totals (and same-day
shares such as ``sales / category_daily_sum``) onto each row. Those contain
the row's own target, so a model could "predict" sales it had already been
shown. Nothing here uses target values younger than ``min_lag`` days.
"""

from __future__ import annotations

import logging

import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HierarchicalFeaturesConfig(BaseModel):
    """Configuration for hierarchical features."""

    hierarchy_levels: list[str] = Field(
        default=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        description="Grouping columns; levels missing from the data are skipped",
    )
    rolling_window: int = Field(default=28, ge=1, description="Rolling window in days")
    min_lag: int = Field(default=28, ge=1, description="Minimum age in days of target values")


class HierarchicalFeatures:
    """Generate lagged, level-aggregated demand features.

    Example:
        >>> hierarchical = HierarchicalFeatures()
        >>> features = hierarchical.transform(df)
    """

    def __init__(self, config: HierarchicalFeaturesConfig | None = None) -> None:
        self.config = config or HierarchicalFeaturesConfig()

    def transform(
        self,
        df: pl.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        group_col: str = "id",
    ) -> pl.DataFrame:
        """Add level aggregates and lagged shares for every level present in ``df``."""
        window, lag = self.config.rolling_window, self.config.min_lag
        own_roll = f"_own_roll_mean_{window}"
        df = df.sort([group_col, date_col]).with_columns(
            pl.col(target_col)
            .shift(lag)
            .rolling_mean(window_size=window)
            .over(group_col)
            .alias(own_roll)
        )

        for level in self.config.hierarchy_levels:
            if level in df.columns:
                df = self._level_features(df, target_col, level, date_col)
                name = level.removesuffix("_id")
                df = df.with_columns(
                    (pl.col(own_roll) / (pl.col(f"{name}_agg_roll_mean_{window}") + 1e-8)).alias(
                        f"share_of_{name}"
                    )
                )
        return df.drop(own_roll)

    def _level_features(
        self, df: pl.DataFrame, target_col: str, level_col: str, date_col: str
    ) -> pl.DataFrame:
        name = level_col.removesuffix("_id")
        window, lag = self.config.rolling_window, self.config.min_lag
        target = pl.col(target_col)

        # A level-day total is null (unknown) when every member is null, for
        # example after the forecast origin where the target is masked.
        daily = (
            df.group_by([level_col, date_col])
            .agg(pl.when(target.is_null().all()).then(None).otherwise(target.sum()).alias("_total"))
            .sort([level_col, date_col])
        )
        shifted = pl.col("_total").shift(lag)
        daily = daily.with_columns(
            shifted.rolling_mean(window_size=window)
            .over(level_col)
            .alias(f"{name}_agg_roll_mean_{window}"),
            shifted.rolling_std(window_size=window)
            .over(level_col)
            .alias(f"{name}_agg_roll_std_{window}"),
        ).drop("_total")

        return df.join(daily, on=[level_col, date_col], how="left")

    def get_feature_names(self, columns: list[str] | None = None) -> list[str]:
        """Feature names produced for the given input columns (all levels if None)."""
        window = self.config.rolling_window
        names = []
        for level in self.config.hierarchy_levels:
            if columns is not None and level not in columns:
                continue
            name = level.removesuffix("_id")
            names += [
                f"{name}_agg_roll_mean_{window}",
                f"{name}_agg_roll_std_{window}",
                f"share_of_{name}",
            ]
        return names

    def aggregate_forecasts(self, forecasts: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """Bottom-up aggregation: add ``forecast_<level>`` totals for each level present."""
        result = forecasts
        for level in self.config.hierarchy_levels:
            if level in result.columns:
                totals = result.group_by([level, date_col]).agg(
                    pl.col("forecast").sum().alias(f"forecast_{level}")
                )
                result = result.join(totals, on=[level, date_col], how="left")
        return result

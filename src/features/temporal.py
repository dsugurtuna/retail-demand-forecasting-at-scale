"""
Temporal features: lags, rolling statistics, calendar and holiday signals.

Leakage rule: every feature built from the target uses values at least
``min_lag`` days old. With ``min_lag`` equal to the forecast horizon (28 days
for M5), a single model can forecast every day of the horizon directly from
information available at the forecast origin, with no recursive feeding of
its own predictions.

Calendar and holiday features depend only on the date, so they are known in
advance and need no shift. ``add_calendar_features`` is shared with the
serving layer so training and serving compute them the same way.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import holidays
import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HOLIDAY_WINDOW_DAYS = 14


class TemporalFeaturesConfig(BaseModel):
    """Configuration for temporal features."""

    lags: list[int] = Field(
        default=[7, 14, 21, 28, 35, 42, 49, 56],
        description="Candidate lags; only those >= min_lag are used",
    )
    rolling_windows: list[int] = Field(
        default=[7, 14, 28, 56, 112], description="Window sizes for rolling statistics"
    )
    rolling_statistics: list[str] = Field(
        default=["mean", "std", "min", "max"],
        description="Statistics for each rolling window (mean, std, min, max)",
    )
    expanding_windows: bool = Field(default=True, description="Add expanding-window features")
    cyclical_encoding: bool = Field(default=True, description="Add sin/cos calendar encodings")
    holiday_countries: list[str] = Field(
        default=["UK", "US"], description="Country codes for the `holidays` package"
    )
    min_lag: int = Field(
        default=28, ge=1, description="Minimum age in days of any target value used"
    )


def add_calendar_features(
    df: pl.DataFrame, date_col: str = "date", cyclical: bool = True
) -> pl.DataFrame:
    """Add date-derived features. ``day_of_week`` is 0 = Monday ... 6 = Sunday."""
    date = pl.col(date_col)
    dow = (date.dt.weekday() - 1).cast(pl.Int8)
    df = df.with_columns(
        dow.alias("day_of_week"),
        date.dt.day().alias("day_of_month"),
        date.dt.ordinal_day().alias("day_of_year"),
        date.dt.week().alias("week_of_year"),
        date.dt.month().alias("month"),
        date.dt.quarter().alias("quarter"),
        date.dt.year().alias("year"),
        (dow >= 5).cast(pl.Int8).alias("is_weekend"),
        (date.dt.day() == 1).cast(pl.Int8).alias("is_month_start"),
        (date.dt.day() <= 7).cast(pl.Int8).alias("is_month_first_week"),
        (date.dt.day() >= 25).cast(pl.Int8).alias("is_month_end"),
    )
    if cyclical:
        two_pi = 2 * np.pi
        df = df.with_columns(
            (two_pi * pl.col("day_of_week") / 7).sin().alias("dow_sin"),
            (two_pi * pl.col("day_of_week") / 7).cos().alias("dow_cos"),
            (two_pi * pl.col("day_of_month") / 31).sin().alias("dom_sin"),
            (two_pi * pl.col("day_of_month") / 31).cos().alias("dom_cos"),
            (two_pi * pl.col("week_of_year") / 52).sin().alias("woy_sin"),
            (two_pi * pl.col("week_of_year") / 52).cos().alias("woy_cos"),
            (two_pi * pl.col("month") / 12).sin().alias("month_sin"),
            (two_pi * pl.col("month") / 12).cos().alias("month_cos"),
        )
    return df


class TemporalFeatures:
    """Generate leakage-safe temporal features for demand forecasting.

    Example:
        >>> temporal = TemporalFeatures(TemporalFeaturesConfig(lags=[28, 35]))
        >>> features = temporal.transform(df)
    """

    def __init__(self, config: TemporalFeaturesConfig | None = None) -> None:
        self.config = config or TemporalFeaturesConfig()
        self._holiday_calendars: dict[str, Any] = {}
        for country in self.config.holiday_countries:
            try:
                self._holiday_calendars[country] = holidays.country_holidays(country)
            except (KeyError, NotImplementedError) as exc:
                logger.warning("Could not load holidays for %s: %s", country, exc)

    @property
    def used_lags(self) -> list[int]:
        """Lags that respect ``min_lag``."""
        return [lag for lag in self.config.lags if lag >= self.config.min_lag]

    def transform(
        self,
        df: pl.DataFrame,
        target_col: str = "sales",
        group_col: str = "id",
        date_col: str = "date",
    ) -> pl.DataFrame:
        """Add all temporal features. Input need not be sorted."""
        df = df.sort([group_col, date_col])
        df = self._lag_features(df, target_col, group_col)
        df = self._rolling_features(df, target_col, group_col)
        if self.config.expanding_windows:
            df = self._expanding_features(df, target_col, group_col)
        df = add_calendar_features(df, date_col, self.config.cyclical_encoding)
        df = self._holiday_features(df, date_col)
        return df.with_columns(pl.int_range(1, pl.len() + 1).over(group_col).alias("time_idx"))

    def _lag_features(self, df: pl.DataFrame, target_col: str, group_col: str) -> pl.DataFrame:
        target = pl.col(target_col)
        exprs = []
        for lag in self.used_lags:
            exprs.append(target.shift(lag).over(group_col).alias(f"{target_col}_lag_{lag}"))
            exprs.append(
                (target.shift(lag) - target.shift(lag + 7))
                .over(group_col)
                .alias(f"{target_col}_lag_{lag}_diff_7")
            )
        return df.with_columns(exprs) if exprs else df

    def _rolling_features(self, df: pl.DataFrame, target_col: str, group_col: str) -> pl.DataFrame:
        shifted = pl.col(target_col).shift(self.config.min_lag)
        stats = self.config.rolling_statistics
        exprs = []
        for window in self.config.rolling_windows:
            if "mean" in stats:
                exprs.append(
                    shifted.rolling_mean(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_mean_{window}")
                )
            if "std" in stats:
                exprs.append(
                    shifted.rolling_std(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_std_{window}")
                )
            if "min" in stats:
                exprs.append(
                    shifted.rolling_min(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_min_{window}")
                )
            if "max" in stats:
                exprs.append(
                    shifted.rolling_max(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_max_{window}")
                )
            exprs.append(
                (
                    shifted.rolling_std(window_size=window)
                    / (shifted.rolling_mean(window_size=window) + 1e-8)
                )
                .over(group_col)
                .alias(f"{target_col}_roll_cv_{window}")
            )
        return df.with_columns(exprs) if exprs else df

    def _expanding_features(
        self, df: pl.DataFrame, target_col: str, group_col: str
    ) -> pl.DataFrame:
        shifted = pl.col(target_col).shift(self.config.min_lag)
        return df.with_columns(
            (shifted.cum_sum() / shifted.cum_count())
            .over(group_col)
            .alias(f"{target_col}_expanding_mean"),
            (shifted == 0).cum_sum().over(group_col).alias(f"{target_col}_zero_days_cum"),
        )

    def _holiday_features(self, df: pl.DataFrame, date_col: str) -> pl.DataFrame:
        """Holiday flags and distances, computed once per unique date and joined back."""
        if not self._holiday_calendars:
            return df
        unique_dates = df.get_column(date_col).unique().sort()
        py_dates = unique_dates.to_list()
        table: dict[str, Any] = {date_col: unique_dates}
        window = range(1, HOLIDAY_WINDOW_DAYS + 1)
        for country, cal in self._holiday_calendars.items():
            table[f"is_holiday_{country}"] = [int(d in cal) for d in py_dates]
            table[f"days_to_holiday_{country}"] = [
                next((k for k in window if d + timedelta(days=k) in cal), -1) for d in py_dates
            ]
            table[f"days_from_holiday_{country}"] = [
                next((k for k in window if d - timedelta(days=k) in cal), -1) for d in py_dates
            ]
        return df.join(pl.DataFrame(table), on=date_col, how="left")

    def get_feature_names(self, target_col: str = "sales") -> list[str]:
        """Names of the features this configuration produces."""
        names = []
        for lag in self.used_lags:
            names += [f"{target_col}_lag_{lag}", f"{target_col}_lag_{lag}_diff_7"]
        for window in self.config.rolling_windows:
            names += [f"{target_col}_roll_{s}_{window}" for s in self.config.rolling_statistics]
            names.append(f"{target_col}_roll_cv_{window}")
        if self.config.expanding_windows:
            names += [f"{target_col}_expanding_mean", f"{target_col}_zero_days_cum"]
        names += [
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "is_weekend",
            "is_month_start",
            "is_month_first_week",
            "is_month_end",
        ]
        if self.config.cyclical_encoding:
            names += [f"{p}_{f}" for p in ("dow", "dom", "woy", "month") for f in ("sin", "cos")]
        for country in self._holiday_calendars:
            names += [
                f"is_holiday_{country}",
                f"days_to_holiday_{country}",
                f"days_from_holiday_{country}",
            ]
        names.append("time_idx")
        return names

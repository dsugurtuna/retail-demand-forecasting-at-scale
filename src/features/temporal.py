"""
Temporal feature engineering for time series forecasting.

Implements comprehensive temporal features including:
- Lag features with various windows
- Rolling statistics (mean, std, min, max, quantiles)
- Expanding window features
- Calendar features with cyclical encoding
- Holiday and event features
"""

from __future__ import annotations

import logging
from typing import Any

import holidays
import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TemporalFeaturesConfig(BaseModel):
    """Configuration for temporal features."""

    # Lag features
    lags: list[int] = Field(
        default=[7, 14, 21, 28, 35, 42, 49, 56], description="Lag periods for creating lag features"
    )

    # Rolling features
    rolling_windows: list[int] = Field(
        default=[7, 14, 28, 56, 112], description="Window sizes for rolling statistics"
    )
    rolling_statistics: list[str] = Field(
        default=["mean", "std", "min", "max"],
        description="Statistics to compute for rolling windows",
    )

    # Expanding features
    expanding_windows: bool = Field(
        default=True, description="Whether to include expanding window features"
    )

    # Calendar features
    cyclical_encoding: bool = Field(
        default=True, description="Use sin/cos encoding for cyclical features"
    )

    # Holidays
    holiday_countries: list[str] = Field(
        default=["UK", "US"], description="Countries for holiday calendars"
    )

    # Shift for preventing leakage
    min_lag: int = Field(
        default=28, description="Minimum lag to prevent data leakage (forecast horizon)"
    )


class TemporalFeatures:
    """
    Temporal feature engineering for demand forecasting.

    Generates comprehensive temporal features including:
    - Lag features at various horizons
    - Rolling window statistics
    - Expanding window features
    - Calendar features with cyclical encoding
    - Holiday indicators

    Example:
        >>> config = TemporalFeaturesConfig(lags=[7, 14, 28])
        >>> temporal = TemporalFeatures(config)
        >>> features = temporal.transform(df)
    """

    def __init__(self, config: TemporalFeaturesConfig | None = None) -> None:
        """
        Initialize temporal feature generator.

        Args:
            config: Feature configuration
        """
        self.config = config or TemporalFeaturesConfig()
        self._holiday_calendars: dict[str, Any] = {}
        self._setup_holidays()

    def _setup_holidays(self) -> None:
        """Initialize holiday calendars."""
        for country in self.config.holiday_countries:
            try:
                self._holiday_calendars[country] = holidays.country_holidays(country)
            except Exception as e:
                logger.warning(f"Could not load holidays for {country}: {e}")

    def transform(
        self,
        df: pl.DataFrame,
        target_col: str = "sales",
        group_col: str = "id",
        date_col: str = "date",
    ) -> pl.DataFrame:
        """
        Generate all temporal features.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            group_col: Column to group by for lag features
            date_col: Date column name

        Returns:
            DataFrame with temporal features added
        """
        logger.info("Generating temporal features...")

        # Ensure sorted by group and date
        df = df.sort([group_col, date_col])

        # Generate features
        df = self._create_lag_features(df, target_col, group_col)
        df = self._create_rolling_features(df, target_col, group_col)

        if self.config.expanding_windows:
            df = self._create_expanding_features(df, target_col, group_col)

        df = self._create_calendar_features(df, date_col)
        df = self._create_holiday_features(df, date_col)
        df = self._create_trend_features(df, group_col, date_col)

        n_features = len([c for c in df.columns if c not in [target_col, group_col, date_col]])
        logger.info(f"Generated {n_features} temporal features")

        return df

    def _create_lag_features(
        self, df: pl.DataFrame, target_col: str, group_col: str
    ) -> pl.DataFrame:
        """Create lag features."""
        lag_exprs = []

        for lag in self.config.lags:
            if lag >= self.config.min_lag:  # Only use lags that don't leak
                lag_exprs.append(
                    pl.col(target_col).shift(lag).over(group_col).alias(f"{target_col}_lag_{lag}")
                )

                # Lag difference
                lag_exprs.append(
                    (pl.col(target_col).shift(lag) - pl.col(target_col).shift(lag + 7))
                    .over(group_col)
                    .alias(f"{target_col}_lag_{lag}_diff_7")
                )

        if lag_exprs:
            df = df.with_columns(lag_exprs)

        return df

    def _create_rolling_features(
        self, df: pl.DataFrame, target_col: str, group_col: str
    ) -> pl.DataFrame:
        """Create rolling window features."""
        roll_exprs = []

        for window in self.config.rolling_windows:
            base_shift = self.config.min_lag  # Shift to prevent leakage

            if "mean" in self.config.rolling_statistics:
                roll_exprs.append(
                    pl.col(target_col)
                    .shift(base_shift)
                    .rolling_mean(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_mean_{window}")
                )

            if "std" in self.config.rolling_statistics:
                roll_exprs.append(
                    pl.col(target_col)
                    .shift(base_shift)
                    .rolling_std(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_std_{window}")
                )

            if "min" in self.config.rolling_statistics:
                roll_exprs.append(
                    pl.col(target_col)
                    .shift(base_shift)
                    .rolling_min(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_min_{window}")
                )

            if "max" in self.config.rolling_statistics:
                roll_exprs.append(
                    pl.col(target_col)
                    .shift(base_shift)
                    .rolling_max(window_size=window)
                    .over(group_col)
                    .alias(f"{target_col}_roll_max_{window}")
                )

            # Coefficient of variation
            roll_exprs.append(
                (
                    pl.col(target_col).shift(base_shift).rolling_std(window_size=window)
                    / (pl.col(target_col).shift(base_shift).rolling_mean(window_size=window) + 1e-8)
                )
                .over(group_col)
                .alias(f"{target_col}_roll_cv_{window}")
            )

        if roll_exprs:
            df = df.with_columns(roll_exprs)

        return df

    def _create_expanding_features(
        self, df: pl.DataFrame, target_col: str, group_col: str
    ) -> pl.DataFrame:
        """Create expanding window features (cumulative statistics)."""
        base_shift = self.config.min_lag

        expand_exprs = [
            # Expanding mean
            pl.col(target_col)
            .shift(base_shift)
            .cum_sum()
            .over(group_col)
            .truediv(pl.lit(1).cum_sum().over(group_col))
            .alias(f"{target_col}_expanding_mean"),
            # Days since last zero
            (pl.col(target_col) == 0).cum_sum().over(group_col).alias(f"{target_col}_cumsum_zeros"),
        ]

        df = df.with_columns(expand_exprs)
        return df

    def _create_calendar_features(self, df: pl.DataFrame, date_col: str) -> pl.DataFrame:
        """Create calendar-based features."""
        # Basic calendar features
        calendar_exprs = [
            pl.col(date_col).dt.weekday().alias("day_of_week"),
            pl.col(date_col).dt.day().alias("day_of_month"),
            pl.col(date_col).dt.ordinal_day().alias("day_of_year"),
            pl.col(date_col).dt.week().alias("week_of_year"),
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.quarter().alias("quarter"),
            pl.col(date_col).dt.year().alias("year"),
            # Binary flags
            (pl.col(date_col).dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
            (pl.col(date_col).dt.day() == 1).cast(pl.Int8).alias("is_month_start"),
            (pl.col(date_col).dt.day() <= 7).cast(pl.Int8).alias("is_month_first_week"),
            (pl.col(date_col).dt.day() >= 25).cast(pl.Int8).alias("is_month_end"),
        ]

        df = df.with_columns(calendar_exprs)

        # Cyclical encoding
        if self.config.cyclical_encoding:
            cyclical_exprs = [
                # Day of week
                (2 * np.pi * pl.col("day_of_week") / 7).sin().alias("dow_sin"),
                (2 * np.pi * pl.col("day_of_week") / 7).cos().alias("dow_cos"),
                # Day of month
                (2 * np.pi * pl.col("day_of_month") / 31).sin().alias("dom_sin"),
                (2 * np.pi * pl.col("day_of_month") / 31).cos().alias("dom_cos"),
                # Week of year
                (2 * np.pi * pl.col("week_of_year") / 52).sin().alias("woy_sin"),
                (2 * np.pi * pl.col("week_of_year") / 52).cos().alias("woy_cos"),
                # Month
                (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
                (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),
            ]
            df = df.with_columns(cyclical_exprs)

        return df

    def _create_holiday_features(self, df: pl.DataFrame, date_col: str) -> pl.DataFrame:
        """Create holiday indicator features."""
        if not self._holiday_calendars:
            return df

        # Convert to pandas for holiday lookup (more convenient)
        dates = df.select(date_col).to_pandas()[date_col]

        for country, cal in self._holiday_calendars.items():
            holiday_flags = []
            days_to_holiday = []
            days_from_holiday = []

            for date in dates:
                try:
                    is_holiday = date in cal
                    holiday_flags.append(int(is_holiday))

                    # Days to/from nearest holiday (simplified)
                    nearest_before = 999
                    nearest_after = 999

                    for offset in range(1, 15):
                        check_before = date - np.timedelta64(offset, "D")
                        check_after = date + np.timedelta64(offset, "D")

                        if check_before in cal and offset < nearest_before:
                            nearest_before = offset
                        if check_after in cal and offset < nearest_after:
                            nearest_after = offset

                    days_to_holiday.append(nearest_after if nearest_after < 999 else -1)
                    days_from_holiday.append(nearest_before if nearest_before < 999 else -1)

                except Exception:
                    holiday_flags.append(0)
                    days_to_holiday.append(-1)
                    days_from_holiday.append(-1)

            df = df.with_columns(
                [
                    pl.Series(f"is_holiday_{country}", holiday_flags),
                    pl.Series(f"days_to_holiday_{country}", days_to_holiday),
                    pl.Series(f"days_from_holiday_{country}", days_from_holiday),
                ]
            )

        return df

    def _create_trend_features(
        self, df: pl.DataFrame, group_col: str, date_col: str
    ) -> pl.DataFrame:
        """Create trend-based features."""
        trend_exprs = [
            # Time index within each series
            pl.lit(1).cum_sum().over(group_col).alias("time_idx"),
            # Normalized time (0 to 1)
            (
                pl.lit(1).cum_sum().over(group_col)
                / pl.lit(1).cum_sum().over(group_col).max().over(group_col)
            ).alias("time_normalized"),
        ]

        df = df.with_columns(trend_exprs)
        return df

    def get_feature_names(self) -> list[str]:
        """Get list of feature names that will be generated."""
        features = []

        # Lag features
        for lag in self.config.lags:
            if lag >= self.config.min_lag:
                features.append(f"sales_lag_{lag}")
                features.append(f"sales_lag_{lag}_diff_7")

        # Rolling features
        for window in self.config.rolling_windows:
            for stat in self.config.rolling_statistics:
                features.append(f"sales_roll_{stat}_{window}")
            features.append(f"sales_roll_cv_{window}")

        # Calendar features
        features.extend(
            [
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
        )

        if self.config.cyclical_encoding:
            features.extend(
                [
                    "dow_sin",
                    "dow_cos",
                    "dom_sin",
                    "dom_cos",
                    "woy_sin",
                    "woy_cos",
                    "month_sin",
                    "month_cos",
                ]
            )

        return features

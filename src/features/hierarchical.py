"""
Hierarchical feature engineering for multi-level forecasting.

Implements cross-level aggregation features:
- Item-level features aggregated from stores
- Category-level features
- Store-level features
- Regional features
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
        description="Hierarchy levels from bottom to top",
    )

    aggregations: list[str] = Field(
        default=["mean", "std", "sum", "count"], description="Aggregation functions to apply"
    )

    rolling_window: int = Field(default=28, description="Rolling window for aggregations")

    min_lag: int = Field(default=28, description="Minimum lag to prevent leakage")


class HierarchicalFeatures:
    """
    Hierarchical feature engineering for multi-level demand forecasting.

    Generates cross-level aggregation features that capture:
    - Item performance across stores
    - Category-level trends
    - Store-level performance
    - Regional patterns

    This is critical for hierarchical forecasting where predictions
    at different levels need to be coherent.

    Example:
        >>> config = HierarchicalFeaturesConfig()
        >>> hierarchical = HierarchicalFeatures(config)
        >>> features = hierarchical.transform(df)
    """

    def __init__(self, config: HierarchicalFeaturesConfig | None = None) -> None:
        """
        Initialize hierarchical feature generator.

        Args:
            config: Feature configuration
        """
        self.config = config or HierarchicalFeaturesConfig()

    def transform(
        self, df: pl.DataFrame, target_col: str = "sales", date_col: str = "date"
    ) -> pl.DataFrame:
        """
        Generate all hierarchical features.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            date_col: Date column name

        Returns:
            DataFrame with hierarchical features added
        """
        logger.info("Generating hierarchical features...")

        # Generate aggregations at each hierarchy level
        for level in self.config.hierarchy_levels:
            if level in df.columns:
                df = self._create_level_features(df, target_col, level, date_col)

        # Create cross-level features
        df = self._create_cross_level_features(df, target_col)

        n_features = len([c for c in df.columns if "_agg_" in c or "_share" in c])
        logger.info(f"Generated {n_features} hierarchical features")

        return df

    def _create_level_features(
        self, df: pl.DataFrame, target_col: str, level_col: str, date_col: str
    ) -> pl.DataFrame:
        """Create aggregated features at a hierarchy level."""
        level_name = level_col.replace("_id", "")
        base_shift = self.config.min_lag

        # Daily aggregation at this level
        daily_agg = df.group_by([level_col, date_col]).agg(
            [
                pl.col(target_col).sum().alias(f"{level_name}_daily_sum"),
                pl.col(target_col).mean().alias(f"{level_name}_daily_mean"),
                pl.col(target_col).std().alias(f"{level_name}_daily_std"),
                pl.col(target_col).count().alias(f"{level_name}_daily_count"),
            ]
        )

        df = df.join(daily_agg, on=[level_col, date_col], how="left")

        # Rolling aggregations at this level
        rolling_exprs = []
        window = self.config.rolling_window

        if f"{level_name}_daily_sum" in df.columns:
            rolling_exprs.extend(
                [
                    pl.col(f"{level_name}_daily_sum")
                    .shift(base_shift)
                    .rolling_mean(window_size=window)
                    .over(level_col)
                    .alias(f"{level_name}_agg_roll_mean_{window}"),
                    pl.col(f"{level_name}_daily_sum")
                    .shift(base_shift)
                    .rolling_std(window_size=window)
                    .over(level_col)
                    .alias(f"{level_name}_agg_roll_std_{window}"),
                ]
            )

        if rolling_exprs:
            df = df.with_columns(rolling_exprs)

        return df

    def _create_cross_level_features(self, df: pl.DataFrame, target_col: str) -> pl.DataFrame:
        """Create features that compare across hierarchy levels."""
        exprs = []

        # Share of category
        if "cat_daily_sum" in df.columns:
            exprs.append(
                (pl.col(target_col) / (pl.col("cat_daily_sum") + 1e-8)).alias("share_of_category")
            )

        # Share of store
        if "store_daily_sum" in df.columns:
            exprs.append(
                (pl.col(target_col) / (pl.col("store_daily_sum") + 1e-8)).alias("share_of_store")
            )

        # Share of state/region
        if "state_daily_sum" in df.columns:
            exprs.append(
                (pl.col(target_col) / (pl.col("state_daily_sum") + 1e-8)).alias("share_of_state")
            )

        # Item performance vs category average
        if "cat_daily_mean" in df.columns:
            exprs.append(
                (pl.col(target_col) / (pl.col("cat_daily_mean") + 1e-8)).alias("vs_category_avg")
            )

        # Item performance vs store average
        if "store_daily_mean" in df.columns:
            exprs.append(
                (pl.col(target_col) / (pl.col("store_daily_mean") + 1e-8)).alias("vs_store_avg")
            )

        if exprs:
            df = df.with_columns(exprs)

        return df

    def aggregate_forecasts(
        self, forecasts: pl.DataFrame, method: str = "bottom_up"
    ) -> pl.DataFrame:
        """
        Aggregate forecasts across hierarchy levels.

        Args:
            forecasts: DataFrame with bottom-level forecasts
            method: Aggregation method ('bottom_up', 'top_down', 'middle_out')

        Returns:
            DataFrame with aggregated forecasts at all levels
        """
        if method == "bottom_up":
            return self._bottom_up_aggregate(forecasts)
        elif method == "top_down":
            return self._top_down_aggregate(forecasts)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _bottom_up_aggregate(self, forecasts: pl.DataFrame) -> pl.DataFrame:
        """Aggregate forecasts from bottom to top of hierarchy."""
        result = forecasts.clone()

        # Aggregate up each level
        for level in reversed(self.config.hierarchy_levels[:-1]):
            if level in result.columns:
                level_agg = result.group_by([level, "date"]).agg(
                    pl.col("forecast").sum().alias(f"forecast_{level}")
                )
                result = result.join(level_agg, on=[level, "date"], how="left")

        return result

    def _top_down_aggregate(self, forecasts: pl.DataFrame) -> pl.DataFrame:
        """Distribute forecasts from top to bottom using proportions."""
        # This is a simplified implementation
        # Full implementation would use historical proportions
        return forecasts

    def get_feature_names(self) -> list[str]:
        """Get list of feature names that will be generated."""
        features = []

        for level in self.config.hierarchy_levels:
            level_name = level.replace("_id", "")
            features.extend(
                [
                    f"{level_name}_daily_sum",
                    f"{level_name}_daily_mean",
                    f"{level_name}_daily_std",
                    f"{level_name}_daily_count",
                    f"{level_name}_agg_roll_mean_{self.config.rolling_window}",
                    f"{level_name}_agg_roll_std_{self.config.rolling_window}",
                ]
            )

        features.extend(
            [
                "share_of_category",
                "share_of_store",
                "share_of_state",
                "vs_category_avg",
                "vs_store_avg",
            ]
        )

        return features

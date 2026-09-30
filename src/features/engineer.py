"""
Main feature engineering orchestrator.

Coordinates all feature generators and provides a unified interface
for feature engineering pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import BaseModel, Field

from src.features.hierarchical import HierarchicalFeatures, HierarchicalFeaturesConfig
from src.features.price import PriceFeatures, PriceFeaturesConfig
from src.features.temporal import TemporalFeatures, TemporalFeaturesConfig

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineerConfig(BaseModel):
    """Configuration for feature engineering."""

    temporal: TemporalFeaturesConfig = Field(default_factory=TemporalFeaturesConfig)
    price: PriceFeaturesConfig = Field(default_factory=PriceFeaturesConfig)
    hierarchical: HierarchicalFeaturesConfig = Field(default_factory=HierarchicalFeaturesConfig)

    # General settings
    target_col: str = Field(default="sales")
    group_col: str = Field(default="id")
    date_col: str = Field(default="date")

    # Feature selection
    enable_temporal: bool = Field(default=True)
    enable_price: bool = Field(default=True)
    enable_hierarchical: bool = Field(default=True)

    # Post-processing
    drop_na_rows: bool = Field(default=False)
    fill_na_value: float | None = Field(default=0.0)
    drop_zero_variance: bool = Field(default=True)


class FeatureEngineer:
    """
    Main feature engineering orchestrator.

    Coordinates multiple feature generators to produce a comprehensive
    feature set for demand forecasting. Provides:

    - Unified interface for all feature types
    - Automatic feature naming and documentation
    - Feature selection and filtering
    - Post-processing and quality checks

    Example:
        >>> config = FeatureEngineerConfig()
        >>> engineer = FeatureEngineer(config)
        >>> features = engineer.fit_transform(data)
        >>> print(engineer.get_feature_importance())
    """

    def __init__(
        self, config: FeatureEngineerConfig | None = None, app_config: Config | None = None
    ) -> None:
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration
            app_config: Main application configuration
        """
        self.config = config or FeatureEngineerConfig()
        self._app_config = app_config

        # Initialize feature generators
        self._temporal = TemporalFeatures(self.config.temporal)
        self._price = PriceFeatures(self.config.price)
        self._hierarchical = HierarchicalFeatures(self.config.hierarchical)

        # State
        self._fitted = False
        self._feature_names: list[str] = []
        self._zero_variance_cols: list[str] = []

    def fit(self, data: pl.DataFrame) -> FeatureEngineer:
        """
        Fit the feature engineer to training data.

        This learns any statistics needed for transformation
        (e.g., zero variance columns to drop).

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature engineer...")

        # Transform to learn statistics
        features = self._transform(data)

        # Identify zero variance columns
        if self.config.drop_zero_variance:
            numeric_cols = [
                c
                for c in features.columns
                if features.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            ]

            for col in numeric_cols:
                std = features.select(pl.col(col).std()).item()
                if std is not None and std == 0:
                    self._zero_variance_cols.append(col)

            if self._zero_variance_cols:
                logger.info(f"Identified {len(self._zero_variance_cols)} zero-variance columns")

        # Store feature names (excluding target and identifiers)
        exclude_cols = {
            self.config.target_col,
            self.config.group_col,
            self.config.date_col,
            "d",
            "wm_yr_wk",
            "id",
        }
        self._feature_names = [
            c
            for c in features.columns
            if c not in exclude_cols and c not in self._zero_variance_cols
        ]

        self._fitted = True
        logger.info(f"Feature engineer fitted with {len(self._feature_names)} features")
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Transform data to create features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with features
        """
        if not self._fitted:
            logger.warning("Feature engineer not fitted, fitting on transform data")
            self.fit(data)

        features = self._transform(data)

        # Drop zero variance columns
        if self._zero_variance_cols:
            features = features.drop(self._zero_variance_cols)

        return features

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Internal transform method."""
        df = data.clone()

        # Apply feature generators
        if self.config.enable_temporal:
            logger.info("Generating temporal features...")
            df = self._temporal.transform(
                df,
                target_col=self.config.target_col,
                group_col=self.config.group_col,
                date_col=self.config.date_col,
            )

        if self.config.enable_price:
            logger.info("Generating price features...")
            df = self._price.transform(
                df, group_col=self.config.group_col, date_col=self.config.date_col
            )

        if self.config.enable_hierarchical:
            logger.info("Generating hierarchical features...")
            df = self._hierarchical.transform(
                df, target_col=self.config.target_col, date_col=self.config.date_col
            )

        # Post-processing
        df = self._postprocess(df)

        return df

    def _postprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Post-process features."""
        # Handle missing values
        if self.config.fill_na_value is not None:
            numeric_cols = [
                c
                for c in df.columns
                if df.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            ]

            fill_exprs = [pl.col(c).fill_null(self.config.fill_na_value) for c in numeric_cols]
            if fill_exprs:
                df = df.with_columns(fill_exprs)

        # Drop rows with any remaining NaN
        if self.config.drop_na_rows:
            df = df.drop_nulls()

        # Replace infinities
        numeric_cols = [c for c in df.columns if df.schema[c] in (pl.Float64, pl.Float32)]

        for col in numeric_cols:
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(pl.lit(None))
                .otherwise(pl.col(col))
                .alias(col)
            )

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._feature_names.copy()

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Get features organized by group/type."""
        groups = {"temporal": [], "price": [], "hierarchical": [], "calendar": [], "other": []}

        for feature in self._feature_names:
            if any(x in feature for x in ["lag_", "roll_", "expanding"]):
                groups["temporal"].append(feature)
            elif any(x in feature for x in ["price", "promotion", "discount"]):
                groups["price"].append(feature)
            elif any(x in feature for x in ["_agg_", "share_", "_daily_"]):
                groups["hierarchical"].append(feature)
            elif any(x in feature for x in ["day_", "week_", "month_", "is_", "dow_", "dom_"]):
                groups["calendar"].append(feature)
            else:
                groups["other"].append(feature)

        return groups

    def get_feature_summary(self) -> dict[str, Any]:
        """Get summary statistics about features."""
        groups = self.get_feature_groups()

        return {
            "total_features": len(self._feature_names),
            "group_counts": {k: len(v) for k, v in groups.items()},
            "zero_variance_dropped": len(self._zero_variance_cols),
            "feature_groups": groups,
        }

    def select_features(
        self, df: pl.DataFrame, include_target: bool = True, include_identifiers: bool = False
    ) -> pl.DataFrame:
        """
        Select only feature columns from DataFrame.

        Args:
            df: DataFrame with features
            include_target: Whether to include target column
            include_identifiers: Whether to include id columns

        Returns:
            DataFrame with selected columns
        """
        cols = self._feature_names.copy()

        if include_target:
            cols.append(self.config.target_col)

        if include_identifiers:
            cols.extend([self.config.group_col, self.config.date_col])

        available_cols = [c for c in cols if c in df.columns]
        return df.select(available_cols)

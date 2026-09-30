"""
Feature engineering orchestrator.

Runs the temporal, price and hierarchical generators in order, cleans the
result and decides which columns the model sees.

One ``min_lag`` value is pushed down to every generator so the temporal and
hierarchical features cannot drift apart. Set it to the forecast horizon.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from src.features.hierarchical import HierarchicalFeatures, HierarchicalFeaturesConfig
from src.features.price import PriceFeatures, PriceFeaturesConfig
from src.features.temporal import TemporalFeatures, TemporalFeaturesConfig

logger = logging.getLogger(__name__)

# Columns that are identifiers, labels or duplicates, never model inputs.
NON_FEATURE_COLUMNS = {"d", "wm_yr_wk", "weekday", "wday"}

CATEGORICAL_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]


def to_model_input(df: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select model columns as pandas, with string columns as categoricals.

    LightGBM reads pandas ``category`` columns as categorical features and
    stores their levels in the model, so the same mapping is applied at
    prediction time even if a later frame has fewer or different levels.
    """
    subset = df.select(columns)
    string_cols = [c for c, dtype in subset.schema.items() if dtype in (pl.String, pl.Categorical)]
    frame = subset.to_pandas()
    for col in string_cols:
        frame[col] = frame[col].astype("category")
    return frame


class FeatureEngineerConfig(BaseModel):
    """Configuration for feature engineering."""

    temporal: TemporalFeaturesConfig = Field(default_factory=TemporalFeaturesConfig)
    price: PriceFeaturesConfig = Field(default_factory=PriceFeaturesConfig)
    hierarchical: HierarchicalFeaturesConfig = Field(default_factory=HierarchicalFeaturesConfig)

    target_col: str = Field(default="sales")
    group_col: str = Field(default="id")
    date_col: str = Field(default="date")

    min_lag: int = Field(
        default=28, ge=1, description="Overrides min_lag in the temporal and hierarchical configs"
    )

    enable_temporal: bool = Field(default=True)
    enable_price: bool = Field(default=True)
    enable_hierarchical: bool = Field(default=True)

    fill_na_value: float | None = Field(
        default=None,
        description="Fill value for numeric feature nulls. None keeps them missing, which "
        "LightGBM handles natively; filling with 0 would claim 'zero sales' where the truth "
        "is 'unknown'.",
    )
    drop_constant_columns: bool = Field(default=True)
    downcast_floats: bool = Field(default=True, description="Store features as float32")


class FeatureEngineer:
    """Build the model's feature matrix.

    Example:
        >>> engineer = FeatureEngineer(FeatureEngineerConfig(min_lag=28))
        >>> features = engineer.fit_transform(data)
        >>> X = features.select(engineer.get_feature_names())
    """

    def __init__(self, config: FeatureEngineerConfig | None = None) -> None:
        self.config = config or FeatureEngineerConfig()
        lag = self.config.min_lag
        self._temporal = TemporalFeatures(self.config.temporal.model_copy(update={"min_lag": lag}))
        self._price = PriceFeatures(self.config.price)
        self._hierarchical = HierarchicalFeatures(
            self.config.hierarchical.model_copy(update={"min_lag": lag})
        )
        self._fitted = False
        self._feature_names: list[str] = []
        self._constant_cols: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, data: pl.DataFrame) -> FeatureEngineer:
        """Learn which columns are features (and which are constant) from ``data``."""
        self.fit_transform(data)
        return self

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Build features once, learn the feature list from them and return them."""
        features = self._transform(data)
        candidates = self._candidate_columns(features)

        self._constant_cols = []
        if self.config.drop_constant_columns:
            self._constant_cols = [
                c for c in candidates if features[c].drop_nulls().n_unique() <= 1
            ]
            if self._constant_cols:
                logger.info("Dropping %d constant columns", len(self._constant_cols))

        self._feature_names = [c for c in candidates if c not in self._constant_cols]
        self._fitted = True
        logger.info("Feature engineer fitted with %d features", len(self._feature_names))
        return features.drop(self._constant_cols)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Build features for new data using the column decisions made in ``fit``."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer is not fitted. Call fit() or fit_transform().")
        features = self._transform(data)
        return features.drop([c for c in self._constant_cols if c in features.columns])

    def _candidate_columns(self, features: pl.DataFrame) -> list[str]:
        excluded = NON_FEATURE_COLUMNS | {
            self.config.target_col,
            self.config.group_col,
            self.config.date_col,
            "id",
        }
        return [c for c in features.columns if c not in excluded]

    def _transform(self, data: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        df = data
        if cfg.enable_temporal:
            df = self._temporal.transform(df, cfg.target_col, cfg.group_col, cfg.date_col)
        if cfg.enable_price:
            df = self._price.transform(df, group_col=cfg.group_col, date_col=cfg.date_col)
        if cfg.enable_hierarchical:
            df = self._hierarchical.transform(df, cfg.target_col, cfg.date_col, cfg.group_col)
        return self._postprocess(df).sort([cfg.group_col, cfg.date_col])

    def _postprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        target = self.config.target_col
        float_cols = [
            c for c, dtype in df.schema.items() if dtype in (pl.Float32, pl.Float64) and c != target
        ]
        # Ratios can divide by zero. Treat inf and NaN as missing, not as numbers.
        exprs = [
            pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in float_cols
        ]
        df = df.with_columns(exprs) if exprs else df

        if self.config.fill_na_value is not None:
            numeric = [c for c, dtype in df.schema.items() if dtype.is_numeric() and c != target]
            df = df.with_columns(pl.col(numeric).fill_null(self.config.fill_na_value))

        if self.config.downcast_floats and float_cols:
            df = df.with_columns(pl.col(float_cols).cast(pl.Float32))
        return df

    def get_feature_names(self) -> list[str]:
        """Model input columns, in order."""
        return self._feature_names.copy()

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Feature names grouped by family, for reporting."""
        groups: dict[str, list[str]] = {
            "temporal": [],
            "price": [],
            "hierarchical": [],
            "calendar": [],
            "categorical": [],
            "other": [],
        }
        calendar_markers = ("day_", "week_", "month", "quarter", "year", "is_", "dow_", "dom_")
        for name in self._feature_names:
            if name in CATEGORICAL_COLUMNS:
                groups["categorical"].append(name)
            elif "_agg_" in name or name.startswith("share_of_"):
                groups["hierarchical"].append(name)
            elif "price" in name or "promotion" in name or "discount" in name:
                groups["price"].append(name)
            elif any(m in name for m in ("_lag_", "_roll_", "expanding", "zero_days")):
                groups["temporal"].append(name)
            elif (
                "holiday" in name
                or name.startswith(calendar_markers)
                or name.startswith("woy_")
                or name == "snap"
            ):
                groups["calendar"].append(name)
            else:
                groups["other"].append(name)
        return groups

    def get_feature_summary(self) -> dict[str, Any]:
        """Counts per feature family, as written to the run's metrics file."""
        groups = self.get_feature_groups()
        return {
            "total_features": len(self._feature_names),
            "group_counts": {k: len(v) for k, v in groups.items()},
            "constant_columns_dropped": list(self._constant_cols),
        }

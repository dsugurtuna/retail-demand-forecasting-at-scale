"""
Data preprocessing and transformation module.

Handles data cleaning, normalization, and preparation for feature engineering.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""

    fill_missing_sales: str = Field(
        default="zero",
        description="Strategy for missing sales: 'zero', 'forward_fill', 'interpolate'",
    )
    clip_outliers: bool = Field(default=True)
    outlier_quantile: float = Field(default=0.99)
    normalize_prices: bool = Field(default=False)
    remove_zero_variance: bool = Field(default=True)


class DataPreprocessor:
    """
    Data preprocessing for the forecasting pipeline.

    Handles:
    - Missing value imputation
    - Outlier treatment
    - Data type optimization
    - Memory efficiency improvements

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> clean_data = preprocessor.transform(raw_data)
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self._fitted = False
        self._statistics: dict[str, Any] = {}

    def fit(self, data: pl.DataFrame | pd.DataFrame) -> DataPreprocessor:
        """
        Fit preprocessor to data, learning statistics for transformation.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        df = self._to_polars(data)

        # Learn statistics
        if self.config.clip_outliers:
            self._statistics["outlier_caps"] = {}
            numeric_cols = [
                c
                for c in df.columns
                if df.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            ]

            for col in numeric_cols:
                if col in ["sales", "sell_price"]:
                    upper = df.select(pl.col(col).quantile(self.config.outlier_quantile)).item()
                    self._statistics["outlier_caps"][col] = upper

        # Learn price normalization stats
        if self.config.normalize_prices and "sell_price" in df.columns:
            self._statistics["price_mean"] = df.select(pl.col("sell_price").mean()).item()
            self._statistics["price_std"] = df.select(pl.col("sell_price").std()).item()

        self._fitted = True
        logger.info("Preprocessor fitted to data")
        return self

    def transform(self, data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        """
        Transform data using fitted statistics.

        Args:
            data: Data to transform

        Returns:
            Transformed Polars DataFrame
        """
        df = self._to_polars(data)

        # Handle missing values
        df = self._handle_missing(df)

        # Clip outliers
        if self.config.clip_outliers and self._fitted:
            df = self._clip_outliers(df)

        # Normalize prices
        if self.config.normalize_prices and self._fitted:
            df = self._normalize_prices(df)

        # Optimize dtypes
        df = self._optimize_dtypes(df)

        logger.info(f"Transformed data: {df.shape}")
        return df

    def fit_transform(self, data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _to_polars(self, data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        return data

    def _handle_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values based on configuration."""
        # Handle missing sales
        if "sales" in df.columns:
            if self.config.fill_missing_sales == "zero":
                df = df.with_columns(pl.col("sales").fill_null(0))
            elif self.config.fill_missing_sales == "forward_fill":
                df = df.with_columns(pl.col("sales").fill_null(strategy="forward").over("id"))

        # Handle missing prices - forward fill within item-store
        if "sell_price" in df.columns:
            df = df.with_columns(
                pl.col("sell_price").fill_null(strategy="forward").over(["item_id", "store_id"])
            )

        return df

    def _clip_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clip outliers to learned quantiles."""
        for col, upper in self._statistics.get("outlier_caps", {}).items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).clip(0, upper))
        return df

    def _normalize_prices(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize prices using learned statistics."""
        if "sell_price" in df.columns:
            mean = self._statistics.get("price_mean", 0)
            std = self._statistics.get("price_std", 1)
            if std > 0:
                df = df.with_columns(
                    ((pl.col("sell_price") - mean) / std).alias("sell_price_normalized")
                )
        return df

    def _optimize_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimize data types for memory efficiency."""
        # Convert large integers to smaller types where possible
        int_cols = [c for c in df.columns if df.schema[c] in (pl.Int64,)]

        for col in int_cols:
            max_val = df.select(pl.col(col).max()).item()
            min_val = df.select(pl.col(col).min()).item()

            if max_val is None or min_val is None:
                continue

            if min_val >= 0:
                if max_val <= 255:
                    df = df.with_columns(pl.col(col).cast(pl.UInt8))
                elif max_val <= 65535:
                    df = df.with_columns(pl.col(col).cast(pl.UInt16))
                elif max_val <= 4294967295:
                    df = df.with_columns(pl.col(col).cast(pl.UInt32))
            else:
                if min_val >= -128 and max_val <= 127:
                    df = df.with_columns(pl.col(col).cast(pl.Int8))
                elif min_val >= -32768 and max_val <= 32767:
                    df = df.with_columns(pl.col(col).cast(pl.Int16))
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df = df.with_columns(pl.col(col).cast(pl.Int32))

        # Downcast floats
        float_cols = [c for c in df.columns if df.schema[c] in (pl.Float64,)]
        for col in float_cols:
            df = df.with_columns(pl.col(col).cast(pl.Float32))

        return df

    def get_memory_usage(self, df: pl.DataFrame) -> dict[str, Any]:
        """Get memory usage statistics."""
        estimated_size = df.estimated_size("mb")
        return {
            "total_mb": estimated_size,
            "rows": len(df),
            "columns": len(df.columns),
            "bytes_per_row": (estimated_size * 1024 * 1024) / len(df) if len(df) > 0 else 0,
        }

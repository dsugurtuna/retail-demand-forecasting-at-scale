"""
Price-based feature engineering for demand forecasting.

Implements features capturing price dynamics including:
- Price momentum and changes
- Relative price positioning
- Promotion detection
- Price elasticity proxies
- Competitor price features
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PriceFeaturesConfig(BaseModel):
    """Configuration for price features."""
    
    price_col: str = Field(default="sell_price")
    
    # Momentum features
    momentum_windows: list[int] = Field(
        default=[7, 14, 28],
        description="Windows for price momentum calculation"
    )
    
    # Price change detection
    change_threshold: float = Field(
        default=0.05,
        description="Threshold for detecting significant price changes"
    )
    
    # Elasticity estimation
    elasticity_window: int = Field(
        default=28,
        description="Window for price elasticity calculation"
    )
    
    # Cross-item features
    include_category_prices: bool = Field(
        default=True,
        description="Include category-level price features"
    )
    
    include_store_prices: bool = Field(
        default=True,
        description="Include store-level price features"
    )


class PriceFeatures:
    """
    Price-based feature engineering for demand forecasting.
    
    Generates features capturing:
    - Price changes and momentum
    - Relative price positioning (vs category, store)
    - Promotion indicators
    - Price elasticity proxies
    
    Example:
        >>> config = PriceFeaturesConfig()
        >>> price_features = PriceFeatures(config)
        >>> features = price_features.transform(df)
    """
    
    def __init__(self, config: PriceFeaturesConfig | None = None) -> None:
        """
        Initialize price feature generator.
        
        Args:
            config: Feature configuration
        """
        self.config = config or PriceFeaturesConfig()
    
    def transform(
        self,
        df: pl.DataFrame,
        group_col: str = "id",
        item_col: str = "item_id",
        store_col: str = "store_id",
        category_col: str = "cat_id",
        date_col: str = "date"
    ) -> pl.DataFrame:
        """
        Generate all price features.
        
        Args:
            df: Input DataFrame
            group_col: Column for item-store grouping
            item_col: Item identifier column
            store_col: Store identifier column
            category_col: Category identifier column
            date_col: Date column name
            
        Returns:
            DataFrame with price features added
        """
        logger.info("Generating price features...")
        
        price_col = self.config.price_col
        
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found, skipping price features")
            return df
        
        # Ensure sorted
        df = df.sort([group_col, date_col])
        
        # Generate features
        df = self._create_price_change_features(df, price_col, group_col)
        df = self._create_momentum_features(df, price_col, group_col)
        df = self._create_promotion_features(df, price_col, group_col)
        
        if self.config.include_category_prices:
            df = self._create_category_price_features(df, price_col, category_col, date_col)
        
        if self.config.include_store_prices:
            df = self._create_store_price_features(df, price_col, store_col, date_col)
        
        df = self._create_relative_price_features(df, price_col, group_col)
        
        n_features = len([c for c in df.columns if "price" in c.lower()])
        logger.info(f"Generated {n_features} price features")
        
        return df
    
    def _create_price_change_features(
        self,
        df: pl.DataFrame,
        price_col: str,
        group_col: str
    ) -> pl.DataFrame:
        """Create price change detection features."""
        exprs = [
            # Price lag (previous week)
            pl.col(price_col)
            .shift(7)
            .over(group_col)
            .alias("price_lag_7"),
            
            # Price change ratio
            (pl.col(price_col) / pl.col(price_col).shift(7).over(group_col))
            .alias("price_change_ratio_7"),
            
            # Absolute price change
            (pl.col(price_col) - pl.col(price_col).shift(7).over(group_col))
            .alias("price_change_abs_7"),
            
            # Binary: price decreased
            (pl.col(price_col) < pl.col(price_col).shift(7).over(group_col))
            .cast(pl.Int8)
            .alias("price_decreased"),
            
            # Binary: significant price change
            (
                ((pl.col(price_col) - pl.col(price_col).shift(7).over(group_col)).abs()
                / pl.col(price_col).shift(7).over(group_col))
                > self.config.change_threshold
            )
            .cast(pl.Int8)
            .alias("price_changed_significant"),
        ]
        
        df = df.with_columns(exprs)
        return df
    
    def _create_momentum_features(
        self,
        df: pl.DataFrame,
        price_col: str,
        group_col: str
    ) -> pl.DataFrame:
        """Create price momentum features."""
        exprs = []
        
        for window in self.config.momentum_windows:
            # Price momentum: current / rolling mean
            exprs.append(
                (pl.col(price_col) / 
                 pl.col(price_col).rolling_mean(window_size=window).over(group_col))
                .alias(f"price_momentum_{window}")
            )
            
            # Price relative to rolling max
            exprs.append(
                (pl.col(price_col) / 
                 pl.col(price_col).rolling_max(window_size=window).over(group_col))
                .alias(f"price_vs_max_{window}")
            )
            
            # Price relative to rolling min
            exprs.append(
                (pl.col(price_col) / 
                 pl.col(price_col).rolling_min(window_size=window).over(group_col))
                .alias(f"price_vs_min_{window}")
            )
            
            # Price volatility
            exprs.append(
                (pl.col(price_col).rolling_std(window_size=window).over(group_col) /
                 (pl.col(price_col).rolling_mean(window_size=window).over(group_col) + 1e-8))
                .alias(f"price_volatility_{window}")
            )
        
        if exprs:
            df = df.with_columns(exprs)
        
        return df
    
    def _create_promotion_features(
        self,
        df: pl.DataFrame,
        price_col: str,
        group_col: str
    ) -> pl.DataFrame:
        """Create promotion detection features."""
        # Calculate baseline price (mode or median of recent prices)
        # Using rolling median as proxy for "regular" price
        
        exprs = [
            # Price vs median (promotion indicator)
            (pl.col(price_col) / 
             pl.col(price_col).rolling_median(window_size=28).over(group_col))
            .alias("price_vs_median_28"),
            
            # Promotion flag: significantly below rolling median
            (pl.col(price_col) < 
             pl.col(price_col).rolling_median(window_size=28).over(group_col) * 0.9)
            .cast(pl.Int8)
            .alias("is_promotion"),
            
            # Deep discount flag
            (pl.col(price_col) < 
             pl.col(price_col).rolling_median(window_size=28).over(group_col) * 0.8)
            .cast(pl.Int8)
            .alias("is_deep_discount"),
            
            # Days since last price change
            (pl.col(price_col) != pl.col(price_col).shift(1).over(group_col))
            .cum_sum()
            .over(group_col)
            .alias("price_change_cumcount"),
        ]
        
        df = df.with_columns(exprs)
        return df
    
    def _create_category_price_features(
        self,
        df: pl.DataFrame,
        price_col: str,
        category_col: str,
        date_col: str
    ) -> pl.DataFrame:
        """Create category-level price features."""
        if category_col not in df.columns:
            return df
        
        # Category average price
        category_avg = (
            df.group_by([category_col, date_col])
            .agg(pl.col(price_col).mean().alias("category_avg_price"))
        )
        
        df = df.join(category_avg, on=[category_col, date_col], how="left")
        
        # Price relative to category
        df = df.with_columns([
            (pl.col(price_col) / pl.col("category_avg_price"))
            .alias("price_vs_category"),
            
            (pl.col(price_col) - pl.col("category_avg_price"))
            .alias("price_diff_category"),
        ])
        
        return df
    
    def _create_store_price_features(
        self,
        df: pl.DataFrame,
        price_col: str,
        store_col: str,
        date_col: str
    ) -> pl.DataFrame:
        """Create store-level price features."""
        if store_col not in df.columns:
            return df
        
        # Store average price
        store_avg = (
            df.group_by([store_col, date_col])
            .agg(pl.col(price_col).mean().alias("store_avg_price"))
        )
        
        df = df.join(store_avg, on=[store_col, date_col], how="left")
        
        # Price relative to store
        df = df.with_columns([
            (pl.col(price_col) / pl.col("store_avg_price"))
            .alias("price_vs_store"),
        ])
        
        return df
    
    def _create_relative_price_features(
        self,
        df: pl.DataFrame,
        price_col: str,
        group_col: str
    ) -> pl.DataFrame:
        """Create relative price position features."""
        exprs = [
            # Price percentile within item history
            pl.col(price_col)
            .rank()
            .over(group_col)
            .truediv(pl.len().over(group_col))
            .alias("price_percentile"),
            
            # Price z-score within item history
            (
                (pl.col(price_col) - pl.col(price_col).mean().over(group_col))
                / (pl.col(price_col).std().over(group_col) + 1e-8)
            ).alias("price_zscore"),
        ]
        
        df = df.with_columns(exprs)
        return df
    
    def get_feature_names(self) -> list[str]:
        """Get list of feature names that will be generated."""
        features = [
            "price_lag_7",
            "price_change_ratio_7",
            "price_change_abs_7",
            "price_decreased",
            "price_changed_significant",
            "price_vs_median_28",
            "is_promotion",
            "is_deep_discount",
            "price_change_cumcount",
            "price_percentile",
            "price_zscore",
        ]
        
        for window in self.config.momentum_windows:
            features.extend([
                f"price_momentum_{window}",
                f"price_vs_max_{window}",
                f"price_vs_min_{window}",
                f"price_volatility_{window}",
            ])
        
        if self.config.include_category_prices:
            features.extend(["category_avg_price", "price_vs_category", "price_diff_category"])
        
        if self.config.include_store_prices:
            features.extend(["store_avg_price", "price_vs_store"])
        
        return features

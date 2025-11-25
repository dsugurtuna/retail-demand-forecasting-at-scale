"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.temporal import TemporalFeatureGenerator
from src.features.price import PriceFeatureGenerator


class TestTemporalFeatureGenerator:
    """Tests for temporal feature generation."""
    
    @pytest.fixture
    def generator(self):
        return TemporalFeatureGenerator(
            lag_days=[7, 14],
            rolling_windows=[7],
            rolling_aggregations=["mean", "std"]
        )
    
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        return pl.DataFrame({
            "id": ["ITEM_1"] * 100,
            "date": dates,
            "sales": np.random.exponential(10, 100),
        })
    
    def test_fit_does_not_fail(self, generator, sample_df):
        """Fit should complete without errors."""
        generator.fit(sample_df)
    
    def test_transform_creates_lag_features(self, generator, sample_df):
        """Transform should create lag features."""
        generator.fit(sample_df)
        result = generator.transform(sample_df)
        
        assert "sales_lag_7" in result.columns
        assert "sales_lag_14" in result.columns
    
    def test_transform_creates_rolling_features(self, generator, sample_df):
        """Transform should create rolling features."""
        generator.fit(sample_df)
        result = generator.transform(sample_df)
        
        assert "sales_rolling_mean_7" in result.columns
        assert "sales_rolling_std_7" in result.columns
    
    def test_transform_creates_calendar_features(self, generator, sample_df):
        """Transform should create calendar features."""
        generator.fit(sample_df)
        result = generator.transform(sample_df)
        
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
    
    def test_cyclical_encoding(self, generator, sample_df):
        """Should create cyclical encoded features."""
        generator.fit(sample_df)
        result = generator.transform(sample_df)
        
        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
    
    def test_lag_features_correct_values(self, generator):
        """Lag features should have correct values."""
        df = pl.DataFrame({
            "id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "sales": list(range(10)),
        })
        
        gen = TemporalFeatureGenerator(lag_days=[1], rolling_windows=[])
        gen.fit(df)
        result = gen.transform(df)
        
        # sales_lag_1 should be shifted by 1
        result_pd = result.to_pandas()
        assert pd.isna(result_pd["sales_lag_1"].iloc[0])
        assert result_pd["sales_lag_1"].iloc[1] == 0
        assert result_pd["sales_lag_1"].iloc[2] == 1


class TestPriceFeatureGenerator:
    """Tests for price feature generation."""
    
    @pytest.fixture
    def generator(self):
        return PriceFeatureGenerator(
            price_lags=[7],
            include_promotions=True
        )
    
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        return pl.DataFrame({
            "id": ["ITEM_1"] * 50,
            "date": dates,
            "sales": np.random.exponential(10, 50),
            "sell_price": np.random.uniform(5, 15, 50),
        })
    
    def test_fit_transform(self, generator, sample_df):
        """Should fit and transform without errors."""
        generator.fit(sample_df)
        result = generator.transform(sample_df)
        
        assert result is not None
        assert len(result) == len(sample_df)
    
    def test_price_momentum_features(self, generator, sample_df):
        """Should create price momentum features."""
        generator.fit(sample_df)
        result = generator.transform(sample_df)
        
        assert "price_momentum_7" in result.columns
    
    def test_promotion_detection(self, generator):
        """Should detect promotions."""
        df = pl.DataFrame({
            "id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "sales": [10] * 10,
            "sell_price": [10, 10, 5, 5, 5, 10, 10, 10, 10, 10],  # Promotion days 2-4
        })
        
        gen = PriceFeatureGenerator(price_lags=[], include_promotions=True)
        gen.fit(df)
        result = gen.transform(df)
        
        # Price drop should be detected
        assert "price_change" in result.columns or "is_promotion" in result.columns


class TestFeatureConsistency:
    """Tests for feature engineering consistency."""
    
    def test_features_are_deterministic(self, sample_sales_data):
        """Features should be deterministic across calls."""
        gen = TemporalFeatureGenerator(lag_days=[7])
        
        gen.fit(sample_sales_data)
        result1 = gen.transform(sample_sales_data)
        result2 = gen.transform(sample_sales_data)
        
        # Results should be identical
        assert result1.frame_equal(result2)
    
    def test_handles_missing_values(self):
        """Should handle missing values gracefully."""
        df = pl.DataFrame({
            "id": ["A"] * 10,
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "sales": [1, None, 3, None, 5, 6, 7, 8, 9, 10],
        })
        
        gen = TemporalFeatureGenerator(lag_days=[1], rolling_windows=[3])
        gen.fit(df)
        
        # Should not raise
        result = gen.transform(df)
        assert result is not None

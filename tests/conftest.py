"""Test configuration and fixtures."""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_sales_data() -> pl.DataFrame:
    """Generate sample sales data for testing."""
    np.random.seed(42)
    n_rows = 1000
    
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    items = [f"ITEM_{i}" for i in range(10)]
    stores = ["STORE_1", "STORE_2"]
    
    data = []
    for store in stores:
        for item in items:
            for date in dates:
                # Simulate sales with seasonality
                base = 10 + np.random.normal(0, 2)
                day_effect = 2 if date.dayofweek >= 5 else 0  # Weekend boost
                month_effect = np.sin(2 * np.pi * date.month / 12) * 3
                
                sales = max(0, base + day_effect + month_effect + np.random.normal(0, 3))
                
                data.append({
                    "id": f"{store}_{item}",
                    "item_id": item,
                    "store_id": store,
                    "date": date,
                    "sales": round(sales, 2),
                    "sell_price": round(5 + np.random.uniform(-1, 1), 2),
                })
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_calendar_data() -> pl.DataFrame:
    """Generate sample calendar data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    
    data = []
    for date in dates:
        data.append({
            "date": date,
            "d": f"d_{(date - pd.Timestamp('2020-01-01')).days + 1}",
            "wm_yr_wk": date.isocalendar()[1],
            "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][date.dayofweek],
            "month": date.month,
            "year": date.year,
            "snap_CA": int(np.random.random() > 0.7),
            "snap_TX": int(np.random.random() > 0.7),
            "snap_WI": int(np.random.random() > 0.7),
            "event_name_1": np.random.choice([None, "Holiday", "Event"], p=[0.9, 0.05, 0.05]),
        })
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_prices_data() -> pl.DataFrame:
    """Generate sample price data."""
    items = [f"ITEM_{i}" for i in range(10)]
    stores = ["STORE_1", "STORE_2"]
    weeks = range(1, 15)
    
    data = []
    for store in stores:
        for item in items:
            for week in weeks:
                data.append({
                    "store_id": store,
                    "item_id": item,
                    "wm_yr_wk": week,
                    "sell_price": round(5 + np.random.uniform(-1, 1), 2),
                })
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_features_df(sample_sales_data: pl.DataFrame) -> pd.DataFrame:
    """Generate sample feature DataFrame."""
    df = sample_sales_data.to_pandas()
    
    # Add lag features
    for lag in [7, 14, 28]:
        df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
    
    # Add rolling features
    for window in [7, 14]:
        df[f"sales_rolling_mean_{window}"] = (
            df.groupby("id")["sales"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    
    # Add temporal features
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    return df.dropna()


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample predictions for metric testing."""
    np.random.seed(42)
    n = 100
    
    y_true = np.random.exponential(10, n)
    noise = np.random.normal(0, 2, n)
    y_pred = np.maximum(0, y_true + noise)
    
    return y_true, y_pred


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test artifacts."""
    test_dir = tmp_path / "test_artifacts"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def mock_config() -> dict:
    """Generate mock configuration."""
    return {
        "environment": "development",
        "debug": True,
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
        },
        "model": {
            "name": "test_model",
            "version": "1.0.0",
            "n_estimators": 100,
            "learning_rate": 0.1,
        },
        "training": {
            "n_folds": 3,
            "random_seed": 42,
        },
    }


# Markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")

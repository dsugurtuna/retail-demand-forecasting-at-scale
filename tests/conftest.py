"""Shared fixtures. Everything is synthetic and generated in memory: no network, no keys."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.data.loader import merge_m5_frames
from src.data.synthetic import SyntheticDataConfig, SyntheticDataGenerator


@pytest.fixture(scope="session")
def m5_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Small synthetic M5-style (sales, calendar, prices): 14 items x 5 stores x 300 days."""
    config = SyntheticDataConfig(n_items=14, n_stores=5, n_days=300, random_seed=7)
    return SyntheticDataGenerator(config).generate_all()


@pytest.fixture(scope="session")
def m5_data(m5_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """The merged long frame for ``m5_frames``."""
    return merge_m5_frames(*m5_frames)


@pytest.fixture
def sample_sales_data() -> pl.DataFrame:
    """Simple long-format sales for 20 series x 100 days, with a weekend effect."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rows = []
    for store in ["STORE_1", "STORE_2"]:
        for item in [f"ITEM_{i}" for i in range(10)]:
            for date in dates:
                base = 10 + rng.normal(0, 2) + (2 if date.dayofweek >= 5 else 0)
                rows.append(
                    {
                        "id": f"{store}_{item}",
                        "item_id": item,
                        "store_id": store,
                        "date": date,
                        "sales": round(max(0.0, base + rng.normal(0, 3)), 2),
                        "sell_price": round(5 + rng.uniform(-1, 1), 2),
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Actuals and noisy forecasts for metric tests."""
    rng = np.random.default_rng(42)
    y_true = rng.exponential(10, 100)
    y_pred = np.maximum(0, y_true + rng.normal(0, 2, 100))
    return y_true, y_pred

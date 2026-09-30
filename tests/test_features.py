"""Original feature tests from the first version of the project.

They targeted a ``src.feature_engineering`` module that was later replaced by
``src.features``. The assertions are unchanged; only the calls now use the
current API.
"""

import unittest

import numpy as np
import pandas as pd
import polars as pl

from src.features.temporal import TemporalFeatures, TemporalFeaturesConfig, add_calendar_features


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # min_lag=1 allows the short lags this test uses; the default (28)
        # would drop them to prevent leakage over a 28-day horizon.
        self.fe = TemporalFeatures(
            TemporalFeaturesConfig(lags=[1, 2], rolling_windows=[2], min_lag=1)
        )

        dates = pd.date_range(start="2023-01-01", periods=10)
        self.df = pl.from_pandas(
            pd.DataFrame(
                {
                    "id": ["item_1"] * 10,
                    "date": dates,
                    "sales": np.arange(10),
                    "sell_price": [10] * 10,
                }
            )
        )

    def test_lag_features(self):
        df = self.fe.transform(self.df).to_pandas()
        # Lag 1 of sales=1 (at index 1) should be 0 (from index 0)
        self.assertEqual(df.iloc[1]["sales_lag_1"], 0)

    def test_date_features(self):
        df = add_calendar_features(self.df).to_pandas()
        self.assertIn("day_of_week", df.columns)
        self.assertEqual(df.iloc[0]["day_of_week"], 6)  # 2023-01-01 is Sunday (6)


if __name__ == "__main__":
    unittest.main()

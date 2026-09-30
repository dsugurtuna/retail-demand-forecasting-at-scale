"""Unit tests for feature engineering, including leakage tests."""

from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.engineer import FeatureEngineer, FeatureEngineerConfig, to_model_input
from src.features.hierarchical import HierarchicalFeatures
from src.features.price import PriceFeatures, PriceFeaturesConfig
from src.features.temporal import TemporalFeatures, TemporalFeaturesConfig, add_calendar_features


class TestTemporalFeatures:
    @pytest.fixture
    def generator(self):
        return TemporalFeatures(
            TemporalFeaturesConfig(
                lags=[7, 14], rolling_windows=[7], rolling_statistics=["mean", "std"], min_lag=7
            )
        )

    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(0)
        return pl.DataFrame(
            {
                "id": ["ITEM_1"] * 100,
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "sales": rng.exponential(10, 100),
            }
        )

    def test_creates_lag_features(self, generator, sample_df):
        result = generator.transform(sample_df)
        assert "sales_lag_7" in result.columns
        assert "sales_lag_14" in result.columns

    def test_lags_below_min_lag_are_dropped(self, sample_df):
        gen = TemporalFeatures(TemporalFeaturesConfig(lags=[1, 7, 28], min_lag=7))
        result = gen.transform(sample_df)
        assert "sales_lag_1" not in result.columns
        assert {"sales_lag_7", "sales_lag_28"} <= set(result.columns)

    def test_creates_rolling_features(self, generator, sample_df):
        result = generator.transform(sample_df)
        assert "sales_roll_mean_7" in result.columns
        assert "sales_roll_std_7" in result.columns

    def test_creates_calendar_features(self, generator, sample_df):
        result = generator.transform(sample_df)
        assert {"day_of_week", "month", "is_weekend"} <= set(result.columns)

    def test_cyclical_encoding(self, generator, sample_df):
        result = generator.transform(sample_df)
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_lag_features_correct_values(self):
        df = pl.DataFrame(
            {
                "id": ["A"] * 10,
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "sales": list(range(10)),
            }
        )
        gen = TemporalFeatures(TemporalFeaturesConfig(lags=[1], rolling_windows=[], min_lag=1))
        result = gen.transform(df).to_pandas()
        assert pd.isna(result["sales_lag_1"].iloc[0])
        assert result["sales_lag_1"].iloc[1] == 0
        assert result["sales_lag_1"].iloc[2] == 1

    def test_rolling_mean_is_shifted_by_min_lag(self):
        df = pl.DataFrame(
            {
                "id": ["A"] * 10,
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "sales": [float(i) for i in range(10)],
            }
        )
        gen = TemporalFeatures(TemporalFeaturesConfig(lags=[], rolling_windows=[3], min_lag=2))
        result = gen.transform(df)["sales_roll_mean_3"].to_list()
        # Day 5 (index 4) sees days 0-2 -> mean 1.0; nothing newer than 2 days.
        assert result[4] == pytest.approx(1.0)
        assert result[3] is None

    def test_weekend_flag_is_saturday_and_sunday(self):
        # 2024-01-05 is a Friday.
        df = pl.DataFrame({"date": pd.date_range("2024-01-05", periods=3, freq="D")})
        result = add_calendar_features(df)
        assert result["day_of_week"].to_list() == [4, 5, 6]
        assert result["is_weekend"].to_list() == [0, 1, 1]

    def test_holiday_flags(self):
        df = pl.DataFrame({"date": pd.date_range("2020-12-23", periods=5, freq="D"), "id": "A"})
        df = df.with_columns(pl.lit(1.0).alias("sales"))
        result = TemporalFeatures(TemporalFeaturesConfig(holiday_countries=["UK"])).transform(df)
        flags = dict(
            zip(result["date"].dt.day().to_list(), result["is_holiday_UK"].to_list(), strict=True)
        )
        assert flags[25] == 1
        assert flags[23] == 0
        assert result.filter(pl.col("date").dt.day() == 23)["days_to_holiday_UK"].item() == 2

    def test_features_are_deterministic(self, sample_sales_data):
        gen = TemporalFeatures(TemporalFeaturesConfig(lags=[7], min_lag=7))
        assert gen.transform(sample_sales_data).equals(gen.transform(sample_sales_data))

    def test_handles_missing_values(self):
        df = pl.DataFrame(
            {
                "id": ["A"] * 10,
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "sales": [1, None, 3, None, 5, 6, 7, 8, 9, 10],
            }
        )
        gen = TemporalFeatures(TemporalFeaturesConfig(lags=[1], rolling_windows=[3], min_lag=1))
        result = gen.transform(df)
        assert len(result) == 10


class TestPriceFeatures:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(1)
        return pl.DataFrame(
            {
                "id": ["ITEM_1"] * 50,
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "sales": rng.exponential(10, 50),
                "sell_price": rng.uniform(5, 15, 50),
            }
        )

    def test_transform_keeps_rows(self, sample_df):
        result = PriceFeatures().transform(sample_df)
        assert len(result) == len(sample_df)

    def test_price_momentum_features(self, sample_df):
        result = PriceFeatures(PriceFeaturesConfig(momentum_windows=[7])).transform(sample_df)
        assert "price_momentum_7" in result.columns

    def test_promotion_detection(self):
        df = pl.DataFrame(
            {
                "id": ["A"] * 40,
                "date": pd.date_range("2020-01-01", periods=40, freq="D"),
                "sales": [10.0] * 40,
                "sell_price": [10.0] * 35 + [5.0] * 5,  # 50% off for the last 5 days
            }
        )
        result = PriceFeatures().transform(df)
        assert result["is_promotion"].to_list()[-1] == 1
        assert result["is_promotion"].to_list()[30] == 0

    def test_missing_price_column_is_skipped(self):
        df = pl.DataFrame({"id": ["A"], "date": [pd.Timestamp("2020-01-01")], "sales": [1.0]})
        assert PriceFeatures().transform(df).columns == df.columns


def _hide_after(data: pl.DataFrame, cutoff) -> pl.DataFrame:
    return data.with_columns(
        pl.when(pl.col("date") > cutoff).then(None).otherwise(pl.col("sales")).alias("sales")
    )


class TestNoLeakage:
    """Features at day t may only use target values from day t - min_lag or earlier."""

    def test_future_target_values_do_not_change_features(self, m5_data):
        min_lag = 28
        cutoff = m5_data["date"].max() - timedelta(days=60)
        rng = np.random.default_rng(3)
        scrambled = m5_data.with_columns(
            pl.when(pl.col("date") > cutoff)
            .then(pl.Series(rng.integers(0, 500, len(m5_data))).cast(pl.Float64))
            .otherwise(pl.col("sales"))
            .alias("sales")
        )
        config = FeatureEngineerConfig(min_lag=min_lag, drop_constant_columns=False)
        original = FeatureEngineer(config).fit_transform(m5_data)
        changed = FeatureEngineer(config).fit_transform(scrambled)

        # Rows up to cutoff + min_lag can only see sales up to the cutoff.
        visible = pl.col("date") <= cutoff + timedelta(days=min_lag)
        columns = [c for c in original.columns if c != "sales"]
        assert (
            original.filter(visible).select(columns).equals(changed.filter(visible).select(columns))
        )

    def test_same_day_hierarchy_totals_are_not_features(self, m5_data):
        result = HierarchicalFeatures().transform(m5_data)
        assert not any(c.endswith(("_daily_sum", "_daily_mean")) for c in result.columns)
        assert "share_of_cat" in result.columns

    def test_hidden_target_gives_null_features_after_min_lag(self, m5_data):
        origin = m5_data["date"].max() - timedelta(days=28)
        engineer = FeatureEngineer(FeatureEngineerConfig(min_lag=28))
        features = engineer.fit_transform(_hide_after(m5_data, origin))
        window = features.filter(pl.col("date") > origin)
        # A 28-day horizon with min_lag=28: every forecast day still has its lag-28 value.
        assert window["sales_lag_28"].null_count() == 0


class TestFeatureEngineer:
    def test_fit_transform_reports_feature_groups(self, m5_data):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(m5_data)
        summary = engineer.get_feature_summary()
        assert summary["total_features"] == len(engineer.get_feature_names())
        assert summary["group_counts"]["temporal"] > 0
        assert summary["group_counts"]["hierarchical"] > 0
        assert set(engineer.get_feature_names()) <= set(features.columns)

    def test_identifiers_and_target_are_not_features(self, m5_data):
        engineer = FeatureEngineer()
        engineer.fit(m5_data)
        names = set(engineer.get_feature_names())
        assert not names & {"sales", "id", "date", "d", "wm_yr_wk"}

    def test_transform_before_fit_raises(self, m5_data):
        with pytest.raises(RuntimeError):
            FeatureEngineer().transform(m5_data)

    def test_no_infinities_and_float32(self, m5_data):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(m5_data)
        floats = [c for c, t in features.schema.items() if t == pl.Float32]
        assert floats
        assert (
            not features.select(pl.any_horizontal(pl.col(floats).is_infinite())).to_series().any()
        )

    def test_to_model_input_makes_categoricals(self, m5_data):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(m5_data)
        frame = to_model_input(features, engineer.get_feature_names())
        assert isinstance(frame["item_id"].dtype, pd.CategoricalDtype)
        assert list(frame.columns) == engineer.get_feature_names()

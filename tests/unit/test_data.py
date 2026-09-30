"""Unit tests for synthetic data generation and M5 file loading."""

import polars as pl
import pytest

from src.data.loader import DataLoader, merge_m5_frames, sales_to_long, sales_to_wide
from src.data.synthetic import SyntheticDataConfig, SyntheticDataGenerator


class TestSyntheticData:
    def test_shapes_and_columns(self, m5_frames):
        sales, calendar, prices = m5_frames
        assert len(calendar) == 300
        assert sales["id"].n_unique() == 14 * 5
        assert len(sales) == 14 * 5 * 300
        assert {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales"} == set(
            sales.columns
        )
        assert {"snap_CA", "snap_TX", "snap_WI", "event_name_1"} <= set(calendar.columns)
        assert {"store_id", "item_id", "wm_yr_wk", "sell_price"} == set(prices.columns)

    def test_same_seed_same_data(self):
        config = SyntheticDataConfig(n_items=7, n_stores=2, n_days=60, random_seed=1)
        first = SyntheticDataGenerator(config).generate_all()
        second = SyntheticDataGenerator(config).generate_all()
        assert all(a.equals(b) for a, b in zip(first, second, strict=True))

    def test_counts_are_non_negative_integers_with_zeros(self, m5_frames):
        sales = m5_frames[0]["sales"]
        assert sales.dtype == pl.Int64
        assert sales.min() >= 0
        assert (sales == 0).sum() > 0  # slow movers produce zero days

    def test_promotions_raise_demand(self, m5_data):
        # The generator's price response should be visible in the data.
        with_ratio = m5_data.with_columns(
            (pl.col("sell_price") / pl.col("sell_price").median().over("id")).alias("ratio")
        )
        promo = with_ratio.filter(pl.col("ratio") < 0.9)["sales"].mean()
        regular = with_ratio.filter(pl.col("ratio") >= 0.97)["sales"].mean()
        assert promo > regular


class TestM5Files:
    def test_wide_long_round_trip(self, m5_frames):
        sales = m5_frames[0]
        back = sales_to_long(sales_to_wide(sales))
        assert back.sort(["id", "d"]).equals(
            sales.with_columns(pl.col("sales").cast(pl.Float64)).sort(["id", "d"])
        )

    def test_loader_reads_what_the_generator_writes(self, tmp_path, m5_data):
        config = SyntheticDataConfig(n_items=14, n_stores=5, n_days=300, random_seed=7)
        SyntheticDataGenerator(config).write_m5_files(tmp_path)
        loaded = DataLoader(data_dir=tmp_path).load_all(stores=["CA_1", "TX_1"])
        expected = m5_data.filter(pl.col("store_id").is_in(["CA_1", "TX_1"]))
        assert loaded.select(expected.columns).equals(expected)

    def test_missing_file_is_an_error_not_synthetic_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--smoke"):
            DataLoader(data_dir=tmp_path).load_sales()

    def test_unknown_store_is_an_error(self, tmp_path):
        SyntheticDataGenerator(
            SyntheticDataConfig(n_items=7, n_stores=1, n_days=60)
        ).write_m5_files(tmp_path)
        with pytest.raises(ValueError, match="No sales rows"):
            DataLoader(data_dir=tmp_path).load_all(stores=["XX_9"])

    def test_merge_adds_own_state_snap_flag(self, m5_frames):
        merged = merge_m5_frames(*m5_frames)
        assert "snap" in merged.columns
        assert not any(c.startswith("snap_") for c in merged.columns)
        calendar = m5_frames[1]
        tx = merged.filter(pl.col("state_id") == "TX").join(
            calendar.select("date", "snap_TX"), on="date"
        )
        assert (tx["snap"] == tx["snap_TX"]).all()

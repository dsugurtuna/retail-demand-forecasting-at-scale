"""Unit tests for the local feature cache."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.features.store import FeatureStore


@pytest.fixture
def features() -> pl.DataFrame:
    return pl.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})


def test_save_and_load(tmp_path, features):
    store = FeatureStore(tmp_path)
    meta = store.save("train", features, version="v1", parameters={"min_lag": 28})
    assert meta.row_count == 3
    assert meta.parameters == {"min_lag": 28}
    assert FeatureStore(tmp_path).load("train", "v1").equals(features)


def test_existing_version_is_protected(tmp_path, features):
    store = FeatureStore(tmp_path)
    store.save("train", features, version="v1")
    with pytest.raises(FileExistsError):
        store.save("train", features, version="v1")
    store.save("train", features.head(1), version="v1", overwrite=True)
    assert len(store.load("train", "v1")) == 1


def test_hash_is_deterministic(tmp_path, features):
    first = FeatureStore(tmp_path / "a").save("f", features, version="v1")
    second = FeatureStore(tmp_path / "b").save("f", features, version="v1")
    assert first.source_hash == second.source_hash


def test_latest_and_as_of(tmp_path, features):
    store = FeatureStore(tmp_path, cache_enabled=False)
    store.save("f", features, version="first")
    store.save("f", features.head(2), version="second")
    assert len(store.load("f")) == 2  # newest by creation time
    assert set(store.list_versions("f")) == {"first", "second"}

    past = datetime.now(UTC) - timedelta(days=1)
    with pytest.raises(ValueError):
        store.load_version_as_of("f", past)
    subset = store.load_version_as_of("f", datetime.now(UTC), entity_ids=["a"])
    assert subset["id"].to_list() == ["a"]


def test_delete(tmp_path, features):
    store = FeatureStore(tmp_path)
    store.save("f", features, version="v1")
    store.delete("f", "v1")
    assert store.list_versions("f") == []

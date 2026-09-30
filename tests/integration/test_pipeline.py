"""Integration tests for rolling-origin backtesting."""

import itertools

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.evaluation.backtesting import BacktestConfig, BacktestEngine
from src.features.engineer import FeatureEngineer, FeatureEngineerConfig
from src.models.lightgbm_model import LightGBMConfig, LightGBMForecaster

FAST = LightGBMConfig(n_estimators=80, learning_rate=0.1, num_leaves=15, min_data_in_leaf=20)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    rows = []
    for item in ["ITEM_1", "ITEM_2"]:
        for i, date in enumerate(dates):
            base = 10 + np.sin(2 * np.pi * i / 7) * 3  # weekly pattern
            rows.append(
                {
                    "id": f"STORE_1_{item}",
                    "item_id": item,
                    "store_id": "STORE_1",
                    "date": date,
                    "sales": round(max(0.0, base + rng.normal(0, 2)), 2),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def mock_model(mocker):
    """A stand-in model whose predictions always have the right length."""
    model = mocker.MagicMock()
    model.clone.return_value = model
    model.predict.side_effect = lambda X, **kwargs: np.full(len(X), 10.0)
    model.metadata = None
    return model


def test_backtest_config_defaults():
    config = BacktestConfig()
    assert config.n_folds == 5
    assert config.test_days == 28
    assert config.gap_days == 0


def test_backtest_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(n_folds=0)
    with pytest.raises(ValueError):
        BacktestConfig(test_days=-1)


def test_backtest_runs_complete(sample_data, mock_model):
    result = BacktestEngine(BacktestConfig(n_folds=2, test_days=14, gap_days=7)).run(
        sample_data, mock_model
    )
    assert len(result.fold_results) == 2
    assert mock_model.fit.call_count == 2


def test_backtest_produces_metrics(sample_data, mock_model):
    result = BacktestEngine(BacktestConfig(n_folds=2, test_days=14, gap_days=7)).run(
        sample_data, mock_model
    )
    assert result.aggregate_metrics.rmse > 0
    assert len(result.metrics_by_fold) == 2
    assert result.metrics_by_horizon is not None


def test_folds_are_ordered_and_do_not_overlap(sample_data, mock_model):
    result = BacktestEngine(BacktestConfig(n_folds=3, test_days=14, gap_days=7)).run(
        sample_data, mock_model
    )
    folds = result.fold_results
    for fold in folds:
        assert (fold.test_start - fold.train_end).days == 7 + 1
    for earlier, later in itertools.pairwise(folds):
        assert earlier.test_end < later.test_start


def test_model_never_sees_test_period_targets(sample_data, mock_model):
    """Training labels passed to fit() must all come from on or before the origin."""
    BacktestEngine(BacktestConfig(n_folds=1, test_days=14, validation_days=14)).run(
        sample_data, mock_model
    )
    _, kwargs = mock_model.fit.call_args
    y_fit, y_val = mock_model.fit.call_args.args[1], kwargs["y_valid"]
    n_train_days = 365 - 14
    assert len(y_fit) + len(y_val) == 2 * n_train_days


def test_expanding_vs_sliding(sample_data, mock_model):
    expanding = BacktestEngine(BacktestConfig(n_folds=2, expanding_window=True, test_days=14)).run(
        sample_data, mock_model
    )
    sliding = BacktestEngine(
        BacktestConfig(n_folds=2, expanding_window=False, min_train_days=100, test_days=14)
    ).run(sample_data, mock_model)
    assert expanding.fold_results[0].train_size > sliding.fold_results[0].train_size
    assert sliding.fold_results[0].train_size == sliding.fold_results[1].train_size


@pytest.mark.slow
def test_lightgbm_backtest_on_m5_style_data(m5_data):
    engine = BacktestEngine(BacktestConfig(n_folds=2, test_days=28, min_train_days=150))
    result = engine.run(
        m5_data, LightGBMForecaster(config=FAST), FeatureEngineer(FeatureEngineerConfig())
    )
    assert len(result.fold_results) == 2
    assert result.mean_wrmsse is not None
    assert 0 < result.mean_wrmsse < 2
    assert all(
        f.predictions is not None and len(f.predictions) == 70 * 28 for f in result.fold_results
    )
    assert result.metrics_by_horizon is not None
    assert set(result.metrics_by_horizon["horizon"]) == set(range(1, 29))

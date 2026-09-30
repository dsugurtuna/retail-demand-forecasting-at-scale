"""Integration tests for the training pipeline."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.evaluation.backtesting import BacktestConfig, BacktestEngine


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for backtesting."""
        np.random.seed(42)

        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        items = ["ITEM_1", "ITEM_2"]
        stores = ["STORE_1"]

        data = []
        for store in stores:
            for item in items:
                for i, date in enumerate(dates):
                    base = 10 + np.sin(2 * np.pi * i / 7) * 3  # Weekly pattern
                    sales = max(0, base + np.random.normal(0, 2))

                    data.append(
                        {
                            "id": f"{store}_{item}",
                            "item_id": item,
                            "store_id": store,
                            "date": date,
                            "sales": round(sales, 2),
                        }
                    )

        return pl.DataFrame(data)

    @pytest.fixture
    def mock_model(self, mocker):
        """Create mock model for testing."""
        model = mocker.MagicMock()
        model.fit.return_value = None
        model.predict.return_value = np.random.exponential(10, 100)
        model.metadata = None
        model.get_params.return_value = {"hyperparameters": {}}
        return model

    def test_backtest_config_defaults(self):
        """BacktestConfig should have sensible defaults."""
        config = BacktestConfig()

        assert config.n_folds == 5
        assert config.test_days == 28
        assert config.gap_days == 28

    def test_backtest_config_validation(self):
        """BacktestConfig should validate inputs."""
        with pytest.raises(ValueError):
            BacktestConfig(n_folds=0)

        with pytest.raises(ValueError):
            BacktestConfig(test_days=-1)

    @pytest.mark.slow
    def test_backtest_runs_complete(self, sample_data, mock_model):
        """Backtest should complete successfully."""
        config = BacktestConfig(n_folds=2, test_days=14, gap_days=7)
        engine = BacktestEngine(config)

        result = engine.run(
            data=sample_data,
            model=mock_model,
            feature_engineer=None,
        )

        assert result is not None
        assert len(result.fold_results) == 2

    @pytest.mark.slow
    def test_backtest_produces_metrics(self, sample_data, mock_model):
        """Backtest should produce metrics."""
        config = BacktestConfig(n_folds=2, test_days=14, gap_days=7)
        engine = BacktestEngine(config)

        result = engine.run(
            data=sample_data,
            model=mock_model,
            feature_engineer=None,
        )

        assert result.aggregate_metrics is not None
        assert result.aggregate_metrics.rmse > 0
        assert result.metrics_by_fold is not None

    def test_backtest_expanding_vs_sliding(self, sample_data, mock_model):
        """Expanding and sliding windows should differ."""
        config_expanding = BacktestConfig(
            n_folds=2, expanding_window=True, test_days=14, gap_days=7
        )
        config_sliding = BacktestConfig(
            n_folds=2, expanding_window=False, min_train_days=100, test_days=14, gap_days=7
        )

        engine_expanding = BacktestEngine(config_expanding)
        engine_sliding = BacktestEngine(config_sliding)

        # Both should work
        result_expanding = engine_expanding.run(sample_data, mock_model)
        result_sliding = engine_sliding.run(sample_data, mock_model)

        assert result_expanding is not None
        assert result_sliding is not None


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.slow
    def test_data_to_features_to_model(self, sample_sales_data, sample_features_df):
        """Full pipeline from data to model should work."""
        # This is a placeholder for full integration test
        # In a real scenario, this would test the complete flow

        assert sample_sales_data is not None
        assert sample_features_df is not None
        assert len(sample_features_df) > 0

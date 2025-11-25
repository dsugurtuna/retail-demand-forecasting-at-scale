"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    mae,
    mape,
    mase,
    Metrics,
    MetricsResult,
    rmse,
    smape,
    wrmsse,
)


class TestBasicMetrics:
    """Tests for basic metric functions."""
    
    def test_rmse_perfect_predictions(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert rmse(y_true, y_pred) == pytest.approx(0.0)
    
    def test_rmse_calculation(self):
        """RMSE should calculate correctly."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        expected = np.sqrt(np.mean(0.25 * np.ones(5)))
        assert rmse(y_true, y_pred) == pytest.approx(expected)
    
    def test_mae_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mae(y_true, y_pred) == pytest.approx(0.0)
    
    def test_mae_calculation(self):
        """MAE should calculate correctly."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        assert mae(y_true, y_pred) == pytest.approx(1.0)
    
    def test_mape_calculation(self):
        """MAPE should calculate correctly."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 22, 33, 44, 55])
        assert mape(y_true, y_pred) == pytest.approx(10.0)
    
    def test_mape_with_zeros(self):
        """MAPE should handle zeros in y_true."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        result = mape(y_true, y_pred)
        assert result == float('inf')
    
    def test_smape_calculation(self):
        """sMAPE should calculate correctly."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        assert smape(y_true, y_pred) == pytest.approx(0.0)
    
    def test_smape_symmetric(self):
        """sMAPE should be symmetric."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([15, 25, 35])
        
        result1 = smape(y_true, y_pred)
        result2 = smape(y_pred, y_true)
        
        assert result1 == pytest.approx(result2)
    
    def test_mase_calculation(self):
        """MASE should calculate correctly."""
        y_train = np.array([10, 12, 14, 16, 18, 20, 22, 24])
        y_true = np.array([26, 28])
        y_pred = np.array([25, 29])
        
        # Naive forecast error (no seasonality)
        naive_errors = np.abs(np.diff(y_train))
        scale = np.mean(naive_errors)  # Should be 2.0
        
        expected_mase = np.mean(np.abs(y_true - y_pred)) / scale
        assert mase(y_true, y_pred, y_train, seasonality=1) == pytest.approx(expected_mase)


class TestMetricsClass:
    """Tests for the Metrics class."""
    
    def test_evaluate_returns_result(self, sample_predictions):
        """Evaluate should return MetricsResult."""
        y_true, y_pred = sample_predictions
        
        metrics = Metrics()
        result = metrics.evaluate(y_true, y_pred)
        
        assert isinstance(result, MetricsResult)
        assert result.rmse > 0
        assert result.mae > 0
        assert result.smape >= 0
    
    def test_evaluate_with_training_data(self, sample_predictions):
        """Evaluate should compute MASE when training data provided."""
        y_true, y_pred = sample_predictions
        y_train = np.random.exponential(10, 200)
        
        metrics = Metrics()
        result = metrics.evaluate(y_true, y_pred, y_train=y_train)
        
        assert result.mase is not None
        assert result.mase > 0
    
    def test_evaluate_per_series(self):
        """Evaluate per series should return DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame({
            "id": ["A"] * 50 + ["B"] * 50,
            "sales": np.random.exponential(10, 100),
            "prediction": np.random.exponential(10, 100),
        })
        
        metrics = Metrics()
        result = metrics.evaluate_per_series(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "rmse" in result.columns
    
    def test_evaluate_by_horizon(self):
        """Evaluate by horizon should return DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame({
            "horizon": np.repeat([1, 7, 14, 28], 25),
            "sales": np.random.exponential(10, 100),
            "prediction": np.random.exponential(10, 100),
        })
        
        metrics = Metrics()
        result = metrics.evaluate_by_horizon(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
    
    def test_compare_models(self, sample_predictions):
        """Compare models should rank correctly."""
        y_true, y_pred = sample_predictions
        
        # Create worse predictions
        y_pred_worse = y_pred + np.random.normal(0, 5, len(y_pred))
        
        metrics = Metrics()
        result = metrics.compare_models(y_true, {
            "good_model": y_pred,
            "bad_model": y_pred_worse,
        })
        
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["model"] == "good_model"
    
    def test_compute_coverage(self, sample_predictions):
        """Coverage should compute correctly."""
        y_true, y_pred = sample_predictions
        
        lower = y_pred - 5
        upper = y_pred + 5
        
        metrics = Metrics()
        coverage = metrics.compute_coverage(y_true, lower, upper)
        
        assert 0 <= coverage <= 1


class TestMetricsResult:
    """Tests for MetricsResult model."""
    
    def test_metrics_result_creation(self):
        """MetricsResult should create correctly."""
        result = MetricsResult(
            rmse=1.5,
            mae=1.2,
            smape=10.0,
            mean_prediction=100.0,
            std_prediction=20.0,
            mean_actual=98.0,
            std_actual=18.0,
        )
        
        assert result.rmse == 1.5
        assert result.mae == 1.2
        assert result.smape == 10.0
    
    def test_metrics_result_optional_fields(self):
        """MetricsResult should handle optional fields."""
        result = MetricsResult(
            rmse=1.5,
            mae=1.2,
            smape=10.0,
            mean_prediction=100.0,
            std_prediction=20.0,
            mean_actual=98.0,
            std_actual=18.0,
            mape=None,
            mase=None,
            wrmsse=None,
        )
        
        assert result.mape is None
        assert result.mase is None
        assert result.wrmsse is None

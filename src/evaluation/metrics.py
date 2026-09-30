"""
Evaluation metrics for demand forecasting.

Implements standard and custom metrics including:
- RMSE, MAE, MAPE, sMAPE
- WRMSSE (M5 competition metric)
- MASE (Mean Absolute Scaled Error)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Note: Undefined when y_true contains zeros.
    Uses epsilon to avoid division by zero.
    """
    mask = y_true > epsilon
    if not mask.any():
        return float("inf")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    More robust than MAPE for values near zero.
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 0

    if not mask.any():
        return 0.0

    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])) * 100


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.

    Scales error by the in-sample naive forecast error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data for computing scale
        seasonality: Seasonality period for naive forecast

    Returns:
        MASE score
    """
    # Compute scale from training data (naive forecast error)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale < 1e-8:
        return float("inf")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def wrmsse(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, scales: np.ndarray
) -> float:
    """
    Weighted Root Mean Squared Scaled Error.

    The M5 competition metric that weights errors by:
    1. Scale: Based on historical variance
    2. Weight: Based on dollar sales value

    Args:
        y_true: Actual values
        y_pred: Predicted values
        weights: Item/series weights (sum to 1)
        scales: Scaling factors for each series

    Returns:
        WRMSSE score
    """
    # Squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Scaled squared errors
    scaled_errors = squared_errors / (scales + 1e-8)

    # Root mean squared scaled error per series
    rmsse_per_series = np.sqrt(np.mean(scaled_errors))

    # Weighted average
    wrmsse_score = float(np.sum(weights * rmsse_per_series))

    return wrmsse_score


class MetricsResult(BaseModel):
    """Result container for evaluation metrics."""

    rmse: float
    mae: float
    mape: float | None = None
    smape: float
    mase: float | None = None
    wrmsse: float | None = None

    # Additional statistics
    mean_prediction: float
    std_prediction: float
    mean_actual: float
    std_actual: float

    # Directional accuracy
    direction_accuracy: float | None = None

    # Coverage (if intervals provided)
    coverage_95: float | None = None


class Metrics:
    """
    Comprehensive metrics calculator for demand forecasting.

    Provides:
    - Standard regression metrics
    - Forecasting-specific metrics (MASE, WRMSSE)
    - Per-series and aggregated metrics
    - Statistical summaries

    Example:
        >>> metrics = Metrics()
        >>> result = metrics.evaluate(y_true, y_pred)
        >>> print(result.rmse, result.mape)
    """

    def __init__(self, seasonality: int = 7, compute_wrmsse: bool = True) -> None:
        """
        Initialize metrics calculator.

        Args:
            seasonality: Seasonality for MASE calculation
            compute_wrmsse: Whether to compute WRMSSE
        """
        self.seasonality = seasonality
        self.compute_wrmsse = compute_wrmsse

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        y_train: np.ndarray | pd.Series | None = None,
        weights: np.ndarray | None = None,
        scales: np.ndarray | None = None,
    ) -> MetricsResult:
        """
        Compute all metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_train: Training data (for MASE)
            weights: Weights for WRMSSE
            scales: Scales for WRMSSE

        Returns:
            MetricsResult with all metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic metrics
        result_dict = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "smape": smape(y_true, y_pred),
            "mean_prediction": float(np.mean(y_pred)),
            "std_prediction": float(np.std(y_pred)),
            "mean_actual": float(np.mean(y_true)),
            "std_actual": float(np.std(y_true)),
        }

        # MAPE (with safeguard)
        if np.any(y_true > 0):
            result_dict["mape"] = mape(y_true, y_pred)

        # MASE (requires training data)
        if y_train is not None:
            y_train = np.asarray(y_train)
            if len(y_train) > self.seasonality:
                result_dict["mase"] = mase(y_true, y_pred, y_train, self.seasonality)

        # WRMSSE (requires weights and scales)
        if self.compute_wrmsse and weights is not None and scales is not None:
            result_dict["wrmsse"] = wrmsse(y_true, y_pred, weights, scales)

        # Directional accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            result_dict["direction_accuracy"] = float(np.mean(actual_direction == pred_direction))

        return MetricsResult(**result_dict)

    def evaluate_per_series(
        self,
        df: pd.DataFrame,
        actual_col: str = "sales",
        pred_col: str = "prediction",
        series_col: str = "id",
    ) -> pd.DataFrame:
        """
        Compute metrics per series.

        Args:
            df: DataFrame with actuals and predictions
            actual_col: Column name for actual values
            pred_col: Column name for predictions
            series_col: Column name for series identifier

        Returns:
            DataFrame with metrics per series
        """
        metrics_list = []

        for series_id, group in df.groupby(series_col):
            y_true = group[actual_col].values
            y_pred = group[pred_col].values

            metrics_list.append(
                {
                    series_col: series_id,
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "smape": smape(y_true, y_pred),
                    "n_samples": len(y_true),
                    "mean_actual": np.mean(y_true),
                    "mean_prediction": np.mean(y_pred),
                }
            )

        return pd.DataFrame(metrics_list)

    def evaluate_by_horizon(
        self,
        df: pd.DataFrame,
        actual_col: str = "sales",
        pred_col: str = "prediction",
        horizon_col: str = "horizon",
    ) -> pd.DataFrame:
        """
        Compute metrics by forecast horizon.

        Args:
            df: DataFrame with predictions at different horizons
            actual_col: Column name for actual values
            pred_col: Column name for predictions
            horizon_col: Column name for forecast horizon

        Returns:
            DataFrame with metrics by horizon
        """
        metrics_list = []

        for horizon, group in df.groupby(horizon_col):
            y_true = group[actual_col].values
            y_pred = group[pred_col].values

            metrics_list.append(
                {
                    "horizon": horizon,
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "smape": smape(y_true, y_pred),
                    "n_samples": len(y_true),
                }
            )

        return pd.DataFrame(metrics_list)

    def compute_coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Compute prediction interval coverage.

        Args:
            y_true: Actual values
            lower: Lower bound of interval
            upper: Upper bound of interval

        Returns:
            Coverage percentage (0-1)
        """
        within_interval = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(within_interval))

    def compare_models(
        self, y_true: np.ndarray, predictions: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            y_true: Actual values
            predictions: Dict of model_name -> predictions

        Returns:
            DataFrame comparing model performance
        """
        results = []

        for model_name, y_pred in predictions.items():
            metrics = self.evaluate(y_true, y_pred)
            results.append(
                {
                    "model": model_name,
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "smape": metrics.smape,
                    "mape": metrics.mape,
                }
            )

        df = pd.DataFrame(results)

        # Add rank columns
        for col in ["rmse", "mae", "smape"]:
            df[f"{col}_rank"] = df[col].rank()

        return df.sort_values("rmse")

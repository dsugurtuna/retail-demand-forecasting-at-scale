"""
Evaluation metrics for demand forecasting.

- Point metrics: RMSE, MAE, MAPE, sMAPE, MASE.
- WRMSSE, the M5 accuracy competition metric: ``wrmsse`` for a set of series,
  ``series_scales`` for the per-series scale, and ``hierarchical_wrmsse`` for
  the 12-level version used to rank M5 entries.

Reference: the M5 competition page, https://www.kaggle.com/c/m5-forecasting-accuracy
(see its Evaluation tab and competitors' guide).
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
    Weighted root mean squared scaled error across a set of series.

    For each series i, RMSSE_i = sqrt(mean_h (y - y_hat)^2 / scale_i), where
    scale_i is the mean squared one-step naive error over that series' training
    history (see ``series_scales``). The result is sum_i w_i * RMSSE_i.

    Args:
        y_true: Actuals, shape (n_series, horizon). 1-D input is one series.
        y_pred: Forecasts, same shape as ``y_true``.
        weights: Non-negative weight per series, shape (n_series,). Renormalised
            to sum to 1 over the series that have a positive scale.
        scales: Scale per series, shape (n_series,). Series with scale 0 (never
            sold, or constant) have no defined RMSSE and are left out.

    Returns:
        WRMSSE. 0 is a perfect forecast.
    """
    actual = np.atleast_2d(np.asarray(y_true, dtype=float))
    forecast = np.atleast_2d(np.asarray(y_pred, dtype=float))
    w = np.asarray(weights, dtype=float).ravel()
    sc = np.asarray(scales, dtype=float).ravel()
    if actual.shape != forecast.shape:
        raise ValueError(f"y_true {actual.shape} and y_pred {forecast.shape} differ in shape")
    if not (len(w) == len(sc) == actual.shape[0]):
        raise ValueError("weights and scales need one value per series (row of y_true)")

    valid = sc > 0
    if not valid.any():
        raise ValueError("Every series has scale 0; WRMSSE is undefined")
    rmsse = np.sqrt(np.mean((actual[valid] - forecast[valid]) ** 2, axis=1) / sc[valid])
    w = w[valid]
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    return float(np.sum(w * rmsse))


def series_scales(history: np.ndarray) -> np.ndarray:
    """
    M5 scale per series: mean squared day-on-day change in the training history,
    counted from each series' first non-zero sale.

    Args:
        history: Training sales, shape (n_series, n_days), oldest day first.

    Returns:
        Scale per series, shape (n_series,). 0 where there are fewer than two
        observations after the first sale.
    """
    x = np.atleast_2d(np.nan_to_num(np.asarray(history, dtype=float)))
    started = np.maximum.accumulate(x != 0, axis=1)
    pair_counted = started[:, :-1]
    squared_diffs = np.diff(x, axis=1) ** 2
    counts = pair_counted.sum(axis=1)
    totals = (squared_diffs * pair_counted).sum(axis=1)
    return np.divide(totals, counts, out=np.zeros(len(x)), where=counts > 0)


def wrmsse_row_weights(
    train: pd.DataFrame,
    series_col: str = "id",
    date_col: str = "date",
    target_col: str = "sales",
    price_col: str = "sell_price",
    weight_days: int = 28,
) -> pd.Series:
    """
    Per-row training weights w_i / scale_i for the bottom-level WRMSSE surrogate.

    w_i is series i's share of dollar sales over the last ``weight_days`` of
    ``train``; scale_i is its ``series_scales`` value. Series with scale 0 get
    weight 0. Weights are rescaled to average 1 over rows, so the learning rate
    means the same thing as without weights.

    Returns:
        A Series aligned to ``train.index``.
    """
    ordered = train.sort_values([series_col, date_col])
    history = ordered.pivot_table(
        index=series_col, columns=date_col, values=target_col, aggfunc="sum"
    ).sort_index(axis=1)
    scales = pd.Series(series_scales(history.to_numpy(dtype=float)), index=history.index)

    dates = pd.to_datetime(train[date_col])
    recent = dates > dates.max() - pd.Timedelta(days=weight_days)
    price = train[price_col].fillna(0) if price_col in train.columns else 1.0
    dollars = (train[target_col].fillna(0) * price).where(recent, 0.0)
    series_weight = dollars.groupby(train[series_col]).sum()
    if series_weight.sum() > 0:
        series_weight = series_weight / series_weight.sum()
    else:
        series_weight = pd.Series(1.0, index=series_weight.index)

    ratio = (series_weight / scales).where(scales > 0, 0.0).fillna(0.0)
    row_weights = train[series_col].map(ratio).astype(float)
    mean = row_weights.mean()
    return row_weights / mean if mean > 0 else row_weights


# The 12 aggregation levels of the M5 accuracy competition. The bottom level
# ("series") is one item in one store, which is the ``id`` column.
M5_LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("category", ["cat_id"]),
    ("department", ["dept_id"]),
    ("state_category", ["state_id", "cat_id"]),
    ("state_department", ["state_id", "dept_id"]),
    ("store_category", ["store_id", "cat_id"]),
    ("store_department", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("series", ["id"]),
]


def _level_matrix(df: pd.DataFrame, keys: list[str], date_col: str, value_col: str) -> pd.DataFrame:
    """Sum ``value_col`` per (level key, date) and pivot to one row per aggregated series."""
    grouped = df.assign(_all="all").groupby([*(keys or ["_all"]), date_col], observed=True)
    summed = grouped[value_col].sum(min_count=1).rename(value_col).reset_index()
    return summed.pivot_table(
        index=keys or ["_all"],
        columns=date_col,
        values=value_col,
        aggfunc="sum",
        dropna=False,
        observed=True,
    ).sort_index(axis=1)


def hierarchical_wrmsse(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_col: str = "sales",
    prediction_col: str = "prediction",
    date_col: str = "date",
    price_col: str = "sell_price",
    weight_days: int = 28,
    levels: list[tuple[str, list[str]]] | None = None,
) -> dict[str, object]:
    """
    WRMSSE averaged over hierarchy levels, following the M5 accuracy competition.

    At each level, series are summed to that level (for example all items in a
    store), scaled by their own training history and weighted by their share of
    dollar sales (units x price) over the last ``weight_days`` of training.
    Each level scores sum_i w_i * RMSSE_i; the overall score is the unweighted
    mean over the levels whose columns exist in the data.

    Args:
        train: Long-format training rows (history up to the forecast origin).
        evaluation: Long-format rows for the forecast horizon with actuals in
            ``target_col`` and forecasts in ``prediction_col``.
        levels: Override the level list. Defaults to ``M5_LEVELS``.

    Returns:
        ``{"wrmsse": float, "levels": {level_name: float}}``.
    """
    levels = levels if levels is not None else M5_LEVELS
    usable = [(name, keys) for name, keys in levels if set(keys) <= set(train.columns)]
    if not usable:
        raise ValueError("None of the hierarchy levels' columns are present")

    train = train.copy()
    last_day = pd.to_datetime(train[date_col]).max()
    recent = pd.to_datetime(train[date_col]) > last_day - pd.Timedelta(days=weight_days)
    if price_col in train.columns:
        dollars = train[target_col].fillna(0) * train[price_col].fillna(0)
    else:
        dollars = train[target_col].fillna(0)
    train["_dollars"] = dollars.where(recent, 0.0)

    scores: dict[str, float] = {}
    for name, keys in usable:
        history = _level_matrix(train, keys, date_col, target_col)
        weights = (
            train.assign(_all="all")
            .groupby(keys or ["_all"], observed=True)["_dollars"]
            .sum()
            .reindex(history.index)
        )
        actual = _level_matrix(evaluation, keys, date_col, target_col).reindex(history.index)
        forecast = _level_matrix(evaluation, keys, date_col, prediction_col).reindex(history.index)
        scores[name] = wrmsse(
            actual.to_numpy(dtype=float),
            forecast.to_numpy(dtype=float),
            weights.to_numpy(dtype=float),
            series_scales(history.to_numpy(dtype=float)),
        )

    return {"wrmsse": float(np.mean(list(scores.values()))), "levels": scores}


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
    Point-forecast metrics for one set of forecasts.

    ``evaluate`` pools all rows (RMSE, MAE, sMAPE, MAPE, optional MASE and
    WRMSSE); ``evaluate_per_series`` and ``evaluate_by_horizon`` break the
    errors down. For the M5 metric on long-format data use
    ``hierarchical_wrmsse``.

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
            y_true = group[actual_col].to_numpy(dtype=float)
            y_pred = group[pred_col].to_numpy(dtype=float)

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
            y_true = group[actual_col].to_numpy(dtype=float)
            y_pred = group[pred_col].to_numpy(dtype=float)

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

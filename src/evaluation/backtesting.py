"""
Rolling-origin backtesting.

Each fold picks a forecast origin, trains on data up to that origin and scores
forecasts for the next ``test_days``. Folds step backwards from the end of the
data, one test window at a time.

How leakage is prevented, fold by fold:

1. Target values after the origin are replaced with nulls *before* features
   are built. Lag and rolling features for the test window can then only use
   what was known at the origin, whatever feature settings are used.
2. The feature engineer is fitted inside the fold, on the masked data.
3. Early stopping uses the last ``validation_days`` before the origin, chosen
   by date, never test rows.
4. A fresh copy of the model is trained in every fold.

``gap_days`` inserts days between the origin and the test window, to mimic
data that arrives late. Those days' targets are hidden too. With ``min_lag``
equal to the horizon, keep ``gap_days`` at 0, otherwise the last test days
have no lag features at all.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from src.evaluation.metrics import Metrics, MetricsResult, hierarchical_wrmsse
from src.features.engineer import NON_FEATURE_COLUMNS, to_model_input

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for rolling-origin backtests."""

    n_folds: int = Field(default=5, ge=1, description="Number of forecast origins")
    test_days: int = Field(default=28, ge=1, description="Forecast horizon per fold")
    gap_days: int = Field(
        default=0, ge=0, description="Hidden days between the origin and the test window"
    )
    min_train_days: int = Field(default=365, ge=28, description="Training length when sliding")
    expanding_window: bool = Field(
        default=True, description="Train from the start of the data (True) or a sliding window"
    )
    validation_days: int = Field(
        default=28, ge=0, description="Days before the origin held out for early stopping"
    )
    evaluate_by_horizon: bool = Field(default=True)


@dataclass
class FoldResult:
    """Results from one fold."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    metrics: MetricsResult
    wrmsse: float | None = None
    predictions: pd.DataFrame | None = None
    model_metadata: dict[str, Any] | None = None


@dataclass
class BacktestResult:
    """Results from a full backtest."""

    config: BacktestConfig
    fold_results: list[FoldResult]
    aggregate_metrics: MetricsResult
    metrics_by_fold: pd.DataFrame
    metrics_by_horizon: pd.DataFrame | None = None
    total_time_seconds: float = 0.0

    @property
    def mean_wrmsse(self) -> float | None:
        values = [f.wrmsse for f in self.fold_results if f.wrmsse is not None]
        return float(np.mean(values)) if values else None


class BacktestEngine:
    """Run rolling-origin backtests for any model with ``fit``/``predict``.

    Example:
        >>> engine = BacktestEngine(BacktestConfig(n_folds=3))
        >>> result = engine.run(data, LightGBMForecaster(), FeatureEngineer())
        >>> result.mean_wrmsse
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self._metrics = Metrics()

    def run(
        self,
        data: pl.DataFrame | pd.DataFrame,
        model: Any,
        feature_engineer: Any | None = None,
        target_col: str = "sales",
        date_col: str = "date",
        group_col: str = "id",
        callbacks: list[Callable[[FoldResult], None]] | None = None,
    ) -> BacktestResult:
        """Run every fold and aggregate the results."""
        start = time.time()
        frame = pl.from_pandas(data) if isinstance(data, pd.DataFrame) else data
        frame = frame.with_columns(pl.col(date_col).cast(pl.Datetime("us")))

        min_lag = getattr(getattr(feature_engineer, "config", None), "min_lag", None)
        horizon = self.config.gap_days + self.config.test_days
        if min_lag is not None and horizon > min_lag:
            logger.warning(
                "gap_days + test_days = %d exceeds min_lag = %d: the last test days "
                "will have no lag features",
                horizon,
                min_lag,
            )

        folds = self._generate_folds(frame, date_col)
        fold_results = []
        for fold_id, fold_dates in enumerate(folds):
            logger.info(
                "Fold %d/%d: origin %s, test %s to %s",
                fold_id + 1,
                len(folds),
                fold_dates["train_end"].date(),
                fold_dates["test_start"].date(),
                fold_dates["test_end"].date(),
            )
            result = self._run_fold(
                frame, model, feature_engineer, fold_id, fold_dates, target_col, date_col, group_col
            )
            fold_results.append(result)
            for callback in callbacks or []:
                callback(result)

        if not fold_results:
            raise ValueError("No valid folds: not enough history for this configuration")

        metrics_by_horizon = None
        predictions = [f.predictions for f in fold_results if f.predictions is not None]
        if self.config.evaluate_by_horizon and predictions:
            metrics_by_horizon = self._metrics.evaluate_by_horizon(
                pd.concat(predictions, ignore_index=True), actual_col=target_col
            )

        backtest = BacktestResult(
            config=self.config,
            fold_results=fold_results,
            aggregate_metrics=self._aggregate_metrics(fold_results),
            metrics_by_fold=self._fold_summary(fold_results),
            metrics_by_horizon=metrics_by_horizon,
            total_time_seconds=time.time() - start,
        )
        logger.info(
            "Backtest finished in %.1fs: RMSE %.4f, mean WRMSSE %s",
            backtest.total_time_seconds,
            backtest.aggregate_metrics.rmse,
            f"{backtest.mean_wrmsse:.4f}" if backtest.mean_wrmsse is not None else "n/a",
        )
        return backtest

    def _generate_folds(self, frame: pl.DataFrame, date_col: str) -> list[dict[str, datetime]]:
        """Fold date ranges, oldest first."""
        cfg = self.config
        min_date = frame[date_col].min()
        max_date = frame[date_col].max()
        assert isinstance(min_date, datetime) and isinstance(max_date, datetime)
        day = pd.Timedelta(days=1)

        folds = []
        for k in range(cfg.n_folds):
            test_end = max_date - k * cfg.test_days * day
            test_start = test_end - (cfg.test_days - 1) * day
            train_end = test_start - (cfg.gap_days + 1) * day
            train_start = (
                min_date if cfg.expanding_window else train_end - (cfg.min_train_days - 1) * day
            )
            if train_start < min_date or train_end - train_start < cfg.validation_days * day:
                logger.warning("Skipping fold %d: not enough training history", k + 1)
                continue
            folds.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
        return list(reversed(folds))

    def _run_fold(
        self,
        frame: pl.DataFrame,
        model: Any,
        feature_engineer: Any | None,
        fold_id: int,
        fold_dates: dict[str, datetime],
        target_col: str,
        date_col: str,
        group_col: str,
    ) -> FoldResult:
        date = pl.col(date_col)
        history = frame.filter(date <= fold_dates["test_end"])
        masked = history.with_columns(
            pl.when(date > fold_dates["train_end"])
            .then(None)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )

        if feature_engineer is not None:
            features = feature_engineer.fit_transform(masked)
            feature_cols = feature_engineer.get_feature_names()
        else:
            features = masked
            excluded = NON_FEATURE_COLUMNS | {target_col, date_col, group_col, "id"}
            feature_cols = [c for c in features.columns if c not in excluded]

        in_train = date.is_between(fold_dates["train_start"], fold_dates["train_end"])
        in_test = date.is_between(fold_dates["test_start"], fold_dates["test_end"])
        train = features.filter(in_train)
        test_actuals = history.filter(in_test).select(group_col, date_col, target_col)
        test = (
            features.filter(in_test)
            .drop(target_col)
            .join(test_actuals, on=[group_col, date_col], how="left")
            .sort([group_col, date_col])
        )

        val_start = fold_dates["train_end"] - pd.Timedelta(days=self.config.validation_days - 1)
        fit_part = train.filter(date < val_start) if self.config.validation_days else train
        val_part = train.filter(date >= val_start) if self.config.validation_days else None

        model_instance = model.clone() if hasattr(model, "clone") else copy.deepcopy(model)
        model_instance.fit(
            to_model_input(fit_part, feature_cols),
            fit_part[target_col].to_numpy(),
            X_valid=to_model_input(val_part, feature_cols) if val_part is not None else None,
            y_valid=val_part[target_col].to_numpy() if val_part is not None else None,
        )

        predictions = np.maximum(
            np.asarray(model_instance.predict(to_model_input(test, feature_cols)), dtype=float), 0
        )
        y_test = test[target_col].to_numpy()
        metrics = self._metrics.evaluate(y_test, predictions, y_train=train[target_col].to_numpy())

        predictions_df = test.select(group_col, date_col, target_col).to_pandas()
        predictions_df["prediction"] = predictions
        predictions_df["fold"] = fold_id
        predictions_df["horizon"] = (
            predictions_df[date_col] - pd.Timestamp(fold_dates["test_start"])
        ).dt.days + 1

        # Score on the untouched rows: feature building may have dropped
        # constant columns (such as state_id) that WRMSSE levels still need.
        fold_wrmsse = None
        eval_frame = history.filter(in_test).sort([group_col, date_col]).to_pandas()
        if (eval_frame[group_col].to_numpy() == test[group_col].to_numpy()).all():
            eval_frame["prediction"] = predictions
            train_history = history.filter(date <= fold_dates["train_end"]).to_pandas()
            try:
                scores = hierarchical_wrmsse(
                    train_history, eval_frame, target_col=target_col, date_col=date_col
                )
                fold_wrmsse = float(scores["wrmsse"])  # type: ignore[arg-type]
            except (ValueError, KeyError) as exc:
                logger.warning("WRMSSE not computed for fold %d: %s", fold_id + 1, exc)

        metadata = getattr(model_instance, "metadata", None)
        return FoldResult(
            fold_id=fold_id,
            train_start=fold_dates["train_start"],
            train_end=fold_dates["train_end"],
            test_start=fold_dates["test_start"],
            test_end=fold_dates["test_end"],
            train_size=len(train),
            test_size=len(test),
            metrics=metrics,
            wrmsse=fold_wrmsse,
            predictions=predictions_df,
            model_metadata=metadata.model_dump() if metadata is not None else None,
        )

    @staticmethod
    def _aggregate_metrics(fold_results: list[FoldResult]) -> MetricsResult:
        """Test-size-weighted averages of the per-fold metrics."""
        sizes = np.array([f.test_size for f in fold_results], dtype=float)

        def weighted(values: list[float]) -> float:
            return float(np.average(values, weights=sizes))

        mapes = [f.metrics.mape for f in fold_results if f.metrics.mape is not None]
        return MetricsResult(
            rmse=weighted([f.metrics.rmse for f in fold_results]),
            mae=weighted([f.metrics.mae for f in fold_results]),
            smape=weighted([f.metrics.smape for f in fold_results]),
            mape=float(np.mean(mapes)) if mapes else None,
            mean_prediction=float(np.mean([f.metrics.mean_prediction for f in fold_results])),
            std_prediction=float(np.mean([f.metrics.std_prediction for f in fold_results])),
            mean_actual=float(np.mean([f.metrics.mean_actual for f in fold_results])),
            std_actual=float(np.mean([f.metrics.std_actual for f in fold_results])),
        )

    @staticmethod
    def _fold_summary(fold_results: list[FoldResult]) -> pd.DataFrame:
        records = [
            {
                "fold": f.fold_id + 1,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "train_size": f.train_size,
                "test_size": f.test_size,
                "rmse": f.metrics.rmse,
                "mae": f.metrics.mae,
                "smape": f.metrics.smape,
                "wrmsse": f.wrmsse,
            }
            for f in fold_results
        ]
        return pd.DataFrame(records)

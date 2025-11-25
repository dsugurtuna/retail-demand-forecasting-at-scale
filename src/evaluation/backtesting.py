"""
Time-series backtesting framework.

Implements rigorous backtesting for demand forecasting with:
- Rolling origin cross-validation
- Gap periods to prevent leakage
- Multiple fold strategies
- Comprehensive reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from src.evaluation.metrics import Metrics, MetricsResult

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    
    n_folds: int = Field(default=5, ge=1, description="Number of CV folds")
    test_days: int = Field(default=28, ge=1, description="Test period length in days")
    gap_days: int = Field(default=28, ge=0, description="Gap between train and test")
    min_train_days: int = Field(default=365, ge=28, description="Minimum training days")
    
    # Strategy
    expanding_window: bool = Field(
        default=True,
        description="Use expanding (True) or sliding (False) window"
    )
    
    # Feature regeneration
    regenerate_features: bool = Field(
        default=True,
        description="Regenerate features for each fold"
    )
    
    # Evaluation
    evaluate_per_series: bool = Field(default=False)
    evaluate_by_horizon: bool = Field(default=True)


@dataclass
class FoldResult:
    """Results from a single backtest fold."""
    
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    metrics: MetricsResult
    predictions: pd.DataFrame | None = None
    model_metadata: dict | None = None


@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    config: BacktestConfig
    fold_results: list[FoldResult]
    aggregate_metrics: MetricsResult
    metrics_by_fold: pd.DataFrame
    metrics_by_horizon: pd.DataFrame | None = None
    total_time_seconds: float = 0.0


class BacktestEngine:
    """
    Time-series backtesting engine.
    
    Implements production-grade backtesting with:
    - Rolling origin cross-validation
    - Gap periods to prevent data leakage
    - Expanding or sliding windows
    - Comprehensive metrics and reporting
    
    Example:
        >>> config = BacktestConfig(n_folds=5, test_days=28, gap_days=28)
        >>> engine = BacktestEngine(config)
        >>> results = engine.run(
        ...     data=df,
        ...     model=LightGBMForecaster(),
        ...     feature_engineer=FeatureEngineer()
        ... )
        >>> print(results.aggregate_metrics)
    """
    
    def __init__(self, config: BacktestConfig | None = None) -> None:
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
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
        callbacks: list[Callable] | None = None
    ) -> BacktestResult:
        """
        Run backtesting.
        
        Args:
            data: Full dataset
            model: Forecasting model (with fit/predict interface)
            feature_engineer: Feature engineering pipeline
            target_col: Target column name
            date_col: Date column name
            group_col: Group/series column name
            callbacks: Optional callbacks after each fold
            
        Returns:
            BacktestResult with all metrics and details
        """
        import time
        start_time = time.time()
        
        logger.info(
            f"Starting backtest with {self.config.n_folds} folds, "
            f"{self.config.test_days} day test periods"
        )
        
        # Convert to pandas for easier manipulation
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Generate fold splits
        folds = self._generate_folds(df, date_col)
        
        fold_results = []
        all_predictions = []
        
        for fold_id, fold_dates in enumerate(folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Fold {fold_id + 1}/{self.config.n_folds}")
            logger.info(f"{'='*60}")
            
            fold_result = self._run_fold(
                df=df,
                model=model,
                feature_engineer=feature_engineer,
                fold_id=fold_id,
                fold_dates=fold_dates,
                target_col=target_col,
                date_col=date_col,
                group_col=group_col
            )
            
            fold_results.append(fold_result)
            
            if fold_result.predictions is not None:
                all_predictions.append(fold_result.predictions)
            
            logger.info(f"Fold {fold_id + 1} RMSE: {fold_result.metrics.rmse:.4f}")
            
            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(fold_result)
        
        # Aggregate results
        aggregate_metrics = self._aggregate_metrics(fold_results)
        metrics_by_fold = self._create_fold_summary(fold_results)
        
        # Metrics by horizon if requested
        metrics_by_horizon = None
        if self.config.evaluate_by_horizon and all_predictions:
            combined_preds = pd.concat(all_predictions, ignore_index=True)
            metrics_by_horizon = self._metrics.evaluate_by_horizon(
                combined_preds,
                actual_col=target_col,
                pred_col="prediction"
            )
        
        total_time = time.time() - start_time
        
        result = BacktestResult(
            config=self.config,
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            metrics_by_fold=metrics_by_fold,
            metrics_by_horizon=metrics_by_horizon,
            total_time_seconds=total_time
        )
        
        logger.info(f"\nBacktest completed in {total_time:.2f}s")
        logger.info(f"Average RMSE: {aggregate_metrics.rmse:.4f}")
        logger.info(f"Average SMAPE: {aggregate_metrics.smape:.2f}%")
        
        return result
    
    def _generate_folds(
        self,
        df: pd.DataFrame,
        date_col: str
    ) -> list[dict]:
        """Generate fold date ranges."""
        max_date = df[date_col].max()
        min_date = df[date_col].min()
        
        folds = []
        
        for fold_id in range(self.config.n_folds):
            # Calculate dates working backwards from max_date
            test_end = max_date - pd.Timedelta(days=fold_id * self.config.test_days)
            test_start = test_end - pd.Timedelta(days=self.config.test_days - 1)
            
            train_end = test_start - pd.Timedelta(days=self.config.gap_days + 1)
            
            if self.config.expanding_window:
                train_start = min_date
            else:
                # Sliding window
                train_start = train_end - pd.Timedelta(days=self.config.min_train_days)
            
            # Validate fold
            if train_start >= train_end:
                logger.warning(f"Skipping fold {fold_id}: insufficient training data")
                continue
            
            folds.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })
            
            logger.info(
                f"Fold {fold_id + 1}: Train [{train_start.date()} - {train_end.date()}], "
                f"Test [{test_start.date()} - {test_end.date()}]"
            )
        
        return folds
    
    def _run_fold(
        self,
        df: pd.DataFrame,
        model: Any,
        feature_engineer: Any | None,
        fold_id: int,
        fold_dates: dict,
        target_col: str,
        date_col: str,
        group_col: str
    ) -> FoldResult:
        """Run a single backtest fold."""
        # Split data
        train_mask = (
            (df[date_col] >= fold_dates["train_start"]) &
            (df[date_col] <= fold_dates["train_end"])
        )
        test_mask = (
            (df[date_col] >= fold_dates["test_start"]) &
            (df[date_col] <= fold_dates["test_end"])
        )
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Generate features if feature engineer provided
        if feature_engineer is not None and self.config.regenerate_features:
            logger.info("Generating features for this fold...")
            
            # Fit on training data only
            feature_engineer.fit(pl.from_pandas(train_df))
            
            train_features = feature_engineer.transform(pl.from_pandas(train_df)).to_pandas()
            test_features = feature_engineer.transform(pl.from_pandas(test_df)).to_pandas()
        else:
            train_features = train_df
            test_features = test_df
        
        # Prepare X and y
        exclude_cols = {target_col, group_col, date_col, "d", "wm_yr_wk", "id"}
        feature_cols = [c for c in train_features.columns if c not in exclude_cols]
        
        X_train = train_features[feature_cols]
        y_train = train_features[target_col]
        X_test = test_features[feature_cols]
        y_test = test_features[target_col]
        
        # Handle categorical features
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
        
        # Train model
        logger.info(f"Training on {len(X_train):,} samples...")
        
        # Create a fresh model instance for this fold
        model_instance = model.__class__(
            **model.get_params().get("hyperparameters", {})
        ) if hasattr(model, "get_params") else model
        
        # Fit with validation
        if hasattr(model_instance, "fit"):
            # Use last portion of training data as validation
            val_size = min(len(X_train) // 5, self.config.test_days * len(df[group_col].unique()))
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
            
            model_instance.fit(
                X_train_fit, y_train_fit,
                X_valid=X_val, y_valid=y_val,
                categorical_features=cat_cols if cat_cols else None
            )
        
        # Predict
        logger.info(f"Predicting {len(X_test):,} samples...")
        predictions = model_instance.predict(X_test)
        
        # Evaluate
        metrics = self._metrics.evaluate(
            y_test.values,
            predictions,
            y_train=y_train.values
        )
        
        # Create predictions DataFrame
        predictions_df = test_df[[group_col, date_col, target_col]].copy()
        predictions_df["prediction"] = predictions
        predictions_df["fold"] = fold_id
        
        # Add horizon (days from test start)
        predictions_df["horizon"] = (
            predictions_df[date_col] - fold_dates["test_start"]
        ).dt.days + 1
        
        return FoldResult(
            fold_id=fold_id,
            train_start=fold_dates["train_start"],
            train_end=fold_dates["train_end"],
            test_start=fold_dates["test_start"],
            test_end=fold_dates["test_end"],
            train_size=len(train_df),
            test_size=len(test_df),
            metrics=metrics,
            predictions=predictions_df,
            model_metadata=model_instance.metadata.model_dump() if hasattr(model_instance, "metadata") and model_instance.metadata else None
        )
    
    def _aggregate_metrics(self, fold_results: list[FoldResult]) -> MetricsResult:
        """Aggregate metrics across folds."""
        rmse_values = [f.metrics.rmse for f in fold_results]
        mae_values = [f.metrics.mae for f in fold_results]
        smape_values = [f.metrics.smape for f in fold_results]
        
        # Weighted average by test size
        total_test_size = sum(f.test_size for f in fold_results)
        
        weighted_rmse = sum(
            f.metrics.rmse * f.test_size for f in fold_results
        ) / total_test_size
        
        weighted_mae = sum(
            f.metrics.mae * f.test_size for f in fold_results
        ) / total_test_size
        
        weighted_smape = sum(
            f.metrics.smape * f.test_size for f in fold_results
        ) / total_test_size
        
        return MetricsResult(
            rmse=weighted_rmse,
            mae=weighted_mae,
            smape=weighted_smape,
            mape=np.mean([f.metrics.mape for f in fold_results if f.metrics.mape is not None]),
            mean_prediction=np.mean([f.metrics.mean_prediction for f in fold_results]),
            std_prediction=np.mean([f.metrics.std_prediction for f in fold_results]),
            mean_actual=np.mean([f.metrics.mean_actual for f in fold_results]),
            std_actual=np.mean([f.metrics.std_actual for f in fold_results]),
        )
    
    def _create_fold_summary(self, fold_results: list[FoldResult]) -> pd.DataFrame:
        """Create summary DataFrame of fold results."""
        records = []
        
        for f in fold_results:
            records.append({
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
                "mape": f.metrics.mape,
            })
        
        df = pd.DataFrame(records)
        
        # Add summary row
        summary = pd.DataFrame([{
            "fold": "Average",
            "rmse": df["rmse"].mean(),
            "mae": df["mae"].mean(),
            "smape": df["smape"].mean(),
            "mape": df["mape"].mean() if df["mape"].notna().any() else None,
        }])
        
        return pd.concat([df, summary], ignore_index=True)


class Evaluator:
    """Legacy evaluator class for backward compatibility."""
    
    def __init__(self, train_df=None, valid_df=None, weights=None):
        self.train_df = train_df
        self.valid_df = valid_df
        self.weights = weights
        self._metrics = Metrics()
    
    def generate_report(self, y_true, y_pred):
        result = self._metrics.evaluate(np.array(y_true), np.array(y_pred))
        return {
            "RMSE": result.rmse,
            "MAE": result.mae,
            "SMAPE": result.smape,
        }

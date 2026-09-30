"""
LightGBM forecaster.

Defaults to LightGBM's built-in Tweedie objective, which suits non-negative,
often-zero count data. An optional custom objective (weighted squared error)
becomes a surrogate for bottom-level WRMSSE when each row carries the weight
w_i / scale_i from ``src.evaluation.metrics.wrmsse_row_weights``. It is a
surrogate, not WRMSSE itself: WRMSSE takes a square root per series, which a
per-row objective cannot express.

Prediction intervals from ``predict_interval`` are a fixed +/-20% heuristic,
not a calibrated model of uncertainty.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.models.base import BaseForecaster, ModelMetadata

logger = logging.getLogger(__name__)


def _available_cpus() -> int:
    """CPUs this process may run on (cgroup/affinity aware on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS and Windows
        return os.cpu_count() or 1


class LightGBMConfig(BaseModel):
    """Configuration for LightGBM model."""

    # Objective
    objective: str = Field(default="tweedie", description="Tweedie for count data")
    tweedie_variance_power: float = Field(default=1.5, ge=1.0, le=2.0)

    # Tree parameters
    num_leaves: int = Field(default=128, ge=2, le=4096)
    max_depth: int = Field(default=-1, description="-1 for no limit")
    min_data_in_leaf: int = Field(default=100, ge=1)

    # Learning parameters
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    n_estimators: int = Field(default=2000, ge=1)

    # Regularization
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.7, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l1: float = Field(default=0.1, ge=0)
    lambda_l2: float = Field(default=0.1, ge=0)

    # Training
    early_stopping_rounds: int = Field(default=50, ge=1)
    verbose: int = Field(default=-1)
    seed: int = Field(default=42)

    # Hardware
    n_jobs: int = Field(
        default=0, description="Threads; 0 = the number of CPUs this process may use"
    )
    device: str = Field(default="cpu")


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM-based demand forecaster.

    - Tweedie objective by default, for count and intermittent demand.
    - Optional weighted squared-error objective (a WRMSSE surrogate; see module docstring).
    - Pandas ``category`` columns are used as categorical features.
    - Early stopping when a validation set is given.

    Example:
        >>> config = LightGBMConfig(learning_rate=0.05, num_leaves=128)
        >>> model = LightGBMForecaster(config=config)
        >>> model.fit(X_train, y_train, X_valid, y_valid)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        config: LightGBMConfig | None = None,
        categorical_features: list[str] | None = None,
        use_custom_objective: bool = False,
        model_name: str = "lightgbm_forecaster",
    ) -> None:
        """
        Initialize LightGBM forecaster.

        Args:
            config: Model configuration
            categorical_features: List of categorical feature names
            use_custom_objective: Use the weighted squared-error objective instead of
                ``config.objective``. Pass ``sample_weight`` to ``fit`` to weight rows.
            model_name: Name for this model instance
        """
        self.config = config or LightGBMConfig()
        self.categorical_features = categorical_features or []
        self.use_custom_objective = use_custom_objective

        super().__init__(
            model_name=model_name, model_type="lightgbm", hyperparameters=self.config.model_dump()
        )

        self._booster: lgb.Booster | None = None
        self._evals_result: dict = {}

    @property
    def booster(self) -> lgb.Booster:
        """The trained LightGBM booster; raises if the model is not fitted."""
        if self._booster is None:
            raise RuntimeError(f"Model '{self.model_name}' is not fitted. Call fit() first.")
        return self._booster

    def clone(self) -> LightGBMForecaster:
        """Return an unfitted copy with the same configuration."""
        return LightGBMForecaster(
            config=self.config.model_copy(deep=True),
            categorical_features=list(self.categorical_features),
            use_custom_objective=self.use_custom_objective,
            model_name=self.model_name,
        )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        callbacks: list | None = None,
        **kwargs: Any,
    ) -> LightGBMForecaster:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight: Sample weights for training
            callbacks: Additional LightGBM callbacks
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        logger.info(f"Training LightGBM model '{self.model_name}'...")
        start_time = time.time()

        # Extract feature names
        self._feature_names = self._extract_feature_names(X_train)
        if isinstance(X_train, pd.DataFrame) and not self.categorical_features:
            # Remember pandas categoricals so callers (e.g. the API) can rebuild them.
            self.categorical_features = [
                str(c)
                for c, dtype in X_train.dtypes.items()
                if isinstance(dtype, pd.CategoricalDtype)
            ]

        # Prepare categorical features
        cat_features = self._get_categorical_indices(X_train)

        # Create datasets
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight,
            categorical_feature=cat_features if cat_features else "auto",
        )

        valid_sets = [train_set]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None:
            valid_set = lgb.Dataset(
                X_valid,
                label=y_valid,
                reference=train_set,
                categorical_feature=cat_features if cat_features else "auto",
            )
            valid_sets.append(valid_set)
            valid_names.append("valid")

        # Prepare parameters
        params = self._get_lgb_params()

        # Custom objective if requested
        if self.use_custom_objective:
            # RMSE stays as the evaluation metric so early stopping still works.
            params["objective"] = self._weighted_squared_error

        default_callbacks: list[Callable[..., Any]] = [
            lgb.log_evaluation(period=100 if self.config.verbose > 0 else 0)
        ]
        if len(valid_sets) > 1:  # early stopping needs a validation set
            default_callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=self.config.verbose > 0,
                )
            )

        if callbacks:
            default_callbacks.extend(callbacks)

        # Train
        self._evals_result = {}
        self._booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=default_callbacks,
        )

        # Keep the trained trees, drop the binned training data they came from.
        self._booster.free_dataset()
        self._model = self._booster
        self._is_fitted = True

        # Create metadata
        training_time = time.time() - start_time
        metrics = self._get_training_metrics()
        metrics["training_time_seconds"] = training_time

        self._metadata = self._create_metadata(training_rows=len(y_train), metrics=metrics)

        logger.info(
            f"Training completed in {training_time:.2f}s, "
            f"best iteration: {self.booster.best_iteration}"
        )

        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        return_std: bool = False,
        num_iteration: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Args:
            X: Features for prediction
            return_std: Whether to return uncertainty estimates
            num_iteration: Number of iterations to use (None for best)
            **kwargs: Additional arguments

        Returns:
            Predictions (and optionally uncertainty estimates)
        """
        self._check_fitted()

        start_time = time.time()

        if isinstance(X, pd.DataFrame) and self._feature_names:
            X = X[self._feature_names]
        predictions = self.booster.predict(
            X, num_iteration=num_iteration or self.booster.best_iteration
        )

        # Ensure non-negative for count data
        predictions = np.maximum(predictions, 0)

        inference_time = (time.time() - start_time) * 1000
        logger.debug(f"Inference completed in {inference_time:.2f}ms for {len(X)} samples")

        if return_std:
            return predictions, self._estimate_uncertainty(predictions)

        return predictions

    def predict_interval(
        self, X: pd.DataFrame | np.ndarray, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with a heuristic band: prediction +/- z * 0.2 * prediction.

        This is a placeholder, not a calibrated interval. Its coverage has not
        been measured. Use quantile models or conformal prediction before
        relying on it for stock decisions.

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions, std = self.predict(X, return_std=True)

        from scipy import stats

        z = stats.norm.ppf((1 + confidence) / 2)

        lower = np.maximum(predictions - z * std, 0)
        upper = predictions + z * std

        return predictions, lower, upper

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'split', 'cover')

        Returns:
            DataFrame with feature importance
        """
        self._check_fitted()

        importance = self.booster.feature_importance(importance_type=importance_type)
        feature_names = self.booster.feature_name()

        df = pd.DataFrame({"feature": feature_names, "importance": importance})

        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_normalized"] = df["importance"] / df["importance"].sum()
        df["cumulative_importance"] = df["importance_normalized"].cumsum()

        return df

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        self._check_fitted()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save booster
        self.booster.save_model(str(path / "model.txt"))

        # Save metadata and config
        if self._metadata is not None:
            (path / "metadata.json").write_text(self._metadata.model_dump_json(indent=2))
        (path / "config.json").write_text(self.config.model_dump_json(indent=2))
        (path / "categorical_features.json").write_text(json.dumps(self.categorical_features))

        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path to load model from
        """
        path = Path(path)

        # Load booster
        self._booster = lgb.Booster(model_file=str(path / "model.txt"))
        self._model = self._booster

        # Load metadata
        self._metadata = ModelMetadata.model_validate_json((path / "metadata.json").read_text())
        self.config = LightGBMConfig.model_validate_json((path / "config.json").read_text())
        if (path / "categorical_features.json").exists():
            self.categorical_features = json.loads((path / "categorical_features.json").read_text())

        self._feature_names = self._metadata.features
        self._is_fitted = True

        logger.info(f"Model loaded from {path}")

    def _get_lgb_params(self) -> dict[str, Any]:
        """Get LightGBM parameters dict."""
        return {
            "objective": self.config.objective,
            "tweedie_variance_power": self.config.tweedie_variance_power,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "lambda_l1": self.config.lambda_l1,
            "lambda_l2": self.config.lambda_l2,
            "verbose": self.config.verbose,
            "seed": self.config.seed,
            "n_jobs": self.config.n_jobs or _available_cpus(),
            "device": self.config.device,
            "metric": "rmse",
            "force_row_wise": True,
            "deterministic": True,  # same data + params + platform -> same model
        }

    def _get_categorical_indices(self, X: pd.DataFrame | np.ndarray) -> list[int] | None:
        """Get indices of categorical features."""
        if not self.categorical_features:
            return None

        if isinstance(X, pd.DataFrame):
            indices = []
            for cat_name in self.categorical_features:
                if cat_name in X.columns:
                    indices.append(list(X.columns).index(cat_name))
            return indices if indices else None

        return None

    def _get_training_metrics(self) -> dict[str, float]:
        """Extract metrics from training history."""
        metrics: dict[str, float] = {
            "best_iteration": self.booster.best_iteration,
            "num_features": self.booster.num_feature(),
            "num_trees": self.booster.num_trees(),
        }

        if self.booster.best_score:
            for dataset, scores in self.booster.best_score.items():
                for metric_name, value in scores.items():
                    metrics[f"{dataset}_{metric_name}"] = float(value)

        return metrics

    @staticmethod
    def _estimate_uncertainty(predictions: np.ndarray) -> np.ndarray:
        """Heuristic spread: 20% of the prediction. Not calibrated."""
        return np.abs(predictions) * 0.2

    @staticmethod
    def _weighted_squared_error(
        preds: np.ndarray, train_data: lgb.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Weighted squared error: loss = 0.5 * sum_r weight_r * (pred_r - y_r)^2.

        With row weights w_i / scale_i (series weight over series scale) this is
        the weighted scaled squared error inside WRMSSE, without the per-series
        square root. Without weights it is plain squared error.
        """
        labels = np.asarray(train_data.get_label(), dtype=float)
        raw_weights = train_data.get_weight()
        weights = (
            np.ones_like(labels) if raw_weights is None else np.asarray(raw_weights, dtype=float)
        )
        grad = weights * (preds - labels)  # d loss / d pred
        hess = weights  # d2 loss / d pred2, constant for squared error
        return grad, hess

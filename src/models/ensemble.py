"""
Ensemble forecaster combining multiple models.

Combines fitted forecasters by weighted average (weights can be fitted on
validation data with SLSQP) or by median. Only LightGBM members exist in this
repository; there are no deep-learning members.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.models.base import BaseForecaster, ModelMetadata

logger = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble model."""

    method: str = Field(
        default="weighted_average",
        description="'weighted_average' or 'median'; anything else falls back to weighted_average",
    )

    weights: dict[str, float] | None = Field(
        default=None, description="Model weights for weighted average"
    )

    optimize_weights: bool = Field(
        default=True, description="Whether to optimize weights on validation data"
    )

    weight_optimization_metric: str = Field(
        default="rmse", description="Metric to optimize weights for"
    )


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster combining multiple base models.

    Strategies:
    - Weighted average, with fixed weights or weights fitted on validation data
    - Median of member forecasts

    Example:
        >>> lgb_model = LightGBMForecaster()
        >>> lgb_model.fit(X_train, y_train)
        >>>
        >>> ensemble = EnsembleForecaster()
        >>> ensemble.add_model("lgb_a", lgb_model, weight=0.5)
        >>> ensemble.add_model("lgb_b", other_lgb_model, weight=0.5)
        >>>
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self, config: EnsembleConfig | None = None, model_name: str = "ensemble_forecaster"
    ) -> None:
        """
        Initialize ensemble forecaster.

        Args:
            config: Ensemble configuration
            model_name: Name for this ensemble
        """
        self.config = config or EnsembleConfig()

        super().__init__(
            model_name=model_name, model_type="ensemble", hyperparameters=self.config.model_dump()
        )

        self._models: dict[str, BaseForecaster] = {}
        self._weights: dict[str, float] = {}
        self._is_weight_optimized = False

    def add_model(
        self, name: str, model: BaseForecaster, weight: float = 1.0
    ) -> EnsembleForecaster:
        """
        Add a model to the ensemble.

        Args:
            name: Name for this model in ensemble
            model: Fitted forecaster model
            weight: Initial weight for this model

        Returns:
            Self for method chaining
        """
        if not model.is_fitted:
            raise ValueError(f"Model '{name}' must be fitted before adding to ensemble")

        self._models[name] = model
        self._weights[name] = weight

        logger.info(f"Added model '{name}' to ensemble with weight {weight}")
        return self

    def remove_model(self, name: str) -> EnsembleForecaster:
        """Remove a model from the ensemble."""
        if name in self._models:
            del self._models[name]
            del self._weights[name]
            logger.info(f"Removed model '{name}' from ensemble")
        return self

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> EnsembleForecaster:
        """
        Fit the ensemble (optimize weights if configured).

        Note: Individual models should already be fitted before
        adding to the ensemble. This method optimizes ensemble weights.

        Args:
            X_train: Training features (for weight optimization)
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        if len(self._models) == 0:
            raise ValueError("No models in ensemble. Add models with add_model() first.")

        logger.info(f"Fitting ensemble with {len(self._models)} models")

        # Extract feature names from first model
        first_model = next(iter(self._models.values()))
        self._feature_names = first_model.feature_names

        # Optimize weights if configured and validation data available
        if self.config.optimize_weights and X_valid is not None and y_valid is not None:
            self._optimize_weights(X_valid, y_valid)
        elif self.config.weights:
            # Use provided weights
            for name, weight in self.config.weights.items():
                if name in self._weights:
                    self._weights[name] = weight

        # Normalize weights
        self._normalize_weights()

        self._is_fitted = True

        # Create metadata
        self._metadata = self._create_metadata(
            training_rows=len(y_train) if hasattr(y_train, "__len__") else 0,
            metrics={"n_models": len(self._models)},
        )

        logger.info(f"Ensemble fitted with weights: {self._weights}")
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        return_std: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Generate ensemble predictions.

        Args:
            X: Features for prediction
            return_std: Also return the spread between member models
            **kwargs: ``return_all_predictions=True`` also returns each member's forecasts

        Returns:
            Ensemble predictions (and optionally std and individual predictions)
        """
        if len(self._models) == 0:
            raise ValueError("No models in ensemble")
        return_all_predictions = bool(kwargs.pop("return_all_predictions", False))

        all_preds = {
            name: np.asarray(model.predict(X), dtype=float) for name, model in self._models.items()
        }

        # Combine predictions
        if self.config.method == "weighted_average":
            predictions = self._weighted_average(all_preds)
        elif self.config.method == "median":
            predictions = self._median_ensemble(all_preds)
        else:
            predictions = self._weighted_average(all_preds)

        # Ensure non-negative
        predictions = np.maximum(predictions, 0)

        result: list[Any] = [predictions]

        if return_std:
            # Estimate uncertainty from model disagreement
            preds_array = np.array(list(all_preds.values()))
            std = np.std(preds_array, axis=0)
            result.append(std)

        if return_all_predictions:
            result.append(all_preds)

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def predict_with_contributions(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Predict and return each model's contribution.

        Args:
            X: Features for prediction

        Returns:
            Tuple of (ensemble_prediction, model_contributions)
        """
        all_preds: dict[str, np.ndarray] = {}
        contributions: dict[str, np.ndarray] = {}

        for name, model in self._models.items():
            pred = np.asarray(model.predict(X), dtype=float)
            all_preds[name] = pred
            contributions[name] = pred * self._weights[name]

        ensemble_pred = self._weighted_average(all_preds)

        return ensemble_pred, contributions

    def get_feature_importance(self, aggregation: str = "mean") -> pd.DataFrame:
        """
        Get aggregated feature importance across models.

        Args:
            aggregation: How to aggregate ('mean', 'max', 'weighted')

        Returns:
            DataFrame with feature importance
        """
        if len(self._models) == 0:
            raise ValueError("No models in ensemble")

        importance_dfs = []

        for name, model in self._models.items():
            try:
                imp = model.get_feature_importance()
                imp["model"] = name
                imp["weight"] = self._weights[name]
                importance_dfs.append(imp)
            except Exception as e:
                logger.warning(f"Could not get importance from model '{name}': {e}")

        if not importance_dfs:
            return pd.DataFrame(columns=["feature", "importance"])

        combined = pd.concat(importance_dfs, ignore_index=True)

        if aggregation == "mean":
            result = combined.groupby("feature")["importance"].mean().reset_index()
        elif aggregation == "max":
            result = combined.groupby("feature")["importance"].max().reset_index()
        elif aggregation == "weighted":
            combined["weighted_importance"] = combined["importance"] * combined["weight"]
            result = combined.groupby("feature")["weighted_importance"].sum().reset_index()
            result.columns = ["feature", "importance"]
        else:
            result = combined.groupby("feature")["importance"].mean().reset_index()

        return result.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save individual models
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)

        model_info = {}
        for name, model in self._models.items():
            model_path = models_dir / name
            model.save(model_path)
            model_info[name] = {"type": model.model_type, "weight": self._weights[name]}

        # Save ensemble config and weights
        (path / "ensemble_config.json").write_text(
            json.dumps(
                {
                    "config": self.config.model_dump(),
                    "models": model_info,
                    "weights": self._weights,
                },
                indent=2,
            )
        )

        # Save metadata
        if self._metadata:
            (path / "metadata.json").write_text(self._metadata.model_dump_json(indent=2))

        logger.info(f"Ensemble saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load ensemble from disk."""
        path = Path(path)

        # Load config
        data = json.loads((path / "ensemble_config.json").read_text())

        self.config = EnsembleConfig.model_validate(data["config"])
        self._weights = data["weights"]

        # Load individual models
        from src.models.lightgbm_model import LightGBMForecaster

        model_classes = {
            "lightgbm": LightGBMForecaster,
        }

        for name, info in data["models"].items():
            model_type = info["type"]
            model_path = path / "models" / name

            if model_type in model_classes:
                model = model_classes[model_type](model_name=name)
                model.load(model_path)
                self._models[name] = model

        # Load metadata
        if (path / "metadata.json").exists():
            self._metadata = ModelMetadata.model_validate_json((path / "metadata.json").read_text())

        self._is_fitted = True
        logger.info(f"Ensemble loaded from {path} with {len(self._models)} models")

    def _weighted_average(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted average of predictions."""
        total_weight = sum(self._weights.values())

        result = np.zeros_like(next(iter(predictions.values())), dtype=float)

        for name, pred in predictions.items():
            weight = self._weights.get(name, 1.0) / total_weight
            result += weight * pred

        return result

    def _median_ensemble(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Compute median of predictions."""
        preds_array = np.array(list(predictions.values()))
        return np.median(preds_array, axis=0)

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def _optimize_weights(
        self, X_valid: pd.DataFrame | np.ndarray, y_valid: pd.Series | np.ndarray
    ) -> None:
        """Optimize ensemble weights using validation data."""
        from scipy.optimize import minimize

        # Get predictions from all models
        all_preds = {
            name: np.asarray(model.predict(X_valid), dtype=float)
            for name, model in self._models.items()
        }
        model_names = list(all_preds.keys())
        preds_array = np.array([all_preds[name] for name in model_names])

        y_true = np.array(y_valid)

        def objective(weights: np.ndarray) -> float:
            """Objective function to minimize."""
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = np.sum(preds_array.T * weights, axis=1)

            if self.config.weight_optimization_metric == "rmse":
                return np.sqrt(np.mean((y_true - ensemble_pred) ** 2))
            elif self.config.weight_optimization_metric == "mae":
                return np.mean(np.abs(y_true - ensemble_pred))
            else:
                return np.sqrt(np.mean((y_true - ensemble_pred) ** 2))

        # Initial weights
        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models

        # Bounds: weights between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Optimize
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            optimized_weights = result.x / result.x.sum()
            self._weights = dict(zip(model_names, map(float, optimized_weights), strict=True))
            self._is_weight_optimized = True
            logger.info(f"Optimized weights: {self._weights}")
        else:
            logger.warning("Weight optimization failed, using equal weights")

    @property
    def model_names(self) -> list[str]:
        """Get names of models in ensemble."""
        return list(self._models.keys())

    @property
    def weights(self) -> dict[str, float]:
        """Get current model weights."""
        return self._weights.copy()

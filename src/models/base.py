"""
Abstract base class for forecasting models.

Defines the interface that all forecasting models must implement,
ensuring consistency and enabling model swapping.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""
    
    model_name: str
    model_type: str
    version: str
    trained_at: datetime
    training_rows: int
    feature_count: int
    features: list[str]
    hyperparameters: dict[str, Any]
    metrics: dict[str, float] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    """Result from model prediction."""
    
    predictions: list[float]
    model_version: str
    inference_time_ms: float
    confidence_intervals: list[tuple[float, float]] | None = None
    feature_contributions: dict[str, list[float]] | None = None


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    Defines the interface that all models must implement:
    - fit: Train the model
    - predict: Generate predictions
    - save/load: Model persistence
    - get_feature_importance: Feature analysis
    
    Subclasses should implement the abstract methods while inheriting
    common functionality like logging and metadata management.
    
    Example:
        >>> class MyForecaster(BaseForecaster):
        ...     def fit(self, X, y, **kwargs):
        ...         # Implementation
        ...         pass
        ...     def predict(self, X, **kwargs):
        ...         return predictions
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        hyperparameters: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize base forecaster.
        
        Args:
            model_name: Name for this model instance
            model_type: Type of model (e.g., 'lightgbm', 'tft')
            hyperparameters: Model hyperparameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        
        self._model: Any = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._metadata: ModelMetadata | None = None
        self._training_history: list[dict] = []
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        **kwargs: Any
    ) -> BaseForecaster:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features (optional)
            y_valid: Validation target (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        return_std: bool = False,
        **kwargs: Any
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Args:
            X: Features for prediction
            return_std: Whether to return uncertainty estimates
            **kwargs: Additional prediction arguments
            
        Returns:
            Predictions (and optionally standard deviations)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        pass
    
    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    @property
    def feature_names(self) -> list[str]:
        """Get feature names used in training."""
        return self._feature_names.copy()
    
    @property
    def metadata(self) -> ModelMetadata | None:
        """Get model metadata."""
        return self._metadata
    
    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"Model '{self.model_name}' is not fitted. "
                "Call fit() before predict()."
            )
    
    def _extract_feature_names(self, X: pd.DataFrame | np.ndarray) -> list[str]:
        """Extract feature names from input data."""
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        elif isinstance(X, np.ndarray):
            return [f"feature_{i}" for i in range(X.shape[1])]
        else:
            return []
    
    def _create_metadata(
        self,
        training_rows: int,
        metrics: dict[str, float] | None = None
    ) -> ModelMetadata:
        """Create model metadata after training."""
        return ModelMetadata(
            model_name=self.model_name,
            model_type=self.model_type,
            version=datetime.now().strftime("v%Y%m%d_%H%M%S"),
            trained_at=datetime.now(),
            training_rows=training_rows,
            feature_count=len(self._feature_names),
            features=self._feature_names,
            hyperparameters=self.hyperparameters,
            metrics=metrics or {}
        )
    
    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters.copy()
        }
    
    def set_params(self, **params: Any) -> BaseForecaster:
        """Set model parameters."""
        for key, value in params.items():
            if key == "hyperparameters":
                self.hyperparameters.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.model_name}', {fitted_str})"

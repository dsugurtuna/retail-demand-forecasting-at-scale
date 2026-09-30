"""
Prediction service for model inference.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PredictionStats(BaseModel):
    """Statistics for prediction service."""

    total_predictions: int = 0
    total_errors: int = 0
    total_inference_time_ms: float = 0.0
    last_prediction_time: datetime | None = None

    @property
    def avg_inference_time_ms(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_inference_time_ms / self.total_predictions

    @property
    def error_rate(self) -> float:
        total = self.total_predictions + self.total_errors
        if total == 0:
            return 0.0
        return self.total_errors / total


class PredictionService:
    """
    Production prediction service.

    Handles model loading, caching, and inference with:
    - Lazy model loading
    - Feature engineering at inference time
    - Confidence intervals
    - Performance monitoring

    Example:
        >>> service = PredictionService(model_path="models/lightgbm_v1.pkl")
        >>> result = service.predict(item_id="FOODS_3_090", store_id="CA_1", date="2024-01-15")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_store_path: str | Path | None = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """
        Initialize prediction service.

        Args:
            model_path: Path to serialized model
            feature_store_path: Path to feature store
            cache_ttl_seconds: Cache TTL for features
        """
        self.model_path = Path(model_path) if model_path else None
        self.feature_store_path = Path(feature_store_path) if feature_store_path else None
        self.cache_ttl_seconds = cache_ttl_seconds

        self._model = None
        self._feature_engineer = None
        self._feature_cache: dict = {}
        self._stats = PredictionStats()
        self._start_time = datetime.utcnow()

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None and self.model_path and self.model_path.exists():
            self._load_model()
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def stats(self) -> PredictionStats:
        """Get prediction statistics."""
        return self._stats

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime."""
        return (datetime.utcnow() - self._start_time).total_seconds()

    def _load_model(self) -> None:
        """Load model from disk."""
        import joblib

        logger.info(f"Loading model from {self.model_path}")
        start_time = time.time()

        self._model = joblib.load(self.model_path)

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")

    def load_model(self, model: Any) -> None:
        """
        Load model directly.

        Args:
            model: Trained model instance
        """
        self._model = model
        logger.info("Model loaded directly into prediction service")

    def predict(
        self,
        item_id: str,
        store_id: str,
        date: str | datetime,
        features: dict | None = None,
        return_interval: bool = True,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Make a single prediction.

        Args:
            item_id: Item identifier
            store_id: Store identifier
            date: Prediction date
            features: Optional pre-computed features
            return_interval: Whether to return prediction interval
            confidence_level: Confidence level for interval

        Returns:
            Dictionary with prediction and metadata
        """
        start_time = time.time()

        try:
            # Build features
            X = self._build_features(item_id, store_id, date, features)

            # Make prediction
            if hasattr(self.model, "predict_with_interval") and return_interval:
                pred, lower, upper = self.model.predict_with_interval(
                    X, confidence_level=confidence_level
                )
            else:
                pred = self.model.predict(X)
                lower, upper = None, None

            # Ensure non-negative
            pred = max(0, float(pred[0]) if hasattr(pred, "__len__") else float(pred))

            inference_time = (time.time() - start_time) * 1000

            # Update stats
            self._stats.total_predictions += 1
            self._stats.total_inference_time_ms += inference_time
            self._stats.last_prediction_time = datetime.utcnow()

            result = {
                "item_id": item_id,
                "store_id": store_id,
                "date": str(date),
                "prediction": pred,
                "inference_time_ms": inference_time,
            }

            if return_interval and lower is not None:
                result["prediction_lower"] = max(
                    0, float(lower[0]) if hasattr(lower, "__len__") else float(lower)
                )
                result["prediction_upper"] = (
                    float(upper[0]) if hasattr(upper, "__len__") else float(upper)
                )

            return result

        except Exception as e:
            self._stats.total_errors += 1
            logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(
        self, items: list[dict], return_intervals: bool = True, confidence_level: float = 0.95
    ) -> list[dict]:
        """
        Make batch predictions.

        Args:
            items: List of dicts with item_id, store_id, date
            return_intervals: Whether to return prediction intervals
            confidence_level: Confidence level for intervals

        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()

        results = []
        for item in items:
            result = self.predict(
                item_id=item["item_id"],
                store_id=item["store_id"],
                date=item["date"],
                features=item.get("features"),
                return_interval=return_intervals,
                confidence_level=confidence_level,
            )
            results.append(result)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch prediction: {len(items)} items in {total_time:.2f}ms")

        return results

    def predict_horizon(
        self,
        item_id: str,
        store_id: str,
        start_date: str | datetime,
        horizon_days: int = 28,
        return_intervals: bool = True,
    ) -> list[dict]:
        """
        Predict for multiple days into the future.

        Args:
            item_id: Item identifier
            store_id: Store identifier
            start_date: First prediction date
            horizon_days: Number of days to predict
            return_intervals: Whether to return intervals

        Returns:
            List of predictions for each day
        """
        from datetime import timedelta

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)

        items = [
            {
                "item_id": item_id,
                "store_id": store_id,
                "date": (start_date + timedelta(days=d)).strftime("%Y-%m-%d"),
            }
            for d in range(horizon_days)
        ]

        return self.predict_batch(items, return_intervals=return_intervals)

    def _build_features(
        self,
        item_id: str,
        store_id: str,
        date: str | datetime,
        provided_features: dict | None = None,
    ) -> pd.DataFrame:
        """Build feature vector for prediction."""
        # Start with basic features
        features = {
            "item_id": item_id,
            "store_id": store_id,
        }

        # Parse date
        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        # Add temporal features
        features.update(
            {
                "day_of_week": date.weekday(),
                "day_of_month": date.day,
                "week_of_year": date.isocalendar()[1],
                "month": date.month,
                "quarter": (date.month - 1) // 3 + 1,
                "year": date.year,
                "is_weekend": int(date.weekday() >= 5),
                "day_of_year": date.timetuple().tm_yday,
            }
        )

        # Add cyclical encoding
        features["day_sin"] = np.sin(2 * np.pi * date.weekday() / 7)
        features["day_cos"] = np.cos(2 * np.pi * date.weekday() / 7)
        features["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * date.month / 12)

        # Override with provided features
        if provided_features:
            features.update(provided_features)

        # Fetch from feature store if available
        if self._feature_engineer is not None:
            cache_key = f"{item_id}_{store_id}_{date}"
            if cache_key not in self._feature_cache:
                store_features = self._fetch_store_features(item_id, store_id, date)
                self._feature_cache[cache_key] = store_features
            features.update(self._feature_cache[cache_key])

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Align with model features
        if hasattr(self.model, "feature_names_"):
            missing = set(self.model.feature_names_) - set(df.columns)
            for col in missing:
                df[col] = 0
            df = df[self.model.feature_names_]

        return df

    def _fetch_store_features(self, item_id: str, store_id: str, date: datetime) -> dict:
        """Fetch features from feature store."""
        # Placeholder - would integrate with actual feature store
        return {}

    def get_model_info(self) -> dict:
        """Get model information."""
        info = {
            "loaded": self.is_loaded,
            "path": str(self.model_path) if self.model_path else None,
        }

        if self.is_loaded and hasattr(self.model, "metadata") and self.model.metadata:
            info.update(
                {
                    "name": self.model.metadata.name,
                    "version": self.model.metadata.version,
                    "features": self.model.metadata.feature_names,
                    "training_date": str(self.model.metadata.training_date),
                }
            )

        return info

    def health_check(self) -> dict:
        """Perform health check."""
        return {
            "status": "healthy" if self.is_loaded else "degraded",
            "model_loaded": self.is_loaded,
            "model_version": getattr(getattr(self.model, "metadata", None), "version", None)
            if self.is_loaded
            else None,
            "uptime_seconds": self.uptime_seconds,
            "total_predictions": self._stats.total_predictions,
            "error_rate": self._stats.error_rate,
            "avg_inference_time_ms": self._stats.avg_inference_time_ms,
        }

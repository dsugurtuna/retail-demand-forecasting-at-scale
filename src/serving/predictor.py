"""
Prediction service used by the API.

Loads a model written by ``python -m src.train`` (a LightGBM model directory)
or, for other models, a joblib file, and builds a feature row per request.

Request-time features are limited to what can be derived from the request:
calendar features for the date (computed by the same function used in
training), the item and store IDs, and optional price, SNAP and event fields.
All history-based features are passed as missing. See ``src.serving.api``.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

from src.features.temporal import add_calendar_features

logger = logging.getLogger(__name__)


class PredictionStats(BaseModel):
    """Running counters for the service."""

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
        return self.total_errors / total if total else 0.0


class PredictionService:
    """Load a model lazily and make predictions for single rows.

    Example:
        >>> service = PredictionService(model_path="artefacts/smoke/model")
        >>> service.predict(item_id="FOODS_1_001", store_id="CA_1", date="2016-05-01")
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._model: Any = None
        self._stats = PredictionStats()
        self._start_time = datetime.now(UTC)

    @property
    def model(self) -> Any:
        """The model, loaded from ``model_path`` on first access."""
        if self._model is None and self.model_path is not None and self.model_path.exists():
            self._load_model()
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def stats(self) -> PredictionStats:
        return self._stats

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(UTC) - self._start_time).total_seconds()

    def _load_model(self) -> None:
        assert self.model_path is not None
        start = time.perf_counter()
        if self.model_path.is_dir() and (self.model_path / "model.txt").exists():
            from src.models.lightgbm_model import LightGBMForecaster

            model = LightGBMForecaster()
            model.load(self.model_path)
            self._model = model
        else:
            import joblib

            # Only load joblib files you created: unpickling runs code.
            self._model = joblib.load(self.model_path)
        logger.info("Model loaded from %s in %.2fs", self.model_path, time.perf_counter() - start)

    def load_model(self, model: Any) -> None:
        """Use an in-memory model (for tests and notebooks)."""
        self._model = model

    def predict(
        self,
        item_id: str,
        store_id: str,
        date: str | datetime,
        features: dict[str, Any] | None = None,
        return_interval: bool = True,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Predict demand for one item in one store on one date."""
        start = time.perf_counter()
        try:
            X = self._build_features(item_id, store_id, date, features)
            lower = upper = None
            if return_interval and hasattr(self.model, "predict_interval"):
                pred, lower, upper = self.model.predict_interval(X, confidence=confidence_level)
            else:
                pred = self.model.predict(X)
            value = max(0.0, float(np.ravel(pred)[0]))
        except Exception:
            self._stats.total_errors += 1
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._stats.total_predictions += 1
        self._stats.total_inference_time_ms += elapsed_ms
        self._stats.last_prediction_time = datetime.now(UTC)

        result: dict[str, Any] = {
            "item_id": item_id,
            "store_id": store_id,
            "date": str(date),
            "prediction": value,
            "inference_time_ms": elapsed_ms,
        }
        if lower is not None and upper is not None:
            result["prediction_lower"] = max(0.0, float(np.ravel(lower)[0]))
            result["prediction_upper"] = float(np.ravel(upper)[0])
        return result

    def predict_batch(
        self,
        items: list[dict[str, Any]],
        return_intervals: bool = True,
        confidence_level: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Predict each ``{"item_id", "store_id", "date"}`` in turn."""
        return [
            self.predict(
                item_id=item["item_id"],
                store_id=item["store_id"],
                date=item["date"],
                features=item.get("features"),
                return_interval=return_intervals,
                confidence_level=confidence_level,
            )
            for item in items
        ]

    def predict_horizon(
        self,
        item_id: str,
        store_id: str,
        start_date: str | datetime,
        horizon_days: int = 28,
        return_intervals: bool = True,
    ) -> list[dict[str, Any]]:
        """Predict one item-store for ``horizon_days`` consecutive days."""
        first = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
        items = [
            {
                "item_id": item_id,
                "store_id": store_id,
                "date": (first + timedelta(days=d)).isoformat(),
            }
            for d in range(horizon_days)
        ]
        return self.predict_batch(items, return_intervals=return_intervals)

    def _build_features(
        self,
        item_id: str,
        store_id: str,
        when: str | date | datetime,
        provided: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """One feature row. Anything the request cannot supply is left missing."""
        day = date.fromisoformat(when) if isinstance(when, str) else when
        row = add_calendar_features(pl.DataFrame({"date": [day]}, schema={"date": pl.Date}))
        values: dict[str, Any] = {k: v[0] for k, v in row.drop("date").to_dict().items()}
        values["item_id"] = item_id
        values["store_id"] = store_id
        values.update(provided or {})

        names: list[str] = list(getattr(self.model, "feature_names", []) or [])
        frame = pd.DataFrame([{name: values.get(name, np.nan) for name in names}])
        if frame.empty or not names:
            frame = pd.DataFrame([values])
        categorical = set(getattr(self.model, "categorical_features", []) or [])
        for col in frame.columns:
            if col in categorical or frame[col].dtype == object:
                frame[col] = frame[col].astype("category")
        return frame

    def get_model_info(self) -> dict[str, Any]:
        """Model metadata for ``/model/info`` and response headers."""
        info: dict[str, Any] = {
            "loaded": self.is_loaded,
            "path": str(self.model_path) if self.model_path else None,
        }
        metadata = getattr(self.model, "metadata", None) if self.is_loaded else None
        if metadata is not None:
            info.update(
                {
                    "name": metadata.model_name,
                    "version": metadata.version,
                    "framework": metadata.model_type,
                    "features": metadata.features,
                    "trained_at": metadata.trained_at,
                    "hyperparameters": metadata.hyperparameters,
                    "metrics": {
                        k: float(v)
                        for k, v in metadata.metrics.items()
                        if isinstance(v, int | float)
                    },
                }
            )
        return info

    def health_check(self) -> dict[str, Any]:
        info = self.get_model_info()
        return {
            "status": "healthy" if self.is_loaded else "degraded",
            "model_loaded": self.is_loaded,
            "model_version": info.get("version"),
            "uptime_seconds": self.uptime_seconds,
            "last_prediction_time": self._stats.last_prediction_time,
            "total_predictions": self._stats.total_predictions,
            "error_rate": self._stats.error_rate,
        }

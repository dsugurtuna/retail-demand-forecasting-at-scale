"""
Pydantic schemas for API request/response models.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionItem(BaseModel):
    """Single item prediction."""
    
    item_id: str
    store_id: str
    date: date
    prediction: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    
    @field_validator("prediction", "lower_bound", "upper_bound", mode="before")
    @classmethod
    def round_predictions(cls, v):
        if v is not None:
            return round(float(v), 4)
        return v


class ForecastRequest(BaseModel):
    """Request for a single forecast."""
    
    item_id: str = Field(..., description="Product/item identifier")
    store_id: str = Field(..., description="Store identifier")
    forecast_date: date = Field(..., description="Date to forecast")
    
    # Optional features
    price: float | None = Field(None, ge=0, description="Item price")
    snap_enabled: bool | None = Field(None, description="SNAP eligibility")
    event_name: str | None = Field(None, description="Event/holiday name")
    
    model_config = {"json_schema_extra": {"example": {
        "item_id": "FOODS_3_090",
        "store_id": "CA_1",
        "forecast_date": "2024-01-15",
        "price": 2.49,
        "snap_enabled": True,
    }}}


class ForecastResponse(BaseModel):
    """Response for a single forecast."""
    
    item_id: str
    store_id: str
    forecast_date: date
    prediction: float
    prediction_lower: float | None = None
    prediction_upper: float | None = None
    model_version: str
    inference_time_ms: float
    
    model_config = {"json_schema_extra": {"example": {
        "item_id": "FOODS_3_090",
        "store_id": "CA_1",
        "forecast_date": "2024-01-15",
        "prediction": 12.5,
        "prediction_lower": 8.2,
        "prediction_upper": 16.8,
        "model_version": "v2.1.0",
        "inference_time_ms": 15.3,
    }}}


class BatchForecastRequest(BaseModel):
    """Request for batch forecasts."""
    
    items: list[ForecastRequest] = Field(..., max_length=1000)
    horizon_days: int = Field(default=28, ge=1, le=90)
    return_intervals: bool = Field(default=True)
    
    @field_validator("items")
    @classmethod
    def validate_items(cls, v):
        if not v:
            raise ValueError("At least one item is required")
        return v


class BatchForecastResponse(BaseModel):
    """Response for batch forecasts."""
    
    predictions: list[PredictionItem]
    n_items: int
    total_predictions: int
    model_version: str
    inference_time_ms: float
    
    @property
    def predictions_df(self):
        """Convert predictions to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([p.model_dump() for p in self.predictions])


class HorizonForecastRequest(BaseModel):
    """Request for multi-horizon forecast."""
    
    item_id: str
    store_id: str
    start_date: date
    horizon_days: int = Field(default=28, ge=1, le=90)
    return_intervals: bool = Field(default=True)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)


class HorizonForecastResponse(BaseModel):
    """Response for multi-horizon forecast."""
    
    item_id: str
    store_id: str
    forecasts: list[PredictionItem]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float
    last_prediction_time: datetime | None = None
    total_predictions: int = 0
    error_rate: float = 0.0


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    name: str
    version: str
    framework: str
    trained_at: datetime | None
    training_dataset: str | None
    features: list[str]
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = None

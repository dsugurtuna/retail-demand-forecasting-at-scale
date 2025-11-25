"""
FastAPI application for demand forecasting service.

Production-ready API with:
- Health checks and readiness probes
- Structured logging
- Request validation
- Error handling
- Metrics export
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.predictor import PredictionService
from src.serving.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HorizonForecastRequest,
    HorizonForecastResponse,
    ModelInfoResponse,
    PredictionItem,
)

logger = logging.getLogger(__name__)

# Global prediction service
prediction_service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global prediction_service
    
    # Startup
    logger.info("Starting demand forecasting service...")
    
    model_path = Path(app.state.config.get("model_path", "models/model.pkl"))
    prediction_service = PredictionService(model_path=model_path)
    
    if model_path.exists():
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down demand forecasting service...")


def create_app(config: dict | None = None) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        config: Application configuration
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Retail Demand Forecasting API",
        description="Production-grade API for retail demand forecasting",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    app.state.config = config or {}
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc)
            ).model_dump(),
        )
    
    return app


# Create default app
app = create_app()


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "service": "Retail Demand Forecasting API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers."""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    health = prediction_service.health_check()
    return HealthResponse(**health)


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe for Kubernetes."""
    if prediction_service is None or not prediction_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "ready"}


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information."""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    info = prediction_service.get_model_info()
    
    if not info.get("loaded"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        name=info.get("name", "Unknown"),
        version=info.get("version", "0.0.0"),
        framework=info.get("framework", "lightgbm"),
        trained_at=info.get("training_date"),
        training_dataset=info.get("training_dataset"),
        features=info.get("features", []),
        hyperparameters=info.get("hyperparameters", {}),
        metrics=info.get("metrics", {}),
    )


@app.post("/predict", response_model=ForecastResponse, tags=["Predictions"])
async def predict(request: ForecastRequest):
    """
    Make a single prediction.
    
    Predicts demand for a specific item at a specific store on a specific date.
    """
    if prediction_service is None or not prediction_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Build features from request
        features = {}
        if request.price is not None:
            features["sell_price"] = request.price
        if request.snap_enabled is not None:
            features["snap_CA"] = int(request.snap_enabled)
        if request.event_name is not None:
            features["event_name_1"] = request.event_name
        
        result = prediction_service.predict(
            item_id=request.item_id,
            store_id=request.store_id,
            date=request.forecast_date.isoformat(),
            features=features if features else None,
        )
        
        return ForecastResponse(
            item_id=result["item_id"],
            store_id=result["store_id"],
            forecast_date=request.forecast_date,
            prediction=result["prediction"],
            prediction_lower=result.get("prediction_lower"),
            prediction_upper=result.get("prediction_upper"),
            model_version=prediction_service.get_model_info().get("version", "1.0.0"),
            inference_time_ms=result["inference_time_ms"],
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post("/predict/batch", response_model=BatchForecastResponse, tags=["Predictions"])
async def predict_batch(request: BatchForecastRequest):
    """
    Make batch predictions.
    
    Efficiently process multiple predictions in a single request.
    Limited to 1000 items per request.
    """
    if prediction_service is None or not prediction_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        items = [
            {
                "item_id": item.item_id,
                "store_id": item.store_id,
                "date": item.forecast_date.isoformat(),
            }
            for item in request.items
        ]
        
        results = prediction_service.predict_batch(
            items=items,
            return_intervals=request.return_intervals,
        )
        
        predictions = [
            PredictionItem(
                item_id=r["item_id"],
                store_id=r["store_id"],
                date=r["date"],
                prediction=r["prediction"],
                lower_bound=r.get("prediction_lower"),
                upper_bound=r.get("prediction_upper"),
            )
            for r in results
        ]
        
        inference_time = (time.time() - start_time) * 1000
        
        return BatchForecastResponse(
            predictions=predictions,
            n_items=len(request.items),
            total_predictions=len(predictions),
            model_version=prediction_service.get_model_info().get("version", "1.0.0"),
            inference_time_ms=inference_time,
        )
        
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


@app.post("/predict/horizon", response_model=HorizonForecastResponse, tags=["Predictions"])
async def predict_horizon(request: HorizonForecastRequest):
    """
    Predict for multiple days into the future.
    
    Generate forecasts for a specified horizon (1-90 days).
    """
    if prediction_service is None or not prediction_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        results = prediction_service.predict_horizon(
            item_id=request.item_id,
            store_id=request.store_id,
            start_date=request.start_date.isoformat(),
            horizon_days=request.horizon_days,
            return_intervals=request.return_intervals,
        )
        
        forecasts = [
            PredictionItem(
                item_id=r["item_id"],
                store_id=r["store_id"],
                date=r["date"],
                prediction=r["prediction"],
                lower_bound=r.get("prediction_lower"),
                upper_bound=r.get("prediction_upper"),
            )
            for r in results
        ]
        
        inference_time = (time.time() - start_time) * 1000
        
        return HorizonForecastResponse(
            item_id=request.item_id,
            store_id=request.store_id,
            forecasts=forecasts,
            model_version=prediction_service.get_model_info().get("version", "1.0.0"),
            inference_time_ms=inference_time,
        )
        
    except Exception as e:
        logger.exception(f"Horizon prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Horizon prediction failed"
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get Prometheus-compatible metrics.
    
    Exposes prediction statistics for monitoring.
    """
    if prediction_service is None:
        return ""
    
    stats = prediction_service.stats
    
    # Prometheus format
    metrics = []
    metrics.append(f"forecasting_predictions_total {stats.total_predictions}")
    metrics.append(f"forecasting_errors_total {stats.total_errors}")
    metrics.append(f"forecasting_inference_time_ms_avg {stats.avg_inference_time_ms:.2f}")
    metrics.append(f"forecasting_error_rate {stats.error_rate:.4f}")
    metrics.append(f"forecasting_uptime_seconds {prediction_service.uptime_seconds:.0f}")
    
    return "\n".join(metrics)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

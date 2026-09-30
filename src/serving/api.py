"""
FastAPI application for the forecasting model.

Routes are registered inside ``create_app`` so every app the factory builds
has them (earlier they were attached only to the module-level ``app``, so
``create_app()`` returned an app that answered 404 to everything).

Limitation: request-time features are the calendar fields for the requested
date plus an optional price. The lag, rolling and hierarchical features the
model was trained on need recent sales history, which this service does not
look up yet, so they are passed as missing. Treat the API as a working
skeleton for wiring and contracts, not as a source of usable forecasts.

Run locally:
    FORECAST_MODEL_PATH=artefacts/smoke/model uvicorn src.serving.api:app
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from src import __version__
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

router = APIRouter()


def _service(request: Request) -> PredictionService:
    service: PredictionService | None = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service not initialised")
    return service


def _loaded_service(request: Request) -> PredictionService:
    service = _service(request)
    if not service.is_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return service


def _items(results: list[dict[str, Any]]) -> list[PredictionItem]:
    return [
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


@router.get("/", tags=["Info"])
async def root() -> dict[str, str]:
    return {"service": "Retail Demand Forecasting API", "version": __version__, "docs": "/docs"}


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request) -> HealthResponse:
    """Service status. Reports 'degraded' when no model is loaded."""
    return HealthResponse(**_service(request).health_check())


@router.get("/ready", tags=["Health"])
async def readiness_check(request: Request) -> dict[str, str]:
    """Readiness probe: 200 only when a model is loaded."""
    _loaded_service(request)
    return {"status": "ready"}


@router.get("/live", tags=["Health"])
async def liveness_check() -> dict[str, str]:
    """Liveness probe: 200 whenever the process is serving requests."""
    return {"status": "alive"}


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(request: Request) -> ModelInfoResponse:
    info = _loaded_service(request).get_model_info()
    return ModelInfoResponse(
        name=info.get("name", "unknown"),
        version=info.get("version", "unknown"),
        framework=info.get("framework", "lightgbm"),
        trained_at=info.get("trained_at"),
        training_dataset=info.get("training_dataset"),
        features=info.get("features", []),
        hyperparameters=info.get("hyperparameters", {}),
        metrics=info.get("metrics", {}),
    )


@router.post("/predict", response_model=ForecastResponse, tags=["Predictions"])
async def predict(request: Request, body: ForecastRequest) -> ForecastResponse:
    """Forecast one item in one store on one date."""
    service = _loaded_service(request)
    features: dict[str, Any] = {}
    if body.price is not None:
        features["sell_price"] = body.price
    if body.snap_enabled is not None:
        features["snap"] = int(body.snap_enabled)
    if body.event_name is not None:
        features["event_name_1"] = body.event_name

    try:
        result = service.predict(
            item_id=body.item_id,
            store_id=body.store_id,
            date=body.forecast_date.isoformat(),
            features=features or None,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return ForecastResponse(
        item_id=result["item_id"],
        store_id=result["store_id"],
        forecast_date=body.forecast_date,
        prediction=result["prediction"],
        prediction_lower=result.get("prediction_lower"),
        prediction_upper=result.get("prediction_upper"),
        model_version=service.get_model_info().get("version", "unknown"),
        inference_time_ms=result["inference_time_ms"],
    )


@router.post("/predict/batch", response_model=BatchForecastResponse, tags=["Predictions"])
async def predict_batch(request: Request, body: BatchForecastRequest) -> BatchForecastResponse:
    """Forecast up to 1,000 (item, store, date) rows in one call."""
    service = _loaded_service(request)
    start = time.perf_counter()
    results = service.predict_batch(
        items=[
            {
                "item_id": item.item_id,
                "store_id": item.store_id,
                "date": item.forecast_date.isoformat(),
            }
            for item in body.items
        ],
        return_intervals=body.return_intervals,
    )
    return BatchForecastResponse(
        predictions=_items(results),
        n_items=len(body.items),
        total_predictions=len(results),
        model_version=service.get_model_info().get("version", "unknown"),
        inference_time_ms=(time.perf_counter() - start) * 1000,
    )


@router.post("/predict/horizon", response_model=HorizonForecastResponse, tags=["Predictions"])
async def predict_horizon(
    request: Request, body: HorizonForecastRequest
) -> HorizonForecastResponse:
    """Forecast one item in one store for each of the next ``horizon_days`` days."""
    service = _loaded_service(request)
    start = time.perf_counter()
    results = service.predict_horizon(
        item_id=body.item_id,
        store_id=body.store_id,
        start_date=body.start_date.isoformat(),
        horizon_days=body.horizon_days,
        return_intervals=body.return_intervals,
    )
    return HorizonForecastResponse(
        item_id=body.item_id,
        store_id=body.store_id,
        forecasts=_items(results),
        model_version=service.get_model_info().get("version", "unknown"),
        inference_time_ms=(time.perf_counter() - start) * 1000,
    )


@router.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_metrics(request: Request) -> str:
    """Request counters in Prometheus text format."""
    service: PredictionService | None = getattr(request.app.state, "prediction_service", None)
    if service is None:
        return ""
    stats = service.stats
    return "\n".join(
        [
            f"forecasting_predictions_total {stats.total_predictions}",
            f"forecasting_errors_total {stats.total_errors}",
            f"forecasting_inference_time_ms_avg {stats.avg_inference_time_ms:.2f}",
            f"forecasting_error_rate {stats.error_rate:.4f}",
            f"forecasting_uptime_seconds {service.uptime_seconds:.0f}",
        ]
    )


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Build the API.

    Args:
        config: Optional settings. ``model_path`` points at a model directory
            written by ``python -m src.train`` (or a joblib file). ``None`` or
            a missing path starts the service without a model: health reports
            'degraded' and prediction endpoints return 503.
    """
    config = dict(config or {})
    app = FastAPI(
        title="Retail Demand Forecasting API",
        description="Serves a LightGBM demand model. See README for limitations.",
        version=__version__,
    )
    model_path = config.get("model_path")
    app.state.config = config
    app.state.prediction_service = PredictionService(
        model_path=Path(model_path) if model_path else None
    )
    if model_path and not Path(model_path).exists():
        logger.warning("Model path %s does not exist; starting without a model", model_path)

    @app.middleware("http")
    async def add_timing_header(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.6f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # Log the detail; do not echo internal error text to callers.
        logger.exception("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="Internal server error").model_dump(mode="json"),
        )

    app.include_router(router)
    return app


app = create_app({"model_path": os.environ.get("FORECAST_MODEL_PATH")})

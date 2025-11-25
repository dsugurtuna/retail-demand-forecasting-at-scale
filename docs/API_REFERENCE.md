# =============================================================================
# API Reference Documentation
# =============================================================================

## Overview

The Retail Demand Forecasting API provides RESTful endpoints for generating
demand forecasts. This document covers all available endpoints, request/response
formats, and usage examples.

## Base URL

```
Production: https://forecast.example.com/api/v1
Staging:    https://staging-forecast.example.com/api/v1
Local:      http://localhost:8000
```

## Authentication

API requests require authentication via API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://forecast.example.com/api/v1/predict
```

## Endpoints

### Health & Status

#### GET /health

Health check for load balancers and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2.1.0",
  "uptime_seconds": 86400.5,
  "total_predictions": 125000,
  "error_rate": 0.001
}
```

#### GET /ready

Kubernetes readiness probe.

**Response:**
```json
{
  "status": "ready"
}
```

#### GET /live

Kubernetes liveness probe.

**Response:**
```json
{
  "status": "alive"
}
```

---

### Predictions

#### POST /predict

Generate a single demand forecast.

**Request Body:**
```json
{
  "item_id": "FOODS_3_090",
  "store_id": "CA_1",
  "forecast_date": "2024-01-15",
  "price": 2.49,
  "snap_enabled": true,
  "event_name": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| item_id | string | Yes | Product identifier |
| store_id | string | Yes | Store identifier |
| forecast_date | date | Yes | Date to forecast (YYYY-MM-DD) |
| price | float | No | Item price |
| snap_enabled | boolean | No | SNAP eligibility flag |
| event_name | string | No | Event or holiday name |

**Response:**
```json
{
  "item_id": "FOODS_3_090",
  "store_id": "CA_1",
  "forecast_date": "2024-01-15",
  "prediction": 12.5,
  "prediction_lower": 8.2,
  "prediction_upper": 16.8,
  "model_version": "2.1.0",
  "inference_time_ms": 15.3
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "item_id": "FOODS_3_090",
       "store_id": "CA_1",
       "forecast_date": "2024-01-15"
     }'
```

---

#### POST /predict/batch

Generate forecasts for multiple items.

**Request Body:**
```json
{
  "items": [
    {
      "item_id": "FOODS_3_090",
      "store_id": "CA_1",
      "forecast_date": "2024-01-15"
    },
    {
      "item_id": "FOODS_3_091",
      "store_id": "CA_1",
      "forecast_date": "2024-01-15"
    }
  ],
  "horizon_days": 28,
  "return_intervals": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| items | array | Yes | Array of forecast requests (max 1000) |
| horizon_days | int | No | Number of days to forecast (default: 28) |
| return_intervals | boolean | No | Include prediction intervals (default: true) |

**Response:**
```json
{
  "predictions": [
    {
      "item_id": "FOODS_3_090",
      "store_id": "CA_1",
      "date": "2024-01-15",
      "prediction": 12.5,
      "lower_bound": 8.2,
      "upper_bound": 16.8
    }
  ],
  "n_items": 2,
  "total_predictions": 2,
  "model_version": "2.1.0",
  "inference_time_ms": 45.2
}
```

---

#### POST /predict/horizon

Generate multi-day forecasts for a single item.

**Request Body:**
```json
{
  "item_id": "FOODS_3_090",
  "store_id": "CA_1",
  "start_date": "2024-01-15",
  "horizon_days": 28,
  "return_intervals": true,
  "confidence_level": 0.95
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| item_id | string | Yes | Product identifier |
| store_id | string | Yes | Store identifier |
| start_date | date | Yes | First forecast date |
| horizon_days | int | No | Days to forecast (default: 28, max: 90) |
| return_intervals | boolean | No | Include intervals (default: true) |
| confidence_level | float | No | Interval confidence (default: 0.95) |

**Response:**
```json
{
  "item_id": "FOODS_3_090",
  "store_id": "CA_1",
  "forecasts": [
    {
      "item_id": "FOODS_3_090",
      "store_id": "CA_1",
      "date": "2024-01-15",
      "prediction": 12.5,
      "lower_bound": 8.2,
      "upper_bound": 16.8
    },
    {
      "item_id": "FOODS_3_090",
      "store_id": "CA_1",
      "date": "2024-01-16",
      "prediction": 11.8,
      "lower_bound": 7.5,
      "upper_bound": 16.1
    }
  ],
  "model_version": "2.1.0",
  "inference_time_ms": 125.7
}
```

---

### Model Information

#### GET /model/info

Get detailed model information.

**Response:**
```json
{
  "name": "lightgbm_forecaster",
  "version": "2.1.0",
  "framework": "lightgbm",
  "trained_at": "2024-01-10T14:30:00Z",
  "training_dataset": "m5_sales_v2",
  "features": [
    "sales_lag_7",
    "sales_lag_14",
    "sales_rolling_mean_7",
    "day_of_week",
    "month",
    "sell_price"
  ],
  "hyperparameters": {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63
  },
  "metrics": {
    "cv_rmse": 2.45,
    "cv_smape": 15.2,
    "cv_wrmsse": 0.52
  }
}
```

---

### Monitoring

#### GET /metrics

Prometheus-compatible metrics endpoint.

**Response (text/plain):**
```prometheus
# HELP forecasting_predictions_total Total predictions made
# TYPE forecasting_predictions_total counter
forecasting_predictions_total 125000

# HELP forecasting_errors_total Total prediction errors
# TYPE forecasting_errors_total counter
forecasting_errors_total 125

# HELP forecasting_inference_time_ms_avg Average inference time
# TYPE forecasting_inference_time_ms_avg gauge
forecasting_inference_time_ms_avg 15.30

# HELP forecasting_error_rate Current error rate
# TYPE forecasting_error_rate gauge
forecasting_error_rate 0.0010

# HELP forecasting_uptime_seconds Service uptime
# TYPE forecasting_uptime_seconds counter
forecasting_uptime_seconds 86400
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc-123-def"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 404 | Not Found - Resource not found |
| 422 | Validation Error - Invalid request body |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

### Common Errors

**Invalid date format:**
```json
{
  "error": "Validation Error",
  "detail": "forecast_date must be in YYYY-MM-DD format",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Unknown item:**
```json
{
  "error": "Not Found",
  "detail": "Item 'INVALID_ITEM' not found in training data",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Rate Limiting

| Tier | Requests/min | Batch size |
|------|--------------|------------|
| Free | 60 | 100 |
| Standard | 300 | 500 |
| Enterprise | Unlimited | 1000 |

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705315800
```

---

## Python SDK Example

```python
import requests
from datetime import date

class ForecastClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def predict(
        self,
        item_id: str,
        store_id: str,
        forecast_date: date
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/predict",
            headers=self.headers,
            json={
                "item_id": item_id,
                "store_id": store_id,
                "forecast_date": forecast_date.isoformat(),
            }
        )
        response.raise_for_status()
        return response.json()
    
    def predict_horizon(
        self,
        item_id: str,
        store_id: str,
        start_date: date,
        horizon_days: int = 28
    ) -> list[dict]:
        response = requests.post(
            f"{self.base_url}/predict/horizon",
            headers=self.headers,
            json={
                "item_id": item_id,
                "store_id": store_id,
                "start_date": start_date.isoformat(),
                "horizon_days": horizon_days,
            }
        )
        response.raise_for_status()
        return response.json()["forecasts"]

# Usage
client = ForecastClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

forecast = client.predict(
    item_id="FOODS_3_090",
    store_id="CA_1",
    forecast_date=date(2024, 1, 15)
)
print(f"Predicted demand: {forecast['prediction']}")
```

---

## Changelog

### v2.1.0 (2024-01-15)
- Added confidence intervals to all prediction endpoints
- Improved batch prediction performance (2x faster)
- Added `/predict/horizon` endpoint

### v2.0.0 (2024-01-01)
- Complete API redesign with FastAPI
- Added Prometheus metrics endpoint
- New authentication system

### v1.0.0 (2023-06-01)
- Initial release

# API reference

The FastAPI app in `src/serving/api.py` serves a model written by `python -m src.train`.

> **Read this first.** The API computes calendar features for the requested date and accepts an optional price, SNAP flag and event name. It does **not** look up recent sales history, so every lag, rolling and hierarchical feature reaches the model as missing. Forecasts from the API are therefore driven mostly by the item, store and calendar, and are not the forecasts the training run scores. Treat the API as a working skeleton for contracts and wiring. There is no authentication, rate limiting or TLS.

## Run it

```bash
python -m src.train --smoke                                   # writes artefacts/smoke/model
FORECAST_MODEL_PATH=artefacts/smoke/model uvicorn src.serving.api:app --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs` (Swagger UI) and `/redoc`.

Without `FORECAST_MODEL_PATH` (or with a path that does not exist) the service starts without a model: `/health` reports `degraded`, `/ready` and the prediction endpoints return `503`.

## Endpoints

| Method | Path | Purpose | Without a model |
|---|---|---|---|
| GET | `/` | Service name and version | 200 |
| GET | `/live` | Liveness: the process is serving | 200 |
| GET | `/ready` | Readiness: a model is loaded | 503 |
| GET | `/health` | Status, model version, counters | 200, `"status": "degraded"` |
| GET | `/model/info` | Model name, version, features, hyperparameters, training metrics | 503 |
| POST | `/predict` | One item, one store, one date | 503 |
| POST | `/predict/batch` | Up to 1,000 (item, store, date) rows | 503 |
| POST | `/predict/horizon` | One item-store for 1 to 90 consecutive days | 503 |
| GET | `/metrics` | Request counters in Prometheus text format | 200 |

Request bodies are validated by the pydantic models in `src/serving/schemas.py`; invalid bodies return `422`.

### POST /predict

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"item_id": "FOODS_1_001", "store_id": "CA_1", "forecast_date": "2013-01-05", "price": 2.5}'
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `item_id` | string | yes | Must be a level seen in training to be useful |
| `store_id` | string | yes | |
| `forecast_date` | date | yes | `YYYY-MM-DD` |
| `price` | float >= 0 | no | Passed as `sell_price` |
| `snap_enabled` | bool | no | Passed as `snap` |
| `event_name` | string | no | Passed as `event_name_1` |

Response fields: `item_id`, `store_id`, `forecast_date`, `prediction`, `prediction_lower`, `prediction_upper`, `model_version`, `inference_time_ms`.

`prediction_lower` and `prediction_upper` come from `LightGBMForecaster.predict_interval`, which is a fixed ±20% band scaled by a normal quantile. It is a placeholder, not a calibrated interval.

### POST /predict/batch

```json
{"items": [{"item_id": "FOODS_1_001", "store_id": "CA_1", "forecast_date": "2013-01-05"}],
 "return_intervals": true}
```

At most 1,000 items per request. Rows are predicted one by one.

### POST /predict/horizon

```json
{"item_id": "FOODS_1_001", "store_id": "CA_1", "start_date": "2013-01-05", "horizon_days": 28}
```

### GET /metrics

```
forecasting_predictions_total 4
forecasting_errors_total 0
forecasting_inference_time_ms_avg 3.10
forecasting_error_rate 0.0000
forecasting_uptime_seconds 42
```

Counters are per process and reset on restart. (Values above are an illustrative example.)

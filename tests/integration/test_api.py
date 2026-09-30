"""Integration tests for the API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.lightgbm_model import LightGBMConfig, LightGBMForecaster
from src.serving.api import create_app
from src.serving.predictor import PredictionService


@pytest.fixture
def client():
    return TestClient(create_app({"model_path": None}))


@pytest.fixture
def client_with_mock_service(mocker):
    app = create_app({"model_path": None})
    service = mocker.MagicMock(spec=PredictionService)
    service.is_loaded = True
    service.predict.return_value = {
        "item_id": "ITEM_1",
        "store_id": "STORE_1",
        "date": "2024-01-15",
        "prediction": 12.5,
        "prediction_lower": 8.0,
        "prediction_upper": 17.0,
        "inference_time_ms": 15.3,
    }
    service.get_model_info.return_value = {"version": "1.0.0"}
    app.state.prediction_service = service
    return TestClient(app), service


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    """A tiny real model trained on calendar-style features and saved to disk."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "item_id": pd.Categorical(rng.choice(["FOODS_1_001", "FOODS_1_002"], n)),
            "store_id": pd.Categorical(rng.choice(["CA_1", "TX_1"], n)),
            "day_of_week": rng.integers(0, 7, n),
            "sell_price": rng.uniform(1, 5, n),
            "sales_lag_28": rng.poisson(3, n).astype(float),
        }
    )
    y = rng.poisson(3, n).astype(float)
    model = LightGBMForecaster(config=LightGBMConfig(n_estimators=20, min_data_in_leaf=10))
    model.fit(X, y)
    path = tmp_path_factory.mktemp("model") / "model"
    model.save(path)
    return path


class TestEndpointsWithoutModel:
    def test_root(self, client):
        data = client.get("/").json()
        assert data["service"] == "Retail Demand Forecasting API"
        assert "version" in data

    def test_liveness(self, client):
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_health_reports_degraded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["model_loaded"] is False

    def test_readiness_is_503(self, client):
        assert client.get("/ready").status_code == 503

    def test_openapi_docs(self, client):
        assert client.get("/docs").status_code == 200

    def test_redoc_docs(self, client):
        assert client.get("/redoc").status_code == 200

    def test_predict_request_validation(self, client):
        assert client.post("/predict", json={"item_id": "ITEM_1"}).status_code == 422

    def test_predict_without_model_is_503(self, client):
        body = {"item_id": "FOODS_3_090", "store_id": "CA_1", "forecast_date": "2024-01-15"}
        assert client.post("/predict", json=body).status_code == 503

    def test_batch_predict_max_items(self, client):
        items = [
            {"item_id": f"ITEM_{i}", "store_id": "STORE_1", "forecast_date": "2024-01-15"}
            for i in range(1001)
        ]
        assert client.post("/predict/batch", json={"items": items}).status_code == 422

    def test_metrics_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "forecasting_predictions_total 0" in response.text


class TestEndpointsWithMockService:
    def test_predict_returns_service_result(self, client_with_mock_service):
        client, service = client_with_mock_service
        body = {"item_id": "ITEM_1", "store_id": "STORE_1", "forecast_date": "2024-01-15"}
        response = client.post("/predict", json={**body, "price": 2.5, "snap_enabled": True})
        assert response.status_code == 200
        assert response.json()["prediction"] == 12.5
        kwargs = service.predict.call_args.kwargs
        assert kwargs["features"] == {"sell_price": 2.5, "snap": 1}

    def test_value_error_becomes_400(self, client_with_mock_service):
        client, service = client_with_mock_service
        service.predict.side_effect = ValueError("unknown item")
        body = {"item_id": "X", "store_id": "Y", "forecast_date": "2024-01-15"}
        response = client.post("/predict", json=body)
        assert response.status_code == 400
        assert response.json()["detail"] == "unknown item"


class TestEndpointsWithRealModel:
    @pytest.fixture
    def client(self, trained_model_dir):
        return TestClient(create_app({"model_path": str(trained_model_dir)}))

    def test_ready_and_info(self, client):
        assert client.get("/ready").status_code == 200
        info = client.get("/model/info").json()
        assert info["framework"] == "lightgbm"
        assert "sales_lag_28" in info["features"]

    def test_predict(self, client):
        body = {
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "forecast_date": "2024-01-15",
            "price": 2.0,
        }
        response = client.post("/predict", json=body)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["prediction"] >= 0
        assert data["prediction_lower"] <= data["prediction"] <= data["prediction_upper"]

    def test_horizon(self, client):
        body = {
            "item_id": "FOODS_1_001",
            "store_id": "TX_1",
            "start_date": "2024-01-15",
            "horizon_days": 7,
        }
        response = client.post("/predict/horizon", json=body)
        assert response.status_code == 200, response.text
        forecasts = response.json()["forecasts"]
        assert [f["date"] for f in forecasts][:2] == ["2024-01-15", "2024-01-16"]
        assert len(forecasts) == 7

    def test_batch(self, client):
        items = [
            {"item_id": "FOODS_1_002", "store_id": "CA_1", "forecast_date": "2024-01-15"},
            {"item_id": "FOODS_1_001", "store_id": "TX_1", "forecast_date": "2024-01-16"},
        ]
        response = client.post("/predict/batch", json={"items": items})
        assert response.status_code == 200, response.text
        assert response.json()["total_predictions"] == 2
        assert "forecasting_predictions_total 2" in client.get("/metrics").text

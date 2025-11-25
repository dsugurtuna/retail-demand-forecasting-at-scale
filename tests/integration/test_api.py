"""Integration tests for API endpoints."""

import pytest
from datetime import date
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.serving.api import create_app
        
        app = create_app({"model_path": None})
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Root endpoint should return service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
    
    def test_liveness_probe(self, client):
        """Liveness probe should return alive."""
        response = client.get("/live")
        
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
    
    def test_health_check_without_model(self, client):
        """Health check without model should show status."""
        response = client.get("/health")
        
        # May be 503 if service not initialized
        assert response.status_code in [200, 503]
    
    def test_openapi_docs(self, client):
        """OpenAPI docs should be available."""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_redoc_docs(self, client):
        """ReDoc docs should be available."""
        response = client.get("/redoc")
        
        assert response.status_code == 200


class TestPredictEndpoints:
    """Tests for prediction endpoints."""
    
    @pytest.fixture
    def client_with_mock_model(self, mocker):
        """Create test client with mocked model."""
        from src.serving.api import create_app
        from src.serving.predictor import PredictionService
        
        app = create_app({"model_path": None})
        
        # Mock prediction service
        mock_service = mocker.MagicMock(spec=PredictionService)
        mock_service.is_loaded = True
        mock_service.predict.return_value = {
            "item_id": "ITEM_1",
            "store_id": "STORE_1",
            "date": "2024-01-15",
            "prediction": 12.5,
            "prediction_lower": 8.0,
            "prediction_upper": 17.0,
            "inference_time_ms": 15.3,
        }
        mock_service.get_model_info.return_value = {"version": "1.0.0"}
        
        return TestClient(app), mock_service
    
    def test_predict_request_validation(self, client):
        """Invalid predict request should return 422."""
        response = client.post("/predict", json={
            "item_id": "ITEM_1",
            # Missing required fields
        })
        
        assert response.status_code == 422
    
    def test_predict_valid_request_format(self, client):
        """Valid request format should be accepted."""
        response = client.post("/predict", json={
            "item_id": "FOODS_3_090",
            "store_id": "CA_1",
            "forecast_date": "2024-01-15",
        })
        
        # Should be 503 (model not loaded) not 422 (validation)
        assert response.status_code in [200, 503]
    
    def test_batch_predict_max_items(self, client):
        """Batch predict should enforce max items."""
        items = [
            {"item_id": f"ITEM_{i}", "store_id": "STORE_1", "forecast_date": "2024-01-15"}
            for i in range(1001)  # Exceeds 1000 limit
        ]
        
        response = client.post("/predict/batch", json={
            "items": items,
        })
        
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    @pytest.fixture
    def client(self):
        from src.serving.api import create_app
        app = create_app({"model_path": None})
        return TestClient(app)
    
    def test_metrics_prometheus_format(self, client):
        """Metrics should return Prometheus format."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Basic text format check
        content = response.text
        assert "forecasting_" in content or content == ""

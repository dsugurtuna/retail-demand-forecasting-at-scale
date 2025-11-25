"""Unit tests for configuration management."""

import os
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    Config,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    ServingConfig,
    load_config,
)


class TestConfig:
    """Tests for Config model."""
    
    def test_default_config(self):
        """Default config should create with defaults."""
        config = Config()
        
        assert config.environment == "development"
        assert config.debug is False
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
    
    def test_environment_validation(self):
        """Environment should be validated."""
        with pytest.raises(ValueError):
            Config(environment="invalid")
    
    def test_valid_environments(self):
        """Valid environments should be accepted."""
        for env in ["development", "staging", "production"]:
            config = Config(environment=env)
            assert config.environment == env


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_paths(self):
        """Default paths should be set."""
        config = DataConfig()
        
        assert config.raw_path == Path("data/raw")
        assert config.processed_path == Path("data/processed")
    
    def test_s3_bucket_optional(self):
        """S3 bucket should be optional."""
        config = DataConfig()
        assert config.s3_bucket is None
        
        config = DataConfig(s3_bucket="my-bucket")
        assert config.s3_bucket == "my-bucket"


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_hyperparameter_defaults(self):
        """Default hyperparameters should be set."""
        config = ModelConfig()
        
        assert config.n_estimators == 2000
        assert config.learning_rate == 0.05
        assert config.max_depth == 8
    
    def test_hyperparameter_validation(self):
        """Hyperparameters should be validated."""
        with pytest.raises(ValueError):
            ModelConfig(n_estimators=50)  # Less than minimum
        
        with pytest.raises(ValueError):
            ModelConfig(learning_rate=2.0)  # Greater than 1
        
        with pytest.raises(ValueError):
            ModelConfig(max_depth=25)  # Greater than maximum


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_empty_config(self):
        """Loading without args should return defaults."""
        config = load_config()
        
        assert isinstance(config, Config)
        assert config.environment == "development"
    
    def test_load_from_file(self, tmp_path):
        """Should load config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "environment": "staging",
            "model": {
                "n_estimators": 500,
            }
        }))
        
        config = load_config(config_file)
        
        assert config.environment == "staging"
        assert config.model.n_estimators == 500
    
    def test_environment_override(self, tmp_path):
        """Environment parameter should override file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "environment": "development",
        }))
        
        config = load_config(config_file, environment="production")
        
        assert config.environment == "production"
    
    def test_env_var_override(self, monkeypatch):
        """Environment variables should override config."""
        monkeypatch.setenv("FORECAST_MODEL_NAME", "custom_model")
        monkeypatch.setenv("FORECAST_SERVING_PORT", "9000")
        monkeypatch.setenv("FORECAST_DEBUG", "true")
        
        config = load_config()
        
        assert config.model.name == "custom_model"
        assert config.serving.port == 9000
        assert config.debug is True
    
    def test_nested_env_override(self, monkeypatch):
        """Nested environment variables should work."""
        monkeypatch.setenv("FORECAST_TRAINING_MLFLOW_URI", "http://mlflow:5000")
        
        config = load_config()
        
        assert config.training.mlflow_tracking_uri == "http://mlflow:5000"


class TestFeatureConfig:
    """Tests for FeatureConfig."""
    
    def test_default_lag_days(self):
        """Default lag days should be set."""
        config = FeatureConfig()
        
        assert 7 in config.lag_days
        assert 14 in config.lag_days
        assert 28 in config.lag_days
    
    def test_default_rolling_windows(self):
        """Default rolling windows should be set."""
        config = FeatureConfig()
        
        assert 7 in config.rolling_windows
        assert 14 in config.rolling_windows


class TestServingConfig:
    """Tests for ServingConfig."""
    
    def test_default_host_port(self):
        """Default host and port should be set."""
        config = ServingConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
    
    def test_port_validation(self):
        """Port should be validated."""
        with pytest.raises(ValueError):
            ServingConfig(port=0)
        
        with pytest.raises(ValueError):
            ServingConfig(port=70000)

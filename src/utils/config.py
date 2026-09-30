"""
Configuration management.

Provides type-safe, validated configuration with:
- YAML file loading
- Environment variable overrides
- Pydantic validation
- Hierarchical config merging
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data configuration."""

    raw_path: Path = Field(default=Path("data/raw"))
    processed_path: Path = Field(default=Path("data/processed"))
    features_path: Path = Field(default=Path("data/features"))

    # Data sources
    sales_file: str = Field(default="sales_train_evaluation.csv")
    calendar_file: str = Field(default="calendar.csv")
    prices_file: str = Field(default="sell_prices.csv")

    # S3 configuration (optional)
    s3_bucket: str | None = Field(default=None)
    s3_prefix: str = Field(default="data/")


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    # Lag features
    lag_days: list[int] = Field(default=[7, 14, 28, 365])

    # Rolling features
    rolling_windows: list[int] = Field(default=[7, 14, 28, 56])
    rolling_aggregations: list[str] = Field(default=["mean", "std", "min", "max"])

    # Price features
    price_lags: list[int] = Field(default=[1, 7, 28])

    # Calendar features
    include_holidays: bool = Field(default=True)
    include_events: bool = Field(default=True)

    # Feature store
    feature_store_enabled: bool = Field(default=False)
    feature_store_path: Path = Field(default=Path("data/feature_store"))


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(default="lightgbm_forecaster")
    version: str = Field(default="1.0.0")

    # Hyperparameters
    n_estimators: int = Field(default=2000, ge=100)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    max_depth: int = Field(default=8, ge=1, le=20)
    num_leaves: int = Field(default=63, ge=2)
    min_child_samples: int = Field(default=50, ge=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=0.1, ge=0)

    # Training
    early_stopping_rounds: int = Field(default=100)
    verbose: int = Field(default=100)

    # Objective
    objective: str = Field(default="tweedie")
    tweedie_variance_power: float = Field(default=1.1, ge=1, le=2)

    # Model saving
    save_path: Path = Field(default=Path("models"))


class TrainingConfig(BaseModel):
    """Training configuration."""

    # Data splits
    train_start: str = Field(default="2011-01-29")
    train_end: str = Field(default="2016-04-24")
    valid_start: str = Field(default="2016-04-25")
    valid_end: str = Field(default="2016-05-22")

    # Cross-validation
    n_folds: int = Field(default=5, ge=1)
    cv_gap_days: int = Field(default=28, ge=0)

    # MLflow tracking
    mlflow_tracking_uri: str = Field(default="mlruns")
    mlflow_experiment_name: str = Field(default="retail_forecasting")

    # Random seed
    random_seed: int = Field(default=42)


class ServingConfig(BaseModel):
    """Serving configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)

    # Model
    model_path: Path = Field(default=Path("models/production/model.pkl"))

    # Performance
    batch_size: int = Field(default=1000, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)

    # Feature store
    feature_store_enabled: bool = Field(default=False)
    cache_ttl_seconds: int = Field(default=3600, ge=0)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Path | None = Field(default=None)
    json_format: bool = Field(default=False)


class Config(BaseModel):
    """Root configuration."""

    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v


def load_config(config_path: str | Path | None = None, environment: str | None = None) -> Config:
    """
    Load configuration from file with environment overrides.

    Args:
        config_path: Path to YAML config file
        environment: Environment name (overrides file)

    Returns:
        Validated Config object
    """
    config_dict: dict[str, Any] = {}

    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            config_dict = yaml.safe_load(config_path.read_text()) or {}

    # Load environment-specific config
    if environment:
        env_config_path = Path(f"config/{environment}.yaml")
        if env_config_path.exists():
            env_dict = yaml.safe_load(env_config_path.read_text()) or {}
            config_dict = _merge_dicts(config_dict, env_dict)
        config_dict["environment"] = environment

    # Override with environment variables
    config_dict = _apply_env_overrides(config_dict)

    return Config(**config_dict)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides."""
    # Map of env var prefixes to config paths
    env_mappings = {
        "FORECAST_DATA_RAW_PATH": ("data", "raw_path"),
        "FORECAST_DATA_S3_BUCKET": ("data", "s3_bucket"),
        "FORECAST_MODEL_NAME": ("model", "name"),
        "FORECAST_MODEL_VERSION": ("model", "version"),
        "FORECAST_TRAINING_MLFLOW_URI": ("training", "mlflow_tracking_uri"),
        "FORECAST_SERVING_HOST": ("serving", "host"),
        "FORECAST_SERVING_PORT": ("serving", "port"),
        "FORECAST_LOG_LEVEL": ("logging", "level"),
        "FORECAST_DEBUG": ("debug",),
        "FORECAST_ENVIRONMENT": ("environment",),
    }

    for env_var, path in env_mappings.items():
        raw = os.environ.get(env_var)
        if raw is not None:
            value: Any = raw
            if raw.lower() in ("true", "false"):
                value = raw.lower() == "true"
            elif raw.isdigit():
                value = int(raw)
            _set_nested(config, path, value)

    return config


def _set_nested(d: dict, path: tuple, value: Any) -> None:
    """Set a nested dictionary value."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value

# =============================================================================
# Retail Demand Forecasting - Architecture Documentation
# =============================================================================

## System Overview

This document provides comprehensive technical documentation for the Retail
Demand Forecasting at Scale system. It covers architecture decisions, component
interactions, and deployment patterns.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RETAIL DEMAND FORECASTING                          │
│                              SYSTEM ARCHITECTURE                             │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │    │   Feature   │    │   Model     │    │   Serving   │
│   Sources   │───▶│   Pipeline  │───▶│   Training  │───▶│   Layer     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  S3/Local   │    │   Feature   │    │   MLflow    │    │   FastAPI   │
│   Storage   │    │    Store    │    │   Registry  │    │   Endpoint  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Component Architecture

### 1. Data Layer (`src/data/`)

The data layer handles all data ingestion, validation, and preprocessing.

```python
# Component: DataLoader
# Purpose: Unified data loading from multiple sources

DataLoader
├── load_sales()        # Load sales data with lazy evaluation
├── load_calendar()     # Load calendar/holiday data
├── load_prices()       # Load price data
└── load_from_s3()      # S3 integration for cloud deployment
```

**Key Design Decisions:**
- **Polars over Pandas**: 2-5x faster for large datasets, better memory efficiency
- **Lazy evaluation**: Deferred execution for query optimization
- **Schema validation**: Pandera schemas ensure data quality at ingestion

### 2. Feature Engineering (`src/features/`)

Modular feature generation with temporal awareness.

```python
# Feature Generation Pipeline

FeatureEngineer
├── TemporalFeatureGenerator
│   ├── Lag features (7, 14, 28, 365 days)
│   ├── Rolling statistics (mean, std, min, max)
│   ├── Calendar features (DOW, month, holidays)
│   └── Cyclical encoding (sin/cos transformations)
│
├── PriceFeatureGenerator
│   ├── Price momentum
│   ├── Promotion detection
│   └── Price elasticity proxies
│
├── HierarchicalFeatureGenerator
│   ├── Cross-level aggregations
│   └── Hierarchical encodings
│
└── FeatureStore
    ├── Point-in-time retrieval
    ├── Feature versioning
    └── Online/offline serving
```

**Key Design Decisions:**
- **Temporal leak prevention**: All features use point-in-time calculations
- **Hierarchical aggregations**: Capture item→store→region patterns
- **Feature store**: Enable feature reuse and online serving

### 3. Model Layer (`src/models/`)

Production-grade model implementations with MLOps integration.

```python
# Model Hierarchy

BaseForecaster (Abstract)
├── LightGBMForecaster
│   ├── Custom WRMSSE objective
│   ├── Confidence interval estimation
│   └── Feature importance analysis
│
└── EnsembleForecaster
    ├── Model combination strategies
    ├── Weight optimization
    └── Calibration layer
```

**Hyperparameter Defaults (Optimized for M5):**
```yaml
n_estimators: 2000
learning_rate: 0.05
max_depth: 8
num_leaves: 63
min_child_samples: 50
subsample: 0.8
colsample_bytree: 0.8
objective: tweedie
tweedie_variance_power: 1.1
```

### 4. Evaluation Framework (`src/evaluation/`)

Comprehensive evaluation with backtesting.

```python
# Evaluation Components

Metrics
├── RMSE, MAE, MAPE, sMAPE
├── MASE (scaled error)
├── WRMSSE (M5 competition metric)
└── Coverage metrics

BacktestEngine
├── Rolling origin CV
├── Gap period handling
├── Expanding/sliding windows
└── Metrics aggregation
```

### 5. Serving Layer (`src/serving/`)

Production API with health monitoring.

```python
# API Endpoints

FastAPI Application
├── GET  /health        # Kubernetes health probe
├── GET  /ready         # Readiness probe
├── GET  /live          # Liveness probe
├── POST /predict       # Single prediction
├── POST /predict/batch # Batch predictions
├── POST /predict/horizon # Multi-day forecast
├── GET  /metrics       # Prometheus metrics
└── GET  /model/info    # Model metadata
```

## Data Flow

### Training Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Raw     │───▶│  Validate│───▶│  Feature │───▶│  Train   │
│  Data    │    │  & Clean │    │  Engineer│    │  Model   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                      │                │
                                      ▼                ▼
                              ┌──────────┐    ┌──────────┐
                              │  Feature │    │  MLflow  │
                              │  Store   │    │  Registry│
                              └──────────┘    └──────────┘
```

### Inference Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  API     │───▶│  Feature │───▶│  Model   │───▶│  Post-   │
│  Request │    │  Lookup  │    │  Predict │    │  Process │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │                │
                     ▼                ▼
              ┌──────────┐    ┌──────────┐
              │  Redis   │    │  Metrics │
              │  Cache   │    │  Export  │
              └──────────┘    └──────────┘
```

## Deployment Architecture

### Kubernetes Deployment

```yaml
# Simplified deployment structure

Namespace: forecasting
├── Deployment: api (replicas: 3)
│   └── Container: forecasting-api
│       ├── Resources: 2Gi memory, 1 CPU
│       └── Probes: /health, /ready, /live
│
├── Deployment: training (replicas: 1)
│   └── Container: forecasting-train
│       └── Resources: 8Gi memory, 4 CPU
│
├── Service: api-service (LoadBalancer)
│   └── Port: 8000
│
├── ConfigMap: forecasting-config
│   └── config.yaml
│
├── Secret: forecasting-secrets
│   └── MLflow credentials, S3 keys
│
└── HPA: api-hpa
    └── Scale: 3-10 pods, 70% CPU target
```

### Infrastructure Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| Container Registry | Image storage | GitHub Container Registry |
| Artifact Storage | Model & data storage | AWS S3 / MinIO |
| Experiment Tracking | Model versioning | MLflow |
| Secrets Management | Credential storage | Kubernetes Secrets |
| Monitoring | Metrics & alerting | Prometheus + Grafana |
| Log Aggregation | Centralized logging | ELK Stack / Loki |

## Configuration Management

### Environment Hierarchy

```
config/
├── default.yaml      # Base configuration
├── development.yaml  # Local development
├── staging.yaml      # Staging environment
└── production.yaml   # Production settings
```

### Configuration Loading

```python
# Priority (highest to lowest):
# 1. Environment variables (FORECAST_*)
# 2. Environment-specific config file
# 3. Default config file
# 4. Pydantic defaults

config = load_config(
    config_path="config/default.yaml",
    environment="production"
)
```

## Security Considerations

### API Security
- Rate limiting per client
- Request validation with Pydantic
- Input sanitization

### Data Security
- Encryption at rest (S3 SSE)
- Encryption in transit (TLS 1.3)
- Audit logging for data access

### Secrets Management
- No hardcoded credentials
- Kubernetes Secrets for sensitive data
- Regular credential rotation

## Performance Optimization

### Model Serving
- Model preloading at startup
- Prediction batching
- Redis caching for features
- Async request handling

### Training
- Polars for data processing (2-5x faster than Pandas)
- LightGBM histogram-based learning
- Early stopping to prevent overfitting
- Parallel feature computation

### Benchmarks

| Operation | Latency (p50) | Latency (p99) | Throughput |
|-----------|---------------|---------------|------------|
| Single prediction | 15ms | 50ms | 200 req/s |
| Batch (100 items) | 100ms | 300ms | 50 req/s |
| Model training | 30min | 45min | N/A |

## Monitoring & Alerting

### Key Metrics

```prometheus
# Prometheus metrics exported

forecasting_predictions_total{status}     # Total predictions
forecasting_inference_latency_seconds     # Inference latency histogram
forecasting_model_version{version}        # Current model version
forecasting_errors_total{type}            # Error counts by type
forecasting_feature_cache_hits_total      # Cache hit rate
```

### Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Latency | p99 > 500ms for 5min | Warning |
| Error Rate | >1% errors for 5min | Critical |
| Model Staleness | Model age > 7 days | Warning |
| Service Down | Health check failing | Critical |

## Development Workflow

### Local Development

```bash
# Start local environment
docker-compose up -d

# Run tests
pytest tests/ -v

# Lint and format
ruff check src tests
black src tests

# Type check
mypy src
```

### CI/CD Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Lint   │───▶│  Test   │───▶│  Build  │───▶│ Deploy  │
│  Format │    │  Unit   │    │  Docker │    │ Staging │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                    │              │               │
                    ▼              ▼               ▼
              ┌─────────┐    ┌─────────┐    ┌─────────┐
              │  Test   │    │  Push   │    │ Deploy  │
              │  Integ  │    │  GHCR   │    │  Prod   │
              └─────────┘    └─────────┘    └─────────┘
```

## Future Roadmap

### Phase 1: Foundation (Current)
- [x] Core forecasting models
- [x] API serving layer
- [x] Docker containerization
- [x] CI/CD pipeline

### Phase 2: Enhancement
- [ ] Deep learning models (TFT, N-BEATS)
- [ ] Real-time feature streaming
- [ ] A/B testing framework
- [ ] Automated retraining

### Phase 3: Scale
- [ ] Multi-region deployment
- [ ] GraphQL API
- [ ] Feature platform integration
- [ ] AutoML pipeline

## References

- [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/)

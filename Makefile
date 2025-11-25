# =============================================================================
# Makefile for Retail Demand Forecasting
# =============================================================================
# Convenience commands for development, testing, and deployment
# =============================================================================

.PHONY: help install install-dev lint format test test-unit test-integration \
        coverage clean docker-build docker-up docker-down train serve

# Default target
help:
	@echo "Retail Demand Forecasting - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linting (ruff)"
	@echo "  make format         Format code (black)"
	@echo "  make typecheck      Run type checking (mypy)"
	@echo "  make quality        Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo ""
	@echo "Application:"
	@echo "  make train          Run model training"
	@echo "  make serve          Start API server"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean build artifacts"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

typecheck:
	mypy src --ignore-missing-imports

quality: lint typecheck
	@echo "All quality checks passed!"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m "not slow"

test-slow:
	pytest tests/ -v -m slow

coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-train:
	docker-compose --profile training up training

docker-dev:
	docker-compose --profile dev up jupyter

# =============================================================================
# Application
# =============================================================================

train:
	python -m src.train

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

# =============================================================================
# Data
# =============================================================================

data-download:
	@echo "Downloading M5 competition data..."
	mkdir -p data/raw
	# kaggle competitions download -c m5-forecasting-accuracy -p data/raw

data-process:
	python -m src.data.preprocessor

# =============================================================================
# MLflow
# =============================================================================

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf mlruns/
	rm -rf models/*.pkl
	rm -rf data/processed/
	rm -rf data/features/

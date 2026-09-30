# Convenience commands. Each target is a one-liner you can also run directly.

.PHONY: help install install-dev lint format typecheck test test-fast smoke train-synthetic serve clean

help:
	@echo "install        pip install -e ."
	@echo "install-dev    pip install -e '.[dev]'"
	@echo "lint           ruff check + ruff format --check"
	@echo "format         ruff format + ruff check --fix"
	@echo "typecheck      mypy src"
	@echo "test           full test suite, including the smoke training run"
	@echo "test-fast      tests without the slow end-to-end runs"
	@echo "smoke          python -m src.train --smoke (synthetic data, ~1 minute)"
	@echo "train-synthetic  larger synthetic run (210 items x 10 stores x 1,941 days)"
	@echo "serve          API on :8000 using the smoke model"
	@echo "clean          remove caches and artefacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest

test-fast:
	pytest -m "not slow"

smoke:
	python -m src.train --smoke

train-synthetic:
	python -m src.train --synthetic --n-items 210 --n-stores 10 --n-days 1941 --output-dir artefacts/synthetic

serve:
	FORECAST_MODEL_PATH=artefacts/smoke/model uvicorn src.serving.api:app --host 127.0.0.1 --port 8000

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

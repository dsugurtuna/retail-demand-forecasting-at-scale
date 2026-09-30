# Changelog

## 2.1.0 (unreleased)

A repair and honesty pass. Both GitHub Actions workflows had failed on every run, the test suite could not be imported, and the README described things the code did not do.

### Fixed

- **CI never ran.** A step condition used the `secrets` context, which GitHub does not allow there, so the workflow was rejected at startup with no jobs.
- **Weekly training failed every week.** Installing the old dependency list (PyTorch, pytorch-forecasting, CatBoost, XGBoost, MLflow, Great Expectations, Evidently and more, none of them imported) sent pip's resolver into long backtracking; reproduced locally, it ends in `resolution-too-deep` after about 26 minutes. The job then called `python -m src.train`, which did not exist, and `import src` itself raised `ImportError`.
- **Target leakage** in hierarchical features (same-day totals and shares that contained the row's own sales), in a cumulative zero-count feature and in a normalised time index. The backtest built test features without history and validated on the last series rather than the last dates.
- **WRMSSE** pooled all errors into one number instead of scoring each series; it now follows the M5 definition, including the 12-level version.
- **Weekday numbering** disagreed between training (Polars, 1-7) and serving (Python, 0-6), and flagged Fridays as weekends.
- **API factory** returned an app with no routes; the model path in serving did not match what training saved.
- **Data loader** crashed on construction and silently fell back to synthetic data when files were missing.

### Added

- `python -m src.train` with `--smoke` (synthetic, pass/fail checks), `--synthetic` and `--data-dir` (M5 files) modes, naive baselines and JSON metrics.
- Tests for leakage, WRMSSE, data loading, models, the API with a real saved model, and the end-to-end smoke run.
- `docs/WHY.md`, a `LICENSE` file (MIT, as the project already declared) and this changelog.

### Removed from the documentation (never implemented or not reproducible)

- Temporal Fusion Transformer and N-BEATS models and the ensemble built on them.
- "150+ engineered features", weather, economic and competitor-price features.
- Airflow DAGs, MLflow tracking and model registry, Redis caching, Prometheus/Grafana dashboards, A/B testing, Kubernetes manifests, Terraform, blue-green deployment, API keys and rate limiting.
- All business-impact figures (MAPE, stockouts, inventory savings, analyst requests), the M5 benchmark table (WRMSSE, MAPE, training times), scalability and latency tables, "Top 2% solution" and "600,000+ SKUs". No code in the repository produced any of these numbers.

### Changed

- Python 3.11+; core dependencies cut to the eleven packages the code needs; FastAPI in a `serve` extra.
- Backtest default gap is 0 days (with `min_lag` equal to the horizon a gap only removes lag features).
- Numeric feature nulls are no longer filled with 0.
- `FeatureStore.get_point_in_time_features` renamed to `load_version_as_of`, because it selects by save time, not event time.
- Docker Compose now defines only the training and API services; MLflow, PostgreSQL, MinIO, Redis and Jupyter were never used by the code.

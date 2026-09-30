"""
Train and evaluate the LightGBM demand model.

    python -m src.train --smoke
        Small synthetic dataset, runs in well under a minute on a laptop CPU
        and fails (exit code 1) if any end-to-end check fails. This is what
        CI and the weekly scheduled workflow run.

    python -m src.train --synthetic --n-items 210 --n-stores 10 --n-days 1941
        Larger synthetic run, same pipeline, no pass/fail checks.

    python -m src.train --data-dir data/raw --stores CA_1
        The M5 files from Kaggle (downloaded separately), optionally limited to
        some stores to fit in memory.

Pipeline: load -> validate (Pandera) -> hide the target after the forecast
origin -> build features -> train LightGBM with early stopping on the last
28 days before the origin -> forecast the final ``horizon`` days -> score
against the actuals and two naive baselines -> optional rolling-origin
backtest -> write artefacts to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import lightgbm
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from src import __version__
from src.data.loader import DataLoader, merge_m5_frames
from src.data.synthetic import SyntheticDataConfig, SyntheticDataGenerator
from src.data.validators import DataValidator
from src.evaluation.backtesting import BacktestConfig, BacktestEngine
from src.evaluation.metrics import hierarchical_wrmsse, mae, rmse, wrmsse_row_weights
from src.features.engineer import FeatureEngineer, FeatureEngineerConfig, to_model_input
from src.models.lightgbm_model import LightGBMConfig, LightGBMForecaster
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

TARGET, DATE, SERIES = "sales", "date", "id"
SCORING_COLUMNS = [
    SERIES,
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    DATE,
    TARGET,
    "sell_price",
]

# 21 items x 5 stores (two states) x 2 years: small, but every hierarchy level exists.
SMOKE_DATA = SyntheticDataConfig(n_items=21, n_stores=5, n_days=730, random_seed=42)
SMOKE_MODEL = LightGBMConfig(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=50,
    early_stopping_rounds=30,
)


class TrainConfig(BaseModel):
    """Everything that defines a run. Written to ``run_config.json``."""

    data_dir: Path | None = None
    stores: list[str] | None = None
    synthetic: SyntheticDataConfig | None = None
    horizon: int = Field(default=28, ge=7, le=56)
    validation_days: int = Field(default=28, ge=7)
    backtest_folds: int = Field(default=0, ge=0)
    objective: Literal["tweedie", "wrmsse-surrogate"] = "tweedie"
    model: LightGBMConfig = Field(default_factory=LightGBMConfig)
    output_dir: Path = Path("artefacts/run")
    run_checks: bool = False


class SmokeCheckError(RuntimeError):
    """Raised when one or more end-to-end checks fail."""


# --------------------------------------------------------------------------- data


def load_data(config: TrainConfig) -> tuple[pl.DataFrame, str]:
    """Return the merged long frame and a short description of its source."""
    if config.synthetic is not None:
        sales, calendar, prices = SyntheticDataGenerator(config.synthetic).generate_all()
        return merge_m5_frames(sales, calendar, prices), "synthetic"
    if config.data_dir is None:
        raise ValueError("Give --data-dir for M5 files, or --synthetic / --smoke")
    stores = ",".join(config.stores) if config.stores else "all stores"
    return DataLoader(data_dir=config.data_dir).load_all(config.stores), f"m5 ({stores})"


def hide_target_after(data: pl.DataFrame, origin: date) -> pl.DataFrame:
    """Replace target values after ``origin`` with nulls, so features cannot see them."""
    return data.with_columns(
        pl.when(pl.col(DATE) > origin).then(None).otherwise(pl.col(TARGET)).alias(TARGET)
    )


# ---------------------------------------------------------------------- baselines


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Repeat each series' last observed week: forecast(t) = sales on the same weekday."""
    last_date = train[DATE].max()
    last_week = train[train[DATE] > last_date - pd.Timedelta(days=7)]
    lookup = last_week.assign(_dow=last_week[DATE].dt.dayofweek).set_index([SERIES, "_dow"])[TARGET]
    keys = pd.MultiIndex.from_arrays([test[SERIES], test[DATE].dt.dayofweek])
    return lookup.reindex(keys).fillna(0).to_numpy(dtype=float)


def moving_average(train: pd.DataFrame, test: pd.DataFrame, days: int = 28) -> np.ndarray:
    """Flat forecast at each series' mean over its last ``days`` days."""
    last_date = train[DATE].max()
    recent = train[train[DATE] > last_date - pd.Timedelta(days=days)]
    means = recent.groupby(SERIES)[TARGET].mean()
    return test[SERIES].map(means).fillna(0).to_numpy(dtype=float)


def score(
    train: pd.DataFrame, test: pd.DataFrame, predictions: np.ndarray
) -> dict[str, float | dict[str, float]]:
    """WRMSSE (12-level and series-level), RMSE and MAE for one set of forecasts."""
    evaluation = test.assign(prediction=predictions)
    hierarchical = hierarchical_wrmsse(train, evaluation)
    levels = hierarchical["levels"]
    assert isinstance(levels, dict)
    y = test[TARGET].to_numpy(dtype=float)
    return {
        "wrmsse": float(hierarchical["wrmsse"]),  # type: ignore[arg-type]
        "wrmsse_series_level": float(levels["series"]),
        "rmse": rmse(y, predictions),
        "mae": mae(y, predictions),
        "wrmsse_by_level": {k: round(float(v), 6) for k, v in levels.items()},
    }


# ---------------------------------------------------------------------- pipeline


def run(config: TrainConfig) -> dict[str, Any]:
    """Run the pipeline and return the metrics written to ``metrics.json``."""
    started = time.time()
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    data, source = load_data(config)
    validation = DataValidator(min_history_days=config.horizon * 3).validate(data, "sales")
    if not validation.is_valid:
        raise ValueError(f"Input data failed validation: {validation.errors}")

    last_day: date = data[DATE].max()  # type: ignore[assignment]
    origin = last_day - timedelta(days=config.horizon)
    val_start = origin - timedelta(days=config.validation_days - 1)
    logger.info("Forecast origin %s, horizon %d days", origin, config.horizon)

    engineer = FeatureEngineer(FeatureEngineerConfig(min_lag=config.horizon))
    features = engineer.fit_transform(hide_target_after(data, origin))
    columns = engineer.get_feature_names()

    fit_part = features.filter(pl.col(DATE) < val_start)
    val_part = features.filter(pl.col(DATE).is_between(val_start, origin))
    test_part = (
        features.filter(pl.col(DATE) > origin)
        .drop(TARGET)
        .join(data.select(SERIES, DATE, TARGET), on=[SERIES, DATE], how="left")
    )

    use_surrogate = config.objective == "wrmsse-surrogate"
    sample_weight = None
    if use_surrogate:
        sample_weight = wrmsse_row_weights(fit_part.to_pandas()).to_numpy()

    model = LightGBMForecaster(config=config.model, use_custom_objective=use_surrogate)
    model.fit(
        to_model_input(fit_part, columns),
        fit_part[TARGET].to_numpy(),
        X_valid=to_model_input(val_part, columns),
        y_valid=val_part[TARGET].to_numpy(),
        sample_weight=sample_weight,
    )
    X_test = to_model_input(test_part, columns)
    predictions = np.asarray(model.predict(X_test), dtype=float)

    # Free the training matrices before scoring and backtesting (memory).
    del features, fit_part, val_part, X_test

    # Score against the untouched data (feature building may drop constant
    # columns such as state_id, which WRMSSE's hierarchy levels still need).
    scoring = data.select([c for c in SCORING_COLUMNS if c in data.columns])
    train_pd = scoring.filter(pl.col(DATE) <= origin).to_pandas()
    test_pd = scoring.filter(pl.col(DATE) > origin).sort([SERIES, DATE]).to_pandas()
    if not (
        test_pd[SERIES].to_numpy() == test_part[SERIES].to_numpy()
    ).all():  # pragma: no cover - guards against a silent misalignment
        raise RuntimeError("Forecast rows are not aligned with the actuals")
    for frame in (train_pd, test_pd):
        frame[DATE] = pd.to_datetime(frame[DATE])

    baselines = {
        "seasonal_naive_7d": seasonal_naive(train_pd, test_pd),
        "moving_average_28d": moving_average(train_pd, test_pd),
    }
    holdout = {"model": score(train_pd, test_pd, predictions)}
    holdout.update({name: score(train_pd, test_pd, p) for name, p in baselines.items()})

    backtest = None
    if config.backtest_folds:
        engine = BacktestEngine(
            BacktestConfig(
                n_folds=config.backtest_folds,
                test_days=config.horizon,
                validation_days=config.validation_days,
                min_train_days=max(365, config.horizon * 4),
            )
        )
        bt = engine.run(
            data, model.clone(), FeatureEngineer(FeatureEngineerConfig(min_lag=config.horizon))
        )
        backtest = {
            "folds": [
                {
                    "origin": str(f.train_end.date()),
                    "wrmsse": f.wrmsse,
                    "rmse": f.metrics.rmse,
                    "mae": f.metrics.mae,
                }
                for f in bt.fold_results
            ],
            "mean_wrmsse": bt.mean_wrmsse,
        }

    model_dir = out / "model"
    model.save(model_dir)
    pd.DataFrame(
        {
            SERIES: test_pd[SERIES],
            DATE: test_pd[DATE],
            "actual": test_pd[TARGET],
            "prediction": predictions,
            **baselines,
        }
    ).to_csv(out / "predictions.csv", index=False)
    model.get_feature_importance().to_csv(out / "feature_importance.csv", index=False)
    (out / "run_config.json").write_text(config.model_dump_json(indent=2))

    metrics: dict[str, Any] = {
        "package_version": __version__,
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "data_source": source,
        "n_series": int(data[SERIES].n_unique()),
        "n_rows": len(data),
        "first_date": str(data[DATE].min()),
        "forecast_origin": str(origin),
        "horizon_days": config.horizon,
        "objective": config.objective,
        "features": engineer.get_feature_summary(),
        "best_iteration": model.metadata.metrics.get("best_iteration") if model.metadata else None,
        "holdout": holdout,
        "backtest": backtest,
        "validation_warnings": validation.warnings,
    }

    if config.run_checks:
        metrics["checks"] = run_checks(
            data, origin, engineer, columns, test_part, predictions, holdout, model, model_dir
        )

    metrics["runtime_seconds"] = round(time.time() - started, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def run_checks(
    data: pl.DataFrame,
    origin: date,
    engineer: FeatureEngineer,
    columns: list[str],
    window_features: pl.DataFrame,
    predictions: np.ndarray,
    holdout: dict[str, dict[str, Any]],
    model: LightGBMForecaster,
    model_dir: Path,
) -> dict[str, bool]:
    """End-to-end checks for the smoke run. Raises SmokeCheckError if any fail."""
    n_series = data[SERIES].n_unique()
    last_day = data[DATE].max()
    assert isinstance(last_day, date)
    horizon = (last_day - origin).days
    model_wrmsse = float(holdout["model"]["wrmsse"])
    naive_wrmsse = min(float(holdout[name]["wrmsse"]) for name in holdout if name != "model")

    # Leakage check: scramble the hidden actuals and rebuild features. Features
    # for the forecast window must not change, because nothing may read them.
    rng = np.random.default_rng(0)
    scrambled = data.with_columns(
        pl.when(pl.col(DATE) > origin)
        .then(pl.Series(rng.integers(0, 1000, len(data))).cast(pl.Float64))
        .otherwise(pl.col(TARGET))
        .alias(TARGET)
    )
    rebuilt = FeatureEngineer(engineer.config).fit_transform(hide_target_after(scrambled, origin))
    window = pl.col(DATE) > origin
    unchanged = window_features.select(columns).equals(rebuilt.filter(window).select(columns))

    reloaded = LightGBMForecaster()
    reloaded.load(model_dir)
    test_rows = window_features
    same_after_reload = bool(
        np.allclose(reloaded.predict(to_model_input(test_rows, columns)), predictions)
    )

    checks = {
        "one_forecast_per_series_per_day": len(predictions) == n_series * horizon,
        "forecasts_finite_and_non_negative": bool(
            np.isfinite(predictions).all() and (predictions >= 0).all()
        ),
        "wrmsse_finite": bool(np.isfinite(model_wrmsse)),
        "model_beats_best_naive_baseline": model_wrmsse < naive_wrmsse,
        "hidden_actuals_do_not_change_features": bool(unchanged),
        "saved_model_reloads_with_same_forecasts": same_after_reload,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SmokeCheckError(f"Smoke checks failed: {failed}")
    return checks


# ---------------------------------------------------------------------------- cli


def build_config(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(prog="python -m src.train", description=__doc__.split("\n")[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="small synthetic run with checks")
    mode.add_argument("--synthetic", action="store_true", help="synthetic data, sized below")
    mode.add_argument("--data-dir", type=Path, help="directory with the M5 CSV files")
    parser.add_argument("--stores", default="", help="comma-separated M5 store IDs, e.g. CA_1")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--n-stores", type=int, default=4)
    parser.add_argument("--n-days", type=int, default=1941)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--backtest-folds", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--objective", choices=["tweedie", "wrmsse-surrogate"], default="tweedie")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.smoke:
        synthetic: SyntheticDataConfig | None = SMOKE_DATA
        model = SMOKE_MODEL.model_copy()
        folds = 2 if args.backtest_folds is None else args.backtest_folds
        output = args.output_dir or Path("artefacts/smoke")
    else:
        synthetic = (
            SyntheticDataConfig(
                n_items=args.n_items,
                n_stores=args.n_stores,
                n_days=args.n_days,
                random_seed=args.seed,
            )
            if args.synthetic
            else None
        )
        model = LightGBMConfig(seed=args.seed)
        folds = 3 if args.backtest_folds is None else args.backtest_folds
        output = args.output_dir or Path("artefacts/run")

    updates: dict[str, Any] = {}
    if args.n_estimators is not None:
        updates["n_estimators"] = args.n_estimators
    if args.learning_rate is not None:
        updates["learning_rate"] = args.learning_rate
    if updates:
        model = model.model_copy(update=updates)

    return TrainConfig(
        data_dir=args.data_dir,
        stores=[s.strip() for s in args.stores.split(",") if s.strip()] or None,
        synthetic=synthetic,
        horizon=args.horizon,
        backtest_folds=folds,
        objective=args.objective,
        model=model,
        output_dir=output,
        run_checks=args.smoke,
    )


def print_summary(metrics: dict[str, Any]) -> None:
    print(
        f"\nData: {metrics['data_source']}, {metrics['n_series']} series, origin "
        f"{metrics['forecast_origin']}, horizon {metrics['horizon_days']} days"
    )
    print(f"Features: {metrics['features']['total_features']}")
    print(f"\n{'forecast':<22}{'WRMSSE':>10}{'WRMSSE (series)':>18}{'RMSE':>10}{'MAE':>10}")
    for name, m in metrics["holdout"].items():
        print(
            f"{name:<22}{m['wrmsse']:>10.4f}{m['wrmsse_series_level']:>18.4f}"
            f"{m['rmse']:>10.4f}{m['mae']:>10.4f}"
        )
    if metrics.get("backtest"):
        folds = ", ".join(f"{f['wrmsse']:.4f}" for f in metrics["backtest"]["folds"])
        print(f"\nBacktest WRMSSE by fold: {folds} (mean {metrics['backtest']['mean_wrmsse']:.4f})")
    for name, ok in (metrics.get("checks") or {}).items():
        print(f"check {name}: {'pass' if ok else 'FAIL'}")
    print(f"\nRuntime {metrics['runtime_seconds']}s")


def main(argv: list[str] | None = None) -> int:
    setup_logging("INFO")
    config = build_config(argv)
    try:
        metrics = run(config)
    except SmokeCheckError as exc:
        logger.error("%s", exc)
        return 1
    print_summary(metrics)
    print(f"Artefacts written to {config.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

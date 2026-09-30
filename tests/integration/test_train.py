"""End-to-end test of the training entry point (the same run CI and the weekly job use)."""

import json

import pytest

from src.train import build_config, main


@pytest.mark.slow
def test_smoke_run_passes_all_checks(tmp_path):
    out = tmp_path / "smoke"
    assert main(["--smoke", "--output-dir", str(out)]) == 0

    metrics = json.loads((out / "metrics.json").read_text())
    assert all(metrics["checks"].values())
    assert metrics["data_source"] == "synthetic"
    model, naive = metrics["holdout"]["model"], metrics["holdout"]["seasonal_naive_7d"]
    assert model["wrmsse"] < naive["wrmsse"]
    assert len(metrics["backtest"]["folds"]) == 2
    # The last backtest fold uses the same origin as the holdout, so it must agree.
    assert metrics["backtest"]["folds"][-1]["wrmsse"] == pytest.approx(model["wrmsse"])
    for name in ("predictions.csv", "feature_importance.csv", "run_config.json"):
        assert (out / name).exists()
    assert (out / "model" / "model.txt").exists()


def test_m5_mode_needs_files(tmp_path):
    config = build_config(["--data-dir", str(tmp_path), "--stores", "CA_1, TX_1"])
    assert config.stores == ["CA_1", "TX_1"]
    assert config.synthetic is None
    assert not config.run_checks


def test_cli_overrides(tmp_path):
    config = build_config(
        ["--synthetic", "--n-items", "7", "--n-estimators", "50", "--backtest-folds", "0"]
    )
    assert config.synthetic is not None and config.synthetic.n_items == 7
    assert config.model.n_estimators == 50
    assert config.backtest_folds == 0

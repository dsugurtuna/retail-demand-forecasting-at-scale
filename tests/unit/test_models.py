"""Unit tests for the LightGBM wrapper and the ensemble."""

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleConfig, EnsembleForecaster
from src.models.lightgbm_model import LightGBMConfig, LightGBMForecaster

FAST = LightGBMConfig(n_estimators=60, learning_rate=0.1, num_leaves=15, min_data_in_leaf=20)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "store": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        }
    )
    effect = X["store"].map({"A": 1.0, "B": 2.0, "C": 3.0}).astype(float)
    y = rng.poisson(np.exp(0.5 * X["x1"]) * effect).astype(float)
    return X.iloc[:1600], y[:1600], X.iloc[1600:], y[1600:]


class TestLightGBMForecaster:
    def test_fit_predict_beats_mean(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        model = LightGBMForecaster(config=FAST).fit(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert (preds >= 0).all()
        model_mse = np.mean((preds - y_test) ** 2)
        mean_mse = np.mean((y_train.mean() - y_test) ** 2)
        assert model_mse < mean_mse

    def test_records_categorical_columns(self, regression_data):
        X_train, y_train, *_ = regression_data
        model = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        assert model.categorical_features == ["store"]

    def test_predict_reorders_columns(self, regression_data):
        X_train, y_train, X_test, _ = regression_data
        model = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        shuffled = X_test[["store", "x2", "x1"]]
        assert np.allclose(model.predict(shuffled), model.predict(X_test))

    def test_save_load_round_trip(self, regression_data, tmp_path):
        X_train, y_train, X_test, _ = regression_data
        model = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        model.save(tmp_path / "m")
        loaded = LightGBMForecaster()
        loaded.load(tmp_path / "m")
        assert np.allclose(loaded.predict(X_test), model.predict(X_test))
        assert loaded.feature_names == ["x1", "x2", "store"]

    def test_clone_is_unfitted_with_same_config(self, regression_data):
        X_train, y_train, *_ = regression_data
        model = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        copy = model.clone()
        assert not copy.is_fitted
        assert copy.config == model.config

    def test_predict_before_fit_raises(self, regression_data):
        with pytest.raises(RuntimeError, match="not fitted"):
            LightGBMForecaster().predict(regression_data[2])

    def test_custom_objective_trains_and_respects_weights(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        model = LightGBMForecaster(config=FAST, use_custom_objective=True)
        model.fit(X_train, y_train, X_test, y_test, sample_weight=np.ones(len(y_train)))
        preds = model.predict(X_test)
        assert np.mean((preds - y_test) ** 2) < np.mean((y_train.mean() - y_test) ** 2)

    def test_interval_is_the_documented_heuristic(self, regression_data):
        X_train, y_train, X_test, _ = regression_data
        model = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        pred, lower, upper = model.predict_interval(X_test[:5], confidence=0.95)
        z = 1.959963984540054
        assert np.allclose(upper, pred + z * 0.2 * pred)
        assert (lower <= pred).all()

    def test_deterministic(self, regression_data):
        X_train, y_train, X_test, _ = regression_data
        a = LightGBMForecaster(config=FAST).fit(X_train, y_train).predict(X_test)
        b = LightGBMForecaster(config=FAST).fit(X_train, y_train).predict(X_test)
        assert np.array_equal(a, b)


class TestEnsemble:
    def test_weights_are_optimised_and_normalised(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        good = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        weak = LightGBMForecaster(
            config=FAST.model_copy(update={"n_estimators": 2, "learning_rate": 0.01})
        ).fit(X_train, y_train)
        ensemble = EnsembleForecaster(EnsembleConfig(optimize_weights=True))
        ensemble.add_model("good", good).add_model("weak", weak)
        ensemble.fit(X_train, y_train, X_test, y_test)
        assert sum(ensemble.weights.values()) == pytest.approx(1.0)
        assert ensemble.weights["good"] > ensemble.weights["weak"]
        preds, spread = ensemble.predict(X_test, return_std=True)
        assert preds.shape == spread.shape == y_test.shape

    def test_unfitted_member_is_rejected(self):
        with pytest.raises(ValueError, match="must be fitted"):
            EnsembleForecaster().add_model("m", LightGBMForecaster())

    def test_save_load(self, regression_data, tmp_path):
        X_train, y_train, X_test, _ = regression_data
        member = LightGBMForecaster(config=FAST).fit(X_train, y_train)
        ensemble = EnsembleForecaster(EnsembleConfig(optimize_weights=False))
        ensemble.add_model("lgb", member).fit(X_train, y_train)
        ensemble.save(tmp_path / "ens")
        loaded = EnsembleForecaster()
        loaded.load(tmp_path / "ens")
        assert np.allclose(loaded.predict(X_test), ensemble.predict(X_test))

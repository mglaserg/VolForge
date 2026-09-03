import numpy as np
import pandas as pd
import pytest

from volforge.forecasting import (
    fit_xgboost,
    latest_xgboost_forecasts,
    xgboost_available,
)

pytestmark = pytest.mark.skipif(not xgboost_available(), reason="xgboost optional dependency not installed")


def _history(n=180, seed=13):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    rv3 = np.maximum(0.025 + 0.006 * np.sin(np.arange(n) / 15) + rng.normal(0, 0.0015, n), 0.002)
    rv9 = pd.Series(rv3).rolling(5, min_periods=1).mean().to_numpy()
    rv30 = pd.Series(rv9).rolling(10, min_periods=1).mean().to_numpy()
    mfiv = rv30 + 0.008 + 0.001 * np.cos(np.arange(n) / 11)
    forward = np.maximum(0.003 + 0.25 * rv3 + 0.30 * rv9 + 0.35 * rv30 + rng.normal(0, 0.0008, n), 1e-5)
    return pd.DataFrame({
        "date": dates,
        "mfiv_var": mfiv,
        "trailing_rv_var": rv30,
        "rv_var_3": rv3,
        "rv_var_9": rv9,
        "rv_var_30": rv30,
        "vrp": mfiv - rv30,
        "forward_rv_var": forward,
    })


def test_xgboost_fit_uses_only_explicit_non_forward_features():
    hist = _history()
    fit = fit_xgboost(hist, min_train=80)
    assert fit.nobs == len(hist)
    assert all("forward" not in col for col in fit.feature_cols)
    pred = fit.predict(hist.tail(3))
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()


def test_latest_xgboost_includes_mean_and_q70():
    hist = _history()
    forecasts, importance = latest_xgboost_forecasts(hist, min_train=80, quantiles=(0.70,))
    assert set(forecasts["model"]) == {"XGBoost", "XGBoost q70"}
    assert np.isfinite(forecasts["forecast_rv_var"]).all()
    assert np.allclose(forecasts["expected_vrp"], forecasts["mfiv_var"] - forecasts["forecast_rv_var"])
    assert not importance.empty

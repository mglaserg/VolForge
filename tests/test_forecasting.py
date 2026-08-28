import numpy as np
import pandas as pd

from volforge.forecasting import (
    fit_har,
    fit_heavy_rm,
    forecast_metrics,
    latest_model_forecasts,
    walk_forward_forecasts,
)


def _synthetic_history(n=260, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    base = 0.03 + 0.006 * np.sin(np.arange(n) / 17.0)
    rv3 = np.maximum(base + rng.normal(0, 0.002, n), 0.002)
    rv9 = np.maximum(0.7 * base + 0.3 * pd.Series(rv3).rolling(5, min_periods=1).mean().to_numpy(), 0.002)
    rv30 = np.maximum(0.6 * base + 0.4 * pd.Series(rv9).rolling(10, min_periods=1).mean().to_numpy(), 0.002)
    forward = 0.002 + 0.20 * rv3 + 0.30 * rv9 + 0.45 * rv30
    daily_rm = np.maximum(rv3 / 252.0, 1e-8)
    mfiv = forward + 0.008
    return pd.DataFrame({
        "date": dates,
        "rv_var_3": rv3,
        "rv_var_9": rv9,
        "rv_var_30": rv30,
        "trailing_rv_var": rv30,
        "forward_rv_var": forward,
        "daily_rm": daily_rm,
        "mfiv_var": mfiv,
    })


def test_har_recovers_direct_forward_variance_relationship():
    hist = _synthetic_history(200)
    fit = fit_har(hist)
    assert fit.nobs == 200
    assert np.isclose(fit.intercept, 0.002, atol=1e-8)
    assert np.allclose(fit.coefficients, (0.20, 0.30, 0.45), atol=1e-8)
    pred = fit.predict(hist.tail(5))
    assert np.allclose(pred, hist["forward_rv_var"].tail(5), atol=1e-8)


def test_heavy_rm_estimates_positive_stable_model_and_forecasts():
    rng = np.random.default_rng(11)
    n = 500
    omega, alpha, beta = 1.5e-5, 0.12, 0.78
    rm = np.empty(n)
    mu = np.empty(n)
    mu[0] = omega / (1 - alpha - beta)
    rm[0] = mu[0]
    for t in range(1, n):
        mu[t] = omega + alpha * rm[t - 1] + beta * mu[t - 1]
        rm[t] = mu[t] * rng.exponential(1.0)

    fit = fit_heavy_rm(pd.Series(rm))
    assert fit.success
    assert fit.omega > 0
    assert fit.alpha >= 0
    assert fit.beta >= 0
    assert fit.persistence < 0.999
    assert fit.nobs == n
    forecast = fit.forecast_horizon_variance(30)
    assert np.isfinite(forecast)
    assert forecast > 0


def test_latest_forecasts_include_persistence_har_and_heavy():
    hist = _synthetic_history(180)
    out = latest_model_forecasts(hist, target_days=30)
    assert {"Persistence", "HAR 3/9/30", "HEAVY-RM"} <= set(out["model"])
    assert (out["forecast_rv_var"] >= 0).all()
    assert np.allclose(out["expected_vrp"], out["mfiv_var"] - out["forecast_rv_var"])


def test_walk_forward_is_purged_and_scores_models():
    hist = _synthetic_history(260)
    out = walk_forward_forecasts(hist, target_days=30, min_train=60, refit_every=40)
    assert not out.empty
    assert {"Persistence", "HAR 3/9/30", "HEAVY-RM"} <= set(out["model"])
    assert out["date"].min() > hist["date"].iloc[60]
    metrics = forecast_metrics(out)
    assert {"model", "n", "mse", "mae", "qlike", "bias"} <= set(metrics.columns)
    assert (metrics["n"] > 0).all()
    assert np.isfinite(metrics["qlike"]).all()

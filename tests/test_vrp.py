import numpy as np
import pandas as pd

from volforge.vrp import forward_vrp_label, rolling_zscore, vrp_features


def test_forward_vrp_is_variance_spread():
    idx = pd.date_range("2025-01-01", periods=3)
    mfiv = pd.Series([0.04, 0.05, 0.06], index=idx)
    fwd = pd.Series([0.03, 0.055, 0.04], index=idx)
    out = forward_vrp_label(mfiv, fwd)
    assert np.allclose(out, [0.01, -0.005, 0.02])


def test_zscore_does_not_use_future_observations():
    idx = pd.date_range("2025-01-01", periods=8)
    base = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=idx, dtype=float)
    z1 = rolling_zscore(base, window=4, min_periods=3)
    changed = base.copy()
    changed.iloc[-1] = 10000
    z2 = rolling_zscore(changed, window=4, min_periods=3)
    assert np.allclose(z1.iloc[:-1], z2.iloc[:-1], equal_nan=True)


def test_vrp_feature_columns():
    idx = pd.date_range("2025-01-01", periods=40)
    mfiv = pd.Series(np.linspace(0.04, 0.08, 40), index=idx)
    rv = pd.Series(np.linspace(0.03, 0.05, 40), index=idx)
    f = vrp_features(mfiv, rv, z_window=20, min_periods=5)
    assert {"mfiv_var", "trailing_rv_var", "vrp", "vrp_z", "vrp_percentile", "mfiv_z", "mfiv_percentile"} <= set(f.columns)


def test_realized_term_structure_and_vol_of_vol():
    from volforge.vrp import realized_term_structure, vol_of_vol

    idx = pd.date_range("2025-01-01", periods=80, freq="B")
    daily = pd.Series(np.linspace(0.00005, 0.0002, 80), index=idx)
    ts = realized_term_structure(daily, windows=(3, 9, 30, 60), basis="trading")
    assert {"rv_3", "rv_9", "rv_30", "rv_60", "rv_slope_3_30", "rv_ratio_9_60"} <= set(ts.columns)
    vov = vol_of_vol(pd.Series(np.linspace(0.03, 0.08, 80), index=idx), window=10, min_periods=5)
    assert vov.notna().sum() > 0

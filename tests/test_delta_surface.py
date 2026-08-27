import numpy as np
import pandas as pd
from scipy.stats import norm

from volforge.data.schema import add_derived_columns
from volforge.delta_surface import (
    DeltaVolSurface,
    build_delta_surface,
    constant_tenor_delta_slice,
    delta_lump_scores,
    delta_surface_change_features,
)


def _bs_price(S, K, sigma, T, r, call):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _chain(days=(20, 30, 40, 60), base_vol=0.20, put_skew=0.0):
    spot = 100.0
    r = 0.03
    quote = pd.Timestamp("2026-08-25 16:00", tz="America/New_York").tz_convert("UTC")
    rows = []
    for dte in days:
        expiry = quote + pd.Timedelta(days=dte)
        T = dte / 365.25
        F = spot * np.exp(r * T)
        for K in np.arange(72.0, 131.0, 0.5):
            k = np.log(K / F)
            sigma = base_vol + put_skew * max(-k, 0.0)
            for right, call in (("C", True), ("P", False)):
                price = _bs_price(spot, K, sigma, T, r, call)
                rows.append({
                    "symbol": "SPY", "quote_time": quote, "expiry": expiry,
                    "strike": K, "right": right,
                    "bid": max(price - 0.0005, 1e-6), "ask": price + 0.0005,
                    "underlying_price": spot,
                })
    return add_derived_columns(pd.DataFrame(rows))


def test_flat_surface_has_delta_ratios_near_one():
    surface = build_delta_surface(_chain(), dte_range=(7, 90))
    target = constant_tenor_delta_slice(surface, 30)
    for col in ("delta_ratio_10p", "delta_ratio_15p", "delta_ratio_25p", "delta_ratio_25c", "delta_ratio_15c", "delta_ratio_10c"):
        assert np.isfinite(target[col])
        assert abs(target[col] - 1.0) < 0.02


def test_put_skew_shows_up_in_delta_ratios():
    surface = build_delta_surface(_chain(put_skew=0.55), dte_range=(7, 90))
    target = constant_tenor_delta_slice(surface, 30)
    assert target["delta_ratio_10p"] > target["delta_ratio_15p"] > target["delta_ratio_25p"] > 1.0
    assert abs(target["delta_ratio_25c"] - 1.0) < 0.03


def test_constant_tenor_uses_total_variance_interpolation():
    # Build a minimal hand-constructed surface with ATM 18% at 20d and 22% at 40d.
    iv = pd.DataFrame({
        "atm_iv": [0.18, 0.22],
        "iv_10p": [0.18, 0.22], "iv_15p": [0.18, 0.22], "iv_25p": [0.18, 0.22],
        "iv_25c": [0.18, 0.22], "iv_15c": [0.18, 0.22], "iv_10c": [0.18, 0.22],
    }, index=pd.Index([20.0, 40.0], name="dte"))
    ratios = pd.DataFrame({
        c.replace("iv_", "delta_ratio_"): [1.0, 1.0]
        for c in iv.columns if c != "atm_iv"
    }, index=iv.index)
    surface = DeltaVolSurface("SPY", pd.Timestamp("2026-08-25", tz="UTC"), (0.10, 0.15, 0.25), iv, ratios, pd.Series(index=iv.index, dtype="datetime64[ns]"))
    got = constant_tenor_delta_slice(surface, 30)["atm_iv"]
    t20, t30, t40 = 20/365.25, 30/365.25, 40/365.25
    expected_w = 0.5 * (0.18**2 * t20) + 0.5 * (0.22**2 * t40)
    expected = np.sqrt(expected_w / t30)
    assert np.isclose(got, expected)


def test_lump_detector_flags_middle_expiry():
    idx = pd.Index([20.0, 30.0, 40.0], name="dte")
    iv = pd.DataFrame({"atm_iv": [0.20]*3, "iv_25p": [0.22, 0.28, 0.22]}, index=idx)
    ratios = pd.DataFrame({"delta_ratio_25p": [1.10, 1.40, 1.10]}, index=idx)
    surface = DeltaVolSurface("SPY", pd.Timestamp("2026-08-25", tz="UTC"), (0.25,), iv, ratios, pd.Series(index=idx, dtype="datetime64[ns]"))
    lumps = delta_lump_scores(surface)
    middle = lumps.loc[np.isclose(lumps["dte"], 30.0), "delta_lump_25p"].iloc[0]
    assert np.isclose(middle, 0.30)


def test_surface_change_decomposition_is_transparent():
    hist = pd.DataFrame({
        "underlying_price": [100.0, 101.0],
        "atm_iv": [0.20, 0.22],
        "iv_10p": [0.28, 0.31], "iv_15p": [0.25, 0.28], "iv_25p": [0.23, 0.25],
        "iv_25c": [0.21, 0.23], "iv_15c": [0.205, 0.225], "iv_10c": [0.202, 0.222],
    })
    out = delta_surface_change_features(hist)
    assert np.isclose(out["surface_parallel_shift"].iloc[1], 0.02)
    assert np.isfinite(out["surface_put_skew_change"].iloc[1])
    assert np.isfinite(out["surface_downside_convexity_change"].iloc[1])
    assert np.isclose(out["surface_spot_log_return"].iloc[1], np.log(1.01))

import numpy as np
import pandas as pd
from scipy.stats import norm

from volforge.dashboard import build_dashboard_snapshot, normalise_intraday_bars, prepare_vrp_history
from volforge.data.schema import add_derived_columns


def _bs_price(S, K, sigma, T, r, call):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _chain(days_list=(20, 40), sigma=0.20, spot=100.0, r=0.03):
    quote = pd.Timestamp("2026-08-25 16:00", tz="America/New_York").tz_convert("UTC")
    rows = []
    for days in days_list:
        expiry = quote + pd.Timedelta(days=days)
        T = days / 365.25
        for K in np.arange(40.0, 181.0, 1.0):
            for right, call in (("C", True), ("P", False)):
                p = _bs_price(spot, K, sigma, T, r, call)
                rows.append({
                    "symbol": "TEST", "quote_time": quote, "expiry": expiry,
                    "strike": K, "right": right,
                    "bid": max(p - 0.0005, 1e-6), "ask": p + 0.0005,
                    "underlying_price": spot,
                })
    return add_derived_columns(pd.DataFrame(rows))


def _bars():
    # Two observations per business session over enough calendar history for RV30.
    days = pd.bdate_range("2026-06-15", "2026-08-25", tz="America/New_York")
    rows = []
    price = 100.0
    for i, day in enumerate(days):
        open_ts = day.normalize() + pd.Timedelta(hours=9, minutes=30)
        close_ts = day.normalize() + pd.Timedelta(hours=16)
        price *= np.exp(0.0003 * np.sin(i / 3))
        rows.append((open_ts, price))
        price *= np.exp(0.002 * np.cos(i / 4))
        rows.append((close_ts, price))
    return pd.DataFrame(rows, columns=["timestamp", "close"])


def test_normalise_intraday_bars_accepts_datetime_index_and_close():
    idx = pd.date_range("2026-08-25 09:30", periods=3, freq="5min", tz="America/New_York")
    raw = pd.DataFrame({"Close": [100.0, 100.1, 100.2]}, index=idx)
    got = normalise_intraday_bars(raw)
    assert list(got.columns) == ["timestamp", "close"]
    assert str(got["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_build_dashboard_snapshot_produces_mfiv_and_realized_curves():
    snap = build_dashboard_snapshot(_chain(), _bars(), target_days=30, min_mfiv_strikes=20)
    assert snap.symbol == "TEST"
    assert len(snap.mfiv_curve) == 2
    assert np.isclose(snap.target_mfiv.implied_volatility, 0.20, atol=0.02)
    assert np.isfinite(snap.trailing_target_variance)
    assert set(snap.realized_curve["days"]).issuperset({3, 9, 30})
    assert snap.chain_quality["crossed_rows"] == 0


def test_prepare_vrp_history_adds_no_lookahead_features_and_forward_label():
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    mfiv = pd.Series(np.linspace(0.03, 0.06, len(dates)))
    rv = pd.Series(np.linspace(0.025, 0.04, len(dates)))
    hist = pd.DataFrame({
        "date": dates,
        "mfiv_var": mfiv,
        "trailing_rv_var": rv,
        "forward_rv_var": rv + 0.002,
    })
    out = prepare_vrp_history(hist, z_window=20, min_periods=5)
    assert {"vrp", "vrp_z", "vrp_percentile", "mfiv_vol", "vol_of_vol", "forward_vrp"} <= set(out.columns)
    assert np.isclose(out["forward_vrp"].iloc[-1], mfiv.iloc[-1] - (rv.iloc[-1] + 0.002))

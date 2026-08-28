import numpy as np
import pandas as pd

from volforge import vix_curve


def test_load_cboe_index_parses_official_history(monkeypatch):
    raw = pd.DataFrame({
        "DATE": ["2026-08-26", "2026-08-27"],
        "CLOSE": [14.25, 14.48],
    })
    monkeypatch.setattr(vix_curve, "_read_url_csv", lambda url: raw)

    got = vix_curve.load_cboe_index("vix")

    assert got.name == "VIX"
    assert got.index.tolist() == [pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27")]
    assert got.tolist() == [14.25, 14.48]


def test_vix_curve_uses_raw_spread_for_backwardation_and_prior_10_day_zscore():
    dates = pd.date_range("2026-08-03", periods=12, freq="B")
    vix = pd.Series(20.0, index=dates)
    spreads = pd.Series(np.arange(12, dtype=float), index=dates)
    vix3m = vix + spreads

    got = vix_curve.vix_curve_features(vix, vix3m)

    assert got["vix_curve_z_10d"].iloc[:10].isna().all()
    expected = (10.0 - np.mean(np.arange(10, dtype=float))) / np.std(np.arange(10, dtype=float), ddof=1)
    assert np.isclose(got["vix_curve_z_10d"].iloc[10], expected)
    assert not got["vix_backwardation"].any()


def test_backwardation_is_independent_of_zscore_sign():
    dates = pd.date_range("2026-08-03", periods=12, freq="B")
    vix = pd.Series(20.0, index=dates)
    vix3m = pd.Series(21.0, index=dates)
    vix3m.iloc[-1] = 19.5

    got = vix_curve.vix_curve_features(vix, vix3m)

    assert bool(got["vix_backwardation"].iloc[-1])
    assert got["vix3m_minus_vix"].iloc[-1] == -0.5


def test_load_vix_curve_history_aligns_cboe_series(monkeypatch):
    dates = pd.date_range("2026-08-03", periods=12, freq="B")
    values = {
        "VIX": pd.Series(20.0, index=dates),
        "VIX3M": pd.Series(np.linspace(21.0, 19.0, len(dates)), index=dates),
    }
    monkeypatch.setattr(vix_curve, "load_cboe_index", lambda symbol: values[symbol])

    got = vix_curve.load_vix_curve_history()

    assert {"VIX", "VIX3M", "vix3m_minus_vix", "vix_curve_z_10d", "vix_backwardation"} <= set(got.columns)
    assert len(got) == 12

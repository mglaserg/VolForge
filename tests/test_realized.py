import numpy as np
import pandas as pd

from volforge.realized import daily_integrated_variance, forward_integrated_variance, rolling_integrated_variance


def test_daily_integrated_variance_includes_overnight_once():
    ts = pd.to_datetime([
        "2026-08-24 13:30Z", "2026-08-24 13:35Z", "2026-08-24 20:00Z",
        "2026-08-25 13:30Z", "2026-08-25 13:35Z", "2026-08-25 20:00Z",
    ], utc=True)
    px = np.array([100.0, 101.0, 102.0, 104.0, 103.0, 105.0])
    bars = pd.DataFrame({"timestamp": ts, "close": px})
    rv = daily_integrated_variance(bars, include_overnight=True)

    day2_expected = np.log(104 / 102) ** 2 + np.log(103 / 104) ** 2 + np.log(105 / 103) ** 2
    assert np.isclose(rv.iloc[1], day2_expected)


def test_forward_calendar_variance_excludes_current_day():
    idx = pd.to_datetime(["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"])
    s = pd.Series([1e-4, 2e-4, 3e-4, 4e-4], index=idx)
    fwd = forward_integrated_variance(s, 2, basis="calendar")
    assert np.isclose(fwd.iloc[0], (2e-4 + 3e-4) * 365.25 / 2)
    assert np.isclose(fwd.iloc[1], (3e-4 + 4e-4) * 365.25 / 2)


def test_trading_window_annualization():
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    s = pd.Series(1e-4, index=idx)
    rv = rolling_integrated_variance(s, 5, basis="trading")
    assert np.isclose(rv.iloc[-1], 1e-4 * 252)

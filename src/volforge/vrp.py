"""Forward volatility-risk-premium features and labels.

All internal calculations are in variance units. Volatility-point transforms
belong in presentation code, not in the target definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "forward_vrp_label", "vrp_features", "rolling_zscore", "rolling_percentile",
    "realized_term_structure", "vol_of_vol",
]


def forward_vrp_label(mfiv_variance: pd.Series, forward_rv_variance: pd.Series) -> pd.Series:
    """Ground-truth forward VRP: variance sold today minus future realized variance."""
    a, b = pd.Series(mfiv_variance, dtype=float).align(pd.Series(forward_rv_variance, dtype=float), join="inner")
    out = a - b
    out.name = "forward_vrp"
    return out


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int | None = None) -> pd.Series:
    """Historical z-score using only observations strictly before each row."""
    s = pd.Series(series, dtype=float)
    minp = min_periods if min_periods is not None else max(20, window // 4)
    hist = s.shift(1)
    mean = hist.rolling(window, min_periods=minp).mean()
    std = hist.rolling(window, min_periods=minp).std(ddof=1)
    out = (s - mean) / std.replace(0.0, np.nan)
    out.name = f"{s.name or 'value'}_z"
    return out


def rolling_percentile(series: pd.Series, window: int = 252, min_periods: int | None = None) -> pd.Series:
    """Percentile of today's value versus the strictly prior rolling history."""
    s = pd.Series(series, dtype=float)
    minp = min_periods if min_periods is not None else max(20, window // 4)

    def pct(arr):
        current = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(current) or len(hist) < minp:
            return np.nan
        return float(np.mean(hist <= current))

    # Include current in the rolling array, but compare only with arr[:-1].
    out = s.rolling(window + 1, min_periods=minp + 1).apply(pct, raw=True)
    out.name = f"{s.name or 'value'}_pct"
    return out


def vrp_features(
    mfiv_variance: pd.Series,
    trailing_rv_variance: pd.Series,
    *,
    z_window: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build the first no-lookahead VRP feature set from aligned daily series."""
    mfiv, rv = pd.Series(mfiv_variance, dtype=float).align(pd.Series(trailing_rv_variance, dtype=float), join="inner")
    current = mfiv - rv
    current.name = "vrp"
    frame = pd.DataFrame({
        "mfiv_var": mfiv,
        "trailing_rv_var": rv,
        "vrp": current,
    })
    frame["vrp_z"] = rolling_zscore(current, z_window, min_periods)
    frame["vrp_percentile"] = rolling_percentile(current, z_window, min_periods)
    frame["mfiv_z"] = rolling_zscore(mfiv.rename("mfiv"), z_window, min_periods)
    frame["mfiv_percentile"] = rolling_percentile(mfiv.rename("mfiv"), z_window, min_periods)
    return frame


def realized_term_structure(
    daily_variance: pd.Series,
    windows=(3, 9, 30, 60, 180),
    *,
    basis: str = "trading",
) -> pd.DataFrame:
    """Build trailing realized-variance/volatility term-structure features."""
    from .realized import integrated_volatility, rolling_integrated_variance

    out = pd.DataFrame(index=pd.Series(daily_variance).index)
    for w in windows:
        var = rolling_integrated_variance(daily_variance, int(w), basis=basis)
        out[f"rv_var_{w}"] = var
        out[f"rv_{w}"] = integrated_volatility(var)
    windows = tuple(int(w) for w in windows)
    for short, long in ((3, 30), (9, 30), (9, 60), (30, 60)):
        if short in windows and long in windows:
            out[f"rv_slope_{short}_{long}"] = out[f"rv_{short}"] - out[f"rv_{long}"]
            out[f"rv_ratio_{short}_{long}"] = out[f"rv_{short}"] / out[f"rv_{long}"].replace(0.0, np.nan)
    return out


def vol_of_vol(
    implied_variance: pd.Series,
    window: int = 20,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling volatility of changes in implied volatility.

    Input stays in variance units; it is transformed to volatility before
    differencing so the output has an intuitive 'volatility of vol changes'
    interpretation rather than depending on the arbitrary scale of variance.
    """
    s = pd.Series(implied_variance, dtype=float)
    iv = np.sqrt(s.clip(lower=0.0))
    changes = iv.diff()
    minp = min_periods if min_periods is not None else max(5, window // 4)
    out = changes.rolling(window, min_periods=minp).std(ddof=1)
    out.name = f"vol_of_vol_{window}"
    return out

"""Interpretable surface features.

Everything here is a finite-difference read off the fixed grid rather than a
transform of the SVI parameters. That is deliberate: raw SVI parameters are not
comparable across days. (a, b, rho, m, sigma) trade off against one another --
two very different parameter vectors can describe nearly the same smile -- so a
time series of rho is not a time series of skew. Grid-based features are
comparable by construction because they are measurements of the fitted curve at
fixed, meaningful points.

Skew is quoted per unit log-moneyness in vol points, so a value of -0.60 means
implied vol rises about 6 vol points as you move 0.1 down in log-moneyness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["surface_features", "feature_panel", "standardize"]


def _interp_iv(surface, tenor_days, k):
    """Bilinear read of the IV grid at an arbitrary (tenor, k)."""
    ti = np.interp(tenor_days, surface.tenor_days, np.arange(len(surface.tenor_days)))
    i0, i1 = int(np.floor(ti)), min(int(np.ceil(ti)), len(surface.tenor_days) - 1)
    wt = ti - i0
    row = (1 - wt) * surface.iv[i0] + wt * surface.iv[i1]
    return float(np.interp(k, surface.k_grid, row))


def surface_features(surface, skew_width: float = 0.10) -> dict:
    """Extract the standard descriptive features from one daily Surface.

    Parameters
    ----------
    skew_width : the log-moneyness offset used for slope and curvature. 0.10 is
        roughly the 25-delta region for a 30-day SPY smile.
    """
    f = {}
    tenors = surface.tenor_days
    d = skew_width

    for t in tenors:
        tag = f"{int(t)}d"
        atm = _interp_iv(surface, t, 0.0)
        down = _interp_iv(surface, t, -d)
        up = _interp_iv(surface, t, +d)

        f[f"atm_{tag}"] = atm
        # Central slope: negative for a normal equity smile.
        f[f"skew_{tag}"] = (up - down) / (2 * d)
        # Downside steepness on its own -- often more informative than the
        # symmetric slope, because equity smiles are not symmetric.
        f[f"put_skew_{tag}"] = (atm - down) / d
        f[f"call_skew_{tag}"] = (up - atm) / d
        # Second difference: smile convexity.
        f[f"curvature_{tag}"] = (up - 2 * atm + down) / (d * d)
        # Wing asymmetry: how much steeper the downside is than the upside.
        f[f"wing_asym_{tag}"] = (atm - down) - (up - atm)
        # Normalised versions, comparable across vol regimes.
        if atm > 0:
            f[f"skew_norm_{tag}"] = f[f"skew_{tag}"] / atm
            f[f"curvature_norm_{tag}"] = f[f"curvature_{tag}"] / atm

    # Term structure: short vs long ATM, in both difference and ratio form.
    t_lo, t_hi = float(tenors.min()), float(tenors.max())
    atm_lo = _interp_iv(surface, t_lo, 0.0)
    atm_hi = _interp_iv(surface, t_hi, 0.0)
    f["term_slope"] = atm_hi - atm_lo
    f["term_ratio"] = atm_hi / atm_lo if atm_lo > 0 else np.nan

    if len(tenors) >= 3:
        mid = float(np.median(tenors))
        atm_mid = _interp_iv(surface, mid, 0.0)
        # Positive = humped term structure, negative = bowed.
        f["term_curvature"] = atm_mid - 0.5 * (atm_lo + atm_hi)

    f["spot"] = surface.spot
    f["n_slices"] = float(surface.n_slices_used)
    f["calendar_repair"] = surface.calendar_repair
    f["is_clean"] = float(surface.is_clean)
    return f


def feature_panel(surfaces) -> pd.DataFrame:
    """Daily feature history from a list of Surfaces."""
    surfaces = sorted(surfaces, key=lambda s: s.trade_date)
    return pd.DataFrame(
        [surface_features(s) for s in surfaces],
        index=pd.DatetimeIndex([s.trade_date for s in surfaces], name="trade_date"),
    )


def standardize(df: pd.DataFrame, window: int | None = 252, min_periods: int = 60,
                robust: bool = True) -> pd.DataFrame:
    """Historical z-scores, expanding or rolling.

    `robust=True` uses median and a MAD-based scale, which matters here because
    vol features have fat tails: a single crisis day will otherwise inflate the
    standard deviation enough to suppress every subsequent signal.

    Uses only trailing data at each point, so the result is free of look-ahead.
    """
    if window is None:
        roll = df.expanding(min_periods=min_periods)
    else:
        roll = df.rolling(window, min_periods=min_periods)

    if robust:
        center = roll.median()
        mad = (df - center).abs().rolling(
            window or len(df), min_periods=min_periods).median()
        scale = 1.4826 * mad
    else:
        center = roll.mean()
        scale = roll.std()

    scale = scale.replace(0.0, np.nan)
    return ((df - center) / scale).replace([np.inf, -np.inf], np.nan)

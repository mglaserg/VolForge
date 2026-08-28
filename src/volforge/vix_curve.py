"""Cboe VIX term-structure features used as a VolForge regime input."""

from __future__ import annotations

from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

from .vrp import rolling_zscore


CBOE_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
VIX_CURVE_Z_WINDOW = 10


def _read_url_csv(url: str) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 VolForge/0.4"})
    with urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    return pd.read_csv(StringIO(text))


def load_cboe_index(symbol: str) -> pd.Series:
    """Load official Cboe daily closes for one volatility index."""
    symbol = symbol.upper()
    df = _read_url_csv(CBOE_HISTORY_URL.format(symbol=symbol))
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "DATE" not in df.columns or "CLOSE" not in df.columns:
        raise ValueError(f"Unexpected Cboe columns for {symbol}: {list(df.columns)}")

    series = pd.Series(
        pd.to_numeric(df["CLOSE"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["DATE"], errors="coerce"),
        name=symbol,
    )
    series = series[~series.index.isna()].dropna().sort_index()
    series.index = series.index.tz_localize(None)
    return series


def vix_curve_features(vix: pd.Series, vix3m: pd.Series) -> pd.DataFrame:
    """Build the VIX3M-minus-VIX spread and its prior-only 10-session z-score.

    ``VIX3M - VIX < 0`` is the actual backwardation flag.  The z-score does not
    define backwardation; it measures how unusual today's curve spread is versus
    the ten strictly prior aligned observations.
    """
    frame = pd.concat(
        [pd.Series(vix, dtype=float).rename("VIX"), pd.Series(vix3m, dtype=float).rename("VIX3M")],
        axis=1,
        join="inner",
    ).dropna()
    frame = frame.sort_index()
    frame["vix3m_minus_vix"] = frame["VIX3M"] - frame["VIX"]
    frame["vix_backwardation"] = frame["vix3m_minus_vix"] < 0.0
    frame["vix_curve_z_10d"] = rolling_zscore(
        frame["vix3m_minus_vix"].rename("vix_curve"),
        window=VIX_CURVE_Z_WINDOW,
        min_periods=VIX_CURVE_Z_WINDOW,
    )
    return frame


def load_vix_curve_history() -> pd.DataFrame:
    """Load VIX/VIX3M from Cboe and return the aligned curve feature history."""
    return vix_curve_features(load_cboe_index("VIX"), load_cboe_index("VIX3M"))

"""The canonical option-chain schema.

Every vendor adapter returns a DataFrame in *this* shape. Nothing downstream --
cleaning, forward extraction, calibration -- ever sees a vendor-specific
column name. Swapping yfinance for ORATS is then a new adapter file, not a
rewrite of the pipeline.

Deliberately absent: any vendor's implied volatility, forward, or greeks. We
compute those ourselves from mid prices and our own parity-implied forward, so
that residuals reflect market structure rather than a vendor's rate and
dividend assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "SETTLEMENT_TIMES",
    "validate_chain",
    "add_derived_columns",
    "expiry_datetime",
]

REQUIRED_COLUMNS = {
    "symbol": "str",            # underlying ticker, e.g. SPY
    "quote_time": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns, UTC]",   # exact settlement instant, not midnight
    "strike": "float64",
    "right": "str",             # 'C' or 'P'
    "bid": "float64",
    "ask": "float64",
    "underlying_price": "float64",
}

OPTIONAL_COLUMNS = {
    "last": "float64",
    "volume": "float64",
    "open_interest": "float64",
    "last_trade_time": "datetime64[ns, UTC]",
    "vendor_iv": "float64",     # kept only as a sanity check; never fitted
    "source": "str",
}

# Settlement convention by underlying. Most US ETF options and SPX weeklies are
# PM-settled at the 16:00 ET close. AM-settled contracts (SPX standard monthlies,
# ticker SPXW vs SPX) open-print at 09:30 ET, which materially changes T for
# short-dated slices.
SETTLEMENT_TIMES = {
    "default": ("16:00", "America/New_York"),
    "SPX_AM": ("09:30", "America/New_York"),
}

DAYS_PER_YEAR = 365.25


def expiry_datetime(expiry_date, convention: str = "default") -> pd.Timestamp:
    """Turn a calendar expiry date into a tz-aware settlement instant in UTC."""
    time_str, tz = SETTLEMENT_TIMES[convention]
    d = pd.Timestamp(expiry_date).normalize()
    if d.tz is not None:
        d = d.tz_localize(None)
    hh, mm = map(int, time_str.split(":"))
    local = d + pd.Timedelta(hours=hh, minutes=mm)
    return local.tz_localize(tz, nonexistent="shift_forward").tz_convert("UTC")


def validate_chain(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Check required columns exist and are typed sanely. Returns the frame."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"chain is missing required columns: {sorted(missing)}")

    bad_right = set(df["right"].unique()) - {"C", "P"}
    if bad_right:
        raise ValueError(f"`right` must be 'C' or 'P', found {bad_right}")

    for col in ("quote_time", "expiry"):
        if not isinstance(df[col].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"`{col}` must be timezone-aware; got {df[col].dtype}")

    if strict and (df["expiry"] <= df["quote_time"]).any():
        n = int((df["expiry"] <= df["quote_time"]).sum())
        raise ValueError(f"{n} rows have expiry at or before quote_time")

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach mid, spread, T and dte. Idempotent."""
    df = df.copy()

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rel_spread"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.inf)

    secs = (df["expiry"] - df["quote_time"]).dt.total_seconds()
    df["T"] = secs / (DAYS_PER_YEAR * 86400.0)
    df["dte"] = secs / 86400.0

    return df

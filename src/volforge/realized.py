"""High-frequency integrated realized-variance utilities.

The MFIV research target should use the same *calendar* clock as an option's
DTE. Trading-day windows remain useful descriptive features, so both bases are
supported explicitly rather than silently mixing 252- and 365-day conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "IntegratedVarianceSeries",
    "daily_integrated_variance",
    "rolling_integrated_variance",
    "forward_integrated_variance",
    "integrated_volatility",
]

CALENDAR_DAYS_PER_YEAR = 365.25
TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class IntegratedVarianceSeries:
    """Container for daily integrated variance and common trailing features."""

    daily_variance: pd.Series

    def trailing(self, window: int, *, basis: str = "calendar") -> pd.Series:
        return rolling_integrated_variance(self.daily_variance, window, basis=basis)

    def forward(self, horizon: int, *, basis: str = "calendar") -> pd.Series:
        return forward_integrated_variance(self.daily_variance, horizon, basis=basis)


def daily_integrated_variance(
    bars: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "close",
    session_tz: str = "America/New_York",
    include_overnight: bool = True,
) -> pd.Series:
    """Compute one integrated-variance observation per trading session.

    Intraday squared log returns are summed within each local session. If
    ``include_overnight`` is true, the log return from the prior session's last
    bar to the current session's first bar is added once. This prevents a
    close-only gap from disappearing from the realized-variance measure.
    """
    if timestamp_col not in bars or price_col not in bars:
        raise ValueError(f"bars must contain {timestamp_col!r} and {price_col!r}")
    df = bars[[timestamp_col, price_col]].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna().sort_values(timestamp_col)
    df = df[df[price_col] > 0]
    if len(df) < 2:
        return pd.Series(dtype="float64", name="integrated_variance")

    local = df[timestamp_col].dt.tz_convert(session_tz)
    df["session"] = pd.to_datetime(local.dt.date)
    df["log_price"] = np.log(df[price_col].astype(float))
    df["prev_log"] = df["log_price"].shift(1)
    df["prev_session"] = df["session"].shift(1)

    same_session = df["session"].eq(df["prev_session"])
    df["sq_return"] = np.where(
        same_session,
        (df["log_price"] - df["prev_log"]) ** 2,
        0.0,
    )
    daily = df.groupby("session", sort=True)["sq_return"].sum()

    if include_overnight:
        first = df.groupby("session", sort=True).head(1).copy()
        overnight = (first["log_price"] - first["prev_log"]) ** 2
        overnight = overnight.where(first["prev_log"].notna(), 0.0)
        daily = daily.add(pd.Series(overnight.to_numpy(), index=first["session"]), fill_value=0.0)

    daily.index = pd.DatetimeIndex(daily.index, name="date")
    daily.name = "integrated_variance"
    return daily.astype(float)


def rolling_integrated_variance(
    daily_variance: pd.Series,
    window: int,
    *,
    basis: str = "calendar",
) -> pd.Series:
    """Annualized trailing integrated variance over a calendar/trading window."""
    s = _prepare_daily(daily_variance)
    if window <= 0:
        raise ValueError("window must be positive")
    if basis == "calendar":
        total = s.rolling(f"{int(window)}D", closed="both").sum()
        out = total * (CALENDAR_DAYS_PER_YEAR / float(window))
    elif basis == "trading":
        total = s.rolling(int(window), min_periods=int(window)).sum()
        out = total * (TRADING_DAYS_PER_YEAR / float(window))
    else:
        raise ValueError("basis must be 'calendar' or 'trading'")
    out.name = f"rv_var_{window}{'c' if basis == 'calendar' else 't'}"
    return out


def forward_integrated_variance(
    daily_variance: pd.Series,
    horizon: int,
    *,
    basis: str = "calendar",
) -> pd.Series:
    """Annualized variance realized *after* each date over the future horizon.

    The current date is excluded: a snapshot taken at date ``t`` can therefore
    be joined directly to this series without leaking date-t realized returns
    into the label.
    """
    s = _prepare_daily(daily_variance)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if basis == "trading":
        cols = [s.shift(-i) for i in range(1, int(horizon) + 1)]
        total = pd.concat(cols, axis=1).sum(axis=1, min_count=int(horizon))
        out = total * (TRADING_DAYS_PER_YEAR / float(horizon))
    elif basis == "calendar":
        dates = s.index.to_numpy(dtype="datetime64[ns]")
        values = s.to_numpy(dtype=float)
        csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values, nan=0.0))])
        valid = np.concatenate([[0], np.cumsum(np.isfinite(values).astype(int))])
        out_values = np.full(len(s), np.nan, dtype=float)
        delta = np.timedelta64(int(horizon), "D")
        for i, d in enumerate(dates):
            # Strictly after t through t+horizon (inclusive).
            left = i + 1
            right = int(np.searchsorted(dates, d + delta, side="right"))
            if right <= left:
                continue
            count = valid[right] - valid[left]
            if count <= 0:
                continue
            out_values[i] = (csum[right] - csum[left]) * (CALENDAR_DAYS_PER_YEAR / float(horizon))
        out = pd.Series(out_values, index=s.index)
    else:
        raise ValueError("basis must be 'calendar' or 'trading'")

    out.name = f"forward_rv_var_{horizon}{'c' if basis == 'calendar' else 't'}"
    return out


def integrated_volatility(variance: pd.Series | np.ndarray | float):
    """Convert annualized integrated variance to volatility."""
    return np.sqrt(np.maximum(variance, 0.0))


def _prepare_daily(series: pd.Series) -> pd.Series:
    s = pd.Series(series, copy=True, dtype="float64")
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()

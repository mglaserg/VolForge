"""Pure data preparation for the Forward VRP dashboard.

The Streamlit entry point lives at the repository root in
``volforge_dashboard.py``.  Keeping calculations here (with no Streamlit
imports) makes the dashboard testable and reusable from notebooks/scripts.

The current snapshot intentionally compares 30-day MFIV with *trailing*
integrated realized variance.  That is a live feature/proxy, not the forward
VRP training label.  Forward VRP remains MFIV today minus variance realized
strictly after today and is only available once future data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data.schema import add_derived_columns, validate_chain
from .mfiv import ConstantTenorMFIV, constant_tenor_mfiv, mfiv_term_structure
from .realized import daily_integrated_variance, integrated_volatility, rolling_integrated_variance
from .vrp import forward_vrp_label, realized_term_structure, vol_of_vol, vrp_features

__all__ = [
    "DashboardSnapshot",
    "build_dashboard_snapshot",
    "normalise_intraday_bars",
    "prepare_vrp_history",
]


@dataclass(frozen=True)
class DashboardSnapshot:
    """Provider-neutral data bundle consumed by the dashboard UI."""

    symbol: str
    quote_time: pd.Timestamp
    target_days: float
    price_side: str
    target_mfiv: ConstantTenorMFIV
    mfiv_curve: pd.DataFrame
    daily_variance: pd.Series
    realized_history: pd.DataFrame
    realized_curve: pd.DataFrame
    trailing_target_variance: float
    trailing_target_volatility: float
    current_vrp_variance: float
    current_vol_spread: float
    chain_quality: dict[str, float | int | str]


def normalise_intraday_bars(
    bars: pd.DataFrame,
    *,
    timestamp_col: str | None = None,
    close_col: str | None = None,
) -> pd.DataFrame:
    """Return ``timestamp``/``close`` columns from common vendor bar shapes.

    Accepts either a DatetimeIndex or a timestamp-like column.  Column matching
    is case-insensitive and understands the common ``Datetime``/``Date`` and
    ``Close`` spellings used by yfinance and CSV exports.
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("intraday bars are empty")

    df = bars.copy()
    lower = {str(c).lower(): c for c in df.columns}

    if timestamp_col is None:
        for candidate in ("timestamp", "datetime", "date", "time"):
            if candidate in lower:
                timestamp_col = lower[candidate]
                break
    if timestamp_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        timestamp_col = df.columns[0]
    if timestamp_col is None:
        raise ValueError("could not find timestamp column or DatetimeIndex")

    # Rebuild the case map because reset_index may have added a column.
    lower = {str(c).lower(): c for c in df.columns}
    if close_col is None:
        for candidate in ("close", "adj close", "adj_close", "price", "last"):
            if candidate in lower:
                close_col = lower[candidate]
                break
    if close_col is None:
        raise ValueError("could not find a close/price column")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[timestamp_col], errors="coerce", utc=True),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    out = out.dropna().sort_values("timestamp")
    out = out[out["close"] > 0].drop_duplicates("timestamp", keep="last")
    if len(out) < 2:
        raise ValueError("need at least two valid intraday bars")
    return out.reset_index(drop=True)


def build_dashboard_snapshot(
    chain: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    target_days: float = 30.0,
    price_side: str = "mid",
    rv_windows: Iterable[int] = (3, 9, 30, 60, 180),
    session_tz: str = "America/New_York",
    min_mfiv_strikes: int = 8,
) -> DashboardSnapshot:
    """Compute the live, no-ML Forward VRP dashboard state.

    ``target_days`` uses calendar time for the MFIV-vs-trailing-RV comparison,
    matching the option DTE clock.  The descriptive RV term structure uses
    trading-day windows because 3/9/30/60/180-day realized-vol features are
    conventionally interpreted that way in this project.
    """
    if target_days <= 0:
        raise ValueError("target_days must be positive")

    canonical = add_derived_columns(chain)
    validate_chain(canonical)
    symbols = canonical["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        raise ValueError("dashboard snapshot requires exactly one symbol")
    symbol = symbols[0]
    quote_time = pd.to_datetime(canonical["quote_time"], utc=True).max()

    slices = mfiv_term_structure(
        canonical,
        price_side=price_side,
        min_strikes=min_mfiv_strikes,
    )
    if len(slices) < 1:
        raise ValueError("no expiries produced a valid MFIV slice")
    target = constant_tenor_mfiv(slices, target_days)

    mfiv_curve = pd.DataFrame([
        {
            "expiry": s.expiry,
            "dte": s.dte,
            "implied_variance": s.implied_variance,
            "implied_volatility": s.implied_volatility,
            "total_variance": s.total_variance,
            "forward": s.forward_fit.forward,
            "discount": s.forward_fit.discount,
            "parity_r2": s.forward_fit.r_squared,
            "n_strikes": s.n_strikes,
            "price_side": s.price_side,
        }
        for s in slices
    ]).sort_values("dte").reset_index(drop=True)

    norm_bars = normalise_intraday_bars(bars)
    daily = daily_integrated_variance(
        norm_bars,
        session_tz=session_tz,
        include_overnight=True,
    )
    if daily.empty:
        raise ValueError("intraday bars did not produce realized variance")

    rv_windows = tuple(int(w) for w in rv_windows)
    realized_hist = realized_term_structure(daily, windows=rv_windows, basis="trading")

    # Only use information observed by the option quote timestamp.  This is
    # particularly important when users load a local file containing later bars.
    local_quote_date = quote_time.tz_convert(session_tz).tz_localize(None).normalize()
    eligible_daily = daily.loc[daily.index <= local_quote_date]
    if eligible_daily.empty:
        raise ValueError("intraday history ends before the option quote date")

    trailing_target = rolling_integrated_variance(
        eligible_daily,
        int(round(target_days)),
        basis="calendar",
    ).dropna()
    if trailing_target.empty:
        raise ValueError(
            f"not enough intraday history to compute trailing {target_days:g}-day realized variance"
        )
    trailing_var = float(trailing_target.iloc[-1])
    trailing_vol = float(integrated_volatility(trailing_var))

    eligible_realized = realized_hist.loc[realized_hist.index <= local_quote_date]
    if eligible_realized.empty:
        raise ValueError("no realized term-structure observation aligns with the quote date")
    latest = eligible_realized.iloc[-1]
    curve_rows = []
    for w in rv_windows:
        col = f"rv_{w}"
        if col in latest.index and np.isfinite(latest[col]):
            curve_rows.append({"days": w, "realized_volatility": float(latest[col])})
    realized_curve = pd.DataFrame(curve_rows)

    mid = canonical["mid"].replace([np.inf, -np.inf], np.nan)
    rel = canonical["rel_spread"].replace([np.inf, -np.inf], np.nan)
    chain_quality: dict[str, float | int | str] = {
        "rows": int(len(canonical)),
        "expiries": int(canonical["expiry"].nunique()),
        "strikes": int(canonical["strike"].nunique()),
        "spot": float(canonical["underlying_price"].median()),
        "zero_bid_rows": int((canonical["bid"] <= 0).sum()),
        "crossed_rows": int((canonical["ask"] < canonical["bid"]).sum()),
        "median_spread": float(canonical["spread"].median()),
        "median_rel_spread": float(rel.median()) if rel.notna().any() else np.nan,
        "median_mid": float(mid.median()) if mid.notna().any() else np.nan,
    }

    return DashboardSnapshot(
        symbol=symbol,
        quote_time=quote_time,
        target_days=float(target_days),
        price_side=price_side,
        target_mfiv=target,
        mfiv_curve=mfiv_curve,
        daily_variance=daily,
        realized_history=realized_hist,
        realized_curve=realized_curve,
        trailing_target_variance=trailing_var,
        trailing_target_volatility=trailing_vol,
        current_vrp_variance=float(target.implied_variance - trailing_var),
        current_vol_spread=float(target.implied_volatility - trailing_vol),
        chain_quality=chain_quality,
    )


def prepare_vrp_history(
    history: pd.DataFrame,
    *,
    date_col: str = "date",
    z_window: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Validate and enrich a daily MFIV/RV history for dashboard charts.

    Required columns are ``date``, ``mfiv_var`` and ``trailing_rv_var``.
    ``forward_rv_var`` is optional; when present, the true ex-post forward VRP
    label is added.  Rolling z-scores and percentiles are inherited from
    :func:`volforge.vrp.vrp_features` and therefore use only prior observations.
    """
    required = {date_col, "mfiv_var", "trailing_rv_var"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"VRP history is missing columns: {sorted(missing)}")

    df = history.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for col in ("mfiv_var", "trailing_rv_var", "forward_rv_var"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[date_col, "mfiv_var", "trailing_rv_var"])
    df = df.sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)

    features = vrp_features(
        df["mfiv_var"],
        df["trailing_rv_var"],
        z_window=z_window,
        min_periods=min_periods,
    )
    out = df.join(features.drop(columns=["mfiv_var", "trailing_rv_var"]), how="left")
    out["mfiv_vol"] = integrated_volatility(out["mfiv_var"])
    out["trailing_rv_vol"] = integrated_volatility(out["trailing_rv_var"])
    out["vol_spread"] = out["mfiv_vol"] - out["trailing_rv_vol"]
    out["vol_of_vol"] = vol_of_vol(out["mfiv_var"], window=20)

    if "forward_rv_var" in out:
        out["forward_rv_vol"] = integrated_volatility(out["forward_rv_var"])
        out["forward_vrp"] = forward_vrp_label(out["mfiv_var"], out["forward_rv_var"])
    return out

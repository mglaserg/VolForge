"""Local persistence helpers for provider-tagged intraday and realized data.

Intraday data is intentionally stored separately by provider/feed so an IEX
research archive can later coexist with SIP without silently mixing the two.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "intraday_archive_path",
    "realized_archive_path",
    "load_intraday_archive",
    "save_intraday_archive",
    "load_realized_archive",
    "save_realized_archive",
]


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError("archive component cannot be empty")
    return text


def intraday_archive_path(
    symbol: str,
    *,
    provider: str,
    feed: str,
    timeframe: str = "5Min",
    root: str | Path = "data/intraday",
) -> Path:
    return (
        Path(root)
        / f"provider={_slug(provider)}"
        / f"feed={_slug(feed)}"
        / f"symbol={str(symbol).strip().upper()}"
        / f"bars_{str(timeframe).strip().lower()}.parquet"
    )


def realized_archive_path(
    symbol: str,
    *,
    provider: str,
    feed: str,
    root: str | Path = "data/realized",
) -> Path:
    return (
        Path(root)
        / f"provider={_slug(provider)}"
        / f"feed={_slug(feed)}"
        / f"symbol={str(symbol).strip().upper()}"
        / "daily_variance.parquet"
    )


def _prepare_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame or "close" not in frame:
        raise ValueError("intraday archive needs timestamp and close columns")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"])
    out = out[out["close"] > 0]
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def load_intraday_archive(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return _prepare_bars(pd.read_parquet(path))


def save_intraday_archive(
    bars: pd.DataFrame,
    path: str | Path,
    *,
    merge_existing: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    incoming = _prepare_bars(bars)
    if merge_existing and target.exists():
        prior = load_intraday_archive(target)
        incoming = _prepare_bars(pd.concat([prior, incoming], ignore_index=True, sort=False))
    incoming.to_parquet(target, index=False)
    return target



def load_realized_archive(path: str | Path) -> pd.Series:
    """Load the canonical daily integrated-variance archive."""
    target = Path(path)
    if not target.exists():
        return pd.Series(dtype="float64", name="integrated_variance")
    frame = pd.read_parquet(target)
    if "date" not in frame or "integrated_variance" not in frame:
        raise ValueError("realized archive needs date and integrated_variance columns")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    values = pd.to_numeric(frame["integrated_variance"], errors="coerce")
    out = pd.Series(values.to_numpy(float), index=pd.DatetimeIndex(dates), name="integrated_variance")
    out = out[~out.index.isna()].dropna()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index().groupby(level=0).last()


def save_realized_archive(daily_variance: pd.Series, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    s = pd.Series(daily_variance, dtype="float64", copy=True)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    frame = pd.DataFrame({
        "date": s.index.normalize(),
        "integrated_variance": s.to_numpy(float),
    }).dropna(subset=["date", "integrated_variance"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame.to_parquet(target, index=False)
    return target

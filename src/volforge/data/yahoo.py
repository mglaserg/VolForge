"""yfinance adapter.

Scope and limits, stated plainly:

* yfinance returns only the **current** chain. There is no historical
  option-chain endpoint. You cannot backfill; you can only begin snapshotting
  from today forward. Any research needing multi-year history requires a real
  vendor (ORATS, CBOE DataShop, Databento, OptionMetrics).
* Quotes are delayed and, outside 09:30-16:00 ET, frequently stale, wide, or
  zero on one side. Snapshot midday.
* Yahoo's `impliedVolatility` is carried through as `vendor_iv` for sanity
  checks only. It is never fitted -- its forward and rate assumptions are
  undocumented and it is often visibly wrong in the wings.
* Yahoo publishes expiry as a bare date. We attach a settlement time from
  `schema.SETTLEMENT_TIMES`, which matters a lot for short-dated slices.

Snapshots are written as Parquet, partitioned by symbol and date, so that a
year of daily captures replays cheaply and the ORATS migration later is a
matter of writing a second adapter that lands in the same directory layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    add_derived_columns,
    expiry_datetime,
    validate_chain,
)

__all__ = ["fetch_chain", "save_snapshot", "load_snapshot", "list_snapshots"]

_COLUMN_ORDER = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)


def fetch_chain(
    symbol: str,
    max_expiries: int | None = None,
    dte_range: tuple[float, float] | None = (7.0, 120.0),
    settlement: str = "default",
    ticker=None,
) -> pd.DataFrame:
    """Fetch the current option chain for `symbol` in canonical schema.

    Parameters
    ----------
    symbol : underlying ticker, e.g. 'SPY'
    max_expiries : cap the number of expiries pulled (each is a separate HTTP
        call, so this is your rate-limit lever).
    dte_range : keep only expiries in this day range. None keeps all.
    settlement : key into schema.SETTLEMENT_TIMES.
    ticker : inject a pre-built object exposing `.options`, `.option_chain(exp)`
        and `.fast_info`. Used for testing; leave None in production.
    """
    if ticker is None:
        import yfinance as yf  # imported lazily so the package works without it
        ticker = yf.Ticker(symbol)

    quote_time = pd.Timestamp.now(tz="UTC")
    spot = _get_spot(ticker)

    expiries = list(ticker.options)
    if not expiries:
        raise RuntimeError(f"no expiries returned for {symbol}")

    frames = []
    for exp_str in expiries:
        exp_ts = expiry_datetime(exp_str, settlement)
        dte = (exp_ts - quote_time).total_seconds() / 86400.0
        if dte <= 0:
            continue
        if dte_range is not None and not (dte_range[0] <= dte <= dte_range[1]):
            continue

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as exc:  # one bad expiry should not kill the snapshot
            print(f"  warning: skipping expiry {exp_str}: {exc}")
            continue

        for right, raw in (("C", chain.calls), ("P", chain.puts)):
            if raw is None or len(raw) == 0:
                continue
            frames.append(_normalise(raw, symbol, right, quote_time, exp_ts, spot))

        if max_expiries is not None and len({f["expiry"].iloc[0] for f in frames}) >= max_expiries:
            break

    if not frames:
        raise RuntimeError(f"no usable expiries for {symbol} in dte_range={dte_range}")

    df = pd.concat(frames, ignore_index=True)
    df = validate_chain(df)
    return add_derived_columns(df).sort_values(["expiry", "strike", "right"]).reset_index(drop=True)


def _get_spot(ticker) -> float:
    """Underlying price, with fallbacks -- yfinance's surface here is unstable."""
    for accessor in (
        lambda: ticker.fast_info["last_price"],
        lambda: ticker.fast_info["lastPrice"],
        lambda: ticker.info["regularMarketPrice"],
        lambda: float(ticker.history(period="1d")["Close"].iloc[-1]),
    ):
        try:
            v = float(accessor())
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            continue
    raise RuntimeError("could not determine underlying price")


def _normalise(raw, symbol, right, quote_time, expiry, spot) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "quote_time": quote_time,
            "expiry": expiry,
            "strike": pd.to_numeric(raw["strike"], errors="coerce").astype("float64"),
            "right": right,
            "bid": pd.to_numeric(raw.get("bid"), errors="coerce").astype("float64"),
            "ask": pd.to_numeric(raw.get("ask"), errors="coerce").astype("float64"),
            "underlying_price": float(spot),
            "last": pd.to_numeric(raw.get("lastPrice"), errors="coerce").astype("float64"),
            "volume": pd.to_numeric(raw.get("volume"), errors="coerce").astype("float64"),
            "open_interest": pd.to_numeric(raw.get("openInterest"), errors="coerce").astype("float64"),
            "vendor_iv": pd.to_numeric(raw.get("impliedVolatility"), errors="coerce").astype("float64"),
            "source": "yfinance",
        }
    )
    lt = raw.get("lastTradeDate")
    if lt is not None:
        lt = pd.to_datetime(lt, errors="coerce", utc=True)
    out["last_trade_time"] = lt if lt is not None else pd.NaT
    return out[_COLUMN_ORDER]


# --------------------------------------------------------------------------
# Snapshot persistence
# --------------------------------------------------------------------------

def save_snapshot(df: pd.DataFrame, root="data/chains") -> Path:
    """Write one snapshot to root/symbol=SPY/date=YYYY-MM-DD/chain.parquet."""
    symbol = df["symbol"].iloc[0]
    date = df["quote_time"].iloc[0].tz_convert("America/New_York").date()
    path = Path(root) / f"symbol={symbol}" / f"date={date}"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "chain.parquet"
    df.to_parquet(target, index=False)
    return target


def load_snapshot(symbol: str, date, root="data/chains") -> pd.DataFrame:
    path = Path(root) / f"symbol={symbol}" / f"date={pd.Timestamp(date).date()}" / "chain.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return add_derived_columns(pd.read_parquet(path))


def list_snapshots(symbol: str, root="data/chains") -> list:
    base = Path(root) / f"symbol={symbol}"
    if not base.exists():
        return []
    return sorted(pd.Timestamp(p.name.split("=", 1)[1]).date()
                  for p in base.glob("date=*") if (p / "chain.parquet").exists())

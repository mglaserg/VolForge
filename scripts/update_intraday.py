#!/usr/bin/env python3
"""Update VolForge's local intraday archive and daily integrated variance.

Example:
    python scripts/update_intraday.py --symbol SPY --start 2016-01-01

Alpaca/IEX is the default because it works with the free Basic market-data
plan.  The archive remains feed-tagged so SIP can be compared later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from volforge.data.alpaca import fetch_alpaca_bars
from volforge.data.intraday import (
    intraday_archive_path,
    load_intraday_archive,
    realized_archive_path,
    save_intraday_archive,
    save_realized_archive,
)
from volforge.realized import daily_integrated_variance, regular_session_bars


def _default_incremental_start(existing: pd.DataFrame, fallback: str) -> pd.Timestamp:
    if existing.empty:
        return pd.Timestamp(fallback, tz="UTC")
    latest = pd.to_datetime(existing["timestamp"], errors="coerce", utc=True).max()
    if pd.isna(latest):
        return pd.Timestamp(fallback, tz="UTC")
    # Re-download the last two calendar days.  Archive merge is idempotent and
    # the overlap makes interrupted/partial final days self-healing.
    return latest - pd.Timedelta(days=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--provider", choices=["alpaca"], default="alpaca")
    ap.add_argument("--feed", choices=["iex", "sip"], default="iex")
    ap.add_argument("--timeframe", default="5Min")
    ap.add_argument("--start", default=None, help="Start date/time. First run defaults to 2016-01-01.")
    ap.add_argument("--end", default=None)
    ap.add_argument("--intraday-root", default="data/intraday")
    ap.add_argument("--realized-root", default="data/realized")
    ap.add_argument("--no-realized", action="store_true", help="Only update bars; skip daily integrated variance.")
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()
    bars_path = intraday_archive_path(
        symbol,
        provider=args.provider,
        feed=args.feed,
        timeframe=args.timeframe,
        root=args.intraday_root,
    )
    existing = load_intraday_archive(bars_path)
    start = pd.Timestamp(args.start) if args.start else _default_incremental_start(existing, "2016-01-01")

    print(f"provider/feed: {args.provider}/{args.feed}")
    print(f"symbol/timeframe: {symbol}/{args.timeframe}")
    print(f"archive: {bars_path}")
    if not existing.empty:
        print(
            "existing bars: "
            f"{len(existing):,} · {existing['timestamp'].min()} -> {existing['timestamp'].max()}"
        )
    else:
        print("existing bars: 0")
    print(f"request: {start} -> {args.end or 'latest available'}")

    fresh = fetch_alpaca_bars(
        symbol,
        start=start,
        end=args.end,
        timeframe=args.timeframe,
        feed=args.feed,
    )
    print(f"downloaded bars: {len(fresh):,}")
    if fresh.empty and existing.empty:
        raise RuntimeError("Alpaca returned no bars and no local archive exists")

    if not fresh.empty:
        save_intraday_archive(fresh, bars_path, merge_existing=True)
    merged = load_intraday_archive(bars_path)
    print(f"stored bars: {len(merged):,}")
    if not merged.empty:
        print(f"bar coverage: {merged['timestamp'].min()} -> {merged['timestamp'].max()}")

    if not args.no_realized:
        rth = regular_session_bars(merged)
        daily = daily_integrated_variance(rth, include_overnight=True)
        rv_path = realized_archive_path(
            symbol,
            provider=args.provider,
            feed=args.feed,
            root=args.realized_root,
        )
        save_realized_archive(daily, rv_path)
        print(f"daily realized rows: {len(daily):,}")
        if len(daily):
            print(f"realized coverage: {daily.index.min().date()} -> {daily.index.max().date()}")
        print(f"realized archive: {rv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

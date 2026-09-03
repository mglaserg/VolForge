#!/usr/bin/env python3
"""Build compact historical VRP research datasets from saved option chains.

The builder never downloads historical option chains. Capture option snapshots
first, update the realized-vol archive, then rerun this script. Previously
unavailable 30-day forward labels fill automatically as future RV arrives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from volforge.data.intraday import realized_archive_path
from volforge.data.storage import list_chain_snapshots, select_daily_snapshots
from volforge.history import VRPHistoryConfig, build_vrp_history, load_daily_variance, save_vrp_history
from volforge.realized import daily_integrated_variance, regular_session_bars


def _load_bars(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("bars file must be CSV or Parquet")


def _coverage(series: pd.Series) -> str:
    if series.empty:
        return "empty"
    idx = pd.DatetimeIndex(series.index)
    return f"{idx.min().date()} -> {idx.max().date()} ({len(series):,} sessions)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--provider", default="yahoo", help="Option-chain provider")
    ap.add_argument("--chain-root", default="data/chains")
    source = ap.add_mutually_exclusive_group(required=False)
    source.add_argument("--daily-variance", help="CSV/Parquet with date, integrated_variance")
    source.add_argument("--bars", help="5m/15m bars CSV/Parquet with timestamp, close")
    ap.add_argument("--rv-provider", default="alpaca", help="Default realized-data provider")
    ap.add_argument("--rv-feed", default="iex", help="Default realized-data feed")
    ap.add_argument("--realized-root", default="data/realized")
    ap.add_argument("--output-root", default="data/derived/vrp")
    ap.add_argument("--target-dte", type=float, default=30.0)
    ap.add_argument("--mfiv-tenors", nargs="+", type=int, default=[7, 30, 60, 180])
    ap.add_argument("--rv-windows", nargs="+", type=int, default=[3, 9, 30, 60, 180])
    ap.add_argument("--price-side", choices=["mid", "bid"], default="mid")
    ap.add_argument("--snapshot-policy", choices=["latest", "earliest", "closest"], default="latest")
    ap.add_argument("--target-time", default=None, help="HH:MM ET for --snapshot-policy closest")
    ap.add_argument("--rv-asof", choices=["previous_session", "same_session"], default="previous_session")
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()
    all_refs = list_chain_snapshots(
        symbol,
        provider=args.provider,
        root=args.chain_root,
        include_legacy_yahoo=True,
    )
    selected = select_daily_snapshots(all_refs, policy=args.snapshot_policy, target_time=args.target_time)
    print(f"option archive: {len(all_refs):,} snapshots across {len({r.date for r in all_refs}):,} dates")
    if selected:
        print(f"selected chain dates: {selected[0].date} -> {selected[-1].date} ({len(selected):,} research rows max)")
    else:
        print("selected chain dates: none")

    if args.daily_variance:
        rv_source = Path(args.daily_variance)
        daily = load_daily_variance(rv_source)
        source_desc = f"daily variance: {rv_source}"
    elif args.bars:
        bars_path = Path(args.bars)
        bars = regular_session_bars(_load_bars(bars_path))
        daily = daily_integrated_variance(bars)
        source_desc = f"intraday bars: {bars_path}"
    else:
        rv_source = realized_archive_path(
            symbol,
            provider=args.rv_provider,
            feed=args.rv_feed,
            root=args.realized_root,
        )
        if not rv_source.exists():
            raise FileNotFoundError(
                f"No realized archive at {rv_source}. Run: "
                f"python scripts/update_intraday.py --symbol {symbol} --start 2016-01-01"
            )
        daily = load_daily_variance(rv_source)
        source_desc = f"canonical realized archive: {rv_source}"

    print(f"RV source: {source_desc}")
    print(f"RV coverage: {_coverage(daily)}")

    cfg = VRPHistoryConfig(
        target_days=args.target_dte,
        mfiv_tenors=tuple(args.mfiv_tenors),
        rv_windows=tuple(args.rv_windows),
        price_side=args.price_side,
        rv_asof=args.rv_asof,
        snapshot_policy=args.snapshot_policy,
        target_time=args.target_time,
    )
    history = build_vrp_history(
        symbol,
        daily,
        provider=args.provider,
        chain_root=args.chain_root,
        config=cfg,
    )
    target = save_vrp_history(
        history,
        symbol=symbol,
        provider=args.provider,
        root=args.output_root,
    )
    n_labels = int(history["forward_rv_var"].notna().sum()) if "forward_rv_var" in history else 0
    print(f"wrote {len(history):,} rows to {target}")
    if len(history):
        hdates = pd.to_datetime(history["date"], errors="coerce").dropna()
        if len(hdates):
            print(f"history coverage: {hdates.min().date()} -> {hdates.max().date()}")
    print(f"forward {args.target_dte:g}d labels available: {n_labels:,}/{len(history):,}")
    if selected and len(history) < len(selected):
        print(f"WARNING: {len(selected) - len(history)} selected chain date(s) did not produce a history row")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build compact historical VRP research datasets from saved option chains.

The builder never downloads historical chains itself. Capture/backfill raw data
first, then rerun this script whenever new option snapshots or realized data
arrive; previously unavailable 30-day forward labels will fill automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from volforge.history import VRPHistoryConfig, build_vrp_history, load_daily_variance, save_vrp_history
from volforge.realized import daily_integrated_variance


def _load_bars(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("bars file must be CSV or Parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--provider", default="yahoo")
    ap.add_argument("--chain-root", default="data/chains")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--daily-variance", help="CSV/Parquet with date, integrated_variance")
    source.add_argument("--bars", help="5m/15m bars CSV/Parquet with timestamp, close")
    ap.add_argument("--output-root", default="data/derived/vrp")
    ap.add_argument("--target-dte", type=float, default=30.0)
    ap.add_argument("--mfiv-tenors", nargs="+", type=int, default=[7, 30, 60, 180])
    ap.add_argument("--rv-windows", nargs="+", type=int, default=[3, 9, 30, 60, 180])
    ap.add_argument("--price-side", choices=["mid", "bid"], default="mid")
    ap.add_argument("--snapshot-policy", choices=["latest", "earliest", "closest"], default="latest")
    ap.add_argument("--target-time", default=None, help="HH:MM ET for --snapshot-policy closest")
    ap.add_argument("--rv-asof", choices=["previous_session", "same_session"], default="previous_session")
    args = ap.parse_args()

    if args.daily_variance:
        daily = load_daily_variance(args.daily_variance)
    else:
        bars = _load_bars(args.bars)
        daily = daily_integrated_variance(bars)

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
        args.symbol,
        daily,
        provider=args.provider,
        chain_root=args.chain_root,
        config=cfg,
    )
    target = save_vrp_history(
        history,
        symbol=args.symbol,
        provider=args.provider,
        root=args.output_root,
    )
    n_labels = int(history["forward_rv_var"].notna().sum()) if "forward_rv_var" in history else 0
    print(f"wrote {len(history)} rows to {target}")
    print(f"forward {args.target_dte:g}d labels available: {n_labels}/{len(history)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

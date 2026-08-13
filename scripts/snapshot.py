#!/usr/bin/env python3
"""Daily chain snapshot.

yfinance has no historical endpoint, so history only exists if you capture it.
Every day this does not run is a day permanently missing from your sample.
Schedule it on trading days, midday ET, when quotes are tightest:

    # crontab -e   (times are in the machine's local zone)
    30 12 * * 1-5  cd /path/to/volforge && .venv/bin/python scripts/snapshot.py

Usage:
    python scripts/snapshot.py                     # SPY, 7-120 DTE
    python scripts/snapshot.py --symbols SPY QQQ --dte 5 180
"""

from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices
from volforge.data.yahoo import fetch_chain, save_snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=["SPY"])
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 120.0],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--root", default="data/chains")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and validate but write nothing")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="America/New_York")
    if now.dayofweek >= 5:
        print(f"warning: {now:%Y-%m-%d} is a weekend; quotes will be stale.")

    failures = 0
    for symbol in args.symbols:
        print(f"\n=== {symbol} @ {now:%Y-%m-%d %H:%M %Z} ===")
        try:
            raw = fetch_chain(symbol, dte_range=tuple(args.dte))
            print(f"fetched {len(raw)} quotes across {raw['expiry'].nunique()} expiries")

            # Clean and build slices as a *validation* pass only. The raw
            # snapshot is what gets written -- cleaning rules will change as
            # you learn, and you cannot un-drop a quote you never saved.
            clean, _ = clean_chain(raw, CleanConfig(dte_range=tuple(args.dte)),
                                   verbose=True)
            slices = build_all_slices(clean, verbose=True)
            print(f"validation: {len(slices)} calibratable slices")
            if not slices:
                print("WARNING: nothing calibratable -- check market hours")

            if args.dry_run:
                print("dry run, not written")
            else:
                path = save_snapshot(raw, root=args.root)
                print(f"wrote {path}")

        except Exception:
            failures += 1
            traceback.print_exc()
            print(f"FAILED: {symbol}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

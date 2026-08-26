#!/usr/bin/env python3
"""Capture option chains into VolForge's provider-neutral archive.

Examples
--------
Current Yahoo snapshot::

    python scripts/capture_chain.py --provider yahoo --symbols SPY GLD XLE

When ORATS is enabled::

    python scripts/capture_chain.py --provider orats --symbols SPY --dte 7 180

A historical ORATS date can be requested with ``--trade-date``. Yahoo cannot
backfill historical option chains.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from volforge.data.provider import fetch_chain
from volforge.data.storage import save_chain_snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="yahoo")
    ap.add_argument("--symbols", nargs="+", default=["SPY"])
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 180.0], metavar=("MIN", "MAX"))
    ap.add_argument("--root", default="data/chains")
    ap.add_argument("--trade-date", default=None, help="historical provider date/time when supported")
    ap.add_argument("--intraday", action="store_true", help="request provider intraday endpoint when supported")
    ap.add_argument("--live", action="store_true", help="request live provider endpoint when supported")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    failures = 0
    for symbol in args.symbols:
        try:
            kwargs = {}
            if args.trade_date is not None:
                kwargs["trade_date"] = args.trade_date
            if args.intraday:
                kwargs["intraday"] = True
            if args.live:
                kwargs["live"] = True
            chain = fetch_chain(
                symbol,
                provider=args.provider,
                dte_range=tuple(args.dte),
                **kwargs,
            )
            print(f"{symbol}: fetched {len(chain)} rows across {chain['expiry'].nunique()} expiries")
            if args.dry_run:
                continue
            ref = save_chain_snapshot(chain, provider=args.provider, root=args.root)
            print(f"{symbol}: wrote {ref.path}")
        except Exception:
            failures += 1
            traceback.print_exc()
            print(f"FAILED: {symbol}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fit raw SVI slices and a global SSVI surface from a saved chain snapshot.

Example:
    python scripts/fit_ssvi.py \
        --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
        --symbol SPY --plot --db volforge.db

The raw SVI fits remain the local per-expiry benchmark.  SSVI uses their ATM
variance levels to construct a monotone theta(T) clock, then fits its global
shape parameters directly to the original option observations.
"""

from __future__ import annotations

import argparse

import pandas as pd

from volforge import VolDB, build_surface, calibrate_svi, calibrate_ssvi
from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=None, help="trade date; inferred from quote_time if omitted")
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 120.0])
    ap.add_argument("--db", default=None, help="optional SQLite database path")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    raw = pd.read_parquet(args.chain)
    if args.date is None:
        if "quote_time" not in raw.columns or raw.empty:
            raise ValueError("--date is required when the snapshot has no quote_time column")
        date = pd.Timestamp(raw["quote_time"].iloc[0]).date()
    else:
        date = pd.Timestamp(args.date).date()

    clean, report = clean_chain(raw, CleanConfig(dte_range=tuple(args.dte)), verbose=True)
    slices = build_all_slices(clean, verbose=True)
    pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]

    good = sum(f.is_reliable for _, f in pairs)
    print(f"\nraw SVI: {good}/{len(pairs)} reliable slices")
    ssvi = calibrate_ssvi(pairs)
    print(
        f"SSVI: rho={ssvi.params.rho:+.4f} eta={ssvi.params.eta:.4f} "
        f"gamma={ssvi.params.gamma:.4f}  RMSE={ssvi.rmse_iv*100:.3f}vp\n"
        f"      theta repair={ssvi.theta_repair_fraction:.3%}  "
        f"butterfly={ssvi.butterfly_free} calendar={ssvi.calendar_free}"
    )

    svi_surface = build_surface(pairs, date, args.symbol)
    ssvi_surface = ssvi.to_surface(date, args.symbol)

    if args.db:
        db = VolDB(args.db)
        db.save_svi_fits(args.symbol, date, [(f, s) for s, f in pairs])
        db.save_ssvi_fit(args.symbol, date, ssvi)
        db.save_surface(args.symbol, date, svi_surface)  # legacy raw-SVI grid
        db.save_model_surface(args.symbol, date, "svi", svi_surface)
        db.save_model_surface(args.symbol, date, "ssvi", ssvi_surface)
        print(f"saved to {args.db}")

    if args.plot:
        import matplotlib.pyplot as plt
        from volforge.visualization import plot_ssvi_diagnostics, plot_surface_3d
        plot_ssvi_diagnostics(pairs, ssvi)
        plot_surface_3d(ssvi_surface)
        plt.show()


if __name__ == "__main__":
    main()

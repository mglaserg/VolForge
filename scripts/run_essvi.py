#!/usr/bin/env python3
"""Fit raw SVI, SSVI and extended SSVI from a saved chain snapshot.

The raw SVI slices provide the flexible local benchmark and the ATM-theta
anchors.  eSSVI then allows rho to vary with maturity while enforcing the
Hendriks-Martini continuous calendar condition and the SSVI butterfly bounds.

Example
-------
    python scripts/run_essvi.py \
        --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
        --symbol SPY --db volforge.db --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volforge import VolDB, calibrate_svi, calibrate_ssvi, calibrate_essvi
from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices


def _date(raw, explicit):
    if explicit is not None:
        return pd.Timestamp(explicit).date()
    if "quote_time" not in raw.columns or raw.empty:
        raise ValueError("--date is required when the snapshot has no quote_time column")
    return pd.Timestamp(raw["quote_time"].iloc[0]).date()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 120.0])
    ap.add_argument("--db", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-plots", default=None, metavar="DIR")
    ap.add_argument("--phi-form", choices=["modified_power_law", "power_law"],
                    default="modified_power_law")
    args = ap.parse_args()

    raw = pd.read_parquet(args.chain)
    date = _date(raw, args.date)
    clean, _ = clean_chain(raw, CleanConfig(dte_range=tuple(args.dte)), verbose=True)
    slices = build_all_slices(clean, verbose=True)
    if not slices:
        raise ValueError("no calibratable slices remain after cleaning")

    pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]
    good = sum(f.is_reliable for _, f in pairs)
    print(f"\nraw SVI: {good}/{len(pairs)} reliable slices")

    ssvi = calibrate_ssvi(pairs, phi_form=args.phi_form)
    essvi = calibrate_essvi(pairs, phi_form=args.phi_form)
    improvement = 1.0 - essvi.rmse_iv / ssvi.rmse_iv if ssvi.rmse_iv > 0 else 0.0

    print(
        f"SSVI : rho={ssvi.params.rho:+.4f} eta={ssvi.params.eta:.4f} "
        f"gamma={ssvi.params.gamma:.4f} RMSE={ssvi.rmse_iv*100:.3f}vp"
    )
    p = essvi.params
    print(
        f"eSSVI: rho0={p.rho0:+.4f} rho_m={p.rho_m:+.4f} a={p.a:.4f} "
        f"eta={p.eta:.4f} gamma={p.gamma:.4f} RMSE={essvi.rmse_iv*100:.3f}vp\n"
        f"       improvement vs SSVI={improvement:.1%}  "
        f"theta repair={essvi.theta_repair_fraction:.3%}  "
        f"butterfly={essvi.butterfly_free} calendar={essvi.calendar_free}"
    )

    ssvi_surface = ssvi.to_surface(date, args.symbol)
    essvi_surface = essvi.to_surface(date, args.symbol)

    if args.db:
        db = VolDB(args.db)
        db.save_svi_fits(args.symbol, date, [(f, s) for s, f in pairs])
        db.save_ssvi_fit(args.symbol, date, ssvi)
        db.save_essvi_fit(args.symbol, date, essvi)
        db.save_model_surface(args.symbol, date, "ssvi", ssvi_surface)
        db.save_model_surface(args.symbol, date, "essvi", essvi_surface)
        print(f"saved SSVI/eSSVI fits and surfaces to {args.db}")

    if args.plot or args.save_plots:
        import matplotlib.pyplot as plt
        from volforge.visualization import plot_essvi_diagnostics, plot_surface_3d
        figs = [plot_essvi_diagnostics(pairs, essvi), plot_surface_3d(essvi_surface)]
        if args.save_plots:
            out = Path(args.save_plots)
            out.mkdir(parents=True, exist_ok=True)
            figs[0].savefig(out / "essvi_diagnostics.png", dpi=160, bbox_inches="tight")
            figs[1].savefig(out / "essvi_surface_3d.png", dpi=160, bbox_inches="tight")
            print(f"saved plots to {out}")
            if not args.plot:
                for fig in figs:
                    plt.close(fig)
        if args.plot:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

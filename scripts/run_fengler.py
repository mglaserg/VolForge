#!/usr/bin/env python3
"""Fit Fengler's constrained natural-spline volatility surface.

Fengler works in call-price space, not implied-volatility parameter space.  Each
maturity is fit as a bounded decreasing convex natural cubic spline; maturities
are processed longest-to-shortest with no-crossing constraints at equal
forward moneyness.  The result is converted back to total variance on
VolForge's common grid for direct SVI/SSVI/eSSVI comparison.

Example
-------
    python scripts/run_fengler.py \
        --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
        --symbol SPY --lambda 1e-5 --db volforge.db --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from volforge import VolDB, fit_fengler_surface, prepare_fengler_slices
from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices


def _date(raw, explicit):
    if explicit is not None:
        return pd.Timestamp(explicit).date()
    if "quote_time" not in raw.columns or raw.empty:
        raise ValueError("--date is required when the snapshot has no quote_time column")
    return pd.Timestamp(raw["quote_time"].iloc[0]).date()


def _summary(fit):
    rows = []
    for s in fit.slices:
        rows.append({
            "dte": s.dte,
            "n": len(s.knots),
            "rmse_vp": s.rmse_iv * 100.0,
            "min_gamma": s.min_gamma,
            "left_slope": s.left_slope,
            "right_slope": s.right_slope,
            "strike_arb": s.strike_arb_free,
            "calendar": s.calendar_free,
            "solver": s.solver,
            "seconds": s.solve_time,
        })
    df = pd.DataFrame(rows)
    print("\nFengler slice summary")
    print(df.to_string(index=False, formatters={
        "dte": lambda x: f"{x:6.1f}",
        "rmse_vp": lambda x: f"{x:8.3f}",
        "min_gamma": lambda x: f"{x:10.3g}",
        "left_slope": lambda x: f"{x:10.4f}",
        "right_slope": lambda x: f"{x:11.4f}",
        "seconds": lambda x: f"{x:7.3f}",
    }))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 120.0])
    ap.add_argument("--lambda", dest="smoothing_lambda", type=float, default=1e-5,
                    help="natural-spline roughness penalty (default: 1e-5)")
    ap.add_argument("--calendar-grid", type=int, default=None,
                    help="calendar no-crossing grid; defaults by mode (fast=61, expanded=101, full=181)")
    ap.add_argument("--mode", choices=("full", "expanded", "fast"), default="full",
                    help="full research fit, broader confirmation, or compact target-tenor confirmation")
    ap.add_argument("--target-dte", type=float, default=30.0,
                    help="target tenor used by --mode fast (default: 30)")
    ap.add_argument("--max-maturities", type=int, default=None,
                    help="override maturity cap (mode defaults: fast=5, expanded=9)")
    ap.add_argument("--max-strikes", type=int, default=None,
                    help="override representative strikes per maturity (mode defaults: fast=60, expanded=90)")
    ap.add_argument("--solver", choices=("auto", "osqp", "slsqp"), default="auto",
                    help="QP solver; auto prefers OSQP and falls back to reduced SLSQP")
    ap.add_argument("--solver-tol", type=float, default=1e-9,
                    help="numerical solver tolerance (default: 1e-9)")
    ap.add_argument("--maxiter", type=int, default=2500,
                    help="maximum solver iterations per maturity")
    ap.add_argument("--quiet-progress", action="store_true",
                    help="suppress per-maturity timing output")
    ap.add_argument("--db", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-plots", default=None, metavar="DIR")
    args = ap.parse_args()

    raw = pd.read_parquet(args.chain)
    date = _date(raw, args.date)
    clean, _ = clean_chain(raw, CleanConfig(dte_range=tuple(args.dte)), verbose=True)
    slices = build_all_slices(clean, verbose=True)
    if len(slices) < 3:
        raise ValueError("need at least three calibratable slices")

    if args.mode == "fast":
        max_maturities = 5 if args.max_maturities is None else int(args.max_maturities)
        max_strikes = 60 if args.max_strikes is None else int(args.max_strikes)
        calendar_grid = 61 if args.calendar_grid is None else int(args.calendar_grid)
        slices = prepare_fengler_slices(
            slices,
            target_days=args.target_dte,
            max_maturities=max_maturities,
            max_strikes_per_slice=max_strikes,
        )
    elif args.mode == "expanded":
        max_maturities = 9 if args.max_maturities is None else int(args.max_maturities)
        max_strikes = 90 if args.max_strikes is None else int(args.max_strikes)
        calendar_grid = 101 if args.calendar_grid is None else int(args.calendar_grid)
        slices = prepare_fengler_slices(
            slices,
            target_days=args.target_dte,
            max_maturities=max_maturities,
            max_strikes_per_slice=max_strikes,
        )
    else:
        calendar_grid = 181 if args.calendar_grid is None else int(args.calendar_grid)

    print(
        f"\nFengler fit plan: mode={args.mode}, solver={args.solver}, "
        f"maturities={len(slices)}, strikes={[int(s.n) for s in slices]}, "
        f"calendar_grid={calendar_grid}"
    )

    def progress(event):
        if args.quiet_progress:
            return
        if event["event"] == "start_slice":
            print(
                f"  fitting {event['dte']:6.1f} DTE · {event['n']:3d} strikes · "
                f"{event['calendar_points']:3d} calendar constraints ...",
                flush=True,
            )
        else:
            print(
                f"    -> {event['solver']} {event['seconds']:.3f}s "
                f"success={event['success']}",
                flush=True,
            )

    started = perf_counter()
    fit = fit_fengler_surface(
        slices,
        smoothing_lambda=args.smoothing_lambda,
        calendar_grid_size=calendar_grid,
        solver=args.solver,
        solver_tol=args.solver_tol,
        maxiter=args.maxiter,
        progress=progress,
    )
    _summary(fit)
    print(
        f"\nFengler: RMSE={fit.rmse_iv*100:.3f}vp  "
        f"strike-arbitrage-free={fit.butterfly_free}  "
        f"calendar-free={fit.calendar_free}  reliable={fit.is_reliable}  "
        f"solver={fit.solver}  elapsed={perf_counter()-started:.3f}s"
    )

    surface = fit.to_surface(date, args.symbol)
    print(
        f"fixed-grid Fengler surface: {surface.n_slices_used} slices; "
        f"clean={surface.is_clean} (wing/tenor extrapolation is flagged)"
    )

    if args.db:
        db = VolDB(args.db)
        db.save_fengler_fit(args.symbol, date, fit)
        db.save_model_surface(args.symbol, date, "fengler", surface)
        print(f"saved Fengler run and surface to {args.db}")

    if args.plot or args.save_plots:
        import matplotlib.pyplot as plt
        from volforge.visualization import plot_fengler_diagnostics, plot_surface_3d
        figs = [plot_fengler_diagnostics(fit), plot_surface_3d(surface)]
        if args.save_plots:
            out = Path(args.save_plots)
            out.mkdir(parents=True, exist_ok=True)
            figs[0].savefig(out / "fengler_diagnostics.png", dpi=160, bbox_inches="tight")
            figs[1].savefig(out / "fengler_surface_3d.png", dpi=160, bbox_inches="tight")
            print(f"saved plots to {out}")
            if not args.plot:
                for fig in figs:
                    plt.close(fig)
        if args.plot:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

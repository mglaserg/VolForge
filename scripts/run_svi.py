#!/usr/bin/env python3
"""Run the raw-SVI pipeline from a saved option-chain snapshot.

This is the per-expiry companion to ``scripts/fit_ssvi.py``.  It preserves raw
SVI as the flexible local benchmark: each expiration is calibrated
independently, diagnostics are reported per slice, reliable slices are
interpolated onto VolForge's common fixed grid, and the results can be saved to
the database and/or plotted.

Examples
--------
    python scripts/run_svi.py \
        --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
        --symbol SPY --db volforge.db --plot

    python scripts/run_svi.py \
        --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
        --symbol SPY --save-plots figures/svi/2026-08-20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volforge import VolDB, build_surface, calibrate_svi
from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices


def _infer_trade_date(raw: pd.DataFrame, explicit_date: str | None):
    if explicit_date is not None:
        return pd.Timestamp(explicit_date).date()
    if "quote_time" not in raw.columns or raw.empty:
        raise ValueError("--date is required when the snapshot has no quote_time column")
    return pd.Timestamp(raw["quote_time"].iloc[0]).date()


def _print_fit_summary(pairs) -> None:
    rows = []
    for slc, fit in pairs:
        rows.append({
            "dte": slc.dte,
            "n": fit.n_obs,
            "rmse_vp": fit.rmse_iv * 100.0,
            "max_err_vp": fit.max_abs_err_iv * 100.0,
            "min_g": fit.min_durrleman_g,
            "butterfly": fit.butterfly_free,
            "reliable": fit.is_reliable,
        })

    summary = pd.DataFrame(rows).sort_values("dte")
    print("\nraw SVI slice summary")
    if summary.empty:
        print("  no calibrated slices")
        return

    print(summary.to_string(
        index=False,
        formatters={
            "dte": lambda x: f"{x:6.1f}",
            "rmse_vp": lambda x: f"{x:8.3f}",
            "max_err_vp": lambda x: f"{x:10.3f}",
            "min_g": lambda x: f"{x:9.4f}",
        },
    ))

    n_good = int(summary["reliable"].sum())
    n_bfly = int(summary["butterfly"].sum())
    print(
        f"\nraw SVI: {n_good}/{len(summary)} reliable slices; "
        f"{n_bfly}/{len(summary)} butterfly-free"
    )


def _render_plots(pairs, surface, save_dir: str | None, show: bool) -> None:
    import matplotlib.pyplot as plt
    from volforge.visualization import plot_slice_diagnostics, plot_surface_3d

    out = Path(save_dir) if save_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    for slc, fit in pairs:
        fig = plot_slice_diagnostics(slc, fit)
        if out is not None:
            dte = int(round(float(slc.dte)))
            fig.savefig(out / f"svi_{dte:03d}d.png", dpi=160, bbox_inches="tight")
            if not show:
                plt.close(fig)

    fig = plot_surface_3d(surface)
    if out is not None:
        fig.savefig(out / "svi_surface_3d.png", dpi=160, bbox_inches="tight")
        if not show:
            plt.close(fig)
        print(f"saved plots to {out}")

    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", required=True, help="saved raw chain parquet")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=None,
                    help="trade date; inferred from quote_time if omitted")
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 120.0],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--db", default=None, help="optional SQLite database path")
    ap.add_argument("--plot", action="store_true",
                    help="show per-slice diagnostics and the 3D SVI surface")
    ap.add_argument("--save-plots", default=None, metavar="DIR",
                    help="optional directory for PNG diagnostics")
    ap.add_argument("--include-unreliable-surface", action="store_true",
                    help="allow merely-successful (not reliable) SVI fits into the fixed grid")
    ap.add_argument("--no-calendar-repair", action="store_true",
                    help="do not monotonically repair total variance across the fixed tenor grid")
    args = ap.parse_args()

    raw = pd.read_parquet(args.chain)
    date = _infer_trade_date(raw, args.date)

    clean, report = clean_chain(
        raw,
        CleanConfig(dte_range=tuple(args.dte)),
        verbose=True,
    )
    slices = build_all_slices(clean, verbose=True)
    if not slices:
        raise ValueError("no calibratable slices remain after cleaning")

    pairs = [
        (slc, calibrate_svi(slc.k, slc.w, slc.T, weights=slc.weights))
        for slc in slices
    ]
    _print_fit_summary(pairs)

    surface = build_surface(
        pairs,
        date,
        args.symbol,
        reliable_only=not args.include_unreliable_surface,
        repair=not args.no_calendar_repair,
    )
    print(
        f"fixed-grid surface: {surface.n_slices_used} slices used; "
        f"calendar repair={surface.calendar_repair:.6g}; "
        f"clean={surface.is_clean}"
    )

    if args.db:
        db = VolDB(args.db)
        db.save_svi_fits(args.symbol, date, [(fit, slc) for slc, fit in pairs])
        db.save_surface(args.symbol, date, surface)  # legacy raw-SVI grid
        db.save_model_surface(args.symbol, date, "svi", surface)
        print(f"saved SVI fits and surface to {args.db}")

    if args.plot or args.save_plots:
        _render_plots(pairs, surface, args.save_plots, args.plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

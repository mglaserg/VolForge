"""
Clean and fit and plot
"""


from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices
from volforge.data.yahoo import fetch_chain, save_snapshot
from volforge.svi import calibrate_svi
import volforge.visualization as viz

import matplotlib.pyplot as plt
from volforge import (VolDB, calibrate_svi, build_surface, surface_panel,
                      feature_panel, fit_surface_pca, pca_residuals,
                      zscore_series, forward_convergence)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", nargs=1)
    ap.add_argument("--dte", nargs=2, type=float, default=[7.0, 120.0],
                        metavar=("MIN", "MAX"))
    args = ap.parse_args()
    print(args.chain)
    raw = pd.read_parquet(args.chain[0])
    print(args.chain[0])
    clean, _ = clean_chain(raw, CleanConfig(dte_range=tuple(args.dte)),
                                       verbose=True)
    
    slices = build_all_slices(clean, verbose=True)
    pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]
    build_surface(pairs, '2026-08-20', "LULU")
    # db.save_surface("SPY", date, surf)
    for p in pairs:
        viz.plot_slice_diagnostics(*p)
    plt.show()
if __name__ == "__main__":
    sys.exit(main())
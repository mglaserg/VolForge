"""Full-stack test: simulate a year of trading, run every phase end to end.

Generates 260 days of synthetic SPY chains driven by a stochastic vol process
with a genuine spot-vol correlation, then runs: clean -> slices -> calibrate ->
database -> surface -> features -> PCA -> signals -> plots.

The point is not that the numbers are right, it is that the whole chain runs
and the diagnostics report honestly on data whose truth we control.
"""

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sys.path.insert(0, "../volforge/src")
warnings.filterwarnings("ignore")

from volforge import SVIParams, calibrate_svi
from volforge.blackscholes import black_price, implied_vol_vec
from volforge.data.clean import CleanConfig, clean_chain
from volforge.data.pipeline import build_all_slices
from volforge.data.schema import add_derived_columns, expiry_datetime
from volforge.database import VolDB
from volforge.features import feature_panel, standardize
from volforge.pca import fit_surface_pca, pca_residuals, reconstruct
from volforge.signals import (bucket_by_signal, forward_convergence,
                              mean_reversion_test, zscore_series)
from volforge.surface import build_surface, surface_panel
from volforge import visualization as viz

N_DAYS = 260
# Grid tenors must lie inside the simulated expiry range (10-75 DTE), or every
# surface is correctly flagged as extrapolated and Surface.is_clean goes False.
TENORS = np.array([14.0, 21.0, 30.0, 45.0, 60.0])
SYMBOL = "SPY"
rng = np.random.default_rng(2024)


# ------------------------------------------------------------------ simulate
def simulate_day(day, spot, atm_level, skew_level):
    """One day's chain across four expiries."""
    quote_time = pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(days=day)
    rows = []
    for dte in (10, 25, 45, 75):
        T = dte / 365.25
        # Term structure of vol and skew, anchored on the day's state.
        atm = atm_level * (1.0 + 0.14 * np.log(dte / 30.0))
        sig = 0.055 + 0.0016 * dte
        m = 0.010 + 0.00022 * dte
        rho = np.clip(skew_level, -0.92, -0.35)
        base = rho * (-m) + np.sqrt(m * m + sig * sig)
        b = 0.0045 + 0.00028 * dte
        a = atm ** 2 * T - b * base
        if a <= 0:
            a = 1e-6
        p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sig)

        F = spot * np.exp(0.043 * T)
        K = np.round(np.arange(spot * 0.80, spot * 1.20, 5.0) / 5.0) * 5.0
        k = np.log(K / F)
        iv = np.sqrt(np.maximum(a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2)),
                                1e-8) / T)
        D = np.exp(-0.043 * T)
        exp_ts = expiry_datetime((quote_time + pd.Timedelta(days=dte)).date())

        for right, is_call in (("C", True), ("P", False)):
            mid = np.asarray(black_price(F, K, iv, T, is_call, D), float)
            half = np.maximum(0.01, 0.015 + 0.55 * np.abs(k) ** 1.4) * np.maximum(mid, 0.3) ** 0.3
            mid = mid + rng.normal(0, half / 5)
            rows.append(pd.DataFrame({
                "symbol": SYMBOL, "quote_time": quote_time, "expiry": exp_ts,
                "strike": K, "right": right,
                "bid": np.maximum(mid - half, 0).round(2),
                "ask": (mid + half).round(2),
                "underlying_price": spot,
                "volume": rng.poisson(np.maximum(1500 * np.exp(-7 * np.abs(k)), 1.0)).astype(float),
                "open_interest": rng.poisson(np.maximum(6000 * np.exp(-4 * np.abs(k)), 2.0)).astype(float),
                "last_trade_time": quote_time,
            }))
    return add_derived_columns(pd.concat(rows, ignore_index=True))


print(f"simulating {N_DAYS} trading days ...")
spot, atm, skew = 600.0, 0.15, -0.75
spots, surfaces, all_fits = [], [], []
db = VolDB(":memory:")

for day in range(N_DAYS):
    # Spot-vol correlation is the dominant real effect; build it in explicitly.
    z = rng.standard_normal()
    ret = 0.0085 * z
    spot *= np.exp(ret - 0.5 * 0.0085 ** 2)
    atm = float(np.clip(atm + 0.045 * (0.15 - atm) - 0.55 * ret + 0.004 * rng.standard_normal(), 0.08, 0.45))
    skew = float(np.clip(skew + 0.05 * (-0.75 - skew) + 0.018 * rng.standard_normal(), -0.92, -0.40))

    chain = simulate_day(day, spot, atm, skew)
    clean, _ = clean_chain(chain, CleanConfig(dte_range=(5, 120)), verbose=False)
    slices = build_all_slices(clean, verbose=False)

    pairs = []
    for s in slices:
        try:
            f = calibrate_svi(s.k, s.w, s.T, weights=s.weights)
            pairs.append((s, f))
        except Exception:
            continue
    if len(pairs) < 2:
        continue

    date = chain["quote_time"].iloc[0].date()
    db.save_svi_fits(SYMBOL, date, [(f, s) for s, f in pairs])

    try:
        surf = build_surface(pairs, date, SYMBOL, tenor_days=TENORS, reliable_only=True)
    except ValueError:
        surf = build_surface(pairs, date, SYMBOL, tenor_days=TENORS, reliable_only=False)
    db.save_surface(SYMBOL, date, surf)
    surfaces.append(surf)
    spots.append((pd.Timestamp(date), spot))
    all_fits.extend(pairs)

    if (day + 1) % 65 == 0:
        print(f"  day {day+1}: spot={spot:.1f} atm={atm:.3f} surfaces={len(surfaces)}")

print(f"\n[1] built {len(surfaces)} daily surfaces")
print(f"    db rows: {db.table_counts()}")
clean_frac = np.mean([s.is_clean for s in surfaces])
print(f"    clean surfaces: {clean_frac:.1%}")

fits_df = db.load_svi_fits(SYMBOL)
print(f"    reliable slice fits: {fits_df['is_reliable'].mean():.1%} of {len(fits_df)}")

# ------------------------------------------------------------------ features
feats = feature_panel(surfaces)
for s in surfaces:
    db.save_features(SYMBOL, s.trade_date, {k: v for k, v in
                                            __import__("volforge.features", fromlist=["x"])
                                            .surface_features(s).items()})
print(f"\n[2] features: {feats.shape[0]} days x {feats.shape[1]} columns")
print(f"    ATM 30d: {feats['atm_30d'].mean()*100:.2f}% mean, "
      f"{feats['atm_30d'].std()*100:.2f}% std")
print(f"    skew 30d: {feats['skew_30d'].mean():+.3f} mean "
      f"(negative = normal equity smile)")
zf = standardize(feats[["atm_30d", "skew_30d", "curvature_30d", "term_slope"]],
                 window=None, min_periods=60)
print(f"    z-scores available from day {int(zf['atm_30d'].notna().argmax())}")

# ----------------------------------------------------------------------- PCA
panel = surface_panel(surfaces)
spot_s = pd.Series(dict(spots))
print(f"\n[3] surface panel: {panel.shape[0]} days x {panel.shape[1]} nodes")

model_raw, scores_raw = fit_surface_pca(panel, n_components=5, spot=None)
print(f"    without spot-neutralisation: PC1 = {model_raw.explained_variance_ratio[0]:.1%}")

model, scores = fit_surface_pca(panel, n_components=5, spot=spot_s)
print(f"    with    spot-neutralisation: PC1 = {model.explained_variance_ratio[0]:.1%}")
print(f"    explained: {np.round(model.explained_variance_ratio*100,1)}  "
      f"(cum {model.explained_variance_ratio.cumsum()[-1]:.1%})")
print(model.describe().to_string(index=False,
      float_format=lambda x: f"{x:6.3f}"))

db.save_pca_model("m1", SYMBOL, model)
n = db.save_pca_scores(SYMBOL, "m1", scores)
print(f"    saved {n} score rows; reload matches: "
      f"{np.allclose(db.load_pca_scores(SYMBOL,'m1').to_numpy(), scores.to_numpy(), atol=1e-9)}")

# --------------------------------------------------------------- PCA residuals
X = panel.diff().dropna().reindex(scores.index)
resid = pca_residuals(model, X, scores, n_pcs=3)
print(f"\n[4] PCA residuals: {resid.shape}, "
      f"std {resid.to_numpy().std():.2e} vs input std {X.to_numpy().std():.2e} "
      f"({resid.to_numpy().std()/X.to_numpy().std():.1%} unexplained by 3 PCs)")

# -------------------------------------------------------------------- signals
node = resid.columns[len(resid.columns) // 2]
r = resid[node]
sig = zscore_series(r, window=120, min_periods=40)
print(f"\n[5] signals on node {node}")
print("    NOTE: this residual is in std units, not half-spreads, so the cost")
print("    charge is set to 0 and the verdict is NOT a tradeability claim.")
print("    On real data use residual_signal() so costs mean something.")
for h in (1, 5, 10):
    print("   ", forward_convergence(sig, r / r.std(), horizon=h,
                                     cost_spreads=0.0, min_signal=1.0))
print("    mean reversion of PC1:")
print(mean_reversion_test(scores[1]).to_string(index=False,
      float_format=lambda x: f"{x:8.3f}"))

buckets = bucket_by_signal(sig, r / r.std(), horizon=5, cost_spreads=0.0)
print("\n    convergence by bucket (h=5):")
print(buckets.to_string(float_format=lambda x: f"{x:7.3f}"))

# ---------------------------------------------------------------------- plots
print("\n[6] plots")
out = "/home/claude/volforge/figures"
import os
os.makedirs(out, exist_ok=True)

s0, f0 = all_fits[len(all_fits) // 2]
viz.plot_slice_diagnostics(s0, f0).savefig(f"{out}/slice_diagnostics.png", dpi=110)
viz.plot_surface_3d(surfaces[-1]).savefig(f"{out}/surface_3d.png", dpi=110)
viz.plot_pca_loadings(model, n=4).savefig(f"{out}/pca_loadings.png", dpi=110)

fig, axes = plt.subplots(3, 1, figsize=(11, 10))
viz.plot_term_structure(surfaces, ax=axes[0])
viz.plot_pca_scores(scores, ax=axes[1])
viz.plot_signal_buckets(buckets, ax=axes[2])
fig.tight_layout()
fig.savefig(f"{out}/timeseries.png", dpi=110)

fig, ax = plt.subplots(figsize=(9, 4))
viz.plot_pca_residual_heatmap(resid, ax=ax)
fig.tight_layout()
fig.savefig(f"{out}/pca_residual_heatmap.png", dpi=110)
plt.close("all")

print(f"    wrote {len(os.listdir(out))} figures to {out}")
print("\nFULL STACK OK")

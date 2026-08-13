# VolForge

SVI volatility-surface calibration and relative-value research.

## Status

All phases through 12 are implemented and tested end to end: data ingestion,
quote cleaning, forward extraction, IV inversion, SVI calibration, arbitrage
diagnostics, the parameter database, surface construction, features, PCA,
signal evaluation, and plotting.

Phase 13 (trading research) is deliberately not written. Trade construction
depends on what the signal tests actually show, and writing it now would mean
guessing. The gate is in place: run `forward_convergence` on real residuals
first.

## Install

```bash
git clone <your-repo> volforge && cd volforge
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[data,viz,dev]"
pip install yfinance          # only needed for the Yahoo adapter
```

## Quickstart

```python
from volforge.data.yahoo import fetch_chain
from volforge.data.clean import clean_chain, CleanConfig
from volforge.data.pipeline import build_all_slices
from volforge import calibrate_svi, residual_report

raw = fetch_chain("SPY", dte_range=(7, 120))
clean, report = clean_chain(raw, CleanConfig())
print(report)                              # the drop funnel -- read it every run

for s in build_all_slices(clean):
    fit = calibrate_svi(s.k, s.w, s.T, weights=s.weights)
    print(s, fit.params, f"rmse={fit.rmse_iv*100:.2f}vp",
          f"reliable={fit.is_reliable}", fit.boundary_flags)
```

Accumulate history (yfinance has no historical endpoint — capture or lose it):

```bash
python scripts/snapshot.py --symbols SPY --dte 7 120
# crontab: 30 12 * * 1-5  cd /path/to/volforge && .venv/bin/python scripts/snapshot.py
```

## Layout

```
src/volforge/
├── blackscholes.py    Black-76 in forward space, IV inversion
├── forward.py         put-call parity regression -> F and D jointly
├── svi.py             raw SVI, quasi-explicit calibration, arbitrage checks
├── diagnostics.py     residuals measured in half-spread units
├── surface.py         all expiries -> fixed grid, calendar repair
├── features.py        ATM, skew, curvature, wing asymmetry, term structure
├── pca.py             PCA on surface changes, spot-neutralised
├── signals.py         z-scores, convergence tests, bucket analysis
├── database.py        SQLite persistence, upsert on natural keys
├── visualization.py   smiles, residuals, 3D surface, PCA diagnostics
└── data/
    ├── schema.py      canonical chain schema, settlement conventions
    ├── yahoo.py       yfinance adapter + Parquet snapshots
    ├── clean.py       quote filters with a per-step drop report
    └── pipeline.py    cleaned chain -> calibration-ready Slice
```

## Design decisions, and why

**We never fit a vendor's implied vol.** Vendor IV embeds undocumented forward,
rate, and dividend assumptions. Fitting it means your residuals measure the
vendor's model, not the market. We invert IV ourselves from mid prices using
our own parity-implied forward. `vendor_iv` is retained only as a sanity check.

**The forward comes from a regression, not a single ATM pair.** `C(K) - P(K)`
regressed on `K` gives slope `-D` and intercept `D*F`, so F and D fall out
jointly and the R² is a free data-quality diagnostic. Single-strike parity is
noisy, and you would later mistake that noise for a modelling failure.

**Calibration is quasi-explicit, not naive 5-parameter least squares.**
Substituting `y = (k-m)/sigma` makes the model linear in `(a, d, c)`, reducing
a non-convex 5-D problem to a 2-D search over `(m, sigma)` with an exact convex
solve inside. Naive fitting produces day-to-day parameter jitter that swamps
any time-series signal. Measured stability: ATM vol varies 0.008 vol points
across independent noise draws of the same slice.

**Boundary-pinned fits are flagged, not trusted.** A slice that returns
`rho = -1.000` with a low RMSE looks fine and plots fine but carries no
information — the optimiser ran out of room, usually because too few quotes
survived. Filter on `SVIFit.is_reliable` before anything enters a PCA, or
you will manufacture a principal component out of an artifact.

**Weights match a spread-relative price objective.** Since
`price_err ≈ vega · iv_err` and `iv_err ≈ w_err / (2·iv·T)`, the weight is
`(vega / (2·iv·T·half_spread))²`. Equal weighting lets a 2-cent wing quote
count as much as a $12 ATM one.

**Raw snapshots are saved, cleaned data is not.** Cleaning rules will change as
you learn what the quotes actually look like. You cannot un-drop a quote you
never saved.

## The question to answer before building anything else

`diagnostics.residual_report` expresses each fit residual as a multiple of that
option's own half-spread. On liquid names, SVI residuals are largely model
misspecification and quote noise rather than mispricing. If very little clears
one full spread, there is no tradeable signal, and phases built on top of the
residuals will produce backtests that die on contact with costs.

Run it on day one. It is a cheap answer to an expensive question.

## Tests

```bash
python tests/smoke.py               # model layer: IV round-trip, param recovery, stability
python tests/test_data_pipeline.py  # data layer, against a yfinance mock (no network)
python tests/test_full_stack.py     # 260 simulated days through every phase (~4 min)
```

The full-stack test simulates a year of chains from a stochastic vol process
with a built-in spot-vol correlation, then runs the whole pipeline and writes
diagnostic figures to `figures/`. It is a wiring check, not a validation of
edge: the data is synthetic, so a positive signal result there means the code
computes what it claims, nothing more.

The mock reproduces yfinance's output shape from documentation, not a live
call. Verify the column names against a real response before relying on it.

## Full pipeline

```python
from volforge import (VolDB, calibrate_svi, build_surface, surface_panel,
                      feature_panel, fit_surface_pca, pca_residuals,
                      zscore_series, forward_convergence)
from volforge import visualization as viz

db = VolDB("volforge.db")
pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]
db.save_svi_fits("SPY", date, [(f, s) for s, f in pairs])

surf = build_surface(pairs, date, "SPY")      # fixed (tenor x k) grid
db.save_surface("SPY", date, surf)
viz.plot_slice_diagnostics(*pairs[0])          # the four-panel sanity check

# once you have a year or more of daily surfaces:
panel = surface_panel(surfaces)
model, scores = fit_surface_pca(panel, spot=spot_series)   # spot-neutralised
resid = pca_residuals(model, panel.diff().dropna(), scores, n_pcs=3)
print(forward_convergence(zscore_series(resid[node]), resid[node], horizon=5))
```

## Two more decisions worth knowing about

**`a < 0` is allowed.** The classical Zeliade domain imposes `a >= 0` as a cheap
guarantee of positive total variance. On short-dated slices the optimum
genuinely wants `a < 0`, so the constraint binds and pins the parameter. In
testing this biased every fit, not merely the flagged ones — recovered rho moved
from -0.7445 to the true -0.7500 once the restriction was lifted. Positivity is
now enforced exactly, via `min w = a + sqrt(c^2 - d^2) > 0` in the outer search,
which keeps the inner problem linear. Enabling this took reliable fits from 52%
to 95% on the simulated year.

**Calibration is an exact active-set QP inside a coarse grid search.** The inner
3-variable problem is solved by active-set iteration rather than a
general-purpose optimiser, and the outer `(m, sigma)` search sweeps a coarse
grid before refining the best cells instead of using random restarts. Together
these cut a fit from roughly 460ms to 50ms, which is the difference between a
multi-year calibration taking hours and taking minutes. `calibrate_svi(..., x0=)`
warm-starts from the previous day's `(m, sigma)`.

## Not yet built

Phase 13: delta- and vega-neutral trade construction, structure selection
(butterflies, verticals, calendars, condors), turnover, and out-of-sample
backtesting. Do the Phase 8 gate first.

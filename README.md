# VolForge

Volatility-risk-premium measurement, forward realized-volatility forecasting, and optional surface diagnostics.

## Status

VolForge is now primarily a **volatility-risk-premium (VRP) research and decision tool**.
The core path is deliberately model-light:

`option chains -> raw-strip MFIV -> integrated realized variance -> forward-RV forecasts -> VRP candidates`

The calibrated volatility surfaces (SVI, SSVI, eSSVI and Fengler) remain in the
repository, but they are **optional Advanced Surface Diagnostics** rather than a
requirement for ordinary VRP history, forecasting or candidate screening. They
are useful for smoothing, arbitrage diagnostics, sparse-strike interpolation and
separate surface-relative-value research.

The active forecasting hurdle is Persistence -> HAR -> HEAVY-RM. XGBoost and
quantile XGBoost are implemented in Model Lab as experimental models and do not
graduate to the main decision view unless they improve purged out-of-sample
forecasting.

## Install

```bash
git clone <your-repo> volforge && cd volforge
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[data,viz,dev,ml]"
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

## SSVI quickstart

Raw SVI and SSVI have different jobs. Raw SVI remains the flexible fit for one
expiry. SSVI couples those maturities through ATM total variance and gives a
global term-structure benchmark.

```python
from volforge import calibrate_svi, calibrate_ssvi, build_surface

pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]
ssvi = calibrate_ssvi(pairs)

print(ssvi.params)
print(f"global RMSE: {ssvi.rmse_iv*100:.3f} vol points")
print(f"theta repair: {ssvi.theta_repair_fraction:.2%}")
print(ssvi.butterfly_free, ssvi.calendar_free)

# arbitrary point on the continuous surface; T is in years
T30 = 30 / 365.25
print(ssvi.implied_vol(k=-0.10, T=T30))

# evaluate on the same fixed grid used by PCA and raw SVI
ssvi_surface = ssvi.to_surface("2026-08-20", "SPY")
```

The most useful comparison is the independent-slice surface against the coupled
SSVI surface:

```python
raw_surface = build_surface(pairs, "2026-08-20", "SPY")
diff = raw_surface.total_var - ssvi_surface.total_var
```

That difference asks: *where does an individual expiry want to depart from the
globally coherent term structure?* It is a measurement, not yet an alpha claim.

### One-command SSVI diagnostic run

```bash
python scripts/fit_ssvi.py \
    --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
    --symbol SPY \
    --db volforge.db \
    --plot
```

This cleans the saved chain, calibrates every raw-SVI slice, fits SSVI, writes
both model grids without overwriting one another, and opens the SSVI diagnostic
figure.

### What is stored

`ssvi_parameters` stores the daily global parameters `(rho, eta, gamma)`, the
ATM theta curve, fit quality, repair amount, and arbitrage diagnostics.
`modeled_surface_grid` stores model-tagged fixed-grid nodes, so `svi`, `ssvi`,
and later `essvi` / `fengler` can coexist for the same symbol and date.

```python
db.save_ssvi_fit("SPY", date, ssvi)
db.save_model_surface("SPY", date, "svi", raw_surface)
db.save_model_surface("SPY", date, "ssvi", ssvi_surface)

svi_hist = db.load_model_surface_panel("SPY", "svi")
ssvi_hist = db.load_model_surface_panel("SPY", "ssvi")
```

## eSSVI quickstart

```bash
python scripts/run_essvi.py \
    --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
    --symbol SPY --db volforge.db --plot
```

The runner calibrates raw SVI, nested SSVI, and eSSVI on the same chain and reports
the eSSVI fit improvement. The eSSVI correlation family is
`rho(theta)=rho0+(rho_m-rho0)(theta/theta_max)^a` over the calibrated horizon.
The continuous Hendriks-Martini calendar condition and slice-wise SSVI butterfly
conditions are enforced during calibration. Results are stored in
`essvi_parameters` and `modeled_surface_grid` with `model='essvi'`.

## Fengler quickstart

```bash
python scripts/run_fengler.py \
    --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
    --symbol SPY --lambda 1e-5 --db volforge.db --plot
```

Fengler is intentionally different from SVI-family models: it smooths
forward-normalised call prices with a natural cubic spline. Convexity, monotonicity,
price bounds, endpoint slope bounds, and dense equal-forward-moneyness calendar
constraints are imposed in the spline optimisation. The resulting surface is
converted back to total variance only after the price surface is clean. Results are
stored in `fengler_runs` and the common grid with `model='fengler'`.

The smoothing parameter `--lambda` is explicit in v0.4; automated AIC/GCV selection
is deliberately left as a later calibration enhancement rather than hidden inside
the first implementation.

**SPY caveat:** the theoretical Fengler construction is for European option prices.
SPY options are American-style, so early-exercise/dividend effects can contaminate
strict parity/arbitrage diagnostics. SPY remains useful for workflow research, but
SPX is the cleaner validation market for the exact European theory.

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
├── term_structure.py  monotone ATM total-variance clock for SSVI/eSSVI
├── ssvi.py            coupled SSVI surface + calibration + no-arbitrage checks
├── essvi.py           maturity-dependent-rho eSSVI + calendar conditions
├── fengler.py         constrained natural call-price smoothing splines
├── diagnostics.py     residuals measured in half-spread units
├── surface.py         all expiries -> fixed grid, calendar repair
├── features.py        ATM, skew, curvature, wing asymmetry, term structure
├── delta_surface.py   10Δ/15Δ/25Δ surface, ratios, lumps, surface-change features
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

## Roadmap

The active roadmap is now about data quality, forward realized-volatility
forecasting and a simple VRP decision screen:

1. **Alpaca intraday archive** — implemented. Free Alpaca/IEX 5-minute history is
   stored locally with provider/feed tags so later SIP data cannot be mixed in
   silently.
2. **Integrated realized variance** — implemented from the retained intraday
   archive, including the overnight close/open gap.
3. **Script-owned VRP history** — implemented. The dashboard is a reader;
   `scripts/build_vrp_history.py` is the canonical rebuild path and prints chain,
   RV, row and forward-label coverage.
4. **Persistence / HAR / HEAVY-RM** — implemented as the benchmark forecasting
   ladder with purged walk-forward evaluation. These models train directly from
   the standalone Alpaca daily realized-variance archive; they do **not** require
   option-chain/VRP history.
5. **XGBoost + q70 XGBoost** — implemented experimentally in Model Lab. They must
   beat the simpler models out of sample before being promoted.
6. **Probability / distribution view** — next: convert the forecast distribution
   into quantities such as `P(MFIV > future RV)` rather than another chart.
7. **Simple decision screen** — keep the default dashboard focused on candidate,
   MFIV, forecast RV, expected VRP, regime and a short explanation.
8. **Surface-RV falsifier** — separately run the residual-in-half-spreads test on
   real SPX chains. Calibrated surfaces remain optional unless that experiment
   earns further research budget.
9. **Trade construction** — only after the edge estimate is validated: structure,
   sizing, costs, tail risk and portfolio integration.

SPX is the preferred reference market for strict European-option methodology
validation; SPY remains useful for workflow and retail implementation testing.

## Raw SVI runner

Run the complete per-expiry SVI pipeline from a saved raw chain snapshot:

```bash
python scripts/run_svi.py \
  --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
  --symbol SPY \
  --db volforge.db \
  --plot
```

To write the diagnostic figures without opening matplotlib windows:

```bash
python scripts/run_svi.py \
  --chain data/chains/symbol=SPY/date=2026-08-20/chain.parquet \
  --symbol SPY \
  --db volforge.db \
  --save-plots figures/svi/2026-08-20
```

The runner saves all raw SVI slice parameters to `svi_parameters`, the legacy
fixed-grid surface to `surface_grid`, and the model-tagged SVI grid to
`modeled_surface_grid` with `model='svi'`.  By default only reliable SVI slices
are allowed into the fixed-grid surface and calendar total-variance inversions
are repaired while the repair amount is retained on the `Surface` object.

## Forward VRP dashboard

VolForge includes a Streamlit dashboard, but data-building jobs intentionally
live in scripts. The dashboard should answer the VRP question quickly rather
than behave like an ETL console.

Install the dashboard/data extras. Add `viz` for optional 3D diagnostics and
`ml` for experimental XGBoost:

```bash
pip install -e ".[dashboard,data,viz,ml]"
```

Run it from the repository root:

```bash
streamlit run volforge_dashboard.py
```

On Windows:

```text
scripts\run_dashboard.bat
```

The default workflow is:

```bash
# 1. Update retained 5-minute Alpaca/IEX bars and daily integrated variance.
python scripts/update_intraday.py --symbol SPY

# 2. Capture option chains using the normal capture job / batch file.

# 3. Rebuild compact VRP history.
python scripts/build_vrp_history.py --symbol SPY --provider yahoo

# 4. Open the dashboard; it reads the finished history.
streamlit run volforge_dashboard.py
```

VolForge loads local credentials from a repository-root `.env` file via
`python-dotenv`. Copy the committed template once:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then fill in the services you use:

```dotenv
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
ORATS_API_TOKEN=
```

The real `.env` is git-ignored; `.env.example` is safe to commit. Existing
process environment variables take precedence over `.env`, so CI or scheduled
jobs can still inject secrets normally.

Alpaca's free research path defaults to the `iex` feed and writes to:

```text
data/intraday/provider=alpaca/feed=iex/symbol=SPY/bars_5min.parquet
data/realized/provider=alpaca/feed=iex/symbol=SPY/daily_variance.parquet
```

This feed separation is intentional. IEX is suitable for building and testing
the research pipeline; later SIP-quality data can be compared without rewriting
or contaminating the existing archive.

The dashboard now emphasizes:

- current raw-strip MFIV versus integrated RV;
- a compact VRP-candidate/regime summary;
- one primary volatility term-structure chart;
- one primary MFIV-vs-RV history chart, with secondary diagnostics collapsed;
- **Model Lab** for Persistence, HAR and HEAVY benchmarks trained from the long
  realized-variance archive, plus experimental XGBoost/q70 XGBoost trained from
  the scarcer option-chain/VRP history; and
- **Advanced Surface Diagnostics** for optional SVI/SSVI/eSSVI/Fengler work.

### Advanced Surface Diagnostics

The calibrated surfaces are deliberately off the core path. Use this page for
smoothing, static-arbitrage checks, sparse-strike interpolation, raw-vs-smoothed
MFIV comparisons, or explicitly registered surface-RV research. A visually
interesting 3D lump is not an alpha result.

### Delta surface and delta-ratio features

VolForge also builds a deliberately simple delta-space representation directly
from its own parity/IV pipeline. It does **not** require SVI/SSVI/eSSVI/Fengler:

```python
from volforge import build_delta_surface, constant_tenor_delta_slice

delta_surface = build_delta_surface(chain, dte_range=(7, 180))
d30 = constant_tenor_delta_slice(delta_surface, 30)
print(d30[["atm_iv", "delta_ratio_25p", "delta_ratio_25c"]])
```

For each actual expiry, VolForge interpolates IV to standard 10Δ, 15Δ and 25Δ
put/call buckets and ATM. Across maturity, it interpolates **total variance**
(`IV² × time`) at fixed delta rather than interpolating IV directly. Delta ratios
are then simply `IV(delta) / IV(ATM)`. The historical VRP builder persists the
raw bucket IVs, ratios, prior-only z-scores/percentiles, local term-structure lump
residuals, and daily ATM/skew/convexity change features.

The intended research split is: simple observable features are the primary
signal inputs; SSVI/eSSVI/Fengler remain smoothing, arbitrage and data-quality
confirmation tools. Model Lab deliberately separates the datasets: Persistence/HAR/HEAVY
use the standalone realized archive, while experimental XGBoost uses saved-chain VRP
features. Sparse option history therefore does not block the realized-volatility models.

### Build / update VRP history from scripts

VRP history is intentionally script-owned. `scripts/build_vrp_history.py` reports
how many option snapshots/dates it found, realized-variance coverage, rows
written and completed forward labels. This makes stale history a visible data
problem instead of a silent dashboard state problem.

### VRP regime guide

The dashboard now includes a **How to use** page that can be opened without
fetching market data.  It teaches the intended reading sequence rather than
turning `RV3 - RV30` into a mechanical trade rule.

The live page classifies the realized-volatility context as one of several
states, including **Shock underway** and **Post-shock / IV still elevated**.
The latter requires more than `RV3 < RV30`: the recent RV slope must previously
have been positive and implied variance must still exceed trailing integrated
RV.  This captures the cooling-after-shock pattern discussed in the VRP
research notes.

Headline MFIV on the core dashboard is the raw model-free strip. SVI/SSVI/eSSVI/Fengler
no longer determine the headline value or gate a VRP candidate; disagreements
between raw and smoothed surfaces are optional diagnostics.

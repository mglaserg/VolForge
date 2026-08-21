# VolForge Alpha Research Playbook

VolForge is **not alpha by itself**. It is infrastructure for measuring option-market structure consistently enough that alpha hypotheses can be tested without confusing bad data, interpolation artifacts, or model instability for an edge.

The research loop is:

```text
measure cleanly
    ↓
find something unusual
    ↓
define what should happen next
    ↓
test it historically
    ↓
map it to an executable trade
    ↓
subtract realistic costs
    ↓
validate out of sample
    ↓
promote or kill the idea
```

The most important rule in this document is:

> **Do not turn a surface anomaly into a trade until you have tested whether it predicts a future outcome.**

## If you are lost in the details, do only this

```text
1. Measure one thing today.
2. Define why it is unusual.
3. State exactly what you predict will happen next.
4. Test that future outcome without look-ahead.
5. Try to break the result across time, nodes, models, and data choices.
6. Build the simplest trade that captures the surviving prediction and subtract costs.
7. Validate out of sample. Keep it only if it still works.
```

Everything else in this document is an implementation detail of those seven steps.

---

# 1. Separate measurement from prediction

VolForge's surface models answer measurement questions:

- raw SVI: what does this individual expiry look like?
- SSVI: what does a globally coupled term structure imply?
- eSSVI: does allowing maturity-dependent skew materially improve that global description?
- Fengler: what does a nonparametric arbitrage-constrained price surface imply?
- PCA: what common factors explain historical surface changes?

None of those statements is a trading edge.

Alpha begins only when today's measurement predicts something about the future.

For a signal `S_t` and a future outcome `Y_(t+h)`, the core question is:

```text
Does E[Y_(t+h) | S_t] change meaningfully with S_t?
```

Examples:

```text
Measurement:
30D / -0.20 log-moneyness raw SVI is far above SSVI.

Prediction hypothesis:
That disagreement tends to shrink over the next 1-5 trading days.

Potential trade:
Express the convergence with an option structure only after the prediction survives testing.
```

---

# 2. Build the historical measurement layer first

For every research date, preserve the raw chain and generate the same measurements using the same rules.

Recommended daily sequence:

```text
snapshot.py
    ↓
run_svi.py
    ↓
fit_ssvi.py
    ↓
run_essvi.py
    ↓
run_fengler.py
    ↓
modeled_surface_grid + diagnostics in the database
```

At minimum, retain:

- raw bid and ask
- quote timestamp
- strike and expiry
- forward estimate and diagnostics
- cleaned/included/excluded status
- model parameters
- model reliability flags
- butterfly/calendar diagnostics
- total variance on the common grid
- implied volatility on the common grid
- extrapolation flags
- model repair amounts where applicable

Never overwrite the original market observation with a repaired or smoothed value without retaining both.

## Why the common grid matters

Comparisons should usually be made at constant coordinates such as:

```text
T = 30 days
k = -0.20
```

rather than comparing a specific strike or listed expiration through time.

A fixed `(tenor, log-moneyness)` grid prevents ordinary passage of time from masquerading as a signal.

---

# 3. Create candidate signals

A candidate signal is simply a measurable condition that **might** contain predictive information.

Do not optimize it yet.

## A. Market-versus-model residuals

Example:

```text
market IV - SVI IV
```

Prefer spread-aware versions when working directly with quotes:

```text
(market IV - model IV) / IV half-spread
```

Question:

> Is this option rich or cheap relative to its local smile by enough to matter versus execution width?

## B. Local-versus-global surface disagreement

Example:

```text
SVI total variance - SSVI total variance
```

Question:

> Is this individual expiry departing from the globally coupled term structure?

## C. SSVI-versus-eSSVI disagreement

Example:

```text
SSVI total variance - eSSVI total variance
```

Question:

> Is the constant-skew/correlation assumption in vanilla SSVI unusually restrictive here?

## D. Parametric-versus-nonparametric disagreement

Example:

```text
SVI total variance - Fengler total variance
```

or

```text
eSSVI total variance - Fengler total variance
```

Question:

> Does an apparent anomaly survive when the surface is estimated by a very different model family?

## E. Model dispersion

At one grid node:

```text
Std(SVI, SSVI, eSSVI, Fengler)
```

Interpretation:

- high dispersion: the measurement is model-sensitive; be cautious
- low dispersion: several models broadly agree about fair surface shape

A large market residual combined with low model dispersion is more interesting than a residual produced by one idiosyncratic model.

## F. Interpretable shape features

Examples:

- ATM variance
- downside skew
- wing asymmetry
- curvature
- front-versus-back term structure
- change in each feature
- historical z-score of each feature

## G. PCA factors

After enough history exists, candidate signals include:

- extreme PC scores
- changes in PC scores
- PCA reconstruction residuals
- node-level deviations not explained by common factors

PCA is a compression tool first. Do not assume PC1, PC2, or a reconstruction residual is predictive merely because it looks economically interpretable.

---

# 4. Freeze the hypothesis before testing it

Write the hypothesis down before looking at the forward returns.

A useful template is:

```text
Signal:
What is measured at time t?

Universe:
Which symbol(s), DTE range, moneyness range, liquidity filters?

Direction:
Do I expect mean reversion, momentum, or another conditional effect?

Horizon:
1 day, 3 days, 5 days, etc.

Outcome:
What exact future quantity should respond?

Reason:
Why might this relationship exist economically or structurally?

Invalidation:
What result would make me abandon the idea?
```

Example:

```text
Signal:
SPY SVI minus SSVI total variance at 30D / k=-0.20.

Direction:
Mean reversion.

Horizon:
1, 3, and 5 trading days.

Outcome:
Change in the same constant-tenor, constant-moneyness model gap.

Reason:
A local expiry may temporarily deviate from the broader term-structure constraint.

Invalidation:
No monotonic relationship between signal magnitude and subsequent convergence,
or an effect too small to plausibly survive option spreads.
```

This prevents changing the question after seeing the answer.

---

# 5. Define the future outcome correctly

For a mean-reversion hypothesis, do not merely ask whether the signal is large.

If the signal is:

```text
D_t = SVI_t - SSVI_t
```

then a simple future convergence outcome is:

```text
Delta D_(t,h) = D_(t+h) - D_t
```

For a positive extreme, mean reversion predicts a negative future change.

Useful outcomes include:

- 1-day change in the residual
- 3-day change
- 5-day change
- probability that the residual shrinks
- percentage of the original residual that disappears
- time required to cross zero
- future change in an interpretable feature
- later, realized P&L of an executable trade

Do not begin with trade P&L if a simpler measurement-level convergence test can falsify the idea first.

---

# 6. Test the raw relationship before optimizing anything

Start with the bluntest version of the hypothesis.

For example:

```python
from volforge import VolDB, zscore_series, mean_reversion_test

DB = VolDB("volforge.db")

svi = DB.load_model_surface_panel("SPY", "svi")
ssvi = DB.load_model_surface_panel("SPY", "ssvi")

common = svi.index.intersection(ssvi.index)
gap = svi.loc[common] - ssvi.loc[common]

node = (30, -0.20)
series = gap[node].dropna()

z = zscore_series(series)
print(mean_reversion_test(series))
```

Then examine conditional forward behavior by signal bucket.

For example:

```text
z < -2
-2 <= z < -1
-1 <= z < 0
0 <= z < 1
1 <= z < 2
z >= 2
```

For each bucket, measure future change at the chosen horizons.

What you want to see is **ordered behavior**, not one lucky bucket.

For a mean-reversion hypothesis, progressively more positive signals should generally be followed by progressively more negative future changes, and vice versa.

---

# 7. Distinguish statistical significance from economic significance

A tiny effect can be statistically detectable and still be useless in options.

Track at least:

- number of observations
- average future change
- median future change
- hit rate
- standard deviation
- confidence interval or bootstrap interval
- effect by signal bucket
- effect by year/regime
- effect by liquidity bucket

Then ask:

> Is the magnitude large enough to plausibly exceed bid/ask, commissions, slippage, hedging, and model error?

If not, it is not economically interesting even if a p-value looks attractive.

---

# 8. Guard against look-ahead and data leakage

This is one of the easiest ways to manufacture fake alpha.

## Never use future information in today's signal

Examples of leakage:

- fitting today's z-score using mean/std calculated from future dates
- fitting PCA on the entire history and then reporting historical performance as if those loadings were known at the time
- choosing hyperparameters using the final test period
- selecting the best-performing nodes after examining all of their future returns and then treating them as pre-specified

## Use expanding or rolling estimates

At date `t`, statistics should be estimated only from information available through `t`.

For PCA research, eventually use a walk-forward structure:

```text
history available through t
        ↓
fit scaler + PCA
        ↓
calculate signal at t
        ↓
observe outcome after t
        ↓
advance one date
```

The same principle applies to historical z-scores and model thresholds.

---

# 9. Do not overfit the signal

The first implementation should be deliberately crude.

Good first test:

```text
signal = SVI - SSVI at 30D / k=-0.20
entry condition = |historical z| > 2
horizon = 5 days
```

Bad first test:

```text
Use 37D when VIX is 17.2-21.8, k between -0.17 and -0.13,
exclude Tuesdays, scale by PC3, require rho<-0.42, and exit after 2.7 days.
```

If the broad idea does not work before tuning, tuning usually manufactures a backtest rather than discovering an edge.

Prefer:

- round thresholds
- fixed horizons
- broad liquidity filters
- simple normalization
- few degrees of freedom

---

# 10. Test robustness across dimensions

A promising effect should not depend on one accidental slice of the dataset.

Test it across:

## Time

- early versus late sample
- calm versus volatile periods
- rolling windows

## Surface location

- nearby tenors
- nearby moneyness nodes

If exactly `29D / -0.175` works while neighboring coordinates fail, be suspicious.

## Model definition

Ask whether the conclusion survives reasonable alternatives:

```text
SVI vs SSVI
SVI vs eSSVI
SVI vs Fengler
SSVI vs Fengler
```

## Underlying

When enough history exists:

- SPY
- QQQ
- IWM
- SPX for cleaner European-style validation

A signal can still be symbol-specific, but you should know that rather than assume universality.

## Data choices

Test sensible variations in:

- quote filters
- spread thresholds
- forward estimation
- smoothing strength
- reliable-fit criteria

A result that vanishes under tiny measurement changes is probably not sturdy enough for trading.

---

# 11. Use model disagreement as a diagnostic, not automatically as a signal

Suppose one node shows:

```text
SVI      0.0210 total variance
SSVI     0.0185
eSSVI    0.0186
Fengler  0.0184
```

That tells you raw SVI disagrees with three other constructions.

It does **not** yet tell you which model is right or that you should trade against SVI.

Instead ask:

1. Does this pattern recur historically?
2. What happens afterward?
3. Does the local SVI slice converge toward the other models?
4. Do market quotes move, or is the gap caused by stable model bias?
5. Does the effect survive neighboring nodes and dates?

Persistent model disagreement may simply identify a known model misspecification.

Predictive change is what matters.

---

# 12. Graduate from measurement convergence to trade construction

Only after the measurement-level hypothesis survives should you create an option trade.

The trade should express the variable that predicted the future outcome while minimizing unrelated exposures.

Examples:

- skew signal → skew-sensitive vertical/risk-reversal-type structure
- curvature signal → butterfly-type structure
- term-structure signal → calendar/diagonal structure
- local rich/cheap option → relative structure using nearby strikes/expiries

Then inspect exposures to:

- delta
- gamma
- vega
- theta
- skew
- vol-of-vol
- spot/vol correlation

A successful surface forecast can still produce a bad trade if the chosen structure mainly earns or loses money from something else.

---

# 13. Add realistic execution costs

Option alpha is especially vulnerable to transaction costs.

At minimum model:

- bid/ask spread at entry
- bid/ask spread at exit
- commissions/fees
- slippage
- hedge costs if delta hedging is required
- inability to fill at theoretical mids
- liquidity constraints
- stale quotes

Use market widths available **at the decision time**, not future or average widths.

For quote-level residuals measured in half-spread units, VolForge's `forward_convergence` helper can incorporate a spread-unit execution cost. For model-grid signals such as SVI-minus-SSVI total variance, first establish predictive convergence, then construct the actual option trade and price its execution explicitly.

A useful hierarchy is:

```text
predictive before costs?
    ↓ yes
predictive after conservative costs?
    ↓ yes
stable out of sample?
    ↓ yes
candidate alpha
```

---

# 14. Use a real out-of-sample process

Do not repeatedly inspect the final test set.

A simple structure is:

```text
research/training period
    ↓
develop hypothesis and broad implementation
    ↓
validation period
    ↓
make limited design decisions
    ↓
final untouched test period
```

For a strategy intended to evolve through time, walk-forward testing is often more realistic:

```text
fit using past data only
    ↓
generate today's signal
    ↓
record future outcome
    ↓
roll forward
```

Once the final holdout has been inspected repeatedly, it is no longer a true holdout.

---

# 15. Keep a research ledger

Every experiment should be reproducible.

Record:

```text
Research ID:
Date proposed:
Hypothesis:
Economic/structural rationale:
Signal definition:
Universe:
Filters:
Horizon:
Outcome definition:
Training period:
Validation period:
Final test period:
Costs assumed:
Primary result:
Robustness results:
Decision: KEEP / MODIFY / KILL
Notes:
```

The `Decision` field is important. Alpha research should produce discarded ideas, not just increasingly complicated versions of every idea you started with.

---

# 16. Criteria for promoting a signal

Do not promote a VolForge measurement into a trading strategy merely because it has a good chart.

A candidate should ideally satisfy most of the following:

- clear hypothesis written before final testing
- sensible structural/economic explanation
- meaningful sample size
- ordered relationship between signal strength and future outcome
- stable sign across nearby parameter choices
- survives different time periods
- survives sensible cleaning/model choices
- survives realistic option execution costs
- survives out-of-sample or walk-forward testing
- does not depend on a handful of extreme observations
- maps to a trade whose exposures actually match the forecast

The correct outcome for many ideas is **KILL**.

That is successful research: VolForge helped prevent capital from being allocated to a false edge.

---

# 17. The first VolForge alpha experiment

Start simple.

## Hypothesis

When an independently fitted raw-SVI slice is unusually far from the globally coupled SSVI/eSSVI surface at constant tenor and moneyness, the disagreement partially mean-reverts.

## Initial universe

```text
symbol: SPY
nodes: common VolForge grid
models: SVI vs SSVI, then SVI vs eSSVI
data: daily saved snapshots
```

Do not cherry-pick the best node first. Start with the entire standard grid, while treating the node dimension as multiple hypotheses when interpreting significance.

## Step 1: build the historical gap

```python
from volforge import VolDB

DB = VolDB("volforge.db")

svi = DB.load_model_surface_panel("SPY", "svi")
ssvi = DB.load_model_surface_panel("SPY", "ssvi")

common = svi.index.intersection(ssvi.index)
gap = svi.loc[common] - ssvi.loc[common]
```

## Step 2: calculate an expanding historical normalization

For each node, calculate a z-score using **only past observations**.

Conceptually:

```text
z_t = (D_t - historical_mean_before_t) / historical_std_before_t
```

Do not calculate the mean/std from the full dataset when evaluating historical predictability.

## Step 3: calculate future convergence

For horizons such as 1, 3, and 5 trading days:

```text
future_change_h = D_(t+h) - D_t
```

For mean reversion, the expected sign is opposite the current signal.

A useful signed convergence quantity is:

```text
convergence_h = -sign(D_t) * (D_(t+h) - D_t)
```

Positive values mean the gap moved toward zero.

## Step 4: bucket by signal magnitude

For example:

```text
|z| < 0.5
0.5 <= |z| < 1
1 <= |z| < 2
|z| >= 2
```

Ask whether larger historical anomalies produce larger subsequent convergence.

## Step 5: repeat nearby

Check:

- neighboring tenor nodes
- neighboring moneyness nodes
- SVI vs eSSVI
- SVI vs Fengler when appropriate

The effect should have some local continuity if it represents real surface behavior.

## Step 6: only then design an option structure

If, for example, downside-skew disagreement reliably converges, design the simplest liquid structure that isolates that convergence and explicitly model execution.

Until Step 6, the output is an **alpha-research result**, not a trading strategy.

---

# 18. A second experiment: PCA reconstruction residuals

Only after a meaningful history of clean surfaces exists:

1. use one chosen surface family, initially SSVI or eSSVI
2. calculate daily total-variance surface changes
3. fit PCA using past data only
4. reconstruct each new surface change from the retained PCs
5. calculate the residual at every node
6. test whether large reconstruction residuals reverse, persist, or predict another option-market quantity

The hypothesis is not:

> PCA residuals are alpha.

The hypothesis must be something testable, such as:

> Surface movements unexplained by the first three common factors tend to mean-revert over the next three trading days.

Then repeat the same research loop: prediction, robustness, costs, out-of-sample, trade mapping.

---

# 19. Common ways to create fake alpha

Be suspicious when:

- a result appears only after trying many thresholds
- one tenor/strike node dominates the entire result
- the effect disappears when using bid/ask rather than midpoint
- the effect disappears with one-day execution lag
- PCA was fitted using future data
- the z-score uses full-sample statistics
- unreliable/arbitrage-violating fits are included silently
- the signal works only under one surface model
- a backtest assumes every order fills at midpoint
- a model residual is treated as "fair value" without testing future convergence
- a strategy earns money from delta or volatility level while being described as a skew trade
- the final test set has been used repeatedly to make design changes

VolForge's purpose is partly to make these mistakes visible.

---

# 20. The short version

Whenever the project feels too complicated, return to these seven questions:

```text
1. What exactly am I measuring today?
2. Why is it unusual relative to a valid benchmark?
3. What exactly do I predict will happen next?
4. Does that happen repeatedly in historical data?
5. Is the effect robust and free of look-ahead?
6. Can an actual option trade capture it after realistic costs?
7. Does it survive out of sample?
```

If you cannot answer Question 3, you have a measurement, not an alpha hypothesis.

If you cannot answer Question 6, you may have predictability, but not tradable alpha.

If you cannot answer Question 7, you have an interesting backtest, not yet an edge.

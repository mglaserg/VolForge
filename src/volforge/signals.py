"""Signals, and the tests that decide whether they are real.

The order matters and the roadmap had it right: measure whether residuals
converge before designing any trade around them. This module is built so the
discouraging answer is easy to get and hard to avoid.

Two things it does deliberately.

**Signals are always expressed in spread units, never raw vol.** A residual of
0.4 vol points is large on a 30-day ATM SPY option and invisible on a 7-day
wing. Dividing by the option's own half-spread makes the number comparable and,
more importantly, makes it directly interpretable: below 1.0 you cannot trade
it, because you would pay more than the edge to get in.

**Forward-return tests are computed net of a spread cost by default.** Gross
convergence in vol space is almost always positive and almost always
meaningless. `cost_spreads=1.0` charges one full half-spread each way, which is
optimistic for a real fill but honest enough to kill most spurious signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "residual_signal",
    "zscore_series",
    "forward_convergence",
    "bucket_by_signal",
    "mean_reversion_test",
    "SignalReport",
]


def residual_signal(iv_market, iv_model, half_spread_iv):
    """Signed richness in half-spread units. Positive = market rich vs model."""
    iv_market = np.asarray(iv_market, float)
    iv_model = np.asarray(iv_model, float)
    hs = np.asarray(half_spread_iv, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (iv_market - iv_model) / np.where(hs > 0, hs, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def zscore_series(s: pd.Series, window: int = 252, min_periods: int = 60,
                  robust: bool = True) -> pd.Series:
    """Trailing z-score. No look-ahead: each point uses only prior data."""
    roll = s.rolling(window, min_periods=min_periods)
    if robust:
        center = roll.median()
        scale = 1.4826 * (s - center).abs().rolling(window, min_periods=min_periods).median()
    else:
        center, scale = roll.mean(), roll.std()
    return ((s - center) / scale.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


@dataclass
class SignalReport:
    horizon: int
    n: int
    mean_convergence: float      # in half-spread units, net of cost
    median_convergence: float
    hit_rate: float              # share where the residual shrank toward zero
    t_stat: float
    gross_mean: float
    cost_charged: float

    def __str__(self):
        verdict = "worth pursuing" if (self.t_stat > 2 and self.mean_convergence > 0) \
            else "no evidence of tradeable convergence"
        return (f"h={self.horizon}d  n={self.n}  net={self.mean_convergence:+.3f} sp  "
                f"(gross {self.gross_mean:+.3f}, cost {self.cost_charged:.2f})  "
                f"hit={self.hit_rate:.1%}  t={self.t_stat:+.2f}  -> {verdict}")


def forward_convergence(
    signal: pd.Series,
    resid: pd.Series,
    horizon: int = 1,
    cost_spreads: float = 1.0,
    min_signal: float = 1.0,
) -> SignalReport:
    """Do large residuals shrink over `horizon` days, net of costs?

    Parameters
    ----------
    signal : residual in half-spread units at time t (the entry signal)
    resid  : the same residual series, used to measure where it went
    cost_spreads : half-spreads charged round trip. 1.0 means paying half a
        spread in and half a spread out, which is a friendly assumption.
    min_signal : only evaluate entries whose |signal| clears this. Below 1.0
        the edge is inside the spread and cannot be captured.

    Convergence is measured as the reduction in absolute residual, signed so
    that positive is profitable, then charged the cost.
    """
    df = pd.DataFrame({"sig": signal, "res": resid}).dropna()
    df["fut"] = df["res"].shift(-horizon)
    df = df.dropna()
    df = df[df["sig"].abs() >= min_signal]

    if len(df) < 20:
        return SignalReport(horizon, len(df), np.nan, np.nan, np.nan, np.nan, np.nan,
                            cost_spreads)

    gross = df["res"].abs() - df["fut"].abs()   # positive = converged
    net = gross - cost_spreads
    t = float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if net.std(ddof=1) > 0 else np.nan

    return SignalReport(
        horizon=horizon,
        n=len(df),
        mean_convergence=float(net.mean()),
        median_convergence=float(net.median()),
        hit_rate=float((gross > 0).mean()),
        t_stat=t,
        gross_mean=float(gross.mean()),
        cost_charged=cost_spreads,
    )


def bucket_by_signal(signal: pd.Series, resid: pd.Series, horizon: int = 1,
                     edges=(-np.inf, -3, -2, -1, 1, 2, 3, np.inf),
                     cost_spreads: float = 1.0) -> pd.DataFrame:
    """Convergence by signal size.

    The pattern to look for is monotonicity: bigger residuals should converge
    more. A strong result confined to one bucket, with nothing in the
    neighbouring ones, is usually a handful of outlier days rather than a
    signal, so check `n` alongside the mean.
    """
    df = pd.DataFrame({"sig": signal, "res": resid}).dropna()
    df["fut"] = df["res"].shift(-horizon)
    df = df.dropna()
    df["gross"] = df["res"].abs() - df["fut"].abs()
    df["net"] = df["gross"] - cost_spreads
    df["bucket"] = pd.cut(df["sig"], bins=list(edges))

    g = df.groupby("bucket", observed=True).agg(
        n=("net", "size"), mean_net=("net", "mean"), median_net=("net", "median"),
        mean_gross=("gross", "mean"), hit_rate=("gross", lambda x: (x > 0).mean()),
    )
    g["t_stat"] = df.groupby("bucket", observed=True)["net"].apply(
        lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2
        and x.std(ddof=1) > 0 else np.nan)
    return g


def mean_reversion_test(series: pd.Series, horizons=(1, 5, 10)) -> pd.DataFrame:
    """AR(1) style test: regress forward change on current level.

    A negative, significant slope indicates mean reversion. Applied to PC scores
    this is the Phase 11 question. Note that a z-scored series is mechanically
    somewhat mean-reverting by construction, so run this on the raw score, not
    the z-score, or you will be measuring your own normalisation.
    """
    s = series.dropna()
    rows = []
    for h in horizons:
        y = (s.shift(-h) - s).dropna()
        x = s.reindex(y.index)
        if len(y) < 30:
            continue
        A = np.column_stack([np.ones(len(x)), x.to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - A @ coef
        dof = len(y) - 2
        se = np.sqrt((resid @ resid / dof) * np.linalg.inv(A.T @ A)[1, 1])
        half_life = -np.log(2) / np.log(1 + coef[1]) if -1 < coef[1] < 0 else np.nan
        rows.append({"horizon": h, "slope": coef[1], "t_stat": coef[1] / se,
                     "half_life_days": half_life, "n": len(y)})
    return pd.DataFrame(rows)

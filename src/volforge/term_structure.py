"""Monotone ATM total-variance term structures.

SSVI uses ATM total variance, theta(T) = sigma_ATM(T)^2 T, as its maturity
clock.  Calendar-spread arbitrage requires theta to be non-decreasing in T.
This module deliberately keeps that clock separate from the smile model so it
can be reused by SSVI, eSSVI and later surface constructions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ATMVarianceCurve", "isotonic_increasing", "build_atm_variance_curve"]


def isotonic_increasing(y, weights=None):
    """Least-squares non-decreasing projection via pool-adjacent-violators.

    Returns an array of the same length as ``y``.  Unlike a cumulative maximum,
    PAVA is the minimum weighted-L2 adjustment and is therefore a less biased
    way to repair small calendar inversions in an estimated ATM variance curve.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(y) == 0:
        return y.copy()
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")

    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be positive, finite, and match y")

    # Blocks are [start, end, weighted_mean, total_weight].
    blocks = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([i, i, float(yi), float(wi)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b1 = blocks.pop()
            b0 = blocks.pop()
            wt = b0[3] + b1[3]
            mean = (b0[2] * b0[3] + b1[2] * b1[3]) / wt
            blocks.append([b0[0], b1[1], mean, wt])

    out = np.empty_like(y)
    for start, end, mean, _ in blocks:
        out[start:end + 1] = mean
    return out


@dataclass(frozen=True)
class ATMVarianceCurve:
    """Piecewise-linear monotone theta(T) curve.

    ``t_years`` contains observed positive maturities.  Evaluation is linear
    between nodes.  From T=0 to the first node the curve is anchored at
    theta(0)=0.  Beyond the last node it extends with the last non-negative
    secant slope.  Those choices preserve monotonicity and avoid flat total
    variance immediately outside the fitted range.
    """

    t_years: np.ndarray
    theta: np.ndarray
    raw_theta: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.t_years, dtype=float)
        th = np.asarray(self.theta, dtype=float)
        raw = np.asarray(self.raw_theta, dtype=float)
        wt = np.asarray(self.weights, dtype=float)
        if not (t.ndim == th.ndim == raw.ndim == wt.ndim == 1):
            raise ValueError("curve arrays must be one-dimensional")
        if not (len(t) == len(th) == len(raw) == len(wt)) or len(t) == 0:
            raise ValueError("curve arrays must have matching lengths")
        if len(t) and (np.any(t <= 0) or np.any(np.diff(t) <= 0)):
            raise ValueError("t_years must be strictly increasing and positive")
        if len(th) and (np.any(th <= 0) or np.any(np.diff(th) < -1e-12)):
            raise ValueError("theta must be positive and non-decreasing")
        object.__setattr__(self, "t_years", t)
        object.__setattr__(self, "theta", th)
        object.__setattr__(self, "raw_theta", raw)
        object.__setattr__(self, "weights", wt)

    @property
    def repair_amount(self) -> float:
        return float(np.abs(self.theta - self.raw_theta).sum())

    @property
    def repair_fraction(self) -> float:
        den = float(np.abs(self.raw_theta).sum())
        return self.repair_amount / den if den > 0 else 0.0

    @property
    def is_monotone(self) -> bool:
        return bool(np.all(np.diff(self.theta) >= -1e-12))

    def __call__(self, T):
        """Evaluate theta at maturity T in years; scalar or array."""
        x = np.asarray(T, dtype=float)
        if len(self.t_years) == 0:
            return np.full_like(x, np.nan, dtype=float)

        # Include the origin so short-end interpolation respects theta(0)=0.
        tx = np.concatenate([[0.0], self.t_years])
        th = np.concatenate([[0.0], self.theta])
        out = np.interp(x, tx, th)

        # Non-negative linear extrapolation at the long end.
        if len(self.t_years) >= 2:
            slope = (self.theta[-1] - self.theta[-2]) / (
                self.t_years[-1] - self.t_years[-2]
            )
        else:
            slope = self.theta[-1] / self.t_years[-1]
        slope = max(float(slope), 0.0)
        mask = x > self.t_years[-1]
        if np.any(mask):
            out = np.asarray(out)
            out[mask] = self.theta[-1] + slope * (x[mask] - self.t_years[-1])

        out = np.maximum(out, 0.0)
        return float(out) if np.ndim(T) == 0 else out

    def to_dict(self) -> dict:
        return {
            "t_years": self.t_years.tolist(),
            "theta": self.theta.tolist(),
            "raw_theta": self.raw_theta.tolist(),
            "weights": self.weights.tolist(),
            "repair_amount": self.repair_amount,
            "repair_fraction": self.repair_fraction,
        }


def build_atm_variance_curve(t_years, theta, weights=None, repair=True) -> ATMVarianceCurve:
    """Sort, deduplicate and optionally isotonic-repair ATM total variance."""
    t = np.asarray(t_years, dtype=float)
    y = np.asarray(theta, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (t > 0) & (y > 0) & (w > 0)
    t, y, w = t[ok], y[ok], w[ok]
    if len(t) < 2:
        raise ValueError("need at least two positive ATM variance observations")

    order = np.argsort(t)
    t, y, w = t[order], y[order], w[order]

    # Combine duplicate maturities with weighted means.
    tu, inv = np.unique(t, return_inverse=True)
    yu = np.zeros_like(tu)
    wu = np.zeros_like(tu)
    for i in range(len(tu)):
        m = inv == i
        wu[i] = w[m].sum()
        yu[i] = np.average(y[m], weights=w[m])

    repaired = isotonic_increasing(yu, wu) if repair else yu.copy()
    if np.any(np.diff(repaired) < -1e-12):
        raise ValueError("ATM total variance is not monotone; set repair=True")
    return ATMVarianceCurve(tu, repaired, yu, wu)

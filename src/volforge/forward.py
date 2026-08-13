"""Forward and discount-factor extraction from put-call parity.

Rather than differencing a single ATM pair, we regress across all matched
strikes:

    C(K) - P(K) = D * F - D * K

so the slope is -D and the intercept is D*F. This gives F and D jointly,
plus an R^2 that is a genuinely useful data-quality diagnostic: a slice
whose parity regression is poor has bad quotes, and no amount of clever
calibration downstream will fix that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ForwardFit", "fit_forward"]


@dataclass(frozen=True)
class ForwardFit:
    forward: float
    discount: float
    rate: float          # continuously-compounded, implied by D and T
    r_squared: float
    n_pairs: int
    resid_std: float     # std of regression residuals, in price units
    strikes_used: np.ndarray

    @property
    def is_sane(self) -> bool:
        """Cheap gate for downstream code."""
        return (
            self.n_pairs >= 5
            and 0.5 < self.discount <= 1.02
            and self.r_squared > 0.99
            and np.isfinite(self.forward)
            and self.forward > 0
        )


def fit_forward(
    strikes,
    call_mid,
    put_mid,
    T,
    spot=None,
    moneyness_window=0.10,
    weights=None,
    trim_z=3.0,
) -> ForwardFit:
    """Estimate (F, D) for a single expiry by regressing C-P on K.

    Parameters
    ----------
    strikes, call_mid, put_mid : arrays over a single expiry, aligned by strike
    T : time to expiry in years
    spot : underlying price. If given with `moneyness_window`, restricts the
        regression to strikes within that fractional band of spot, where both
        legs are liquid and parity is most reliable.
    moneyness_window : e.g. 0.10 keeps strikes in [0.9*S, 1.1*S]. None = all.
    weights : optional regression weights (e.g. 1/combined_spread).
    trim_z : drop points whose residual exceeds this many robust sigmas, then
        refit once. Set None to disable.
    """
    K = np.asarray(strikes, dtype=float)
    y = np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)

    mask = np.isfinite(K) & np.isfinite(y)
    if spot is not None and moneyness_window is not None:
        mask &= np.abs(K / spot - 1.0) <= moneyness_window

    w = np.ones_like(K) if weights is None else np.asarray(weights, dtype=float)
    mask &= np.isfinite(w) & (w > 0)

    if mask.sum() < 3:
        raise ValueError(f"need >=3 matched strikes for parity fit, got {mask.sum()}")

    def _wls(m):
        slope, intercept, r2, resid = _weighted_linfit(K[m], y[m], w[m])
        return slope, intercept, r2, resid

    slope, intercept, r2, resid = _wls(mask)

    if trim_z is not None and mask.sum() >= 8:
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if scale > 0:
            keep = np.abs(resid - np.median(resid)) <= trim_z * scale
            new_mask = mask.copy()
            new_mask[mask] = keep
            if new_mask.sum() >= 5:
                mask = new_mask
                slope, intercept, r2, resid = _wls(mask)

    D = -slope
    if D <= 0:
        raise ValueError(f"parity regression gave non-positive discount factor {D:.4f}")
    F = intercept / D
    rate = -np.log(D) / T if T > 0 else np.nan

    return ForwardFit(
        forward=float(F),
        discount=float(D),
        rate=float(rate),
        r_squared=float(r2),
        n_pairs=int(mask.sum()),
        resid_std=float(np.std(resid)),
        strikes_used=K[mask],
    )


def _weighted_linfit(x, y, w):
    sw = w.sum()
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    sxy = (w * (x - xm) * (y - ym)).sum()
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    ss_res = (w * resid ** 2).sum()
    ss_tot = (w * (y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2, resid


def log_moneyness(strikes, forward):
    """k = log(K / F)"""
    return np.log(np.asarray(strikes, dtype=float) / forward)

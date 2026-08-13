"""Black-76 pricing and implied-vol inversion, parameterised on the forward.

Everything here works in *undiscounted* forward space. Discounting is applied
once, at the boundary, using the discount factor D estimated from put-call
parity. This keeps rate/dividend assumptions in exactly one place.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "black_price",
    "black_vega",
    "implied_vol",
    "implied_vol_vec",
    "total_variance_to_iv",
    "iv_to_total_variance",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _d1_d2(F, K, sigma, T):
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return d1, d1 - v


def black_price(F, K, sigma, T, is_call=True, discount=1.0):
    """Black-76 option price.

    Parameters
    ----------
    F : forward price of the underlying to expiry T
    K : strike
    sigma : lognormal implied volatility (annualised)
    T : time to expiry in years
    is_call : True for call, False for put
    discount : discount factor D = exp(-rT). Pass 1.0 for undiscounted price.
    """
    F, K, sigma, T = map(np.asarray, (F, K, sigma, T))
    d1, d2 = _d1_d2(F, K, sigma, T)
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    out = np.where(is_call, call, call - F + K)  # put-call parity, undiscounted
    return discount * out


def black_vega(F, K, sigma, T, discount=1.0):
    """dPrice/dsigma. Same for calls and puts."""
    F, K, sigma, T = map(np.asarray, (F, K, sigma, T))
    d1, _ = _d1_d2(F, K, sigma, T)
    return discount * F * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / _SQRT_2PI


def implied_vol(
    price,
    F,
    K,
    T,
    is_call=True,
    discount=1.0,
    lo=1e-6,
    hi=5.0,
):
    """Invert Black-76 for implied volatility. Returns nan if no solution.

    Returns nan when the price violates the no-arbitrage bounds
    (intrinsic <= price <= F for a call), which is the correct signal to
    drop that quote rather than fudge it.
    """
    if not np.isfinite(price) or price <= 0 or T <= 0 or F <= 0 or K <= 0:
        return np.nan

    fwd_price = price / discount  # undiscounted

    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    upper = F if is_call else K
    # Strict inequalities, with a scale-relative epsilon: at the bounds vega is
    # zero and IV is undefined. Deep-ITM quotes whose time value has vanished
    # into floating-point noise correctly return nan -- they carry no vol
    # information and should be dropped, not rescued.
    eps = 1e-10 * max(F, K)
    if fwd_price <= intrinsic + eps or fwd_price >= upper - eps:
        return np.nan

    def obj(s):
        return float(black_price(F, K, s, T, is_call, 1.0)) - fwd_price

    try:
        if obj(lo) > 0 or obj(hi) < 0:
            return np.nan
        return brentq(obj, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_vec(prices, F, strikes, T, is_call, discount=1.0):
    """Vectorised wrapper around `implied_vol`. is_call may be scalar or array."""
    prices = np.atleast_1d(prices)
    strikes = np.atleast_1d(strikes)
    is_call = np.broadcast_to(np.atleast_1d(is_call), prices.shape)
    return np.array(
        [
            implied_vol(p, F, k, T, bool(c), discount)
            for p, k, c in zip(prices, strikes, is_call)
        ]
    )


def iv_to_total_variance(iv, T):
    """w = iv^2 * T"""
    return np.asarray(iv) ** 2 * T


def total_variance_to_iv(w, T):
    """iv = sqrt(w / T). Negative w -> nan."""
    w = np.asarray(w, dtype=float)
    return np.sqrt(np.where(w < 0, np.nan, w) / T)

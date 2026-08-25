"""Model-free implied variance (variance-swap / VIX-style integration).

The implementation works from VolForge's canonical option chain, estimates the
forward and discount factor from put-call parity, selects OTM puts/calls around
K0, and integrates across strikes. Constant-tenor interpolation is performed in
*total variance* (variance x time), never directly in volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forward import ForwardFit, fit_forward
from .blackscholes import black_price
from .data.clean import matched_pairs
from .data.schema import add_derived_columns

__all__ = [
    "MFIVSlice",
    "ConstantTenorMFIV",
    "mfiv_for_expiry",
    "mfiv_term_structure",
    "constant_tenor_mfiv",
    "mfiv_from_model",
]

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class MFIVSlice:
    expiry: pd.Timestamp
    T: float
    dte: float
    implied_variance: float
    implied_volatility: float
    forward_fit: ForwardFit
    k0: float
    n_strikes: int
    price_side: str

    @property
    def total_variance(self) -> float:
        return self.implied_variance * self.T


@dataclass(frozen=True)
class ConstantTenorMFIV:
    target_days: float
    implied_variance: float
    implied_volatility: float
    lower_expiry: pd.Timestamp
    upper_expiry: pd.Timestamp
    lower_days: float
    upper_days: float
    interpolation_weight: float


def mfiv_for_expiry(
    chain: pd.DataFrame,
    expiry,
    *,
    price_side: str = "mid",
    min_strikes: int = 8,
) -> MFIVSlice:
    """Compute annualized model-free implied variance for one expiry."""
    df = add_derived_columns(chain)
    exp = pd.Timestamp(expiry)
    sl = df[df["expiry"] == exp].copy()
    if sl.empty:
        raise ValueError(f"expiry {exp} not found")
    if price_side not in {"mid", "bid"}:
        raise ValueError("price_side must be 'mid' or 'bid'")

    # Basic executable-quote hygiene without the moneyness truncation used by
    # smile calibration: MFIV needs the widest reliable OTM strip available.
    sl = sl[
        np.isfinite(sl["bid"])
        & np.isfinite(sl["ask"])
        & (sl["bid"] >= 0)
        & (sl["ask"] > 0)
        & (sl["ask"] >= sl["bid"])
    ].copy()
    if price_side == "mid":
        sl["q"] = (sl["bid"] + sl["ask"]) / 2.0
    else:
        sl["q"] = sl["bid"]
    sl = sl[np.isfinite(sl["q"]) & (sl["q"] > 0)]

    # matched_pairs also carries optional liquidity columns for calibration;
    # MFIV only needs prices. Supply NaNs when a minimal canonical chain omits
    # those optional fields.
    for col in ("volume", "open_interest"):
        if col not in sl:
            sl[col] = np.nan
    pairs = matched_pairs(sl)
    if len(pairs) < 3:
        raise ValueError("need matched call/put strikes to infer forward")
    T = float(sl["T"].median())
    spot = float(sl["underlying_price"].median())
    combined = pairs.get("combined_spread", pd.Series(1.0, index=pairs.index)).to_numpy(float)
    weights = 1.0 / np.maximum(combined, 0.01)
    ffit = fit_forward(
        pairs["strike"],
        pairs["mid_c"],
        pairs["mid_p"],
        T,
        spot=spot,
        moneyness_window=0.20,
        weights=weights,
    )
    F = ffit.forward
    D = ffit.discount

    strikes = np.sort(sl["strike"].unique().astype(float))
    below = strikes[strikes <= F]
    if len(below) == 0:
        raise ValueError("no strike at or below forward for K0")
    k0 = float(below[-1])

    calls = sl[sl["right"] == "C"].set_index("strike")["q"]
    puts = sl[sl["right"] == "P"].set_index("strike")["q"]
    selected: dict[float, float] = {}
    for k in strikes:
        if k < k0 and k in puts.index:
            selected[float(k)] = float(puts.loc[k])
        elif k > k0 and k in calls.index:
            selected[float(k)] = float(calls.loc[k])
        elif k == k0:
            vals = []
            if k in puts.index:
                vals.append(float(puts.loc[k]))
            if k in calls.index:
                vals.append(float(calls.loc[k]))
            if vals:
                selected[float(k)] = float(np.mean(vals))

    K = np.array(sorted(k for k, q in selected.items() if np.isfinite(q) and q > 0), dtype=float)
    Q = np.array([selected[k] for k in K], dtype=float)
    if len(K) < min_strikes:
        raise ValueError(f"need >= {min_strikes} OTM strikes for MFIV, got {len(K)}")

    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]

    integral = (2.0 / T) * np.sum((dK / (K ** 2)) * (Q / D))
    forward_adjustment = ((F / k0) - 1.0) ** 2 / T
    variance = float(integral - forward_adjustment)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"MFIV produced non-positive variance {variance}")

    return MFIVSlice(
        expiry=exp,
        T=T,
        dte=T * DAYS_PER_YEAR,
        implied_variance=variance,
        implied_volatility=float(np.sqrt(variance)),
        forward_fit=ffit,
        k0=k0,
        n_strikes=len(K),
        price_side=price_side,
    )


def mfiv_term_structure(chain: pd.DataFrame, *, price_side: str = "mid", min_strikes: int = 8) -> list[MFIVSlice]:
    """Compute MFIV independently for every expiry that can support it."""
    out = []
    for expiry in sorted(pd.to_datetime(chain["expiry"], utc=True).unique()):
        try:
            out.append(mfiv_for_expiry(chain, expiry, price_side=price_side, min_strikes=min_strikes))
        except (ValueError, FloatingPointError):
            continue
    return sorted(out, key=lambda x: x.T)


def constant_tenor_mfiv(
    slices: list[MFIVSlice],
    target_days: float = 30.0,
    *,
    allow_extrapolation: bool = False,
) -> ConstantTenorMFIV:
    """Interpolate a target maturity linearly in total variance."""
    if target_days <= 0:
        raise ValueError("target_days must be positive")
    ss = sorted(slices, key=lambda s: s.dte)
    if not ss:
        raise ValueError("no MFIV slices supplied")
    target_T = target_days / DAYS_PER_YEAR

    for s in ss:
        if abs(s.dte - target_days) < 1e-10:
            return ConstantTenorMFIV(
                target_days, s.implied_variance, s.implied_volatility,
                s.expiry, s.expiry, s.dte, s.dte, 0.0,
            )

    lower = [s for s in ss if s.dte < target_days]
    upper = [s for s in ss if s.dte > target_days]
    if not lower or not upper:
        if not allow_extrapolation or len(ss) < 2:
            raise ValueError(f"target {target_days:g}d is not bracketed by available expiries")
        lo, hi = (ss[0], ss[1]) if not lower else (ss[-2], ss[-1])
    else:
        lo, hi = lower[-1], upper[0]

    weight = (target_T - lo.T) / (hi.T - lo.T)
    theta = lo.total_variance + weight * (hi.total_variance - lo.total_variance)
    variance = float(theta / target_T)
    return ConstantTenorMFIV(
        target_days=float(target_days),
        implied_variance=variance,
        implied_volatility=float(np.sqrt(max(variance, 0.0))),
        lower_expiry=lo.expiry,
        upper_expiry=hi.expiry,
        lower_days=lo.dte,
        upper_days=hi.dte,
        interpolation_weight=float(weight),
    )


def mfiv_from_model(
    *,
    expiry,
    T: float,
    forward_fit: ForwardFit,
    strikes,
    implied_vol_fn,
    label: str = "model",
    min_strikes: int = 8,
) -> MFIVSlice:
    """Integrate model-smoothed option prices on an observed strike support.

    ``implied_vol_fn`` receives log-forward-moneyness ``k = log(K/F)`` and
    returns annualized Black implied volatility.  Using the observed strike
    support keeps raw-vs-model MFIV comparisons honest: the smoother changes
    quote shape, not the integration domain.
    """
    T = float(T)
    if T <= 0:
        raise ValueError("T must be positive")
    F = float(forward_fit.forward)
    D = float(forward_fit.discount)
    K_all = np.unique(np.asarray(strikes, float))
    K_all = K_all[np.isfinite(K_all) & (K_all > 0)]
    K_all.sort()
    below = K_all[K_all <= F]
    if len(below) == 0:
        raise ValueError("no strike at or below forward for K0")
    k0 = float(below[-1])

    logk = np.log(K_all / F)
    iv = np.asarray(implied_vol_fn(logk), float)
    ok = np.isfinite(iv) & (iv > 0)
    K_all, iv = K_all[ok], iv[ok]
    if len(K_all) < min_strikes:
        raise ValueError(f"need >= {min_strikes} model strikes for MFIV, got {len(K_all)}")

    selected_k, selected_q = [], []
    for K, sig in zip(K_all, iv):
        if K < k0:
            q = float(black_price(F, K, sig, T, False, D))
        elif K > k0:
            q = float(black_price(F, K, sig, T, True, D))
        else:
            q = 0.5 * (float(black_price(F, K, sig, T, True, D)) + float(black_price(F, K, sig, T, False, D)))
        if np.isfinite(q) and q > 0:
            selected_k.append(float(K))
            selected_q.append(q)

    K = np.asarray(selected_k, float)
    Q = np.asarray(selected_q, float)
    if len(K) < min_strikes:
        raise ValueError(f"need >= {min_strikes} positive model option prices, got {len(K)}")
    order = np.argsort(K)
    K, Q = K[order], Q[order]
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    integral = (2.0 / T) * np.sum((dK / (K ** 2)) * (Q / D))
    forward_adjustment = ((F / k0) - 1.0) ** 2 / T
    variance = float(integral - forward_adjustment)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"model MFIV produced non-positive variance {variance}")
    return MFIVSlice(
        expiry=pd.Timestamp(expiry),
        T=T,
        dte=T * DAYS_PER_YEAR,
        implied_variance=variance,
        implied_volatility=float(np.sqrt(variance)),
        forward_fit=forward_fit,
        k0=k0,
        n_strikes=len(K),
        price_side=label,
    )

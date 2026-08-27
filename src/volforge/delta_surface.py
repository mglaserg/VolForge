"""Simple delta-space volatility surface and RW-style surface features.

This module intentionally avoids a global parametric volatility model.  It
converts VolForge's own implied-volatility slices into standard delta buckets,
then interpolates *total variance* through maturity at fixed delta.  The result
is a compact, interpretable representation suitable for delta ratios, term
structure diagnostics, historical z-scores and simple surface-change features.

Delta convention
----------------
The bucket deltas are spot deltas implied by VolForge's parity-derived forward
and discount factor.  For an equity/ETF option under Black-Scholes,

    dS = D * F / S * N(d1)              (call)
    dS = D * F / S * (N(d1) - 1)       (put)

so the construction remains vendor-neutral and does not depend on supplied
Greeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data.clean import CleanConfig, clean_chain
from .data.pipeline import build_all_slices

__all__ = [
    "DEFAULT_DELTAS",
    "DeltaVolSurface",
    "build_delta_surface",
    "constant_tenor_delta_slice",
    "delta_ratio_term_structure",
    "delta_lump_scores",
    "delta_surface_change_features",
]

DAYS_PER_YEAR = 365.25
DEFAULT_DELTAS = (0.10, 0.15, 0.25)


def _bucket(delta: float, side: str) -> str:
    return f"{int(round(100 * float(delta)))}{side.lower()}"


def _interp_no_extrap(x, y, target: float) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    frame = pd.DataFrame({"x": x[ok], "y": y[ok]}).groupby("x", as_index=False)["y"].mean()
    xs = frame["x"].to_numpy(float)
    ys = frame["y"].to_numpy(float)
    if target < xs.min() or target > xs.max():
        return np.nan
    return float(np.interp(float(target), xs, ys))


def _spot_deltas(slc) -> np.ndarray:
    iv = np.asarray(slc.iv, float)
    K = np.asarray(slc.strikes, float)
    v = iv * np.sqrt(float(slc.T))
    d1 = (np.log(float(slc.forward) / K) + 0.5 * v * v) / v
    factor = float(slc.forward_fit.discount) * float(slc.forward) / float(slc.spot)
    call = factor * norm.cdf(d1)
    put = factor * (norm.cdf(d1) - 1.0)
    return np.where(np.asarray(slc.is_call, bool), call, put)


@dataclass(frozen=True)
class DeltaVolSurface:
    """Observed-expiry delta surface.

    ``iv`` is indexed by actual expiry DTE and has columns such as ``iv_10p``,
    ``iv_15p``, ``iv_25p``, ``atm_iv``, ``iv_25c`` ... .  ``ratios`` uses the
    same DTE index and divides each wing bucket by ATM IV.
    """

    symbol: str
    quote_time: pd.Timestamp
    deltas: tuple[float, ...]
    iv: pd.DataFrame
    ratios: pd.DataFrame
    expiries: pd.Series

    @property
    def bucket_columns(self) -> tuple[str, ...]:
        puts = tuple(f"iv_{_bucket(d, 'p')}" for d in self.deltas)
        calls = tuple(f"iv_{_bucket(d, 'c')}" for d in reversed(self.deltas))
        return puts + ("atm_iv",) + calls

    def display_frame(self) -> pd.DataFrame:
        return self.iv.loc[:, [c for c in self.bucket_columns if c in self.iv.columns]].copy()


def build_delta_surface(
    chain: pd.DataFrame,
    *,
    deltas: Iterable[float] = DEFAULT_DELTAS,
    dte_range: tuple[float, float] = (7.0, 180.0),
    require_activity: bool = False,
    min_quotes: int = 8,
) -> DeltaVolSurface:
    """Build a non-parametric delta-bucket volatility surface from a chain."""
    deltas = tuple(sorted({float(x) for x in deltas}))
    if not deltas or any((d <= 0 or d >= 0.50) for d in deltas):
        raise ValueError("delta buckets must lie strictly between 0 and 0.50")

    clean, _ = clean_chain(
        chain,
        CleanConfig(dte_range=dte_range, require_activity=require_activity),
        verbose=False,
    )
    # ``matched_pairs`` can carry optional liquidity fields through its pivot.
    # Provider adapters are not required to supply them, so make their absence
    # explicit rather than letting a perfectly usable chain fail downstream.
    for optional in ("volume", "open_interest"):
        if optional not in clean:
            clean[optional] = np.nan
    slices = build_all_slices(clean, verbose=False, min_quotes=min_quotes)
    if not slices:
        raise ValueError("no calibratable expiries for delta surface")

    rows: list[dict] = []
    expiry_map: dict[float, pd.Timestamp] = {}
    for slc in slices:
        delta = _spot_deltas(slc)
        abs_delta = np.abs(delta)
        row: dict[str, float] = {"dte": float(slc.dte)}

        # ATM is interpolated directly in log-moneyness from the observed OTM
        # IVs.  Calls and puts jointly bracket k=0 in a healthy slice.
        row["atm_iv"] = _interp_no_extrap(slc.k, slc.iv, 0.0)

        is_call = np.asarray(slc.is_call, bool)
        for target in deltas:
            for side, mask in (("p", ~is_call), ("c", is_call)):
                row[f"iv_{_bucket(target, side)}"] = _interp_no_extrap(
                    abs_delta[mask], np.asarray(slc.iv, float)[mask], target
                )

        rows.append(row)
        expiry_map[float(slc.dte)] = pd.Timestamp(slc.expiry)

    iv = pd.DataFrame(rows).sort_values("dte").drop_duplicates("dte", keep="last").set_index("dte")
    ratio = pd.DataFrame(index=iv.index)
    for target in deltas:
        for side in ("p", "c"):
            col = f"iv_{_bucket(target, side)}"
            ratio[f"delta_ratio_{_bucket(target, side)}"] = iv[col] / iv["atm_iv"].replace(0.0, np.nan)

    symbols = clean["symbol"].dropna().astype(str).unique()
    symbol = symbols[0] if len(symbols) == 1 else ""
    quote_time = pd.to_datetime(clean["quote_time"], utc=True).max()
    expiries = pd.Series(expiry_map, name="expiry").sort_index()
    expiries.index.name = "dte"
    return DeltaVolSurface(symbol, quote_time, deltas, iv, ratio, expiries)


def _constant_tenor_column(series: pd.Series, target_days: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if s.empty:
        return np.nan
    d = s.index.to_numpy(float)
    target = float(target_days)
    exact = np.where(np.isclose(d, target, atol=1e-8))[0]
    if len(exact):
        return float(s.iloc[int(exact[0])])
    below = np.where(d < target)[0]
    above = np.where(d > target)[0]
    if not len(below) or not len(above):
        return np.nan
    i0, i1 = int(below[-1]), int(above[0])
    d0, d1 = float(d[i0]), float(d[i1])
    v0, v1 = float(s.iloc[i0]), float(s.iloc[i1])
    t0, t1, tt = d0 / DAYS_PER_YEAR, d1 / DAYS_PER_YEAR, target / DAYS_PER_YEAR
    w0, w1 = v0 * v0 * t0, v1 * v1 * t1
    weight = (tt - t0) / (t1 - t0)
    total_var = (1.0 - weight) * w0 + weight * w1
    return float(np.sqrt(max(total_var, 0.0) / tt))


def constant_tenor_delta_slice(surface: DeltaVolSurface, target_days: float = 30.0) -> pd.Series:
    """Interpolate a constant-maturity delta smile in total-variance time."""
    if target_days <= 0:
        raise ValueError("target_days must be positive")
    out: dict[str, float] = {"target_days": float(target_days)}
    for col in surface.iv.columns:
        out[col] = _constant_tenor_column(surface.iv[col], float(target_days))
    atm = out.get("atm_iv", np.nan)
    for target in surface.deltas:
        for side in ("p", "c"):
            key = _bucket(target, side)
            iv = out.get(f"iv_{key}", np.nan)
            out[f"delta_ratio_{key}"] = iv / atm if np.isfinite(iv) and np.isfinite(atm) and atm > 0 else np.nan
    return pd.Series(out, dtype=float)


def delta_ratio_term_structure(surface: DeltaVolSurface) -> pd.DataFrame:
    """Return actual-expiry delta ratios with expiry metadata."""
    out = surface.ratios.copy()
    out.insert(0, "expiry", surface.expiries.reindex(out.index).to_numpy())
    return out.reset_index()


def delta_lump_scores(surface: DeltaVolSurface) -> pd.DataFrame:
    """Local term-structure residuals for delta ratios.

    An interior expiry is compared with the straight line through its immediate
    neighboring expiries.  This is a *cross-sectional* lump diagnostic, not a
    historical z-score.  Historical anomaly scores are added by the VRP history
    builder.
    """
    ratios = surface.ratios.sort_index()
    out = pd.DataFrame(index=ratios.index)
    dte = ratios.index.to_numpy(float)
    for col in ratios.columns:
        values = pd.to_numeric(ratios[col], errors="coerce").to_numpy(float)
        residual = np.full(len(values), np.nan)
        for i in range(1, len(values) - 1):
            if not np.all(np.isfinite(values[i - 1 : i + 2])):
                continue
            x0, x, x1 = dte[i - 1], dte[i], dte[i + 1]
            expected = values[i - 1] + (values[i + 1] - values[i - 1]) * (x - x0) / (x1 - x0)
            residual[i] = values[i] - expected
        out[col.replace("delta_ratio_", "delta_lump_")] = residual
    out["delta_lump_max_abs"] = out.abs().max(axis=1)
    return out.reset_index()


def delta_surface_change_features(history: pd.DataFrame) -> pd.DataFrame:
    """Observable linear decomposition of daily constant-tenor delta-smile changes.

    This is intentionally a transparent *delta-space decomposition*, not an
    exact Vanna-Volga replication.  It separates the ATM level, 25-delta skew
    gradients, and 10/15/25-delta curvature on each wing.  A log spot return is
    included as context for later sticky-strike research.
    """
    df = history.copy()
    out = pd.DataFrame(index=df.index)

    atm = pd.to_numeric(df.get("atm_iv"), errors="coerce")
    out["surface_parallel_shift"] = atm.diff()

    for side in ("p", "c"):
        iv25 = pd.to_numeric(df.get(f"iv_25{side}"), errors="coerce")
        skew = iv25 - atm
        out[f"surface_{'put' if side == 'p' else 'call'}_skew"] = skew
        out[f"surface_{'put' if side == 'p' else 'call'}_skew_change"] = skew.diff()

        iv10 = pd.to_numeric(df.get(f"iv_10{side}"), errors="coerce")
        iv15 = pd.to_numeric(df.get(f"iv_15{side}"), errors="coerce")
        # Difference of adjacent delta slopes; scaling makes the measure
        # comparable despite the unequal 5- and 10-delta intervals.
        slope_tail = (iv15 - iv10) / 0.05
        slope_wing = (iv25 - iv15) / 0.10
        convexity = slope_tail - slope_wing
        name = "downside" if side == "p" else "upside"
        out[f"surface_{name}_convexity"] = convexity
        out[f"surface_{name}_convexity_change"] = convexity.diff()

    if "underlying_price" in df:
        spot = pd.to_numeric(df["underlying_price"], errors="coerce")
        out["surface_spot_log_return"] = np.log(spot).diff()

    change_cols = [
        "surface_parallel_shift",
        "surface_put_skew_change",
        "surface_call_skew_change",
        "surface_downside_convexity_change",
        "surface_upside_convexity_change",
    ]
    available = [c for c in change_cols if c in out]
    if available:
        out["surface_change_magnitude"] = np.sqrt((out[available] ** 2).sum(axis=1, min_count=1))
    return out

"""From a cleaned chain to a calibration-ready slice.

This is the seam between the data layer and the model layer. For one expiry it
runs: parity regression -> forward and discount -> OTM leg selection -> our own
IV inversion -> log-moneyness, total variance, and fit weights.

On weighting. The calibrator minimises squared error in total variance w, but
what you actually care about is error in price relative to the width of the
market. Chaining the two, price_err ~ vega * iv_err and iv_err ~ w_err/(2*iv*T),
so matching a spread-relative price objective means

    weight_i = ( vega_i / (2 * iv_i * T * half_spread_i) )^2

which is what `mode='vega_spread'` computes. Equal weighting lets wide, illiquid
wing quotes dominate a fit that you will then interpret as signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..blackscholes import black_vega, implied_vol_vec
from ..forward import ForwardFit, fit_forward
from .clean import matched_pairs

__all__ = ["Slice", "build_slice", "build_all_slices", "svi_weights"]


@dataclass
class Slice:
    expiry: pd.Timestamp
    T: float
    dte: float
    spot: float
    forward_fit: ForwardFit
    k: np.ndarray            # log-moneyness log(K/F)
    w: np.ndarray            # total implied variance iv^2 * T
    iv: np.ndarray
    strikes: np.ndarray
    is_call: np.ndarray
    half_spread: np.ndarray  # price-space half spread of the used leg
    weights: np.ndarray

    @property
    def forward(self) -> float:
        return self.forward_fit.forward

    @property
    def n(self) -> int:
        return len(self.k)

    def __repr__(self):
        return (f"Slice(dte={self.dte:.1f}, n={self.n}, F={self.forward:.2f}, "
                f"R2={self.forward_fit.r_squared:.5f})")


def svi_weights(iv, T, vega, half_spread, mode="vega_spread"):
    """Fit weights in total-variance space. See module docstring."""
    iv, vega, half_spread = map(lambda x: np.asarray(x, float), (iv, vega, half_spread))
    hs = np.where(half_spread > 0, half_spread, np.nan)

    if mode == "equal":
        wt = np.ones_like(iv)
    elif mode == "vega":
        wt = (vega / (2.0 * iv * T)) ** 2
    elif mode == "spread":
        wt = 1.0 / hs ** 2
    elif mode == "vega_spread":
        wt = (vega / (2.0 * iv * T * hs)) ** 2
    else:
        raise ValueError(f"unknown weighting mode {mode!r}")

    wt = np.where(np.isfinite(wt) & (wt > 0), wt, 0.0)
    return wt / wt.mean() if wt.mean() > 0 else np.ones_like(iv)


def build_slice(
    chain: pd.DataFrame,
    expiry,
    weight_mode: str = "vega_spread",
    parity_window: float = 0.10,
    otm_only: bool = True,
    min_quotes: int = 8,
) -> Slice:
    """Build one calibration-ready slice from a cleaned chain.

    `otm_only=True` uses calls above the forward and puts below it. OTM options
    carry essentially all the vol information and avoid double-counting the
    same strike, which would otherwise let parity noise into the fit twice.
    """
    pairs = matched_pairs(chain, expiry)
    if len(pairs) < 3:
        raise ValueError(f"expiry {expiry}: only {len(pairs)} matched strikes")

    T = float(pairs["T"].iloc[0])
    dte = float(pairs["dte"].iloc[0])
    spot = float(pairs["underlying_price"].iloc[0])

    ff = fit_forward(
        pairs["strike"].to_numpy(),
        pairs["mid_c"].to_numpy(),
        pairs["mid_p"].to_numpy(),
        T,
        spot=spot,
        moneyness_window=parity_window,
        weights=1.0 / np.maximum(pairs["combined_spread"].to_numpy(), 1e-6),
    )

    rows = chain[chain["expiry"] == expiry]
    K = rows["strike"].to_numpy(float)
    mid = rows["mid"].to_numpy(float)
    half = rows["spread"].to_numpy(float) / 2.0
    is_call = (rows["right"] == "C").to_numpy()

    if otm_only:
        keep = np.where(is_call, K >= ff.forward, K < ff.forward)
        K, mid, half, is_call = K[keep], mid[keep], half[keep], is_call[keep]

    iv = implied_vol_vec(mid, ff.forward, K, T, is_call, ff.discount)

    ok = np.isfinite(iv) & (iv > 1e-4)
    if ok.sum() < min_quotes:
        raise ValueError(
            f"expiry {expiry}: only {ok.sum()} quotes survived IV inversion "
            f"(need {min_quotes}). Check the parity fit: R2={ff.r_squared:.4f}"
        )
    K, mid, half, is_call, iv = K[ok], mid[ok], half[ok], is_call[ok], iv[ok]

    vega = np.asarray(black_vega(ff.forward, K, iv, T, ff.discount), float)

    return Slice(
        expiry=pd.Timestamp(expiry),
        T=T,
        dte=dte,
        spot=spot,
        forward_fit=ff,
        k=np.log(K / ff.forward),
        w=iv ** 2 * T,
        iv=iv,
        strikes=K,
        is_call=is_call,
        half_spread=half,
        weights=svi_weights(iv, T, vega, half, weight_mode),
    )


def build_all_slices(chain: pd.DataFrame, verbose: bool = True, **kwargs) -> list[Slice]:
    """Build every expiry in a cleaned chain, skipping those that fail."""
    out = []
    for exp in sorted(chain["expiry"].unique()):
        try:
            out.append(build_slice(chain, exp, **kwargs))
        except Exception as exc:
            if verbose:
                print(f"  skip {pd.Timestamp(exp).date()}: {exc}")
    return out

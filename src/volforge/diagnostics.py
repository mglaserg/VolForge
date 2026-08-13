"""Fit diagnostics, centred on one question worth asking before anything else.

The honest failure mode of a residual-based vol strategy is that SVI residuals
on liquid names are dominated by model misspecification and quote noise rather
than mispricing. `residual_report` measures that directly: it expresses each
residual as a multiple of the option's own half-spread. If almost nothing
clears 1.0, there is no tradeable signal in the residuals, and the phases
built on top of them will produce backtests that die on contact with costs.

Run this on day one. It is a cheap answer to an expensive question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blackscholes import black_vega
from .svi import SVIParams, svi_total_variance

__all__ = ["ResidualReport", "residual_report"]


@dataclass
class ResidualReport:
    iv_resid: np.ndarray        # market IV - SVI IV, in vol decimals
    spread_units: np.ndarray    # |iv_resid| / half-spread in IV terms
    frac_over_half: float       # share of quotes with |resid| > 1 half-spread
    frac_over_one: float        # share with |resid| > 2 half-spreads (a full spread)
    median_spread_units: float
    n: int

    def summary(self) -> str:
        return (
            f"n={self.n}  median |resid| = {self.median_spread_units:.2f} half-spreads  "
            f"| >1 half-spread: {self.frac_over_half:.1%}  "
            f"| >1 full spread: {self.frac_over_one:.1%}"
        )

    @property
    def has_signal_candidate(self) -> bool:
        """Weak gate, not a green light. Below ~5% you are fitting noise."""
        return self.frac_over_one >= 0.05


def residual_report(
    k,
    iv_market,
    params: SVIParams,
    T: float,
    bid=None,
    ask=None,
    forward=None,
    strikes=None,
    iv_half_spread=None,
) -> ResidualReport:
    """Compare fit residuals against the width of the market.

    Supply either `iv_half_spread` directly, or price-space `bid`/`ask` plus
    `forward` and `strikes`, in which case the spread is converted to vol terms
    via vega: half_spread_iv = (ask - bid) / 2 / vega.
    """
    k = np.asarray(k, dtype=float)
    iv_market = np.asarray(iv_market, dtype=float)
    iv_fit = np.sqrt(np.maximum(svi_total_variance(k, params), 0.0) / T)
    resid = iv_market - iv_fit

    if iv_half_spread is None:
        if bid is None or ask is None or forward is None or strikes is None:
            raise ValueError("provide iv_half_spread, or bid/ask + forward + strikes")
        vega = np.asarray(black_vega(forward, np.asarray(strikes, float),
                                     iv_market, T), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            iv_half_spread = (np.asarray(ask, float) - np.asarray(bid, float)) / 2.0 / vega
    iv_half_spread = np.asarray(iv_half_spread, dtype=float)

    ok = np.isfinite(resid) & np.isfinite(iv_half_spread) & (iv_half_spread > 0)
    units = np.abs(resid[ok]) / iv_half_spread[ok]

    return ResidualReport(
        iv_resid=resid,
        spread_units=units,
        frac_over_half=float(np.mean(units > 1.0)),
        frac_over_one=float(np.mean(units > 2.0)),
        median_spread_units=float(np.median(units)),
        n=int(ok.sum()),
    )

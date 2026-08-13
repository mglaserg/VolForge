"""Whole-surface construction.

Turns a day's independently-fitted slices into a single fixed-size matrix, so
that every trading day becomes the same vector and cross-sectional methods
(PCA, z-scores) have something well-defined to operate on.

Two choices here carry real weight.

**Interpolation is linear in total variance against T, at fixed k.** This is
the standard choice because it is the one that preserves the no-calendar-
arbitrage condition: if w is non-decreasing in T at the observed maturities,
linear interpolation keeps it so. Interpolating implied *vol* instead does not
have that property and will manufacture arbitrage between your grid points.

**Slices are fitted independently, so calendar arbitrage is possible.** Nothing
in a per-slice fit knows about neighbouring maturities. `repair_calendar`
enforces monotonicity by raising the later slice's grid values where they dip
below the earlier one, and reports how much it moved. Large repairs are a
signal that a slice is bad, not that the repair is working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .svi import SVIParams, svi_total_variance

__all__ = [
    "DEFAULT_TENORS",
    "DEFAULT_K_GRID",
    "Surface",
    "build_surface",
    "repair_calendar",
]

DEFAULT_TENORS = np.array([7.0, 14.0, 30.0, 60.0, 90.0])
DEFAULT_K_GRID = np.linspace(-0.20, 0.20, 17)

DAYS_PER_YEAR = 365.25


@dataclass
class Surface:
    trade_date: pd.Timestamp
    symbol: str
    tenor_days: np.ndarray
    k_grid: np.ndarray
    total_var: np.ndarray       # shape (n_tenors, n_k)
    iv: np.ndarray
    extrapolated: np.ndarray    # bool per tenor: outside the observed T range
    spot: float
    forwards: np.ndarray        # interpolated forward per tenor
    n_slices_used: int
    calendar_repair: float = 0.0
    node_index: list = field(default_factory=list)  # [(tenor, k), ...] row-major

    @property
    def vector(self) -> np.ndarray:
        """Flatten to the day's surface vector, row-major over (tenor, k)."""
        return self.total_var.ravel()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.iv, index=pd.Index(self.tenor_days, name="tenor_days"),
                            columns=pd.Index(np.round(self.k_grid, 4), name="k"))

    @property
    def is_clean(self) -> bool:
        """No extrapolated tenors and no material calendar repair."""
        return (not self.extrapolated.any()) and self.calendar_repair < 1e-6

    def __repr__(self):
        return (f"Surface({self.symbol} {self.trade_date:%Y-%m-%d}, "
                f"{len(self.tenor_days)}x{len(self.k_grid)}, "
                f"slices={self.n_slices_used}, clean={self.is_clean})")


def build_surface(
    slices_and_params,
    trade_date,
    symbol: str = "",
    tenor_days=DEFAULT_TENORS,
    k_grid=DEFAULT_K_GRID,
    reliable_only: bool = True,
    allow_extrapolation: bool = False,
    repair: bool = True,
) -> Surface:
    """Build the daily fixed-grid surface.

    Parameters
    ----------
    slices_and_params : iterable of (Slice, SVIFit) for one trade date
    reliable_only : drop fits flagged by SVIFit.is_reliable. Strongly advised --
        a boundary-pinned slice contributes pure noise to every grid node it
        touches, and that noise propagates into PCA as a spurious component.
    allow_extrapolation : if False, tenors outside the observed maturity range
        are still produced (flat-extrapolated in variance) but flagged in
        `extrapolated`, and `Surface.is_clean` becomes False.
    """
    tenor_days = np.asarray(tenor_days, float)
    k_grid = np.asarray(k_grid, float)

    usable = [(s, f) for s, f in slices_and_params
              if (f.is_reliable if reliable_only else f.success)]
    if len(usable) < 2:
        raise ValueError(
            f"need >=2 usable slices to build a surface, got {len(usable)} "
            f"(of {len(list(slices_and_params))} supplied). Loosen reliable_only "
            f"only if you accept the noise it lets in."
        )
    usable.sort(key=lambda sf: sf[0].T)

    T_obs = np.array([s.T for s, _ in usable])
    F_obs = np.array([s.forward for s, _ in usable])
    spot = float(usable[0][0].spot)

    # Evaluate each fitted slice on the k grid -> total variance at observed T.
    W_obs = np.vstack([svi_total_variance(k_grid, f.params) for _, f in usable])

    T_grid = tenor_days / DAYS_PER_YEAR
    extrap = (T_grid < T_obs.min()) | (T_grid > T_obs.max())
    if extrap.any() and allow_extrapolation is False:
        pass  # produced anyway, but flagged; see Surface.is_clean

    # Linear in total variance vs T, at each fixed k. Flat beyond the ends.
    W_grid = np.empty((len(T_grid), len(k_grid)))
    for j in range(len(k_grid)):
        W_grid[:, j] = np.interp(T_grid, T_obs, W_obs[:, j])

    repair_amount = 0.0
    if repair:
        W_grid, repair_amount = repair_calendar(W_grid, T_grid)

    F_grid = np.interp(T_grid, T_obs, F_obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = np.sqrt(np.maximum(W_grid, 0.0) / T_grid[:, None])

    return Surface(
        trade_date=pd.Timestamp(trade_date),
        symbol=symbol,
        tenor_days=tenor_days,
        k_grid=k_grid,
        total_var=W_grid,
        iv=iv,
        extrapolated=extrap,
        spot=spot,
        forwards=F_grid,
        n_slices_used=len(usable),
        calendar_repair=repair_amount,
        node_index=[(float(t), float(k)) for t in tenor_days for k in k_grid],
    )


def repair_calendar(W, T_grid, tol=0.0):
    """Enforce total variance non-decreasing in T at each k.

    Returns (repaired_W, total_absolute_adjustment). A large adjustment means a
    slice is wrong; it does not mean the repair succeeded. Track it.
    """
    W = np.array(W, dtype=float, copy=True)
    before = W.copy()
    order = np.argsort(T_grid)
    for j in range(W.shape[1]):
        col = W[order, j]
        col = np.maximum.accumulate(col)
        W[order, j] = col
    return W, float(np.abs(W - before).sum())


def surface_panel(surfaces) -> pd.DataFrame:
    """Stack daily Surfaces into a panel: rows = date, columns = (tenor, k)."""
    surfaces = sorted(surfaces, key=lambda s: s.trade_date)
    cols = pd.MultiIndex.from_tuples(surfaces[0].node_index, names=["tenor_days", "k"])
    return pd.DataFrame(
        np.vstack([s.vector for s in surfaces]),
        index=pd.DatetimeIndex([s.trade_date for s in surfaces], name="trade_date"),
        columns=cols,
    )

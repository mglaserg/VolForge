"""Fengler-style arbitrage-free smoothing splines in call-price space.

The core problem is the constrained natural cubic smoothing spline from
Fengler (2009).  For one maturity we minimise

    1/2 sum_i w_i (g_i - y_i)^2 + lambda/2 * integral [g''(x)]^2 dx

using the value/second-derivative representation ``(g, gamma)``.  The spline
identity ``Q.T @ g = R @ gamma`` is imposed exactly, and linear constraints
ensure the forward-normalised call curve is bounded, decreasing and convex.

A full surface is fitted sequentially from the longest maturity backwards.
The shorter slice is constrained below the already fitted longer slice at a
dense set of equal forward-moneyness points.  This is the calendar-spread
constraint in the forward-moneyness coordinates used by Fengler.  The result
is then evaluated on VolForge's standard total-variance grid so it can be
compared directly with SVI/SSVI/eSSVI and fed to the same PCA machinery.

Prices are represented as forward-normalised undiscounted calls

    c(x,T) = C(K,T) / (D(T) F(T)),  x = K/F(T),

for which the strike-arbitrage bounds become especially simple:

    max(1-x,0) <= c <= 1,   -1 <= dc/dx <= 0,   d2c/dx2 >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .blackscholes import black_price, implied_vol_vec

__all__ = [
    "FenglerSliceFit",
    "FenglerSurfaceFit",
    "spline_qr_matrices",
    "natural_spline_basis",
    "fit_fengler_slice",
    "fit_fengler_surface",
    "calibrate_fengler",
]

_EPS = 1e-12


def spline_qr_matrices(knots):
    """Return Fengler/Green-Silverman Q and R matrices for natural splines."""
    u = np.asarray(knots, dtype=float)
    if u.ndim != 1 or len(u) < 4 or np.any(np.diff(u) <= 0):
        raise ValueError("knots must be a strictly increasing 1D array of length >=4")
    n = len(u)
    h = np.diff(u)
    Q = np.zeros((n, n - 2), dtype=float)
    R = np.zeros((n - 2, n - 2), dtype=float)
    for j in range(1, n - 1):
        c = j - 1
        Q[j - 1, c] = 1.0 / h[j - 1]
        Q[j, c] = -(1.0 / h[j - 1] + 1.0 / h[j])
        Q[j + 1, c] = 1.0 / h[j]
        R[c, c] = (h[j - 1] + h[j]) / 3.0
        if c < n - 3:
            R[c, c + 1] = h[j] / 6.0
            R[c + 1, c] = h[j] / 6.0
    return Q, R


def natural_spline_basis(knots, x_eval):
    """Linear map E such that spline(x_eval) == E @ [g, gamma_interior]."""
    u = np.asarray(knots, float)
    x = np.atleast_1d(np.asarray(x_eval, float))
    n = len(u)
    E = np.zeros((len(x), 2 * n - 2), dtype=float)

    for r, z in enumerate(x):
        if z <= u[0]:
            # Natural linear extrapolation using the left endpoint derivative:
            # s'(u0)=(g1-g0)/h - h*gamma1/6, gamma0=0.
            h = u[1] - u[0]
            dz = z - u[0]
            E[r, 0] = 1.0 - dz / h
            E[r, 1] = dz / h
            E[r, n] = -dz * h / 6.0  # interior gamma at knot 1
            continue
        if z >= u[-1]:
            h = u[-1] - u[-2]
            dz = z - u[-1]
            E[r, n - 2] = -dz / h
            E[r, n - 1] = 1.0 + dz / h
            E[r, n + (n - 3)] = dz * h / 6.0  # gamma at knot n-2
            continue

        i = int(np.searchsorted(u, z) - 1)
        h = u[i + 1] - u[i]
        A = (u[i + 1] - z) / h
        B = (z - u[i]) / h
        E[r, i] += A
        E[r, i + 1] += B

        # gamma endpoints are zero and therefore absent from the state vector.
        if 1 <= i <= n - 2:
            gi = n + (i - 1)
            E[r, gi] += ((u[i + 1] - z) ** 3 / (6.0 * h) - A * h * h / 6.0)
        if 1 <= i + 1 <= n - 2:
            gj = n + i
            E[r, gj] += ((z - u[i]) ** 3 / (6.0 * h) - B * h * h / 6.0)
    return E


def _dedupe_xy(x, y, half_spread):
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]
    hs = np.asarray(half_spread, float)[order]
    uniq = np.unique(np.round(x, 12))
    if len(uniq) == len(x):
        return x, y, hs
    xo, yo, ho = [], [], []
    for q in uniq:
        m = np.isclose(x, q, atol=5e-13, rtol=0)
        w = 1.0 / np.maximum(hs[m], 1e-8) ** 2
        xo.append(float(np.average(x[m], weights=w)))
        yo.append(float(np.average(y[m], weights=w)))
        ho.append(float(np.sqrt(1.0 / np.sum(w))))
    return np.asarray(xo), np.asarray(yo), np.asarray(ho)


@dataclass
class FenglerSliceFit:
    T: float
    dte: float
    forward: float
    discount: float
    spot: float
    knots: np.ndarray
    values: np.ndarray
    gamma: np.ndarray  # full second-derivative vector; endpoints are zero
    market_values: np.ndarray
    weights: np.ndarray
    smoothing_lambda: float
    success: bool
    objective: float
    rmse_price_norm: float
    rmse_price: float
    rmse_iv: float
    max_abs_err_iv: float
    min_gamma: float
    left_slope: float
    right_slope: float
    strike_arb_free: bool
    calendar_free: bool
    calendar_margin_min: float
    calendar_repair: float = 0.0

    @property
    def is_reliable(self):
        return self.success and self.strike_arb_free and self.calendar_free

    @property
    def k_min(self):
        return float(np.log(self.knots[0]))

    @property
    def k_max(self):
        return float(np.log(self.knots[-1]))

    def _state(self):
        return np.concatenate([self.values, self.gamma[1:-1]])

    def normalized_call(self, x, clip_bounds=True):
        xa = np.atleast_1d(np.asarray(x, float))
        vals = natural_spline_basis(self.knots, xa) @ self._state()
        if clip_bounds:
            vals = np.clip(vals, np.maximum(1.0 - xa, 0.0), 1.0)
        return float(vals[0]) if np.ndim(x) == 0 else vals

    def implied_vol(self, k, allow_extrapolation=False):
        ka = np.atleast_1d(np.asarray(k, float))
        x = np.exp(ka)
        outside = (x < self.knots[0]) | (x > self.knots[-1])
        c = self.normalized_call(x)
        iv = implied_vol_vec(c, 1.0, x, self.T, np.ones_like(x, dtype=bool), 1.0)
        if not allow_extrapolation:
            iv[outside] = np.nan
        return float(iv[0]) if np.ndim(k) == 0 else iv

    def total_variance(self, k, allow_extrapolation=False):
        iv = self.implied_vol(k, allow_extrapolation=allow_extrapolation)
        return np.asarray(iv) ** 2 * self.T

    def as_dict(self):
        return {
            "dte": float(self.dte),
            "T": float(self.T),
            "n_knots": int(len(self.knots)),
            "smoothing_lambda": float(self.smoothing_lambda),
            "success": bool(self.success),
            "objective": float(self.objective),
            "rmse_price_norm": float(self.rmse_price_norm),
            "rmse_price": float(self.rmse_price),
            "rmse_iv": float(self.rmse_iv),
            "max_abs_err_iv": float(self.max_abs_err_iv),
            "min_gamma": float(self.min_gamma),
            "left_slope": float(self.left_slope),
            "right_slope": float(self.right_slope),
            "strike_arb_free": bool(self.strike_arb_free),
            "calendar_free": bool(self.calendar_free),
            "calendar_margin_min": float(self.calendar_margin_min),
            "calendar_repair": float(self.calendar_repair),
            "k_min": self.k_min,
            "k_max": self.k_max,
        }


def fit_fengler_slice(
    slc,
    smoothing_lambda: float = 1e-5,
    calendar_upper=None,
    calendar_x=None,
    weight_cap: float = 100.0,
    maxiter: int = 2500,
) -> FenglerSliceFit:
    """Fit one constrained natural cubic spline in forward-moneyness space."""
    x = np.exp(np.asarray(slc.k, float))
    # Market IV inversion has already mapped each selected OTM quote to the
    # corresponding Black price.  Repricing as a call gives the parity-equivalent
    # call at that strike, independent of whether the observed leg was C or P.
    y = np.asarray(black_price(1.0, x, slc.iv, slc.T, True, 1.0), float)
    hs = np.asarray(slc.half_spread, float) / max(slc.forward_fit.discount * slc.forward, 1e-12)
    x, y, hs = _dedupe_xy(x, y, hs)
    if len(x) < 6:
        raise ValueError(f"Fengler slice needs >=6 distinct strikes, got {len(x)}")

    n = len(x)
    Q, R = spline_qr_matrices(x)
    w = 1.0 / np.maximum(hs, np.nanmedian(hs[hs > 0]) * 0.05 if np.any(hs > 0) else 1e-5) ** 2
    w = w / np.mean(w)
    w = np.minimum(w, float(weight_cap))

    nvar = 2 * n - 2
    Aeq = np.zeros((n - 2, nvar), float)
    Aeq[:, :n] = Q.T
    Aeq[:, n:] = -R
    eq = LinearConstraint(Aeq, np.zeros(n - 2), np.zeros(n - 2))

    constraints = [eq]

    # Monotonicity at the knots. Convexity is imposed through gamma>=0, and
    # the endpoint derivative bounds make the whole spline decreasing, but the
    # discrete constraint improves numerical robustness at tiny spreads.
    M = np.zeros((n - 1, nvar), float)
    for i in range(n - 1):
        M[i, i] = -1.0
        M[i, i + 1] = 1.0
    constraints.append(LinearConstraint(M, -np.inf, np.zeros(n - 1)))

    # Endpoint derivative bounds in normalized coordinates: -1 <= c_x <= 0.
    D = np.zeros((2, nvar), float)
    h0 = x[1] - x[0]
    D[0, 0] = -1.0 / h0
    D[0, 1] = 1.0 / h0
    D[0, n] = -h0 / 6.0
    hn = x[-1] - x[-2]
    D[1, n - 2] = -1.0 / hn
    D[1, n - 1] = 1.0 / hn
    D[1, n + n - 3] = hn / 6.0
    constraints.append(LinearConstraint(D, np.array([-1.0, -np.inf]), np.array([np.inf, 0.0])))

    # Calendar no-crossing constraints are linear because spline evaluation is
    # linear in (g, gamma).  Use a dense grid over the overlap, not just knots.
    cal_margin_target = np.inf
    if calendar_upper is not None:
        if calendar_x is None:
            calendar_x = np.linspace(x[0], x[-1], max(121, 2 * n))
        cx = np.asarray(calendar_x, float)
        cx = cx[(cx >= x[0]) & (cx <= x[-1])]
        if len(cx):
            upper = np.asarray(calendar_upper(cx), float)
            lower = np.maximum(1.0 - cx, 0.0)
            # A valid longer slice should always lie above intrinsic. Small
            # numerical violations are repaired only enough to keep the QP feasible.
            upper_safe = np.maximum(upper, lower + 1e-10)
            cal_margin_target = float(np.min(upper_safe - lower))
            E = natural_spline_basis(x, cx)
            constraints.append(LinearConstraint(E, -np.inf, upper_safe))

    lower_g = np.maximum(1.0 - x, 0.0)
    upper_g = np.ones(n)
    lb = np.concatenate([lower_g, np.zeros(n - 2)])
    ub = np.concatenate([upper_g, np.full(n - 2, np.inf)])
    bounds = Bounds(lb, ub)

    lam = float(smoothing_lambda)
    if lam < 0:
        raise ValueError("smoothing_lambda must be >=0")

    def objective(z):
        g = z[:n]
        gamma = z[n:]
        resid = g - y
        return 0.5 * float(np.sum(w * resid * resid) + lam * gamma @ R @ gamma)

    def jac(z):
        g = z[:n]
        gamma = z[n:]
        return np.concatenate([w * (g - y), lam * (R @ gamma)])

    # Smooth convex Black curve as the starting point. Its second derivatives
    # are recovered from the spline identity so equality constraints start close.
    med_iv = float(np.nanmedian(slc.iv))
    g0 = np.asarray(black_price(1.0, x, med_iv, slc.T, True, 1.0), float)
    g0 = np.clip(g0, lower_g + 1e-10, upper_g - 1e-10)
    try:
        gamma0 = np.linalg.solve(R, Q.T @ g0)
    except np.linalg.LinAlgError:
        gamma0 = np.zeros(n - 2)
    z0 = np.concatenate([g0, gamma0])

    res = minimize(
        objective,
        z0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "disp": False},
    )

    z = np.asarray(res.x, float)
    g = z[:n]
    gamma_full = np.zeros(n, float)
    gamma_full[1:-1] = z[n:]

    # Observed-point errors.
    fit_iv = implied_vol_vec(g, 1.0, x, slc.T, np.ones(n, dtype=bool), 1.0)
    iv_err = fit_iv - np.asarray(slc.iv, float)
    price_err = g - y

    left_slope = (g[1] - g[0]) / h0 - h0 * gamma_full[1] / 6.0
    right_slope = (g[-1] - g[-2]) / hn + hn * gamma_full[-2] / 6.0
    strike_ok = bool(
        np.min(gamma_full) >= -2e-7
        and left_slope >= -1.0 - 2e-6
        and right_slope <= 2e-6
        and np.max(np.diff(g)) <= 2e-7
        and np.all(g >= lower_g - 2e-7)
        and np.all(g <= 1.0 + 2e-7)
    )

    calendar_ok = True
    calendar_margin = np.inf
    if calendar_upper is not None:
        cx = np.asarray(calendar_x, float)
        cx = cx[(cx >= x[0]) & (cx <= x[-1])]
        if len(cx):
            vals = natural_spline_basis(x, cx) @ z
            up = np.asarray(calendar_upper(cx), float)
            calendar_margin = float(np.min(up - vals))
            calendar_ok = bool(calendar_margin >= -2e-7)

    return FenglerSliceFit(
        T=float(slc.T),
        dte=float(slc.dte),
        forward=float(slc.forward),
        discount=float(slc.forward_fit.discount),
        spot=float(slc.spot),
        knots=x,
        values=g,
        gamma=gamma_full,
        market_values=y,
        weights=w,
        smoothing_lambda=lam,
        success=bool(res.success or (np.isfinite(res.fun) and strike_ok and calendar_ok)),
        objective=float(res.fun),
        rmse_price_norm=float(np.sqrt(np.mean(price_err * price_err))),
        rmse_price=float(np.sqrt(np.mean(price_err * price_err)) * slc.forward_fit.discount * slc.forward),
        rmse_iv=float(np.sqrt(np.nanmean(iv_err * iv_err))),
        max_abs_err_iv=float(np.nanmax(np.abs(iv_err))),
        min_gamma=float(np.min(gamma_full)),
        left_slope=float(left_slope),
        right_slope=float(right_slope),
        strike_arb_free=strike_ok,
        calendar_free=calendar_ok,
        calendar_margin_min=float(calendar_margin),
        calendar_repair=max(0.0, -float(cal_margin_target)) if np.isfinite(cal_margin_target) else 0.0,
    )


@dataclass
class FenglerSurfaceFit:
    slices: list[FenglerSliceFit]
    smoothing_lambda: float
    success: bool
    butterfly_free: bool
    calendar_free: bool
    rmse_iv: float
    max_abs_err_iv: float
    n_obs: int
    calendar_margin_min: float

    @property
    def is_reliable(self):
        return self.success and self.butterfly_free and self.calendar_free and len(self.slices) >= 3

    @property
    def T_obs(self):
        return np.array([s.T for s in self.slices], float)

    @property
    def forwards(self):
        return np.array([s.forward for s in self.slices], float)

    @property
    def spot(self):
        return float(self.slices[0].spot)

    def to_surface(self, trade_date, symbol="", tenor_days=None, k_grid=None):
        from .surface import Surface, DEFAULT_K_GRID, DEFAULT_TENORS, DAYS_PER_YEAR

        tenor_days = np.asarray(DEFAULT_TENORS if tenor_days is None else tenor_days, float)
        k_grid = np.asarray(DEFAULT_K_GRID if k_grid is None else k_grid, float)
        Ts = self.T_obs
        Wobs = np.full((len(self.slices), len(k_grid)), np.nan)
        any_k_extrap = np.zeros(len(self.slices), dtype=bool)

        for i, fit in enumerate(self.slices):
            inside = (k_grid >= fit.k_min) & (k_grid <= fit.k_max)
            any_k_extrap[i] = not bool(np.all(inside))
            if np.any(inside):
                Wobs[i, inside] = fit.total_variance(k_grid[inside], allow_extrapolation=False)
            # Fengler is nonparametric and should not manufacture wing shape.
            # Outside the observed knot range use the nearest valid total
            # variance only so the fixed matrix remains usable, and mark it as
            # extrapolated. Research comparisons should filter those tenors.
            valid = np.where(np.isfinite(Wobs[i]))[0]
            if len(valid):
                Wobs[i, :valid[0]] = Wobs[i, valid[0]]
                Wobs[i, valid[-1] + 1:] = Wobs[i, valid[-1]]

        Tgrid = tenor_days / DAYS_PER_YEAR
        W = np.empty((len(Tgrid), len(k_grid)), float)
        extrap = (Tgrid < Ts.min()) | (Tgrid > Ts.max())
        for j in range(len(k_grid)):
            good = np.isfinite(Wobs[:, j])
            if good.sum() < 2:
                W[:, j] = np.nan
                continue
            tw = Ts[good]
            ww = Wobs[good, j]
            # Sequential Fengler constraints should already make these
            # non-decreasing; maximum-accumulate only removes tiny solver noise.
            ww = np.maximum.accumulate(ww)
            W[:, j] = np.interp(Tgrid, tw, ww)
            W[Tgrid < tw[0], j] = ww[0]
            W[Tgrid > tw[-1], j] = ww[-1]

        # Mark a tenor extrapolated if time is outside the fitted horizon OR if
        # either adjacent observed slice needed wing extrapolation on this grid.
        for i, t in enumerate(Tgrid):
            idx = int(np.argmin(np.abs(Ts - np.clip(t, Ts.min(), Ts.max()))))
            extrap[i] = extrap[i] or any_k_extrap[idx]

        iv = np.sqrt(np.maximum(W, 0.0) / Tgrid[:, None])
        F = np.interp(Tgrid, Ts, self.forwards)
        return Surface(
            trade_date=__import__("pandas").Timestamp(trade_date),
            symbol=symbol,
            tenor_days=tenor_days,
            k_grid=k_grid,
            total_var=W,
            iv=iv,
            extrapolated=extrap,
            spot=self.spot,
            forwards=F,
            n_slices_used=len(self.slices),
            calendar_repair=0.0,
            node_index=[(float(t), float(k)) for t in tenor_days for k in k_grid],
        )

    def as_row(self):
        return {
            "smoothing_lambda": float(self.smoothing_lambda),
            "success": bool(self.success),
            "butterfly_free": bool(self.butterfly_free),
            "calendar_free": bool(self.calendar_free),
            "rmse_iv": float(self.rmse_iv),
            "max_abs_err_iv": float(self.max_abs_err_iv),
            "n_obs": int(self.n_obs),
            "n_slices": int(len(self.slices)),
            "calendar_margin_min": float(self.calendar_margin_min),
        }


def fit_fengler_surface(
    slices: Iterable,
    smoothing_lambda: float = 1e-5,
    calendar_grid_size: int = 181,
) -> FenglerSurfaceFit:
    """Fit a Fengler arbitrage-free price surface, longest expiry backwards."""
    slcs = sorted(list(slices), key=lambda s: s.T, reverse=True)
    if len(slcs) < 3:
        raise ValueError("need >=3 slices for a Fengler surface")

    desc = []
    longer = None
    for slc in slcs:
        cal_upper = None
        cal_x = None
        if longer is not None:
            lo = max(float(np.min(np.exp(slc.k))), longer.knots[0])
            hi = min(float(np.max(np.exp(slc.k))), longer.knots[-1])
            if hi > lo:
                cal_x = np.linspace(lo, hi, max(int(calendar_grid_size), slc.n))
                cal_upper = longer.normalized_call
        fit = fit_fengler_slice(
            slc,
            smoothing_lambda=smoothing_lambda,
            calendar_upper=cal_upper,
            calendar_x=cal_x,
        )
        desc.append(fit)
        longer = fit

    fits = sorted(desc, key=lambda f: f.T)

    # Dense final calendar check over pairwise overlap.
    margins = []
    for a, b in zip(fits[:-1], fits[1:]):
        lo = max(a.knots[0], b.knots[0])
        hi = min(a.knots[-1], b.knots[-1])
        if hi <= lo:
            continue
        x = np.linspace(lo, hi, max(241, int(calendar_grid_size)))
        margins.append(float(np.min(b.normalized_call(x) - a.normalized_call(x))))
    cal_margin = min(margins) if margins else np.inf
    cal_ok = bool(cal_margin >= -3e-7 and all(f.calendar_free for f in fits))
    bfly_ok = bool(all(f.strike_arb_free for f in fits))
    success = bool(all(f.success for f in fits))

    nobs = sum(len(f.knots) for f in fits)
    rmse_iv = np.sqrt(
        sum(len(f.knots) * f.rmse_iv ** 2 for f in fits) / max(nobs, 1)
    )
    max_err = max(f.max_abs_err_iv for f in fits)

    return FenglerSurfaceFit(
        slices=fits,
        smoothing_lambda=float(smoothing_lambda),
        success=success,
        butterfly_free=bfly_ok,
        calendar_free=cal_ok,
        rmse_iv=float(rmse_iv),
        max_abs_err_iv=float(max_err),
        n_obs=int(nobs),
        calendar_margin_min=float(cal_margin),
    )


calibrate_fengler = fit_fengler_surface

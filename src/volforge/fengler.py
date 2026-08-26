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
from copy import copy
from time import perf_counter
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
    "prepare_fengler_slices",
    "calibrate_fengler",
]

_EPS = 1e-12


@dataclass
class _SolveResult:
    x: np.ndarray
    success: bool
    fun: float
    solver: str
    elapsed_seconds: float
    message: str = ""


def _calendar_grid(lo: float, hi: float, size: int) -> np.ndarray:
    """Return exactly ``size`` calendar-constraint points over an overlap.

    Historically VolForge silently expanded the requested calendar grid to at
    least the number of strikes in the slice. That made ``--calendar-grid 61``
    turn into 100--200+ constraints on dense SPY expiries. Keep the requested
    value as an actual cap instead.
    """
    n = max(int(size), 2)
    return np.linspace(float(lo), float(hi), n)


def _thin_slice(slc, max_strikes: int | None):
    """Return a shallow slice copy with representative strikes retained."""
    if max_strikes is None or int(max_strikes) <= 0 or slc.n <= int(max_strikes):
        return slc

    limit = max(int(max_strikes), 6)
    k = np.asarray(slc.k, float)
    order = np.argsort(k)
    pos = np.linspace(0, len(order) - 1, limit).round().astype(int)
    idx = set(order[pos].tolist())
    idx.add(int(np.nanargmin(np.abs(k))))
    idx.add(int(order[0]))
    idx.add(int(order[-1]))
    idx = np.array(sorted(idx, key=lambda i: k[i]), dtype=int)

    out = copy(slc)
    for name in ("k", "w", "iv", "strikes", "is_call", "half_spread", "weights"):
        if hasattr(slc, name):
            arr = np.asarray(getattr(slc, name))
            if arr.ndim and len(arr) == len(k):
                setattr(out, name, arr[idx].copy())
    try:
        setattr(out, "n", len(idx))
    except Exception:
        pass
    return out


def prepare_fengler_slices(
    slices: Iterable,
    *,
    target_days: float = 30.0,
    max_maturities: int = 5,
    max_strikes_per_slice: int | None = 60,
) -> list:
    """Select a compact local surface for fast MFIV confirmation.

    The selection brackets the target tenor where possible, then fills with the
    nearest maturities until ``max_maturities`` is reached. Each selected slice
    can optionally be thinned to representative strikes. Full research fits do
    not call this helper and therefore retain all maturities/strikes.
    """
    slcs = sorted(list(slices), key=lambda s: float(s.dte))
    if len(slcs) < 3:
        return slcs
    m = max(3, int(max_maturities))
    if len(slcs) <= m:
        chosen = slcs
    else:
        target = float(target_days)
        below = [s for s in slcs if float(s.dte) < target]
        exact = [s for s in slcs if np.isclose(float(s.dte), target, atol=1e-8)]
        above = [s for s in slcs if float(s.dte) > target]
        chosen = below[-2:] + exact[:1] + above[:2]
        seen = {id(s) for s in chosen}
        for s in sorted(slcs, key=lambda x: abs(float(x.dte) - target)):
            if len(chosen) >= m:
                break
            if id(s) not in seen:
                chosen.append(s)
                seen.add(id(s))
        chosen = sorted(chosen[:m], key=lambda s: float(s.dte))
    return [_thin_slice(s, max_strikes_per_slice) for s in chosen]


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


def _solve_reduced_slsqp(P, q, g0, bounds, constraints, *, maxiter: int, tol: float):
    """Fallback QP solve using SciPy on the reduced knot-value state."""
    start = perf_counter()

    def objective(g):
        return 0.5 * float(g @ P @ g) + float(q @ g)

    def jac(g):
        return P @ g + q

    res = minimize(
        objective,
        np.asarray(g0, float),
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": float(tol), "disp": False},
    )
    return _SolveResult(
        x=np.asarray(res.x, float),
        success=bool(res.success),
        fun=float(res.fun),
        solver="slsqp-reduced",
        elapsed_seconds=float(perf_counter() - start),
        message=str(res.message),
    )


def _solve_reduced_osqp(P, q, g0, bounds, constraints, *, maxiter: int, tol: float):
    """Solve the convex Fengler QP with OSQP when installed."""
    try:
        import osqp
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "OSQP is not installed. Install/reinstall VolForge (`pip install -e .`) "
            "or use solver='slsqp'."
        ) from exc

    n = len(g0)
    mats = [sparse.eye(n, format="csc")]
    lowers = [np.asarray(bounds.lb, float)]
    uppers = [np.asarray(bounds.ub, float)]
    for c in constraints:
        A = sparse.csc_matrix(c.A)
        rows = A.shape[0]
        lb = np.asarray(c.lb, float)
        ub = np.asarray(c.ub, float)
        if lb.ndim == 0:
            lb = np.full(rows, float(lb))
        if ub.ndim == 0:
            ub = np.full(rows, float(ub))
        mats.append(A)
        lowers.append(lb)
        uppers.append(ub)

    A = sparse.vstack(mats, format="csc")
    l = np.concatenate(lowers)
    u = np.concatenate(uppers)
    Psym = 0.5 * (np.asarray(P, float) + np.asarray(P, float).T)
    # OSQP reads the upper triangle of P.
    Psp = sparse.csc_matrix(np.triu(Psym))

    solver = osqp.OSQP()
    start = perf_counter()
    solver.setup(
        P=Psp,
        q=np.asarray(q, float),
        A=A,
        l=l,
        u=u,
        verbose=False,
        eps_abs=float(tol),
        eps_rel=float(tol),
        max_iter=int(maxiter),
        polish=True,
    )
    # Available in both old/new OSQP APIs and useful when the synthetic Black
    # starting curve is already close to the optimum.
    try:
        solver.warm_start(x=np.asarray(g0, float))
    except Exception:
        pass
    out = solver.solve()
    elapsed = float(perf_counter() - start)
    status = str(getattr(out.info, "status", "")).lower()
    success = status.startswith("solved") and out.x is not None
    if out.x is None:
        x = np.asarray(g0, float)
    else:
        x = np.asarray(out.x, float)
    fun = 0.5 * float(x @ Psym @ x) + float(np.asarray(q, float) @ x)
    return _SolveResult(
        x=x,
        success=bool(success),
        fun=fun,
        solver="osqp",
        elapsed_seconds=elapsed,
        message=str(getattr(out.info, "status", "")),
    )


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
    solver: str = "unknown"
    solve_time: float = np.nan
    solver_message: str = ""

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
            "solver": self.solver,
            "solve_time": float(self.solve_time),
            "solver_message": self.solver_message,
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
    solver: str = "auto",
    solver_tol: float = 1e-9,
) -> FenglerSliceFit:
    """Fit one constrained natural cubic spline in forward-moneyness space.

    The spline identity is eliminated analytically, so the optimizer works on
    the ``n`` knot values instead of ``2n-2`` values plus equality constraints.
    ``solver='auto'`` uses OSQP when installed and otherwise falls back to the
    reduced SLSQP problem.  The mathematical objective and arbitrage constraints
    are the same in both paths.
    """
    x = np.exp(np.asarray(slc.k, float))
    y = np.asarray(black_price(1.0, x, slc.iv, slc.T, True, 1.0), float)
    hs = np.asarray(slc.half_spread, float) / max(slc.forward_fit.discount * slc.forward, 1e-12)
    x, y, hs = _dedupe_xy(x, y, hs)
    if len(x) < 6:
        raise ValueError(f"Fengler slice needs >=6 distinct strikes, got {len(x)}")

    n = len(x)
    Q, R = spline_qr_matrices(x)
    w = 1.0 / np.maximum(
        hs,
        np.nanmedian(hs[hs > 0]) * 0.05 if np.any(hs > 0) else 1e-5,
    ) ** 2
    w = w / np.mean(w)
    w = np.minimum(w, float(weight_cap))

    # Eliminate the natural-spline equality Q.T@g = R@gamma exactly.
    # B maps knot values to interior second derivatives.
    try:
        B = np.linalg.solve(R, Q.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Fengler spline system is singular") from exc
    state_map = np.vstack([np.eye(n), B])

    constraints = []

    # Convexity: all interior second derivatives are non-negative.
    constraints.append(LinearConstraint(B, np.zeros(n - 2), np.inf))

    # Monotonicity at the knots.
    M = np.zeros((n - 1, n), float)
    for i in range(n - 1):
        M[i, i] = -1.0
        M[i, i + 1] = 1.0
    constraints.append(LinearConstraint(M, -np.inf, np.zeros(n - 1)))

    # Endpoint derivative bounds in normalized coordinates: -1 <= c_x <= 0.
    Dfull = np.zeros((2, 2 * n - 2), float)
    h0 = x[1] - x[0]
    Dfull[0, 0] = -1.0 / h0
    Dfull[0, 1] = 1.0 / h0
    Dfull[0, n] = -h0 / 6.0
    hn = x[-1] - x[-2]
    Dfull[1, n - 2] = -1.0 / hn
    Dfull[1, n - 1] = 1.0 / hn
    Dfull[1, n + n - 3] = hn / 6.0
    D = Dfull @ state_map
    constraints.append(
        LinearConstraint(D, np.array([-1.0, -np.inf]), np.array([np.inf, 0.0]))
    )

    cal_margin_target = np.inf
    if calendar_upper is not None:
        if calendar_x is None:
            calendar_x = _calendar_grid(x[0], x[-1], 121)
        cx = np.asarray(calendar_x, float)
        cx = cx[(cx >= x[0]) & (cx <= x[-1])]
        if len(cx):
            upper = np.asarray(calendar_upper(cx), float)
            lower = np.maximum(1.0 - cx, 0.0)
            upper_safe = np.maximum(upper, lower + 1e-10)
            cal_margin_target = float(np.min(upper_safe - lower))
            E = natural_spline_basis(x, cx) @ state_map
            constraints.append(LinearConstraint(E, -np.inf, upper_safe))

    lower_g = np.maximum(1.0 - x, 0.0)
    upper_g = np.ones(n)
    bounds = Bounds(lower_g, upper_g)

    lam = float(smoothing_lambda)
    if lam < 0:
        raise ValueError("smoothing_lambda must be >=0")

    # Reduced convex-QP objective: 1/2 g'Pg + q'g (+ irrelevant constant).
    P = np.diag(w) + lam * (B.T @ R @ B)
    P = 0.5 * (P + P.T)
    q = -(w * y)

    med_iv = float(np.nanmedian(slc.iv))
    g0 = np.asarray(black_price(1.0, x, med_iv, slc.T, True, 1.0), float)
    g0 = np.clip(g0, lower_g + 1e-10, upper_g - 1e-10)

    solver_name = str(solver).strip().lower()
    if solver_name not in {"auto", "osqp", "slsqp"}:
        raise ValueError("solver must be 'auto', 'osqp', or 'slsqp'")

    if solver_name in {"auto", "osqp"}:
        try:
            res = _solve_reduced_osqp(
                P, q, g0, bounds, constraints,
                maxiter=maxiter, tol=solver_tol,
            )
        except ImportError:
            if solver_name == "osqp":
                raise
            res = _solve_reduced_slsqp(
                P, q, g0, bounds, constraints,
                maxiter=maxiter, tol=solver_tol,
            )
    else:
        res = _solve_reduced_slsqp(
            P, q, g0, bounds, constraints,
            maxiter=maxiter, tol=solver_tol,
        )

    g = np.asarray(res.x, float)
    gamma_inner = B @ g
    gamma_full = np.zeros(n, float)
    gamma_full[1:-1] = gamma_inner
    z = np.concatenate([g, gamma_inner])

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

    objective = 0.5 * float(
        np.sum(w * price_err * price_err) + lam * gamma_inner @ R @ gamma_inner
    )

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
        success=bool(res.success or (np.isfinite(objective) and strike_ok and calendar_ok)),
        objective=float(objective),
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
        solver=res.solver,
        solve_time=res.elapsed_seconds,
        solver_message=res.message,
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
    solver: str = "unknown"
    elapsed_seconds: float = np.nan

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
            "solver": self.solver,
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def fit_fengler_surface(
    slices: Iterable,
    smoothing_lambda: float = 1e-5,
    calendar_grid_size: int = 181,
    solver: str = "auto",
    solver_tol: float = 1e-9,
    maxiter: int = 2500,
    progress=None,
) -> FenglerSurfaceFit:
    """Fit a Fengler arbitrage-free price surface, longest expiry backwards."""
    started = perf_counter()
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
                cal_x = _calendar_grid(lo, hi, int(calendar_grid_size))
                cal_upper = longer.normalized_call
        if progress:
            progress({
                "event": "start_slice",
                "dte": float(slc.dte),
                "n": int(slc.n),
                "calendar_points": 0 if cal_x is None else int(len(cal_x)),
            })
        fit = fit_fengler_slice(
            slc,
            smoothing_lambda=smoothing_lambda,
            calendar_upper=cal_upper,
            calendar_x=cal_x,
            maxiter=maxiter,
            solver=solver,
            solver_tol=solver_tol,
        )
        desc.append(fit)
        longer = fit
        if progress:
            progress({
                "event": "end_slice",
                "dte": float(fit.dte),
                "n": int(len(fit.knots)),
                "solver": fit.solver,
                "seconds": float(fit.solve_time),
                "success": bool(fit.success),
            })

    fits = sorted(desc, key=lambda f: f.T)

    # Dense final calendar check over pairwise overlap.
    margins = []
    for a, b in zip(fits[:-1], fits[1:]):
        lo = max(a.knots[0], b.knots[0])
        hi = min(a.knots[-1], b.knots[-1])
        if hi <= lo:
            continue
        x = _calendar_grid(lo, hi, max(241, int(calendar_grid_size)))
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
        solver=(fits[0].solver if fits and all(f.solver == fits[0].solver for f in fits) else "mixed"),
        elapsed_seconds=float(perf_counter() - started),
    )


calibrate_fengler = fit_fengler_surface

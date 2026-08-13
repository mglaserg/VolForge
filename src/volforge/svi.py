"""Raw SVI: evaluation, quasi-explicit calibration, arbitrage diagnostics.

Raw parameterisation (Gatheral), in total implied variance:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

Calibration follows the quasi-explicit scheme of Zeliade / De Marco & Martini:
substituting y = (k - m)/sigma turns the model into

    w = a + d*y + c*sqrt(y^2 + 1),     c = b*sigma,  d = rho*b*sigma

which is *linear* in (a, d, c). So the 5-parameter non-convex problem reduces
to a 2-dimensional search over (m, sigma) with a small convex QP solved
exactly at each step. This matters more than it sounds: naive 5-parameter
least squares produces day-to-day parameter jitter that will swamp any
time-series signal you try to extract later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations as _combinations

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "SVIParams",
    "svi_total_variance",
    "svi_iv",
    "svi_derivatives",
    "durrleman_g",
    "wing_slopes",
    "is_butterfly_free",
    "check_calendar_arbitrage",
    "SVIFit",
    "calibrate_svi",
]

# Lee's moment formula caps the asymptotic slope of total implied variance
# at 2. b*(1+|rho|) <= 2 is the corresponding SVI constraint.
MAX_WING_SLOPE = 2.0

# Smallest total variance we consider a legitimate fit rather than a degenerate one.
_W_FLOOR = 1e-8


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    def as_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        if self.b < 0:
            raise ValueError("b must be >= 0")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")


# --------------------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------------------

def svi_total_variance(k, p: SVIParams):
    """w(k), total implied variance."""
    k = np.asarray(k, dtype=float)
    km = k - p.m
    return p.a + p.b * (p.rho * km + np.sqrt(km * km + p.sigma * p.sigma))


def svi_iv(k, p: SVIParams, T: float):
    """Implied volatility from the SVI slice at maturity T."""
    w = svi_total_variance(k, p)
    return np.sqrt(np.maximum(w, 0.0) / T)


def svi_derivatives(k, p: SVIParams):
    """(w, w', w'') with respect to k."""
    k = np.asarray(k, dtype=float)
    km = k - p.m
    root = np.sqrt(km * km + p.sigma * p.sigma)
    w = p.a + p.b * (p.rho * km + root)
    w1 = p.b * (p.rho + km / root)
    w2 = p.b * p.sigma ** 2 / root ** 3
    return w, w1, w2


# --------------------------------------------------------------------------
# Arbitrage diagnostics
# --------------------------------------------------------------------------

def wing_slopes(p: SVIParams) -> tuple[float, float]:
    """Asymptotic slopes of w(k) as k -> -inf and +inf. Both must be < 2."""
    return p.b * (1.0 - p.rho), p.b * (1.0 + p.rho)


def durrleman_g(k, p: SVIParams):
    """Durrleman's function. g(k) >= 0 for all k <=> no butterfly arbitrage.

    g = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    """
    w, w1, w2 = svi_derivatives(k, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (1.0 - np.asarray(k) * w1 / (2.0 * w)) ** 2
        term2 = (w1 ** 2 / 4.0) * (1.0 / w + 0.25)
    return term1 - term2 + w2 / 2.0


def is_butterfly_free(p: SVIParams, k_range=(-1.5, 1.5), n=801, tol=-1e-8):
    """Scan Durrleman's g on a grid. Returns (ok, min_g, argmin_k)."""
    k = np.linspace(*k_range, n)
    g = durrleman_g(k, p)
    g = np.where(np.isfinite(g), g, np.inf)
    i = int(np.argmin(g))
    return bool(g[i] >= tol), float(g[i]), float(k[i])


def check_calendar_arbitrage(slices, k_range=(-1.5, 1.5), n=401, tol=-1e-10):
    """`slices` is a list of (T, SVIParams) sorted by T.

    Total variance must be non-decreasing in T at every fixed k. Returns a list
    of (T_short, T_long, worst_violation_k, worst_gap) for offending pairs.
    """
    k = np.linspace(*k_range, n)
    slices = sorted(slices, key=lambda s: s[0])
    violations = []
    for (t0, p0), (t1, p1) in zip(slices, slices[1:]):
        gap = svi_total_variance(k, p1) - svi_total_variance(k, p0)
        i = int(np.argmin(gap))
        if gap[i] < tol:
            violations.append((t0, t1, float(k[i]), float(gap[i])))
    return violations


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

@dataclass
class SVIFit:
    params: SVIParams
    rmse: float           # weighted RMSE in total-variance units
    rmse_iv: float        # unweighted RMSE in vol points
    max_abs_err_iv: float
    n_obs: int
    success: bool
    butterfly_free: bool
    min_durrleman_g: float
    wing_slopes: tuple[float, float]
    n_restarts: int
    boundary_flags: tuple[str, ...] = ()

    @property
    def is_reliable(self) -> bool:
        """A fit safe to feed into time-series work.

        Boundary-pinned parameters are the trap here: rho = -1 is not a
        measurement of skew, it is the optimiser running out of room, usually
        because too few quotes survived. Such a fit still plots fine and still
        reports a low RMSE, but it carries no information and will inject
        spurious variance into any PCA built on top of it.
        """
        return (
            self.success
            and self.butterfly_free
            and not self.boundary_flags
            and self.n_obs >= 10
        )

    def as_row(self) -> dict:
        """Flat dict, ready to write to the svi_parameters table."""
        d = self.params.as_dict()
        d.update(
            rmse=self.rmse,
            rmse_iv=self.rmse_iv,
            max_abs_err_iv=self.max_abs_err_iv,
            n_obs=self.n_obs,
            butterfly_free=self.butterfly_free,
            min_durrleman_g=self.min_durrleman_g,
            slope_left=self.wing_slopes[0],
            slope_right=self.wing_slopes[1],
            success=self.success,
            boundary_flags=",".join(self.boundary_flags),
            is_reliable=self.is_reliable,
        )
        return d


def _boundary_flags(p: SVIParams, k, max_slope: float, a_upper: float,
                    allow_negative_a: bool = True) -> tuple[str, ...]:
    """Detect parameters sitting on a constraint. See SVIFit.is_reliable."""
    flags = []
    if abs(p.rho) > 0.995:
        flags.append("rho_pinned")
    if p.b * (1.0 + abs(p.rho)) > 0.995 * max_slope:
        flags.append("wing_slope_pinned")
    if p.b < 1e-6:
        flags.append("b_degenerate")
    if not allow_negative_a and abs(p.a) < 1e-9:
        flags.append("a_at_zero")
    if p.a > 0.995 * a_upper:
        flags.append("a_at_upper")
    if p.a + p.b * p.sigma * np.sqrt(max(1.0 - p.rho ** 2, 0.0)) < 10 * _W_FLOOR:
        flags.append("min_variance_pinned")
    # m outside the observed strike range means the smile minimum is being
    # extrapolated, not measured.
    if p.m < k.min() or p.m > k.max():
        flags.append("m_outside_data")
    return tuple(flags)


_CONSTRAINT_G = np.array([
    [0.0,  0.0,  1.0],   # 0: c >= 0
    [0.0,  0.0, -1.0],   # 1: c <= lim
    [0.0, -1.0,  1.0],   # 2: d <= c
    [0.0,  1.0,  1.0],   # 3: -d <= c
    [0.0, -1.0, -1.0],   # 4: d <= lim - c
    [0.0,  1.0, -1.0],   # 5: -d <= lim - c
    [1.0,  0.0,  0.0],   # 6: a >= a_lo
    [-1.0, 0.0,  0.0],   # 7: a <= a_upper
])

# Candidate active sets: every subset of up to 3 constraints whose normals are
# linearly independent. The geometry is fixed, so this is computed once at
# import rather than re-tested inside the optimiser loop -- np.linalg.matrix_rank
# is SVD-based and calling it thousands of times per fit dominated the runtime.
_ACTIVE_SETS = tuple(
    c for r in (1, 2, 3) for c in _combinations(range(8), r)
    if np.linalg.matrix_rank(_CONSTRAINT_G[list(c)]) == len(c)
)


def _constraint_system(sigma, max_slope, a_upper, allow_negative_a):
    """Feasible region as G @ theta >= h, with theta = (a, d, c).

    Rows encode, in order: c >= 0; c <= lim; |d| <= c (two rows);
    |d| <= lim - c (two rows); a >= 0; a <= a_upper. Together these give
    b >= 0, |rho| <= 1 and b*(1+|rho|) <= max_slope.
    """
    lim = max_slope * sigma
    h = np.array([0.0, -lim, 0.0, 0.0, -lim, -lim,
                  -a_upper if allow_negative_a else 0.0, -a_upper])
    return _CONSTRAINT_G, h


def _inner_problem(y, w, weights, sigma, max_slope, a_upper, allow_negative_a):
    """Exact solution of the convex sub-problem in (a, d, c) for fixed (m, sigma).

    Design matrix columns: [1, y, sqrt(y^2+1)], so the model is linear in the
    unknowns and the problem is a 3-variable inequality-constrained least
    squares -- a tiny QP with a unique solution.

    Solved by active-set enumeration rather than a general-purpose optimiser.
    With three unknowns at most three constraints can bind, so enumerating the
    candidate active sets and taking the feasible KKT point with the lowest
    objective is both exact and fast. This matters because the outer
    Nelder-Mead search calls this thousands of times per slice; the enumeration
    is roughly an order of magnitude quicker than an SLSQP call, which turns a
    multi-year calibration from hours into minutes.
    """
    X = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    Xw = X * weights[:, None]
    H = X.T @ Xw                      # 3x3 normal matrix
    g = Xw.T @ w                      # 3-vector
    wsq = float(np.dot(weights * w, w))

    def sse(theta):
        return float(theta @ H @ theta - 2.0 * g @ theta + wsq)

    G, h = _constraint_system(sigma, max_slope, a_upper, allow_negative_a)
    finite = np.isfinite(h)

    # Fast path: the unconstrained optimum is usually already feasible.
    try:
        theta = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(H, g, rcond=None)[0]
    if np.all(G[finite] @ theta >= h[finite] - 1e-12):
        return theta, sse(theta)

    idx = np.flatnonzero(finite)
    Gf, hf = G[idx], h[idx]

    # Feasible starting point: d = 0 puts us strictly inside every wing
    # constraint, so only a and c need clipping.
    lim = max_slope * sigma
    theta = np.array([
        float(np.clip(np.mean(w), 0.0 if not allow_negative_a else -abs(a_upper), a_upper)),
        0.0,
        0.5 * lim,
    ])

    working = [i for i in range(len(idx)) if abs(Gf[i] @ theta - hf[i]) < 1e-12]

    for _ in range(30):
        grad = H @ theta - g
        if working:
            A = Gf[working]
            n_a = len(working)
            K = np.zeros((3 + n_a, 3 + n_a))
            K[:3, :3] = H
            K[:3, 3:] = -A.T
            K[3:, :3] = A
            rhs = np.concatenate([-grad, np.zeros(n_a)])
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            p, lam = sol[:3], sol[3:]
        else:
            try:
                p = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                p = np.linalg.lstsq(H, -grad, rcond=None)[0]
            lam = np.array([])

        if np.linalg.norm(p) < 1e-14:
            if lam.size == 0 or np.all(lam >= -1e-12):
                break                             # KKT satisfied: optimal
            working.pop(int(np.argmin(lam)))      # drop the most negative
            continue

        # Step to the nearest blocking constraint not already in the set.
        alpha, blocker = 1.0, None
        for i in range(len(idx)):
            if i in working:
                continue
            denom = Gf[i] @ p
            if denom < -1e-14:
                step = (Gf[i] @ theta - hf[i]) / -denom
                if step < alpha:
                    alpha, blocker = step, i
        theta = theta + alpha * p
        if blocker is not None:
            working.append(blocker)

    return theta, sse(theta)


def calibrate_svi(
    k,
    w,
    T: float,
    weights=None,
    max_slope: float = MAX_WING_SLOPE,
    allow_negative_a: bool = True,
    n_grid_m: int = 11,
    n_grid_s: int = 11,
    n_refine: int = 3,
    x0=None,
) -> SVIFit:
    """Fit a raw-SVI slice to total variance observations.

    Parameters
    ----------
    k : log-moneyness log(K/F)
    w : observed total implied variance, iv^2 * T
    T : time to expiry in years (used only for reporting errors in vol points)
    weights : per-observation weights. Vega or inverse-spread weighting is
        strongly recommended -- unweighted fits let illiquid wings dominate.
    max_slope : wing-slope cap, b*(1+|rho|) <= max_slope. Lee's bound is 2.
    allow_negative_a : permit a < 0. Default True. The classical Zeliade
        domain imposes a >= 0 as a cheap way to keep total variance positive,
        but on short-dated slices the unconstrained optimum genuinely wants
        a < 0 and the constraint binds, producing a pinned parameter that is an
        artefact of the restriction rather than a feature of the market. We
        allow a < 0 and instead enforce positivity directly in the outer search
        via min w = a + sqrt(c^2 - d^2) > 0, which is exact and keeps the inner
        problem linear.
    n_grid_m, n_grid_s : coarse-grid resolution over (m, sigma).
    n_refine : how many of the best grid cells get local refinement.
    x0 : optional (m, sigma) warm start, typically the previous day's fit.
        Cheap, and it keeps consecutive fits in the same local basin, which
        reduces day-to-day parameter jitter in the time series.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[good], w[good]

    if weights is None:
        wt = np.ones_like(w)
    else:
        wt = np.asarray(weights, dtype=float)[good]
        wt = np.where(np.isfinite(wt) & (wt > 0), wt, 0.0)
    if wt.sum() <= 0:
        wt = np.ones_like(w)
    wt = wt / wt.mean()

    n = len(k)
    if n < 5:
        raise ValueError(f"need >=5 usable quotes to fit SVI, got {n}")

    a_upper = float(np.max(w))
    k_span = max(float(np.ptp(k)), 1e-3)

    def outer(z):
        m, log_sigma = z[0], z[1]
        sigma = float(np.exp(log_sigma))
        if not (1e-4 < sigma < 10.0) or abs(m) > 5.0:
            return 1e12
        y = (k - m) / sigma
        theta, sse = _inner_problem(y, w, wt, sigma, max_slope, a_upper,
                                    allow_negative_a)
        # Total variance must stay positive everywhere. Its minimum is
        # a + b*sigma*sqrt(1-rho^2) = a + sqrt(c^2 - d^2). Penalise smoothly
        # rather than rejecting, so the outer simplex keeps a usable gradient.
        av, dv, cv = theta
        min_w = av + np.sqrt(max(cv * cv - dv * dv, 0.0))
        if min_w < _W_FLOOR:
            sse += 1e6 * (_W_FLOOR - min_w) ** 2
        return sse

    # Coarse grid over (m, sigma), then local refinement from the best cells.
    # The inner problem is solved exactly, so the outer surface is smooth and
    # cheap to map; a systematic sweep finds the basin far more reliably than
    # random restarts and costs a fraction of the function evaluations.
    m_grid = np.linspace(k.min() - 0.25 * k_span, k.max() + 0.25 * k_span, n_grid_m)
    s_grid = np.exp(np.linspace(np.log(0.01), np.log(max(1.0, 2.0 * k_span)), n_grid_s))

    coarse = np.empty((len(m_grid), len(s_grid)))
    for i, mv in enumerate(m_grid):
        for j, sv in enumerate(s_grid):
            coarse[i, j] = outer(np.array([mv, np.log(sv)]))

    flat = np.argsort(coarse, axis=None)[:max(1, n_refine)]
    starts = [np.array([m_grid[i], np.log(s_grid[j])])
              for i, j in (np.unravel_index(f, coarse.shape) for f in flat)]
    if x0 is not None:                       # warm start, e.g. yesterday's fit
        starts.insert(0, np.array([float(x0[0]), np.log(float(x0[1]))]))

    best, best_sse = None, np.inf
    for z0 in starts:
        res = minimize(outer, z0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-8, "fatol": 1e-15})
        if res.fun < best_sse:
            best_sse, best = res.fun, res.x

    if best is None or not np.isfinite(best_sse):
        raise RuntimeError("SVI calibration failed to find any feasible fit")

    m = float(best[0])
    sigma = float(np.exp(best[1]))
    theta, sse = _inner_problem((k - m) / sigma, w, wt, sigma,
                                max_slope, a_upper, allow_negative_a)
    a, d, c = theta
    b = c / sigma
    rho = d / c if c > 1e-12 else 0.0
    rho = float(np.clip(rho, -0.999999, 0.999999))

    params = SVIParams(a=float(a), b=float(max(b, 0.0)), rho=rho, m=m, sigma=sigma)

    w_fit = svi_total_variance(k, params)
    iv_mkt = np.sqrt(np.maximum(w, 0) / T)
    iv_fit = np.sqrt(np.maximum(w_fit, 0) / T)
    err_iv = iv_fit - iv_mkt

    ok, min_g, _ = is_butterfly_free(params,
                                     k_range=(k.min() - 0.5, k.max() + 0.5))

    return SVIFit(
        params=params,
        rmse=float(np.sqrt(np.average((w_fit - w) ** 2, weights=wt))),
        rmse_iv=float(np.sqrt(np.mean(err_iv ** 2))),
        max_abs_err_iv=float(np.max(np.abs(err_iv))),
        n_obs=n,
        success=True,
        butterfly_free=ok,
        min_durrleman_g=min_g,
        wing_slopes=wing_slopes(params),
        n_restarts=len(starts),
        boundary_flags=_boundary_flags(params, k, max_slope, a_upper,
                                       allow_negative_a),
    )

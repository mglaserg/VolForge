"""Extended SSVI (eSSVI) with maturity-dependent correlation.

Hendriks & Martini extend SSVI by replacing the constant correlation ``rho``
with a function of ATM total variance, ``rho(theta)``.  This implementation
uses the power interpolation family from their paper,

    rho(theta) = rho0 + (rho_m - rho0) * (theta/theta_max)**a,

on the calibrated theta range and holds rho flat beyond ``theta_max``.  The
curvature function is the same power-law / modified-power-law family used by
VolForge's SSVI implementation.

The continuous-time no-calendar-arbitrage condition is checked through

    gamma(theta) = (1/phi) d(theta*phi)/dtheta
    delta(theta) = theta * d rho / dtheta

and, for the practical power-law families where 0 <= gamma <= 1,

    |delta + rho*gamma| <= gamma.

Each eSSVI slice is still an SSVI slice, so the standard SSVI sufficient
butterfly conditions are applied maturity by maturity as well.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .svi import SVIParams, durrleman_g, svi_total_variance
from .ssvi import SSVIParams, calibrate_ssvi, ssvi_phi, ssvi_phi_prime
from .term_structure import ATMVarianceCurve, build_atm_variance_curve

__all__ = [
    "ESSVIParams",
    "ESSVIFit",
    "essvi_phi",
    "essvi_rho",
    "essvi_rho_prime",
    "essvi_total_variance",
    "essvi_iv",
    "essvi_calendar_terms",
    "essvi_butterfly_conditions",
    "essvi_to_raw_svi",
    "is_essvi_calendar_free",
    "is_essvi_butterfly_free",
    "calibrate_essvi",
    "fit_essvi_surface",
]

_EPS = 1e-12
_RHO_MAX = 0.999


@dataclass(frozen=True)
class ESSVIParams:
    rho0: float
    rho_m: float
    a: float
    eta: float
    gamma: float
    theta_max: float
    phi_form: str = "modified_power_law"

    def __post_init__(self):
        if not -1.0 < self.rho0 < 1.0 or not -1.0 < self.rho_m < 1.0:
            raise ValueError("rho0 and rho_m must lie strictly inside (-1, 1)")
        if self.a < 0:
            raise ValueError("a must be >= 0")
        if self.eta <= 0:
            raise ValueError("eta must be > 0")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        if self.theta_max <= 0:
            raise ValueError("theta_max must be > 0")
        if self.phi_form not in {"modified_power_law", "power_law"}:
            raise ValueError("phi_form must be 'modified_power_law' or 'power_law'")

    def as_dict(self) -> dict:
        return asdict(self)


def _ssvi_params(p: ESSVIParams, rho: float = 0.0) -> SSVIParams:
    return SSVIParams(rho=float(rho), eta=p.eta, gamma=p.gamma, phi_form=p.phi_form)


def essvi_phi(theta, p: ESSVIParams):
    return ssvi_phi(theta, _ssvi_params(p))


def _essvi_phi_prime(theta, p: ESSVIParams):
    return ssvi_phi_prime(theta, _ssvi_params(p))


def essvi_rho(theta, p: ESSVIParams):
    """Maturity-dependent correlation on the calibrated theta clock.

    The Hendriks-Martini power interpolation is used up to theta_max.  Beyond
    the liquid calibration horizon rho is held at rho_m rather than allowing
    the power law to run outside the range for which its calendar conditions
    were designed.
    """
    th = np.asarray(theta, dtype=float)
    x = np.clip(np.maximum(th, 0.0) / p.theta_max, 0.0, 1.0)
    if p.a <= 1e-12:
        # a=0 is interpreted as the constant limiting family.  Using rho_m
        # makes rho0/rho_m equality the natural SSVI nesting point.
        return np.full_like(x, p.rho_m, dtype=float)
    return p.rho0 + (p.rho_m - p.rho0) * x ** p.a


def essvi_rho_prime(theta, p: ESSVIParams):
    th = np.asarray(theta, dtype=float)
    out = np.zeros_like(th, dtype=float)
    if p.a <= 1e-12 or abs(p.rho_m - p.rho0) <= 1e-14:
        return out
    inside = (th > 0.0) & (th < p.theta_max)
    x = np.maximum(th[inside] / p.theta_max, _EPS)
    out[inside] = (
        (p.rho_m - p.rho0)
        * p.a
        * x ** (p.a - 1.0)
        / p.theta_max
    )
    return out


def essvi_total_variance(k, theta, p: ESSVIParams):
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = essvi_phi(theta, p)
    rho = essvi_rho(theta, p)
    u = phi * k + rho
    return 0.5 * theta * (
        1.0 + rho * phi * k + np.sqrt(u * u + 1.0 - rho * rho)
    )


def essvi_iv(k, theta, T, p: ESSVIParams):
    w = essvi_total_variance(k, theta, p)
    return np.sqrt(np.maximum(w, 0.0) / np.asarray(T, dtype=float))


def essvi_to_raw_svi(theta: float, p: ESSVIParams) -> SVIParams:
    phi = float(essvi_phi(theta, p))
    rho = float(essvi_rho(theta, p))
    root = np.sqrt(max(1.0 - rho * rho, 0.0))
    return SVIParams(
        a=0.5 * theta * (1.0 - rho * rho),
        b=0.5 * theta * phi,
        rho=rho,
        m=-rho / phi,
        sigma=root / phi,
    )


def essvi_calendar_terms(theta, p: ESSVIParams):
    """Return (gamma, delta, lhs, margin) for the eSSVI calendar theorem.

    For the two power-law phi families implemented here, gamma lies in [0,1]
    over theta>0, so no calendar spread arbitrage is equivalent to

        abs(delta + rho*gamma) <= gamma.
    """
    th = np.asarray(theta, dtype=float)
    phi = essvi_phi(th, p)
    gamma = 1.0 + th * _essvi_phi_prime(th, p) / np.maximum(phi, _EPS)
    rho = essvi_rho(th, p)
    delta = th * essvi_rho_prime(th, p)
    lhs = np.abs(delta + rho * gamma)
    margin = gamma - lhs
    return gamma, delta, lhs, margin


def essvi_butterfly_conditions(theta, p: ESSVIParams):
    th = np.asarray(theta, dtype=float)
    phi = essvi_phi(th, p)
    rho = np.abs(essvi_rho(th, p))
    c1 = th * phi * (1.0 + rho)
    c2 = th * phi * phi * (1.0 + rho)
    return c1, c2


def is_essvi_calendar_free(theta_values, p: ESSVIParams, n_dense: int = 801):
    theta_values = np.asarray(theta_values, dtype=float)
    lo = max(float(np.min(theta_values)), 1e-10)
    hi = float(np.max(theta_values))
    grid = np.geomspace(lo, hi, max(16, int(n_dense))) if hi > lo else np.array([lo])
    gamma, delta, lhs, margin = essvi_calendar_terms(grid, p)
    ok = bool(
        np.all(gamma >= -1e-10)
        and np.all(gamma <= 1.0 + 1e-8)
        and np.all(margin >= -1e-8)
    )
    return ok, float(np.min(margin)), float(np.max(lhs)), float(np.min(gamma)), float(np.max(gamma))


def is_essvi_butterfly_free(theta_values, p: ESSVIParams, numerical=True,
                             k_range=(-1.5, 1.5), n=801):
    theta = np.asarray(theta_values, dtype=float)
    c1, c2 = essvi_butterfly_conditions(theta, p)
    sufficient = bool(np.all(c1 < 4.0 - 1e-10) and np.all(c2 <= 4.0 + 1e-10))
    min_g = np.inf
    if numerical:
        kk = np.linspace(*k_range, n)
        for th in theta:
            g = durrleman_g(kk, essvi_to_raw_svi(float(th), p))
            finite = g[np.isfinite(g)]
            if len(finite):
                min_g = min(min_g, float(finite.min()))
        numerical_ok = min_g >= -1e-8
    else:
        numerical_ok = True
    return sufficient and numerical_ok, float(np.max(c1)), float(np.max(c2)), float(min_g)


@dataclass
class ESSVIFit:
    params: ESSVIParams
    theta_curve: ATMVarianceCurve
    rmse: float
    rmse_iv: float
    max_abs_err_iv: float
    n_obs: int
    n_slices: int
    n_theta_slices: int
    success: bool
    butterfly_free: bool
    calendar_free: bool
    min_durrleman_g: float
    max_bfly_condition1: float
    max_bfly_condition2: float
    calendar_margin_min: float
    calendar_lhs_max: float
    calendar_gamma_min: float
    calendar_gamma_max: float
    objective: float
    slice_rmse_iv: dict[float, float]
    T_obs: np.ndarray
    forwards: np.ndarray
    spot: float

    @property
    def is_reliable(self) -> bool:
        return self.success and self.butterfly_free and self.calendar_free and self.n_slices >= 3

    @property
    def theta_repair(self) -> float:
        return self.theta_curve.repair_amount

    @property
    def theta_repair_fraction(self) -> float:
        return self.theta_curve.repair_fraction

    def theta(self, T):
        return self.theta_curve(T)

    def rho(self, T):
        return essvi_rho(self.theta(T), self.params)

    def total_variance(self, k, T):
        return essvi_total_variance(k, self.theta(T), self.params)

    def implied_vol(self, k, T):
        return essvi_iv(k, self.theta(T), T, self.params)

    def raw_slice(self, T):
        return essvi_to_raw_svi(float(self.theta(T)), self.params)

    def to_surface(self, trade_date, symbol="", tenor_days=None, k_grid=None):
        from .surface import Surface, DEFAULT_K_GRID, DEFAULT_TENORS, DAYS_PER_YEAR

        tenor_days = np.asarray(DEFAULT_TENORS if tenor_days is None else tenor_days, float)
        k_grid = np.asarray(DEFAULT_K_GRID if k_grid is None else k_grid, float)
        T = tenor_days / DAYS_PER_YEAR
        W = np.vstack([self.total_variance(k_grid, t) for t in T])
        iv = np.sqrt(np.maximum(W, 0.0) / T[:, None])
        extrap = (T < self.T_obs.min()) | (T > self.T_obs.max())
        F = np.interp(T, self.T_obs, self.forwards)
        return Surface(
            trade_date=__import__("pandas").Timestamp(trade_date),
            symbol=symbol,
            tenor_days=tenor_days,
            k_grid=k_grid,
            total_var=W,
            iv=iv,
            extrapolated=extrap,
            spot=float(self.spot),
            forwards=F,
            n_slices_used=int(self.n_slices),
            calendar_repair=0.0,
            node_index=[(float(t), float(k)) for t in tenor_days for k in k_grid],
        )

    def as_row(self) -> dict:
        d = self.params.as_dict()
        d.update(
            rmse=self.rmse,
            rmse_iv=self.rmse_iv,
            max_abs_err_iv=self.max_abs_err_iv,
            n_obs=self.n_obs,
            n_slices=self.n_slices,
            n_theta_slices=self.n_theta_slices,
            success=self.success,
            butterfly_free=self.butterfly_free,
            calendar_free=self.calendar_free,
            min_durrleman_g=self.min_durrleman_g,
            max_bfly_condition1=self.max_bfly_condition1,
            max_bfly_condition2=self.max_bfly_condition2,
            calendar_margin_min=self.calendar_margin_min,
            calendar_lhs_max=self.calendar_lhs_max,
            calendar_gamma_min=self.calendar_gamma_min,
            calendar_gamma_max=self.calendar_gamma_max,
            theta_repair=self.theta_repair,
            theta_repair_fraction=self.theta_repair_fraction,
        )
        return d


def _build_theta_and_observations(
    pairs: list,
    reliable_only: bool,
    include_unreliable_observations: bool,
    repair_theta: bool,
):
    theta_pairs = [(s, f) for s, f in pairs if (f.is_reliable if reliable_only else f.success)]
    if len(theta_pairs) < 3:
        raise ValueError(f"need >=3 usable SVI slices for eSSVI, got {len(theta_pairs)}")
    theta_pairs.sort(key=lambda sf: sf[0].T)
    T_theta = np.array([float(s.T) for s, _ in theta_pairs])
    theta_raw = np.array([float(svi_total_variance(0.0, f.params)) for _, f in theta_pairs])
    theta_weights = np.sqrt(np.array([max(f.n_obs, 1) for _, f in theta_pairs], dtype=float))
    curve = build_atm_variance_curve(T_theta, theta_raw, theta_weights, repair=repair_theta)

    if include_unreliable_observations:
        lo, hi = curve.t_years[0], curve.t_years[-1]
        fit_pairs = [(s, f) for s, f in pairs
                     if f.success and s.n >= 8 and lo - 1e-12 <= s.T <= hi + 1e-12]
    else:
        fit_pairs = list(theta_pairs)
    fit_pairs.sort(key=lambda sf: sf[0].T)
    return theta_pairs, fit_pairs, curve


def calibrate_essvi(
    slices_and_fits: Iterable,
    reliable_only: bool = True,
    include_unreliable_observations: bool = True,
    repair_theta: bool = True,
    phi_form: str = "modified_power_law",
    n_restarts: int = 7,
) -> ESSVIFit:
    """Calibrate Hendriks-Martini eSSVI to one day's option slices.

    This is the global parametric eSSVI construction from the original paper,
    not the later sequential robust-slice algorithm of Corbetta et al.  That
    algorithm can therefore be added later as a separate calibration backend
    without changing the public surface interface.
    """
    pairs = list(slices_and_fits)
    theta_pairs, fit_pairs, curve = _build_theta_and_observations(
        pairs, reliable_only, include_unreliable_observations, repair_theta
    )

    T_obs = np.array([float(s.T) for s, _ in fit_pairs])
    F_obs = np.array([float(s.forward) for s, _ in fit_pairs])
    ks = np.concatenate([np.asarray(s.k, float) for s, _ in fit_pairs])
    Ts = np.concatenate([np.full(s.n, float(s.T)) for s, _ in fit_pairs])
    ws = np.concatenate([np.asarray(s.w, float) for s, _ in fit_pairs])
    weights = np.concatenate([np.asarray(s.weights, float) for s, _ in fit_pairs])
    weights = weights / np.mean(weights)
    theta_obs = curve(Ts)
    theta_max = float(np.max(curve.theta))

    # A plain SSVI fit is a guaranteed feasible nested starting point when
    # rho0 == rho_m.  Starting there makes eSSVI an extension rather than a
    # completely independent optimizer lottery.
    ssvi0 = calibrate_ssvi(
        pairs,
        reliable_only=reliable_only,
        include_unreliable_observations=include_unreliable_observations,
        repair_theta=repair_theta,
        phi_form=phi_form,
        n_restarts=5,
    )

    def make_params(x):
        rho0, rho_m, a, eta, gamma = map(float, x)
        return ESSVIParams(
            rho0=np.clip(rho0, -_RHO_MAX, _RHO_MAX),
            rho_m=np.clip(rho_m, -_RHO_MAX, _RHO_MAX),
            a=max(a, 0.0),
            eta=max(eta, 1e-8),
            gamma=np.clip(gamma, 1e-3, 0.999),
            theta_max=theta_max,
            phi_form=phi_form,
        )

    def loss(x):
        p = make_params(x)
        pred = essvi_total_variance(ks, theta_obs, p)
        err = pred - ws
        return float(np.mean(weights * err * err))

    theta_dense = np.geomspace(max(float(np.min(curve.theta)), 1e-10), theta_max, 121)

    def cal_constraint(x):
        p = make_params(x)
        _, _, _, margin = essvi_calendar_terms(theta_dense, p)
        return margin

    def bfly1_constraint(x):
        p = make_params(x)
        c1, _ = essvi_butterfly_conditions(theta_dense, p)
        return 4.0 - c1

    def bfly2_constraint(x):
        p = make_params(x)
        _, c2 = essvi_butterfly_conditions(theta_dense, p)
        return 4.0 - c2

    rho = float(ssvi0.params.rho)
    base = np.array([rho, rho, 0.5, ssvi0.params.eta, ssvi0.params.gamma])
    bounds = [
        (-0.995, 0.995),
        (-0.995, 0.995),
        (0.0, 8.0),
        (1e-5, 4.0),
        (0.01, 0.99),
    ]
    constraints = [
        {"type": "ineq", "fun": cal_constraint},
        {"type": "ineq", "fun": bfly1_constraint},
        {"type": "ineq", "fun": bfly2_constraint},
    ]

    starts = [base]
    # Nearby departures from constant rho; the first start is always the nested
    # SSVI solution and therefore already satisfies calendar constraints.
    for dr, a0 in [(-0.08, 0.35), (0.08, 0.35), (-0.15, 0.8), (0.15, 0.8),
                   (-0.05, 1.5), (0.05, 1.5)]:
        starts.append(np.array([
            np.clip(rho - dr / 2, -0.95, 0.95),
            np.clip(rho + dr / 2, -0.95, 0.95),
            a0,
            ssvi0.params.eta,
            ssvi0.params.gamma,
        ]))
    starts = starts[:max(1, int(n_restarts))]

    best = None
    for x0 in starts:
        res = minimize(
            loss,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1600, "ftol": 1e-13, "disp": False},
        )
        feasible = (
            np.min(cal_constraint(res.x)) >= -2e-7
            and np.min(bfly1_constraint(res.x)) >= -2e-7
            and np.min(bfly2_constraint(res.x)) >= -2e-7
        )
        if feasible and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        # The nested SSVI point is still a valid eSSVI surface.  Return it rather
        # than producing a non-arbitrage-free optimizer artifact.
        best = minimize(
            loss,
            base,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1600, "ftol": 1e-13, "disp": False},
        )

    p = make_params(best.x)
    pred_w = essvi_total_variance(ks, theta_obs, p)
    pred_iv = np.sqrt(np.maximum(pred_w, 0.0) / Ts)
    market_iv = np.sqrt(np.maximum(ws, 0.0) / Ts)
    iv_err = pred_iv - market_iv

    bfly_ok, c1max, c2max, min_g = is_essvi_butterfly_free(curve.theta, p)
    cal_ok, cal_margin, lhs_max, gmin, gmax = is_essvi_calendar_free(curve.theta, p)
    cal_ok = cal_ok and curve.is_monotone

    per_slice = {}
    for s, _ in fit_pairs:
        th = float(curve(s.T))
        piv = essvi_iv(s.k, th, s.T, p)
        per_slice[float(s.dte)] = float(np.sqrt(np.mean((piv - s.iv) ** 2)))

    return ESSVIFit(
        params=p,
        theta_curve=curve,
        rmse=float(np.sqrt(np.mean(weights * (pred_w - ws) ** 2))),
        rmse_iv=float(np.sqrt(np.mean(iv_err * iv_err))),
        max_abs_err_iv=float(np.max(np.abs(iv_err))),
        n_obs=int(len(ws)),
        n_slices=int(len(fit_pairs)),
        n_theta_slices=int(len(theta_pairs)),
        success=bool(np.isfinite(best.fun)),
        butterfly_free=bfly_ok,
        calendar_free=cal_ok,
        min_durrleman_g=min_g,
        max_bfly_condition1=c1max,
        max_bfly_condition2=c2max,
        calendar_margin_min=cal_margin,
        calendar_lhs_max=lhs_max,
        calendar_gamma_min=gmin,
        calendar_gamma_max=gmax,
        objective=float(best.fun),
        slice_rmse_iv=per_slice,
        T_obs=T_obs,
        forwards=F_obs,
        spot=float(fit_pairs[0][0].spot),
    )


fit_essvi_surface = calibrate_essvi

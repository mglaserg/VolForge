"""Surface SVI (SSVI) with an explicit monotone ATM-variance clock.

The implementation follows Gatheral & Jacquier's SSVI representation

    w(k, theta) = theta/2 * [1 + rho*phi(theta)*k
                    + sqrt((phi(theta)*k + rho)^2 + 1-rho^2)]

and defaults to their modified power-law mixing function

    phi(theta) = eta / [theta^gamma (1+theta)^(1-gamma)]

with 0 < gamma < 1.  Constraining eta*(1+|rho|) <= 2 gives a particularly
convenient globally admissible family.  theta(T) is estimated from reliable
raw-SVI slices and projected onto a non-decreasing term structure before the
three global shape parameters (rho, eta, gamma) are calibrated jointly.

Raw SVI remains useful: it is the flexible per-expiry benchmark.  SSVI is the
coupled no-calendar-arbitrage benchmark against which those slices can be
compared.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .svi import SVIParams, durrleman_g, svi_total_variance
from .term_structure import ATMVarianceCurve, build_atm_variance_curve

__all__ = [
    "SSVIParams",
    "SSVIFit",
    "ssvi_phi",
    "ssvi_phi_prime",
    "ssvi_total_variance",
    "ssvi_iv",
    "ssvi_to_raw_svi",
    "ssvi_calendar_ratio",
    "ssvi_butterfly_conditions",
    "is_ssvi_butterfly_free",
    "calibrate_ssvi",
    "fit_ssvi_surface",
]

_EPS = 1e-12
_GAMMA_LO = 1e-3
_GAMMA_HI = 1.0 - 1e-3
_RHO_MAX = 0.999


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float
    phi_form: str = "modified_power_law"

    def __post_init__(self):
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly inside (-1, 1)")
        if self.eta <= 0:
            raise ValueError("eta must be > 0")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        if self.phi_form not in {"modified_power_law", "power_law"}:
            raise ValueError("phi_form must be 'modified_power_law' or 'power_law'")

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- model

def ssvi_phi(theta, p: SSVIParams):
    theta = np.asarray(theta, dtype=float)
    th = np.maximum(theta, _EPS)
    if p.phi_form == "power_law":
        return p.eta * th ** (-p.gamma)
    return p.eta / (th ** p.gamma * (1.0 + th) ** (1.0 - p.gamma))


def ssvi_phi_prime(theta, p: SSVIParams):
    theta = np.asarray(theta, dtype=float)
    th = np.maximum(theta, _EPS)
    phi = ssvi_phi(th, p)
    if p.phi_form == "power_law":
        return -p.gamma * phi / th
    return -phi * (p.gamma / th + (1.0 - p.gamma) / (1.0 + th))


def ssvi_total_variance(k, theta, p: SSVIParams):
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = ssvi_phi(theta, p)
    u = phi * k + p.rho
    return 0.5 * theta * (
        1.0 + p.rho * phi * k + np.sqrt(u * u + 1.0 - p.rho * p.rho)
    )


def ssvi_iv(k, theta, T, p: SSVIParams):
    w = ssvi_total_variance(k, theta, p)
    return np.sqrt(np.maximum(w, 0.0) / np.asarray(T, dtype=float))


def ssvi_to_raw_svi(theta: float, p: SSVIParams) -> SVIParams:
    """Equivalent raw-SVI slice for one theta; useful for diagnostics."""
    phi = float(ssvi_phi(theta, p))
    root = np.sqrt(max(1.0 - p.rho * p.rho, 0.0))
    return SVIParams(
        a=0.5 * theta * (1.0 - p.rho * p.rho),
        b=0.5 * theta * phi,
        rho=p.rho,
        m=-p.rho / phi,
        sigma=root / phi,
    )


# ----------------------------------------------------------- no-arbitrage

def ssvi_calendar_ratio(theta, p: SSVIParams):
    """[d/dtheta(theta*phi)] / phi, the quantity in the calendar theorem."""
    theta = np.asarray(theta, dtype=float)
    phi = ssvi_phi(theta, p)
    return 1.0 + theta * ssvi_phi_prime(theta, p) / phi


def _calendar_upper_bound(rho: float) -> float:
    if abs(rho) < 1e-12:
        return np.inf
    return (1.0 + np.sqrt(1.0 - rho * rho)) / (rho * rho)


def ssvi_butterfly_conditions(theta, p: SSVIParams):
    """Gatheral-Jacquier sufficient butterfly quantities at theta.

    No-butterfly sufficient conditions are
      theta*phi(theta)*(1+|rho|) < 4
      theta*phi(theta)^2*(1+|rho|) <= 4.
    """
    theta = np.asarray(theta, dtype=float)
    phi = ssvi_phi(theta, p)
    c1 = theta * phi * (1.0 + abs(p.rho))
    c2 = theta * phi * phi * (1.0 + abs(p.rho))
    return c1, c2


def is_ssvi_butterfly_free(theta_values, p: SSVIParams, numerical=True,
                           k_range=(-1.5, 1.5), n=801):
    """Return (ok, max_condition1, max_condition2, min_Durrleman_g)."""
    theta = np.asarray(theta_values, dtype=float)
    c1, c2 = ssvi_butterfly_conditions(theta, p)
    sufficient = bool(np.all(c1 < 4.0 - 1e-10) and np.all(c2 <= 4.0 + 1e-10))

    min_g = np.inf
    if numerical:
        kk = np.linspace(*k_range, n)
        for th in theta:
            raw = ssvi_to_raw_svi(float(th), p)
            g = durrleman_g(kk, raw)
            finite = g[np.isfinite(g)]
            if len(finite):
                min_g = min(min_g, float(finite.min()))
        numerical_ok = min_g >= -1e-8
    else:
        numerical_ok = True
    return sufficient and numerical_ok, float(np.max(c1)), float(np.max(c2)), float(min_g)


def _calendar_conditions_ok(theta_values, p: SSVIParams):
    r = ssvi_calendar_ratio(theta_values, p)
    ub = _calendar_upper_bound(p.rho)
    return bool(np.all(r >= -1e-12) and np.all(r <= ub + 1e-12)), float(np.min(r)), float(np.max(r)), float(ub)


# -------------------------------------------------------------- calibration

def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    # Stable enough over optimiser-scale inputs.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _logit(x):
    x = np.clip(x, 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def _unpack(z, phi_form="modified_power_law") -> SSVIParams:
    rho = _RHO_MAX * np.tanh(float(z[0]))
    gamma = _GAMMA_LO + (_GAMMA_HI - _GAMMA_LO) * float(_sigmoid(z[1]))
    # For the modified power law, eta*(1+|rho|) <= 2 is the convenient
    # global static-arbitrage bound.  Parameterising eta as a fraction of that
    # cap means the optimiser cannot wander into an inadmissible region.
    eta_cap = 2.0 / (1.0 + abs(rho))
    eta = eta_cap * 0.999 * float(_sigmoid(z[2]))
    return SSVIParams(rho=rho, eta=max(eta, 1e-8), gamma=gamma, phi_form=phi_form)


def _pack_guess(rho, gamma, eta_fraction):
    rho = np.clip(float(rho), -0.95, 0.95)
    z0 = np.arctanh(rho / _RHO_MAX)
    gfrac = (np.clip(gamma, _GAMMA_LO + 1e-5, _GAMMA_HI - 1e-5) - _GAMMA_LO) / (
        _GAMMA_HI - _GAMMA_LO
    )
    return np.array([z0, _logit(gfrac), _logit(np.clip(eta_fraction, 0.01, 0.99))])


@dataclass
class SSVIFit:
    params: SSVIParams
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
    calendar_ratio_min: float
    calendar_ratio_max: float
    calendar_ratio_upper: float
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

    def total_variance(self, k, T):
        return ssvi_total_variance(k, self.theta(T), self.params)

    def implied_vol(self, k, T):
        return ssvi_iv(k, self.theta(T), T, self.params)

    def raw_slice(self, T):
        return ssvi_to_raw_svi(float(self.theta(T)), self.params)

    def to_surface(self, trade_date, symbol="", tenor_days=None, k_grid=None):
        """Evaluate SSVI on VolForge's standard Surface grid."""
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
            calendar_ratio_min=self.calendar_ratio_min,
            calendar_ratio_max=self.calendar_ratio_max,
            calendar_ratio_upper=self.calendar_ratio_upper,
            theta_repair=self.theta_repair,
            theta_repair_fraction=self.theta_repair_fraction,
        )
        return d


def calibrate_ssvi(
    slices_and_fits: Iterable,
    reliable_only: bool = True,
    include_unreliable_observations: bool = True,
    repair_theta: bool = True,
    phi_form: str = "modified_power_law",
    n_restarts: int = 9,
) -> SSVIFit:
    """Jointly calibrate SSVI to a day's raw-SVI-calibrated option slices.

    Parameters
    ----------
    slices_and_fits:
        Iterable of ``(Slice, SVIFit)``.  Raw SVI supplies a stable estimate of
        each expiry's ATM total variance theta; the SSVI shape parameters are
        then fitted directly to the original option observations, not to the
        raw-SVI curve.
    reliable_only:
        Use only reliable raw-SVI fits as ATM-theta anchors.  Strongly advised.
    include_unreliable_observations:
        Still let successful interior slices that failed the raw-SVI reliability
        gate contribute their original market observations to the global SSVI
        objective.  Their raw-SVI parameters are *not* trusted; theta at those
        maturities comes from the monotone curve built from reliable neighbours.
        This is useful when SSVI can regularise a slice that raw SVI made
        butterfly-arbitrageable.
    repair_theta:
        Project small ATM total-variance inversions onto the nearest monotone
        curve with isotonic regression.  The repair amount is retained on the
        result and should be monitored.
    phi_form:
        ``modified_power_law`` is the recommended first implementation because
        it admits a simple global static-arbitrage parameter bound.  Plain
        ``power_law`` is exposed for research comparisons but only checked on
        the fitted theta range.
    """
    pairs = list(slices_and_fits)
    theta_pairs = [(s, f) for s, f in pairs if (f.is_reliable if reliable_only else f.success)]
    if len(theta_pairs) < 3:
        raise ValueError(f"need >=3 usable SVI slices for the SSVI theta clock, got {len(theta_pairs)}")
    theta_pairs.sort(key=lambda sf: sf[0].T)

    T_theta = np.array([float(s.T) for s, _ in theta_pairs])
    theta_raw = np.array([float(svi_total_variance(0.0, f.params)) for _, f in theta_pairs])
    # More observations give a more stable ATM estimate, but cap the influence
    # so one very dense expiry cannot dictate the entire term structure.
    theta_weights = np.sqrt(np.array([max(f.n_obs, 1) for _, f in theta_pairs], dtype=float))
    curve = build_atm_variance_curve(T_theta, theta_raw, theta_weights, repair=repair_theta)

    if include_unreliable_observations:
        # A bad raw-SVI shape need not imply bad market observations.  Use the
        # original quotes from successful slices inside the reliable theta
        # range, while sourcing theta from the monotone curve rather than the
        # suspect raw fit itself.
        lo, hi = curve.t_years[0], curve.t_years[-1]
        fit_pairs = [(s, f) for s, f in pairs
                     if f.success and s.n >= 8 and lo - 1e-12 <= s.T <= hi + 1e-12]
    else:
        fit_pairs = list(theta_pairs)
    fit_pairs.sort(key=lambda sf: sf[0].T)

    T_obs = np.array([float(s.T) for s, _ in fit_pairs])
    F_obs = np.array([float(s.forward) for s, _ in fit_pairs])
    ks = np.concatenate([np.asarray(s.k, float) for s, _ in fit_pairs])
    Ts = np.concatenate([np.full(s.n, float(s.T)) for s, _ in fit_pairs])
    ws = np.concatenate([np.asarray(s.w, float) for s, _ in fit_pairs])
    weights = np.concatenate([np.asarray(s.weights, float) for s, _ in fit_pairs])
    weights = weights / np.mean(weights)
    theta_obs_for_quotes = curve(Ts)

    # Use the median raw-SVI rho only as a warm start; it is not treated as an
    # SSVI parameter observation.
    rho0 = float(np.median([f.params.rho for _, f in theta_pairs]))

    def loss(z):
        p = _unpack(z, phi_form)
        pred = ssvi_total_variance(ks, theta_obs_for_quotes, p)
        err = pred - ws
        val = float(np.mean(weights * err * err))
        if phi_form == "power_law":
            # Plain power law is not globally admissible for arbitrary theta;
            # penalise violations on the actually fitted theta range.
            c1, c2 = ssvi_butterfly_conditions(curve.theta, p)
            cal_ok, rmin, rmax, ub = _calendar_conditions_ok(curve.theta, p)
            penalty = np.sum(np.maximum(c1 - 3.999, 0.0) ** 2)
            penalty += np.sum(np.maximum(c2 - 4.0, 0.0) ** 2)
            penalty += max(-rmin, 0.0) ** 2 + max(rmax - ub, 0.0) ** 2
            val += 1e3 * float(penalty)
        return val

    starts = []
    gammas = [0.30, 0.50, 0.70]
    etas = [0.25, 0.50, 0.75]
    for g in gammas:
        for ef in etas:
            starts.append(_pack_guess(rho0, g, ef))
    starts = starts[:max(1, int(n_restarts))]

    best = None
    for x0 in starts:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 2500, "xatol": 1e-7, "fatol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res

    p = _unpack(best.x, phi_form)
    pred_w = ssvi_total_variance(ks, theta_obs_for_quotes, p)
    pred_iv = np.sqrt(np.maximum(pred_w, 0.0) / Ts)
    market_iv = np.sqrt(np.maximum(ws, 0.0) / Ts)
    iv_err = pred_iv - market_iv

    bfly_ok, c1max, c2max, min_g = is_ssvi_butterfly_free(curve.theta, p)
    cal_ok, rmin, rmax, rub = _calendar_conditions_ok(curve.theta, p)
    cal_ok = cal_ok and curve.is_monotone

    per_slice = {}
    for s, _ in fit_pairs:
        th = float(curve(s.T))
        piv = ssvi_iv(s.k, th, s.T, p)
        per_slice[float(s.dte)] = float(np.sqrt(np.mean((piv - s.iv) ** 2)))

    return SSVIFit(
        params=p,
        theta_curve=curve,
        rmse=float(np.sqrt(np.mean(weights * (pred_w - ws) ** 2))),
        rmse_iv=float(np.sqrt(np.mean(iv_err * iv_err))),
        max_abs_err_iv=float(np.max(np.abs(iv_err))),
        n_obs=int(len(ws)),
        n_slices=int(len(fit_pairs)),
        n_theta_slices=int(len(theta_pairs)),
        success=bool(best.success or np.isfinite(best.fun)),
        butterfly_free=bfly_ok,
        calendar_free=cal_ok,
        min_durrleman_g=min_g,
        max_bfly_condition1=c1max,
        max_bfly_condition2=c2max,
        calendar_ratio_min=rmin,
        calendar_ratio_max=rmax,
        calendar_ratio_upper=rub,
        objective=float(best.fun),
        slice_rmse_iv=per_slice,
        T_obs=T_obs,
        forwards=F_obs,
        spot=float(fit_pairs[0][0].spot),
    )


fit_ssvi_surface = calibrate_ssvi

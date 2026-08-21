"""Extended SSVI model and calibration tests."""

from dataclasses import dataclass

import numpy as np

from volforge import (
    ESSVIParams,
    calibrate_essvi,
    calibrate_ssvi,
    calibrate_svi,
    essvi_rho,
    essvi_total_variance,
    is_essvi_butterfly_free,
    is_essvi_calendar_free,
)


@dataclass
class FakeSlice:
    T: float
    dte: float
    k: np.ndarray
    w: np.ndarray
    iv: np.ndarray
    weights: np.ndarray
    forward: float = 500.0
    spot: float = 499.0

    @property
    def n(self):
        return len(self.k)


def _pairs():
    # theta_max is chosen to match the observed synthetic horizon closely.
    dtes = [14.0, 30.0, 60.0, 90.0, 120.0]
    theta = [0.04 * (d / 365.25) + 0.0015 * np.sqrt(d / 365.25) for d in dtes]
    true = ESSVIParams(
        rho0=-0.25,
        rho_m=-0.55,
        a=0.55,
        eta=1.0,
        gamma=0.45,
        theta_max=max(theta),
    )
    k = np.linspace(-0.25, 0.20, 41)
    out = []
    for dte, th in zip(dtes, theta):
        T = dte / 365.25
        w = essvi_total_variance(k, th, true)
        iv = np.sqrt(w / T)
        slc = FakeSlice(T=T, dte=dte, k=k, w=w, iv=iv, weights=np.ones_like(k))
        raw = calibrate_svi(k, w, T, weights=slc.weights)
        assert raw.is_reliable
        out.append((slc, raw))
    return true, out


def test_essvi_no_arbitrage_conditions():
    true, pairs = _pairs()
    theta = np.array([float(p[0].w[np.argmin(np.abs(p[0].k))]) for p in pairs])
    cal, margin, *_ = is_essvi_calendar_free(theta, true)
    bfly, *_ = is_essvi_butterfly_free(theta, true)
    assert cal and margin > 0
    assert bfly
    rho = essvi_rho(theta, true)
    assert rho[-1] < rho[0]


def test_essvi_calibration_beats_nested_ssvi_on_varying_rho_surface():
    _, pairs = _pairs()
    ssvi = calibrate_ssvi(pairs, n_restarts=4)
    essvi = calibrate_essvi(pairs, n_restarts=5)
    assert essvi.success
    assert essvi.calendar_free
    assert essvi.butterfly_free
    assert essvi.is_reliable
    assert essvi.rmse_iv < ssvi.rmse_iv
    assert abs(essvi.params.rho_m - essvi.params.rho0) > 0.02


def test_essvi_common_surface_grid():
    _, pairs = _pairs()
    fit = calibrate_essvi(pairs, n_restarts=4)
    surf = fit.to_surface("2026-08-20", "TEST")
    assert surf.total_var.shape == (5, 17)
    assert np.all(np.diff(surf.total_var, axis=0) >= -2e-8)

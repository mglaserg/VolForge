"""SSVI model, ATM clock, calibration and common-grid integration."""

from dataclasses import dataclass

import numpy as np

from volforge import (
    SSVIParams,
    build_atm_variance_curve,
    calibrate_svi,
    calibrate_ssvi,
    is_ssvi_butterfly_free,
    ssvi_total_variance,
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


def test_atm_curve_isotonic_repair():
    T = np.array([14, 30, 60, 90]) / 365.25
    raw = np.array([0.0020, 0.0045, 0.0042, 0.0100])
    c = build_atm_variance_curve(T, raw, repair=True)
    assert c.is_monotone
    assert c.repair_amount > 0
    assert np.all(np.diff(c.theta) >= 0)
    assert abs(c(0.0)) < 1e-14


def _synthetic_pairs():
    true = SSVIParams(rho=-0.42, eta=1.05, gamma=0.48)
    pairs = []
    k = np.linspace(-0.25, 0.20, 41)
    for dte in [14.0, 30.0, 60.0, 90.0, 120.0]:
        T = dte / 365.25
        theta = 0.045 * T + 0.0015 * np.sqrt(T)
        w = ssvi_total_variance(k, theta, true)
        iv = np.sqrt(w / T)
        s = FakeSlice(T=T, dte=dte, k=k, w=w, iv=iv, weights=np.ones_like(k))
        raw = calibrate_svi(k, w, T, weights=s.weights)
        assert raw.is_reliable
        pairs.append((s, raw))
    return true, pairs


def test_ssvi_calibration_recovers_synthetic_surface():
    true, pairs = _synthetic_pairs()
    fit = calibrate_ssvi(pairs, n_restarts=9)
    assert fit.success
    assert fit.calendar_free
    assert fit.butterfly_free
    assert fit.is_reliable
    assert fit.rmse_iv < 2e-4
    assert abs(fit.params.rho - true.rho) < 0.03
    assert abs(fit.params.gamma - true.gamma) < 0.08
    assert abs(fit.params.eta - true.eta) < 0.08

    ok, _, _, min_g = is_ssvi_butterfly_free(fit.theta_curve.theta, fit.params)
    assert ok
    assert min_g > -1e-8


def test_ssvi_to_common_surface_grid():
    _, pairs = _synthetic_pairs()
    fit = calibrate_ssvi(pairs, n_restarts=4)
    surf = fit.to_surface("2026-08-20", "TEST")
    assert surf.total_var.shape == (5, 17)
    assert surf.iv.shape == surf.total_var.shape
    assert np.all(np.diff(surf.total_var, axis=0) >= -1e-10)


def test_ssvi_can_regularise_unreliable_raw_slice_without_using_its_theta():
    _, pairs = _synthetic_pairs()
    # Mark the middle raw-SVI shape unreliable while leaving its market quotes intact.
    pairs[2][1].butterfly_free = False
    fit = calibrate_ssvi(pairs, n_restarts=4, include_unreliable_observations=True)
    assert fit.n_theta_slices == 4
    assert fit.n_slices == 5
    assert fit.calendar_free and fit.butterfly_free

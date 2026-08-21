"""Fengler constrained natural-spline tests."""

from types import SimpleNamespace

import numpy as np

from volforge import SSVIParams, ssvi_total_variance
from volforge.fengler import (
    fit_fengler_surface,
    natural_spline_basis,
    spline_qr_matrices,
)


class FakeSlice:
    def __init__(self, T, theta):
        self.T = T
        self.dte = T * 365.25
        self.forward = 500.0
        self.spot = 500.0
        self.forward_fit = SimpleNamespace(discount=0.995)
        self.k = np.linspace(-0.22, 0.18, 23)
        self.w = ssvi_total_variance(self.k, theta, SSVIParams(-0.40, 1.0, 0.45))
        self.iv = np.sqrt(self.w / T)
        self.half_spread = np.full_like(self.k, 0.03)
        self.n = len(self.k)


def _slices():
    out = []
    for d in [14.0, 30.0, 60.0, 90.0]:
        T = d / 365.25
        theta = 0.04 * T + 0.001 * np.sqrt(T)
        out.append(FakeSlice(T, theta))
    return out


def test_fengler_qr_and_basis_reproduce_knots():
    u = np.linspace(0.75, 1.25, 9)
    Q, R = spline_qr_matrices(u)
    assert Q.shape == (9, 7)
    assert R.shape == (7, 7)
    assert np.all(np.linalg.eigvalsh(R) > 0)

    g = np.exp(-0.5 * (u - 1.0))
    gamma = np.linalg.solve(R, Q.T @ g)
    state = np.concatenate([g, gamma])
    E = natural_spline_basis(u, u)
    assert np.max(np.abs(E @ state - g)) < 1e-12


def test_fengler_surface_is_strike_and_calendar_arbitrage_free():
    fit = fit_fengler_surface(_slices(), smoothing_lambda=1e-6, calendar_grid_size=101)
    assert fit.success
    assert fit.butterfly_free
    assert fit.calendar_free
    assert fit.is_reliable
    assert fit.rmse_iv < 0.015
    assert all(s.min_gamma >= -2e-7 for s in fit.slices)
    assert all(s.left_slope >= -1.00001 for s in fit.slices)
    assert all(s.right_slope <= 1e-5 for s in fit.slices)


def test_fengler_common_grid_is_calendar_monotone():
    fit = fit_fengler_surface(_slices(), smoothing_lambda=1e-6, calendar_grid_size=101)
    surf = fit.to_surface("2026-08-20", "TEST", k_grid=np.linspace(-0.15, 0.15, 9))
    assert surf.total_var.shape == (5, 9)
    assert np.all(np.diff(surf.total_var, axis=0) >= -1e-10)

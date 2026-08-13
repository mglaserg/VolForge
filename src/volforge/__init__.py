"""VolForge: SVI volatility-surface calibration and relative-value research."""

__version__ = "0.2.0"

from .blackscholes import (
    black_price, black_vega, implied_vol, implied_vol_vec,
    iv_to_total_variance, total_variance_to_iv,
)
from .forward import ForwardFit, fit_forward, log_moneyness
from .svi import (
    SVIParams, SVIFit, calibrate_svi, svi_total_variance, svi_iv,
    svi_derivatives, durrleman_g, wing_slopes, is_butterfly_free,
    check_calendar_arbitrage,
)
from .diagnostics import ResidualReport, residual_report
from .surface import Surface, build_surface, repair_calendar, surface_panel, \
    DEFAULT_TENORS, DEFAULT_K_GRID
from .features import surface_features, feature_panel, standardize
from .pca import PCAModel, fit_surface_pca, reconstruct, pca_residuals
from .signals import (
    residual_signal, zscore_series, forward_convergence, bucket_by_signal,
    mean_reversion_test, SignalReport,
)
from .database import VolDB

__all__ = [
    "black_price", "black_vega", "implied_vol", "implied_vol_vec",
    "iv_to_total_variance", "total_variance_to_iv",
    "ForwardFit", "fit_forward", "log_moneyness",
    "SVIParams", "SVIFit", "calibrate_svi", "svi_total_variance", "svi_iv",
    "svi_derivatives", "durrleman_g", "wing_slopes", "is_butterfly_free",
    "check_calendar_arbitrage",
    "ResidualReport", "residual_report",
    "Surface", "build_surface", "repair_calendar", "surface_panel",
    "DEFAULT_TENORS", "DEFAULT_K_GRID",
    "surface_features", "feature_panel", "standardize",
    "PCAModel", "fit_surface_pca", "reconstruct", "pca_residuals",
    "residual_signal", "zscore_series", "forward_convergence",
    "bucket_by_signal", "mean_reversion_test", "SignalReport",
    "VolDB",
]

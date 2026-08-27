"""VolForge: SVI volatility-surface calibration and relative-value research."""

__version__ = "0.4.0"

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
from .term_structure import ATMVarianceCurve, isotonic_increasing, build_atm_variance_curve
from .ssvi import (
    SSVIParams, SSVIFit, ssvi_phi, ssvi_phi_prime, ssvi_total_variance, ssvi_iv,
    ssvi_to_raw_svi, ssvi_calendar_ratio, ssvi_butterfly_conditions,
    is_ssvi_butterfly_free, calibrate_ssvi, fit_ssvi_surface,
)
from .essvi import (
    ESSVIParams, ESSVIFit, essvi_phi, essvi_rho, essvi_rho_prime,
    essvi_total_variance, essvi_iv, essvi_calendar_terms,
    essvi_butterfly_conditions, essvi_to_raw_svi,
    is_essvi_calendar_free, is_essvi_butterfly_free,
    calibrate_essvi, fit_essvi_surface,
)
from .fengler import (
    FenglerSliceFit, FenglerSurfaceFit, spline_qr_matrices, natural_spline_basis,
    fit_fengler_slice, fit_fengler_surface, prepare_fengler_slices, calibrate_fengler,
)
from .diagnostics import ResidualReport, residual_report
from .surface import Surface, build_surface, repair_calendar, surface_panel, \
    DEFAULT_TENORS, DEFAULT_K_GRID
from .features import surface_features, feature_panel, standardize
from .delta_surface import (
    DeltaVolSurface, build_delta_surface, constant_tenor_delta_slice,
    delta_ratio_term_structure, delta_lump_scores, delta_surface_change_features,
)
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
    "ATMVarianceCurve", "isotonic_increasing", "build_atm_variance_curve",
    "SSVIParams", "SSVIFit", "ssvi_phi", "ssvi_phi_prime",
    "ssvi_total_variance", "ssvi_iv", "ssvi_to_raw_svi",
    "ssvi_calendar_ratio", "ssvi_butterfly_conditions",
    "is_ssvi_butterfly_free", "calibrate_ssvi", "fit_ssvi_surface",
    "ESSVIParams", "ESSVIFit", "essvi_phi", "essvi_rho", "essvi_rho_prime",
    "essvi_total_variance", "essvi_iv", "essvi_calendar_terms",
    "essvi_butterfly_conditions", "essvi_to_raw_svi",
    "is_essvi_calendar_free", "is_essvi_butterfly_free",
    "calibrate_essvi", "fit_essvi_surface",
    "FenglerSliceFit", "FenglerSurfaceFit", "spline_qr_matrices",
    "natural_spline_basis", "fit_fengler_slice", "fit_fengler_surface", "prepare_fengler_slices",
    "calibrate_fengler",
    "ResidualReport", "residual_report",
    "Surface", "build_surface", "repair_calendar", "surface_panel",
    "DEFAULT_TENORS", "DEFAULT_K_GRID",
    "surface_features", "feature_panel", "standardize",
    "DeltaVolSurface", "build_delta_surface", "constant_tenor_delta_slice",
    "delta_ratio_term_structure", "delta_lump_scores", "delta_surface_change_features",
    "PCAModel", "fit_surface_pca", "reconstruct", "pca_residuals",
    "residual_signal", "zscore_series", "forward_convergence",
    "bucket_by_signal", "mean_reversion_test", "SignalReport",
    "VolDB",
]

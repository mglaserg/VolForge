"""Pure data preparation for the Forward VRP dashboard.

The Streamlit entry point lives at the repository root in
``volforge_dashboard.py``.  Keeping calculations here (with no Streamlit
imports) makes the dashboard testable and reusable from notebooks/scripts.

The current snapshot intentionally compares 30-day MFIV with *trailing*
integrated realized variance.  That is a live feature/proxy, not the forward
VRP training label.  Forward VRP remains MFIV today minus variance realized
strictly after today and is only available once future data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data.schema import add_derived_columns, validate_chain
from .mfiv import ConstantTenorMFIV, constant_tenor_mfiv, mfiv_term_structure
from .realized import daily_integrated_variance, integrated_volatility, rolling_integrated_variance
from .vrp import forward_vrp_label, realized_term_structure, vol_of_vol, vrp_features

__all__ = [
    "DashboardSnapshot",
    "build_dashboard_snapshot",
    "normalise_intraday_bars",
    "prepare_vrp_history",
    "VRPContext",
    "VRPCandidate",
    "SurfaceMFIVResult",
    "SurfaceExplorerData",
    "classify_vrp_context",
    "classify_vrp_candidate",
    "build_surface_mfiv_comparison",
    "build_surface_explorer",
]


@dataclass(frozen=True)
class DashboardSnapshot:
    """Provider-neutral data bundle consumed by the dashboard UI."""

    symbol: str
    quote_time: pd.Timestamp
    target_days: float
    price_side: str
    target_mfiv: ConstantTenorMFIV
    mfiv_curve: pd.DataFrame
    daily_variance: pd.Series
    realized_history: pd.DataFrame
    realized_curve: pd.DataFrame
    trailing_target_variance: float
    trailing_target_volatility: float
    current_vrp_variance: float
    current_vol_spread: float
    chain_quality: dict[str, float | int | str]


def normalise_intraday_bars(
    bars: pd.DataFrame,
    *,
    timestamp_col: str | None = None,
    close_col: str | None = None,
) -> pd.DataFrame:
    """Return ``timestamp``/``close`` columns from common vendor bar shapes.

    Accepts either a DatetimeIndex or a timestamp-like column.  Column matching
    is case-insensitive and understands the common ``Datetime``/``Date`` and
    ``Close`` spellings used by yfinance and CSV exports.
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("intraday bars are empty")

    df = bars.copy()
    lower = {str(c).lower(): c for c in df.columns}

    if timestamp_col is None:
        for candidate in ("timestamp", "datetime", "date", "time"):
            if candidate in lower:
                timestamp_col = lower[candidate]
                break
    if timestamp_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        timestamp_col = df.columns[0]
    if timestamp_col is None:
        raise ValueError("could not find timestamp column or DatetimeIndex")

    # Rebuild the case map because reset_index may have added a column.
    lower = {str(c).lower(): c for c in df.columns}
    if close_col is None:
        for candidate in ("close", "adj close", "adj_close", "price", "last"):
            if candidate in lower:
                close_col = lower[candidate]
                break
    if close_col is None:
        raise ValueError("could not find a close/price column")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[timestamp_col], errors="coerce", utc=True),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    out = out.dropna().sort_values("timestamp")
    out = out[out["close"] > 0].drop_duplicates("timestamp", keep="last")
    if len(out) < 2:
        raise ValueError("need at least two valid intraday bars")
    return out.reset_index(drop=True)


def build_dashboard_snapshot(
    chain: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    target_days: float = 30.0,
    price_side: str = "mid",
    rv_windows: Iterable[int] = (3, 9, 30, 60, 180),
    session_tz: str = "America/New_York",
    min_mfiv_strikes: int = 8,
) -> DashboardSnapshot:
    """Compute the live, no-ML Forward VRP dashboard state.

    ``target_days`` uses calendar time for the MFIV-vs-trailing-RV comparison,
    matching the option DTE clock.  The descriptive RV term structure uses
    trading-day windows because 3/9/30/60/180-day realized-vol features are
    conventionally interpreted that way in this project.
    """
    if target_days <= 0:
        raise ValueError("target_days must be positive")

    canonical = add_derived_columns(chain)
    validate_chain(canonical)
    symbols = canonical["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        raise ValueError("dashboard snapshot requires exactly one symbol")
    symbol = symbols[0]
    quote_time = pd.to_datetime(canonical["quote_time"], utc=True).max()

    slices = mfiv_term_structure(
        canonical,
        price_side=price_side,
        min_strikes=min_mfiv_strikes,
    )
    if len(slices) < 1:
        raise ValueError("no expiries produced a valid MFIV slice")
    target = constant_tenor_mfiv(slices, target_days)

    mfiv_curve = pd.DataFrame([
        {
            "expiry": s.expiry,
            "dte": s.dte,
            "implied_variance": s.implied_variance,
            "implied_volatility": s.implied_volatility,
            "total_variance": s.total_variance,
            "forward": s.forward_fit.forward,
            "discount": s.forward_fit.discount,
            "parity_r2": s.forward_fit.r_squared,
            "n_strikes": s.n_strikes,
            "price_side": s.price_side,
        }
        for s in slices
    ]).sort_values("dte").reset_index(drop=True)

    norm_bars = normalise_intraday_bars(bars)
    daily = daily_integrated_variance(
        norm_bars,
        session_tz=session_tz,
        include_overnight=True,
    )
    if daily.empty:
        raise ValueError("intraday bars did not produce realized variance")

    rv_windows = tuple(int(w) for w in rv_windows)
    realized_hist = realized_term_structure(daily, windows=rv_windows, basis="trading")

    # Only use information observed by the option quote timestamp.  This is
    # particularly important when users load a local file containing later bars.
    local_quote_date = quote_time.tz_convert(session_tz).tz_localize(None).normalize()
    eligible_daily = daily.loc[daily.index <= local_quote_date]
    if eligible_daily.empty:
        raise ValueError("intraday history ends before the option quote date")

    trailing_target = rolling_integrated_variance(
        eligible_daily,
        int(round(target_days)),
        basis="calendar",
    ).dropna()
    if trailing_target.empty:
        raise ValueError(
            f"not enough intraday history to compute trailing {target_days:g}-day realized variance"
        )
    trailing_var = float(trailing_target.iloc[-1])
    trailing_vol = float(integrated_volatility(trailing_var))

    eligible_realized = realized_hist.loc[realized_hist.index <= local_quote_date]
    if eligible_realized.empty:
        raise ValueError("no realized term-structure observation aligns with the quote date")
    latest = eligible_realized.iloc[-1]
    curve_rows = []
    for w in rv_windows:
        col = f"rv_{w}"
        if col in latest.index and np.isfinite(latest[col]):
            curve_rows.append({"days": w, "realized_volatility": float(latest[col])})
    realized_curve = pd.DataFrame(curve_rows)

    mid = canonical["mid"].replace([np.inf, -np.inf], np.nan)
    rel = canonical["rel_spread"].replace([np.inf, -np.inf], np.nan)
    chain_quality: dict[str, float | int | str] = {
        "rows": int(len(canonical)),
        "expiries": int(canonical["expiry"].nunique()),
        "strikes": int(canonical["strike"].nunique()),
        "spot": float(canonical["underlying_price"].median()),
        "zero_bid_rows": int((canonical["bid"] <= 0).sum()),
        "crossed_rows": int((canonical["ask"] < canonical["bid"]).sum()),
        "median_spread": float(canonical["spread"].median()),
        "median_rel_spread": float(rel.median()) if rel.notna().any() else np.nan,
        "median_mid": float(mid.median()) if mid.notna().any() else np.nan,
    }

    return DashboardSnapshot(
        symbol=symbol,
        quote_time=quote_time,
        target_days=float(target_days),
        price_side=price_side,
        target_mfiv=target,
        mfiv_curve=mfiv_curve,
        daily_variance=daily,
        realized_history=realized_hist,
        realized_curve=realized_curve,
        trailing_target_variance=trailing_var,
        trailing_target_volatility=trailing_vol,
        current_vrp_variance=float(target.implied_variance - trailing_var),
        current_vol_spread=float(target.implied_volatility - trailing_vol),
        chain_quality=chain_quality,
    )


def prepare_vrp_history(
    history: pd.DataFrame,
    *,
    date_col: str = "date",
    z_window: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Validate and enrich a daily MFIV/RV history for dashboard charts.

    Required columns are ``date``, ``mfiv_var`` and ``trailing_rv_var``.
    ``forward_rv_var`` is optional; when present, the true ex-post forward VRP
    label is added.  Rolling z-scores and percentiles are inherited from
    :func:`volforge.vrp.vrp_features` and therefore use only prior observations.
    """
    required = {date_col, "mfiv_var", "trailing_rv_var"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"VRP history is missing columns: {sorted(missing)}")

    df = history.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for col in ("mfiv_var", "trailing_rv_var", "forward_rv_var"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[date_col, "mfiv_var", "trailing_rv_var"])
    df = df.sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)

    features = vrp_features(
        df["mfiv_var"],
        df["trailing_rv_var"],
        z_window=z_window,
        min_periods=min_periods,
    )
    # Saved history files produced by ``build_vrp_history`` already contain
    # these derived columns.  Recompute them for display so the dashboard uses
    # the current feature definitions, but replace any persisted values rather
    # than joining duplicate column names (which makes repeated builds fail).
    out = df.copy()
    for col in features.columns:
        if col not in {"mfiv_var", "trailing_rv_var"}:
            out[col] = features[col]
    out["mfiv_vol"] = integrated_volatility(out["mfiv_var"])
    out["trailing_rv_vol"] = integrated_volatility(out["trailing_rv_var"])
    out["vol_spread"] = out["mfiv_vol"] - out["trailing_rv_vol"]
    out["vol_of_vol"] = vol_of_vol(out["mfiv_var"], window=20)

    if "forward_rv_var" in out:
        out["forward_rv_vol"] = integrated_volatility(out["forward_rv_var"])
        out["forward_vrp"] = forward_vrp_label(out["mfiv_var"], out["forward_rv_var"])
    return out


@dataclass(frozen=True)
class VRPContext:
    state: str
    explanation: str
    rv_slope: float
    recent_slope_peak: float
    cooling_from_shock: bool
    premium_positive: bool


@dataclass(frozen=True)
class VRPCandidate:
    """Transparent research classification for a possible VRP setup.

    This is deliberately a *candidate* label rather than a trade instruction.
    It only summarizes the observable premium/regime state; position sizing,
    tail risk, execution, and the future-RV forecasting layer remain separate.
    """

    label: str
    level: str
    explanation: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceExplorerData:
    """One calibrated surface plus the raw observations used to interpret it."""

    model: str
    surface: object
    raw_points: pd.DataFrame
    raw_atm_term: pd.DataFrame
    mfiv_curve: pd.DataFrame
    reliable: bool
    rmse_iv: float
    detail: str


@dataclass(frozen=True)
class SurfaceMFIVResult:
    name: str
    target: ConstantTenorMFIV
    curve: pd.DataFrame
    reliable: bool
    rmse_iv: float
    detail: str


def classify_vrp_context(
    snapshot: DashboardSnapshot,
    *,
    mfiv_variance: float | None = None,
    lookback: int = 10,
) -> VRPContext:
    """Describe the RV3/RV30 state without pretending it is a trade rule."""
    hist = snapshot.realized_history
    if not {"rv_3", "rv_30"}.issubset(hist.columns):
        return VRPContext("Insufficient RV history", "Need RV3 and RV30.", np.nan, np.nan, False, False)
    slope = (hist["rv_3"] - hist["rv_30"]).dropna()
    if slope.empty:
        return VRPContext("Insufficient RV history", "Need RV3 and RV30.", np.nan, np.nan, False, False)
    current = float(slope.iloc[-1])
    recent_peak = float(slope.tail(max(int(lookback), 2)).max())
    cooling = bool(current < 0 and recent_peak > 0)
    implied = snapshot.target_mfiv.implied_variance if mfiv_variance is None else float(mfiv_variance)
    premium = bool(implied > snapshot.trailing_target_variance)

    if current > 0:
        state = "Shock underway"
        explanation = "RV3 is above RV30: short-term realized volatility is hotter than the monthly window. Premium may be forming, but the shock is still active."
    elif cooling and premium:
        state = "Post-shock / IV still elevated"
        explanation = "RV3 has fallen below RV30 after a recent positive slope, while implied variance still exceeds trailing RV. This is the cooling-after-shock pattern worth researching."
    elif current < 0 and premium:
        state = "Calm / premium positive"
        explanation = "RV3 is below RV30 and implied variance exceeds trailing RV, but there was no recent RV-slope shock in the selected lookback."
    elif current < 0:
        state = "Calm / little premium"
        explanation = "Realized volatility is calm, but implied variance is not compensating much above trailing RV."
    else:
        state = "Flat RV slope"
        explanation = "RV3 and RV30 are roughly aligned; use VRP stretch and the broader term structure for context."
    return VRPContext(state, explanation, current, recent_peak, cooling, premium)


def classify_vrp_candidate(
    snapshot: DashboardSnapshot,
    context: VRPContext,
    *,
    mfiv_variance: float | None = None,
    mfiv_volatility: float | None = None,
) -> VRPCandidate:
    """Classify the current setup as a VRP *research candidate*.

    The rule is intentionally small and interpretable.  It does not attempt to
    predict option P&L.  A positive implied-vs-trailing-RV spread is required;
    the RV3/RV30 transition then determines whether the premium is merely
    present, still forming during an active shock, or lingering after the shock
    has cooled.
    """
    implied_var = (
        snapshot.target_mfiv.implied_variance
        if mfiv_variance is None
        else float(mfiv_variance)
    )
    implied_vol = (
        snapshot.target_mfiv.implied_volatility
        if mfiv_volatility is None
        else float(mfiv_volatility)
    )
    variance_spread = float(implied_var - snapshot.trailing_target_variance)
    vol_spread = float(implied_vol - snapshot.trailing_target_volatility)

    reasons: list[str] = []
    if variance_spread > 0:
        reasons.append("Implied variance is above trailing integrated realized variance.")
    else:
        reasons.append("Implied variance is not above trailing integrated realized variance.")

    if context.cooling_from_shock:
        reasons.append("RV3 has fallen below RV30 after a recent positive RV3−RV30 slope.")
    elif np.isfinite(context.rv_slope) and context.rv_slope > 0:
        reasons.append("RV3 is above RV30, so the realized-volatility shock is still active.")
    elif np.isfinite(context.rv_slope) and context.rv_slope < 0:
        reasons.append("RV3 is below RV30, indicating calmer very-short-horizon realized volatility.")

    if variance_spread <= 0:
        return VRPCandidate(
            label="Not a VRP candidate",
            level="none",
            explanation=(
                "The live implied-variance premium is not positive versus trailing integrated RV. "
                "There may still be other option edges, but this is not a clean VRP candidate by the current screen."
            ),
            reasons=tuple(reasons),
        )

    if context.cooling_from_shock:
        return VRPCandidate(
            label="Post-shock VRP candidate",
            level="strong",
            explanation=(
                "Realized volatility has cooled after a recent short-horizon spike while option-implied variance "
                "still exceeds trailing realized variance. This is the lingering-premium pattern to research most closely."
            ),
            reasons=tuple(reasons),
        )

    if np.isfinite(context.rv_slope) and context.rv_slope > 0:
        return VRPCandidate(
            label="Developing VRP candidate",
            level="watch",
            explanation=(
                "Premium is positive, but RV3 is still above RV30. The setup may be forming, yet the realized-volatility "
                "shock is still active and deserves more caution than a post-shock state."
            ),
            reasons=tuple(reasons),
        )

    # A positive premium with a calm/flat short-horizon RV state deserves
    # investigation, but without the recent shock -> cooling transition it is
    # not promoted to the post-shock classification.
    spread_text = f"{100 * vol_spread:+.1f} vol points" if np.isfinite(vol_spread) else "positive"
    return VRPCandidate(
        label="VRP candidate",
        level="candidate",
        explanation=(
            f"Implied volatility is {spread_text} above trailing integrated RV. The premium is present, "
            "but the stronger post-shock transition is not currently confirmed."
        ),
        reasons=tuple(reasons),
    )


def build_surface_mfiv_comparison(
    chain: pd.DataFrame,
    *,
    target_days: float = 30.0,
    dte_range: tuple[float, float] = (7.0, 180.0),
    min_strikes: int = 8,
    fengler_lambda: float = 1e-5,
    models: Iterable[str] = ("SSVI", "eSSVI", "Fengler"),
    fengler_mode: str = "fast",
    fengler_max_maturities: int = 5,
    fengler_max_strikes: int | None = 60,
    fengler_calendar_grid: int = 61,
) -> dict[str, SurfaceMFIVResult]:
    """Fit selected surface models and integrate their smoothed option prices.

    Models are integrated on each expiry's observed strike support, then
    converted to a constant tenor in total-variance time.  This makes their
    MFIV directly comparable with the raw-strip calculation.
    """
    from .data.clean import CleanConfig, clean_chain
    from .data.pipeline import build_all_slices
    from .svi import calibrate_svi
    from .ssvi import calibrate_ssvi
    from .essvi import calibrate_essvi
    from .fengler import fit_fengler_surface, prepare_fengler_slices
    from .mfiv import mfiv_from_model

    requested = {str(x).strip().lower() for x in models}
    valid = {"ssvi", "essvi", "fengler"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"unknown surface model(s): {sorted(unknown)}")
    if not requested:
        return {}

    clean, _ = clean_chain(
        chain,
        CleanConfig(dte_range=dte_range, require_activity=False),
        verbose=False,
    )
    slices = build_all_slices(clean, verbose=False)
    if len(slices) < 3:
        raise ValueError("need at least three calibratable expiries for SSVI/Fengler")

    pairs = None
    if requested & {"ssvi", "essvi"}:
        pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]

    def frame(ss):
        return pd.DataFrame([{
            "expiry": x.expiry, "dte": x.dte,
            "implied_variance": x.implied_variance,
            "implied_volatility": x.implied_volatility,
            "n_strikes": x.n_strikes,
        } for x in ss]).sort_values("dte").reset_index(drop=True)

    out: dict[str, SurfaceMFIVResult] = {}

    if "ssvi" in requested:
        ssvi_anchor_mode = "reliable raw-SVI anchors"
        try:
            ssvi = calibrate_ssvi(pairs, reliable_only=True)
        except ValueError:
            ssvi = calibrate_ssvi(pairs, reliable_only=False)
            ssvi_anchor_mode = "successful raw-SVI anchors (diagnostic fallback)"
        ssvi_slices = []
        for s in slices:
            strikes = clean.loc[clean["expiry"] == s.expiry, "strike"].unique()
            try:
                ssvi_slices.append(mfiv_from_model(
                    expiry=s.expiry,
                    T=s.T,
                    forward_fit=s.forward_fit,
                    strikes=strikes,
                    implied_vol_fn=lambda k, T=s.T: ssvi.implied_vol(k, T),
                    label="ssvi",
                    min_strikes=min_strikes,
                ))
            except ValueError:
                continue
        out["SSVI"] = SurfaceMFIVResult(
            "SSVI", constant_tenor_mfiv(ssvi_slices, target_days), frame(ssvi_slices),
            bool(ssvi.is_reliable and ssvi_anchor_mode == "reliable raw-SVI anchors"),
            float(ssvi.rmse_iv),
            f"butterfly={ssvi.butterfly_free}, calendar={ssvi.calendar_free}, "
            f"theta repair={ssvi.theta_repair_fraction:.2%}; {ssvi_anchor_mode}",
        )

    if "essvi" in requested:
        essvi_anchor_mode = "reliable raw-SVI anchors"
        try:
            essvi = calibrate_essvi(pairs, reliable_only=True, n_restarts=5)
        except ValueError:
            essvi = calibrate_essvi(pairs, reliable_only=False, n_restarts=5)
            essvi_anchor_mode = "successful raw-SVI anchors (diagnostic fallback)"
        essvi_slices = []
        for s in slices:
            strikes = clean.loc[clean["expiry"] == s.expiry, "strike"].unique()
            try:
                essvi_slices.append(mfiv_from_model(
                    expiry=s.expiry,
                    T=s.T,
                    forward_fit=s.forward_fit,
                    strikes=strikes,
                    implied_vol_fn=lambda k, T=s.T: essvi.implied_vol(k, T),
                    label="essvi",
                    min_strikes=min_strikes,
                ))
            except ValueError:
                continue
        out["eSSVI"] = SurfaceMFIVResult(
            "eSSVI", constant_tenor_mfiv(essvi_slices, target_days), frame(essvi_slices),
            bool(essvi.is_reliable and essvi_anchor_mode == "reliable raw-SVI anchors"),
            float(essvi.rmse_iv),
            f"butterfly={essvi.butterfly_free}, calendar={essvi.calendar_free}, "
            f"theta repair={essvi.theta_repair_fraction:.2%}, "
            f"rho0={essvi.params.rho0:+.3f}, rho_m={essvi.params.rho_m:+.3f}; {essvi_anchor_mode}",
        )

    if "fengler" in requested:
        mode = str(fengler_mode).strip().lower()
        if mode not in {"fast", "expanded", "full"}:
            raise ValueError("fengler_mode must be 'fast', 'expanded', or 'full'")
        if mode == "full":
            fengler_inputs = slices
        else:
            fengler_inputs = prepare_fengler_slices(
                slices,
                target_days=target_days,
                max_maturities=int(fengler_max_maturities),
                max_strikes_per_slice=fengler_max_strikes,
            )
        fengler = fit_fengler_surface(
            fengler_inputs,
            smoothing_lambda=fengler_lambda,
            calendar_grid_size=int(fengler_calendar_grid),
            solver="auto",
            solver_tol=1e-9,
        )
        fengler_slices = []
        originals = sorted(slices, key=lambda s: s.T)
        for fit in fengler.slices:
            s = min(originals, key=lambda candidate: abs(float(candidate.T) - float(fit.T)))
            strikes = clean.loc[clean["expiry"] == s.expiry, "strike"].unique()
            try:
                fengler_slices.append(mfiv_from_model(
                    expiry=s.expiry,
                    T=s.T,
                    forward_fit=s.forward_fit,
                    strikes=strikes,
                    implied_vol_fn=lambda k, f=fit: f.implied_vol(k, allow_extrapolation=False),
                    label="fengler",
                    min_strikes=min_strikes,
                ))
            except ValueError:
                continue
        out["Fengler"] = SurfaceMFIVResult(
            "Fengler", constant_tenor_mfiv(fengler_slices, target_days), frame(fengler_slices),
            bool(fengler.is_reliable), float(fengler.rmse_iv),
            f"strike-arb={fengler.butterfly_free}, calendar={fengler.calendar_free}; "
            f"mode={mode}, solver={fengler.solver}, {fengler.elapsed_seconds:.2f}s, "
            f"{len(fengler.slices)} maturities, grid={int(fengler_calendar_grid)}",
        )

    return out


def build_surface_explorer(
    chain: pd.DataFrame,
    *,
    model: str = "SVI",
    dte_range: tuple[float, float] = (7.0, 180.0),
    tenor_count: int = 12,
    k_range: tuple[float, float] = (-0.25, 0.25),
    k_points: int = 41,
    fengler_mode: str = "fast",
    fengler_max_maturities: int = 5,
    fengler_max_strikes: int | None = 60,
    fengler_calendar_grid: int = 61,
    price_side: str = "mid",
) -> SurfaceExplorerData:
    """Build the fitted IV surface used by the dashboard Surface Explorer.

    The function deliberately reuses VolForge's production surface objects.
    Raw observations remain visible alongside the fit so a smooth model cannot
    hide sparse/bad quotes.  SVI is the fast default; SSVI/eSSVI/Fengler are
    optional global/arbitrage-aware alternatives.
    """
    from .data.clean import CleanConfig, clean_chain
    from .data.pipeline import build_all_slices
    from .svi import calibrate_svi
    from .surface import build_surface
    from .ssvi import calibrate_ssvi
    from .essvi import calibrate_essvi
    from .fengler import fit_fengler_surface, prepare_fengler_slices

    name = str(model).strip()
    key = name.lower()
    aliases = {"svi": "SVI", "ssvi": "SSVI", "essvi": "eSSVI", "fengler": "Fengler"}
    if key not in aliases:
        raise ValueError("model must be one of SVI, SSVI, eSSVI, Fengler")
    name = aliases[key]

    clean, _ = clean_chain(
        chain,
        CleanConfig(dte_range=dte_range, require_activity=False),
        verbose=False,
    )
    slices = build_all_slices(clean, verbose=False)
    if len(slices) < 2:
        raise ValueError("need at least two calibratable expiries for the surface explorer")

    raw_rows = []
    atm_rows = []
    for slc in slices:
        for k, strike, iv in zip(slc.k, slc.strikes, slc.iv):
            raw_rows.append({
                "expiry": slc.expiry,
                "dte": float(slc.dte),
                "k": float(k),
                "strike": float(strike),
                "iv": float(iv),
            })
        j = int(np.argmin(np.abs(slc.k)))
        atm_rows.append({
            "expiry": slc.expiry,
            "dte": float(slc.dte),
            "raw_atm_iv": float(slc.iv[j]),
            "raw_atm_k": float(slc.k[j]),
        })
    raw_points = pd.DataFrame(raw_rows).sort_values(["dte", "k"]).reset_index(drop=True)
    raw_atm = pd.DataFrame(atm_rows).sort_values("dte").reset_index(drop=True)

    observed_dtes = np.array(sorted(float(s.dte) for s in slices), dtype=float)
    lo_dte, hi_dte = observed_dtes.min(), observed_dtes.max()
    n_tenor = max(3, int(tenor_count))
    if len(observed_dtes) <= n_tenor:
        tenors = observed_dtes
    else:
        tenors = np.linspace(lo_dte, hi_dte, n_tenor)
    k_lo, k_hi = map(float, k_range)
    if not np.isfinite(k_lo) or not np.isfinite(k_hi) or k_hi <= k_lo:
        raise ValueError("k_range must be finite and increasing")
    k_grid = np.linspace(k_lo, k_hi, max(9, int(k_points)))

    pairs = [(s, calibrate_svi(s.k, s.w, s.T, weights=s.weights)) for s in slices]
    reliable = False
    rmse_iv = np.nan
    detail = ""
    trade_date = pd.to_datetime(clean["quote_time"], utc=True).max().date()
    symbol = str(clean["symbol"].iloc[0])

    if name == "SVI":
        try:
            surface = build_surface(
                pairs, trade_date=trade_date, symbol=symbol, tenor_days=tenors,
                k_grid=k_grid, reliable_only=True, repair=True,
            )
            used_reliable = True
        except ValueError:
            surface = build_surface(
                pairs, trade_date=trade_date, symbol=symbol, tenor_days=tenors,
                k_grid=k_grid, reliable_only=False, repair=True,
            )
            used_reliable = False
        ok = [f for _, f in pairs if f.is_reliable]
        reliable = bool(used_reliable and len(ok) >= 2 and surface.calendar_repair < 1e-5)
        rmse_iv = float(np.mean([f.rmse_iv for _, f in pairs])) if pairs else np.nan
        detail = (
            f"{len(ok)}/{len(pairs)} reliable raw-SVI slices; "
            f"calendar repair={surface.calendar_repair:.3g}"
        )

    elif name == "SSVI":
        anchor_mode = "reliable raw-SVI anchors"
        try:
            fit = calibrate_ssvi(pairs, reliable_only=True)
        except ValueError:
            fit = calibrate_ssvi(pairs, reliable_only=False)
            anchor_mode = "successful raw-SVI anchors (diagnostic fallback)"
        surface = fit.to_surface(trade_date, symbol=symbol, tenor_days=tenors, k_grid=k_grid)
        reliable = bool(fit.is_reliable and anchor_mode.startswith("reliable"))
        rmse_iv = float(fit.rmse_iv)
        detail = (
            f"butterfly={fit.butterfly_free}, calendar={fit.calendar_free}, "
            f"theta repair={fit.theta_repair_fraction:.2%}; {anchor_mode}"
        )

    elif name == "eSSVI":
        anchor_mode = "reliable raw-SVI anchors"
        try:
            fit = calibrate_essvi(pairs, reliable_only=True, n_restarts=5)
        except ValueError:
            fit = calibrate_essvi(pairs, reliable_only=False, n_restarts=5)
            anchor_mode = "successful raw-SVI anchors (diagnostic fallback)"
        surface = fit.to_surface(trade_date, symbol=symbol, tenor_days=tenors, k_grid=k_grid)
        reliable = bool(fit.is_reliable and anchor_mode.startswith("reliable"))
        rmse_iv = float(fit.rmse_iv)
        detail = (
            f"butterfly={fit.butterfly_free}, calendar={fit.calendar_free}, "
            f"rho0={fit.params.rho0:+.3f}, rho_m={fit.params.rho_m:+.3f}; {anchor_mode}"
        )

    else:
        mode = str(fengler_mode).strip().lower()
        if mode not in {"fast", "expanded", "full"}:
            raise ValueError("fengler_mode must be 'fast', 'expanded', or 'full'")
        f_inputs = slices if mode == "full" else prepare_fengler_slices(
            slices,
            target_days=30.0,
            max_maturities=int(fengler_max_maturities),
            max_strikes_per_slice=fengler_max_strikes,
        )
        fit = fit_fengler_surface(
            f_inputs,
            calendar_grid_size=int(fengler_calendar_grid),
            solver="auto",
            solver_tol=1e-9,
        )
        # Restrict the display tenor range to maturities Fengler actually fit.
        f_dtes = np.array([float(x.dte) for x in fit.slices])
        f_tenors = tenors[(tenors >= f_dtes.min()) & (tenors <= f_dtes.max())]
        if len(f_tenors) < 3:
            f_tenors = np.linspace(f_dtes.min(), f_dtes.max(), min(max(3, len(f_dtes)), n_tenor))
        surface = fit.to_surface(trade_date, symbol=symbol, tenor_days=f_tenors, k_grid=k_grid)
        reliable = bool(fit.is_reliable)
        rmse_iv = float(fit.rmse_iv)
        detail = (
            f"strike-arb={fit.butterfly_free}, calendar={fit.calendar_free}; "
            f"mode={mode}, solver={fit.solver}, {fit.elapsed_seconds:.2f}s"
        )

    raw_mfiv = mfiv_term_structure(clean, price_side=price_side)
    mfiv_curve = pd.DataFrame([{
        "expiry": x.expiry,
        "dte": float(x.dte),
        "mfiv": float(x.implied_volatility),
        "implied_variance": float(x.implied_variance),
    } for x in raw_mfiv]).sort_values("dte").reset_index(drop=True)

    return SurfaceExplorerData(
        model=name,
        surface=surface,
        raw_points=raw_points,
        raw_atm_term=raw_atm,
        mfiv_curve=mfiv_curve,
        reliable=reliable,
        rmse_iv=rmse_iv,
        detail=detail,
    )

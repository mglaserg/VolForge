"""Streamlit dashboard for VolForge's Forward VRP measurement layer.

Run from the repository root with:

    streamlit run volforge_dashboard.py

This app is deliberately diagnostic.  It does not issue trade recommendations
and it does not confuse the live MFIV-minus-trailing-RV feature with the true
forward VRP label, which only exists after the future realization window.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from volforge.dashboard import (
    build_dashboard_snapshot,
    build_surface_mfiv_comparison,
    build_surface_explorer,
    classify_vrp_candidate,
    classify_vrp_context,
    normalise_intraday_bars,
    prepare_vrp_history,
)
from volforge.data.provider import available_providers, fetch_chain
from volforge.data.storage import (
    list_chain_snapshots,
    load_chain_snapshot,
    save_chain_snapshot,
    select_daily_snapshots,
)
from volforge.history import VRPHistoryConfig, build_vrp_history, load_daily_variance, save_vrp_history
from volforge.realized import daily_integrated_variance
from volforge.delta_surface import (
    build_delta_surface, constant_tenor_delta_slice, delta_lump_scores,
    delta_ratio_term_structure,
)


st.set_page_config(page_title="VolForge · Forward VRP", page_icon="〽", layout="wide")


def _render_demo_page():
    st.title("VolForge · How to Use Forward VRP")
    st.caption("A practical reading guide. These states are research context, not automatic trade signals.")

    st.subheader("The question VolForge is trying to answer")
    st.latex(r"\mathrm{Expected\ VRP}_{30} = \mathrm{MFIV}_{30}^{2} - \widehat{RV}_{t\rightarrow t+30}^{2}")
    st.write(
        "Today the live dashboard compares MFIV with **trailing** integrated RV. Later, the forecasting layer "
        "will replace trailing RV with a genuine forward-RV forecast. The RV3/RV30 slope tells you what the "
        "realized-volatility regime is doing; it is not the edge by itself."
    )

    st.subheader("The shock → cooling → lingering-premium pattern")
    demo = pd.DataFrame({
        "Stage": ["Shock underway", "Cooling", "Post-shock / IV still elevated"],
        "RV3": [0.40, 0.27, 0.16],
        "RV30": [0.22, 0.24, 0.23],
        "MFIV30": [0.35, 0.34, 0.31],
    })
    demo["RV3 − RV30"] = demo["RV3"] - demo["RV30"]
    chart = demo.set_index("Stage")[["RV3", "RV30", "MFIV30"]] * 100
    st.line_chart(chart)
    shown = demo.copy()
    for c in ["RV3", "RV30", "MFIV30", "RV3 − RV30"]:
        shown[c] = (100 * shown[c]).map(lambda x: f"{x:.1f}%")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.info(
        "The third state is the one to investigate most closely: RV3 has cooled below RV30 after a recent shock, "
        "but MFIV30 is still high. That can mean insurance premiums are staying elevated after realized risk has begun to normalize."
    )

    st.subheader("How to read a live symbol")
    st.markdown(
        """
1. **Premium:** Is MFIV30 above trailing RV30, and by a meaningful amount?
2. **Stretch:** With history loaded, is VRP unusually high by z-score/percentile?
3. **Regime:** Is RV3 above RV30 (shock active), falling through RV30 (cooling), or simply calm?
4. **Persistence:** Did RV3 recently spike above RV30 before falling below it? That transition is more informative than either sign alone.
5. **Surface quality:** Compare raw-strip MFIV with SSVI and Fengler. Small differences are reassuring; large differences tell you to inspect quotes/surface fit before trusting the VRP reading.
6. **No automatic trade:** A strong context still needs portfolio/tail-risk rules and, later, the forward-RV model.
"""
    )

    st.subheader("What the VRP-candidate label means")
    st.write(
        "VolForge now promotes the regime summary into a simple research classification: **Not a VRP candidate**, "
        "**Developing VRP candidate**, **VRP candidate**, or **Post-shock VRP candidate**. The label is intentionally "
        "not a trade instruction. It requires a positive implied-vs-trailing-RV premium and then uses the RV3/RV30 "
        "state to explain whether the shock is still active, simply calm, or cooling after a recent spike."
    )

    st.subheader("RV3 / RV30 context matrix")
    matrix = pd.DataFrame([
        {"RV state": "RV3 > RV30", "MFIV state": "High", "Read": "Shock active; premium may be forming, but realized risk is still hot."},
        {"RV state": "RV3 > RV30", "MFIV state": "Not high", "Read": "Poor compensation for an active shock."},
        {"RV state": "RV3 < RV30 after recent positive slope", "MFIV state": "Still high", "Read": "Post-shock / lingering-premium candidate; especially worth researching."},
        {"RV state": "RV3 < RV30", "MFIV state": "Low", "Read": "Calm market, but probably little premium to harvest."},
    ])
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.subheader("Why compare Raw, SSVI, eSSVI, and Fengler?")
    st.write(
        "Raw-strip MFIV uses the observed option quotes directly. SSVI gives a global parametric, static-arbitrage-aware "
        "surface; eSSVI lets correlation vary with maturity; Fengler smooths in option-price space under "
        "convexity/monotonicity/calendar constraints. VolForge integrates "
        "each model over the **same observed strike support** so differences reflect smoothing/model quality rather than a different wing domain."
    )
    st.caption("If a model is flagged unreliable, do not use its smoothed MFIV as confirmation. Treat the disagreement itself as a diagnostic.")


def _pct(x: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _var_pts(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{100 * x:+.2f}"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_chain_cached(symbol: str, provider: str, max_expiries: int, dte_lo: int, dte_hi: int):
    return fetch_chain(
        symbol,
        provider=provider,
        max_expiries=max_expiries,
        dte_range=(float(dte_lo), float(dte_hi)),
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_yahoo_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=True,
        prepost=False,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no {interval} bars for {symbol}")
    return normalise_intraday_bars(raw)


def _surface_focus_dte_range(
    chain: pd.DataFrame,
    *,
    target_days: float,
    user_range: tuple[float, float],
) -> tuple[float, float]:
    """Return a tight expiry window around the target tenor.

    Surface confirmation only needs enough maturities to bracket the target and
    identify the local term structure.  Restricting the fit to nearby expiries
    avoids recalibrating the whole 7--180d surface for a 30d diagnostic.
    """
    lo, hi = map(float, user_range)
    dtes = pd.to_numeric(chain.get("dte"), errors="coerce")
    dtes = np.array(sorted(set(float(x) for x in dtes.dropna() if lo <= float(x) <= hi)))
    if len(dtes) <= 5:
        return lo, hi

    target = float(target_days)
    below = dtes[dtes < target]
    above = dtes[dtes > target]
    exact = dtes[np.isclose(dtes, target, atol=1e-8)]

    chosen: list[float] = []
    chosen.extend(below[-2:].tolist())
    chosen.extend(exact[:1].tolist())
    chosen.extend(above[:2].tolist())

    # SSVI/Fengler need at least three calibratable maturities. Fill from the
    # nearest expiries if one side of the target is sparse.
    if len(set(chosen)) < 3:
        nearest = dtes[np.argsort(np.abs(dtes - target))]
        for dte in nearest:
            if float(dte) not in chosen:
                chosen.append(float(dte))
            if len(set(chosen)) >= 3:
                break

    chosen = sorted(set(chosen))
    if len(chosen) < 3:
        return lo, hi
    return max(lo, chosen[0] - 1e-6), min(hi, chosen[-1] + 1e-6)


@st.cache_data(show_spinner=False, max_entries=16)
def _build_surface_comparison_cached(
    chain: pd.DataFrame,
    target_days: float,
    fit_lo: float,
    fit_hi: float,
    models: tuple[str, ...],
    fengler_mode: str,
    fengler_max_maturities: int,
    fengler_max_strikes: int,
    fengler_calendar_grid: int,
):
    """Cache expensive surface fits by option-chain snapshot/configuration."""
    return build_surface_mfiv_comparison(
        chain,
        target_days=float(target_days),
        dte_range=(float(fit_lo), float(fit_hi)),
        models=models,
        fengler_mode=fengler_mode,
        fengler_max_maturities=int(fengler_max_maturities),
        fengler_max_strikes=(None if int(fengler_max_strikes) <= 0 else int(fengler_max_strikes)),
        fengler_calendar_grid=int(fengler_calendar_grid),
    )


@st.cache_data(show_spinner=False, max_entries=12)
def _build_delta_surface_cached(
    chain: pd.DataFrame,
    dte_lo: float,
    dte_hi: float,
    target_days: float,
):
    surface = build_delta_surface(
        chain,
        dte_range=(float(dte_lo), float(dte_hi)),
        require_activity=False,
    )
    target = constant_tenor_delta_slice(surface, float(target_days))
    ratios = delta_ratio_term_structure(surface)
    lumps = delta_lump_scores(surface)
    return surface, target, ratios, lumps


@st.cache_data(show_spinner=False, max_entries=12)
def _build_surface_explorer_cached(
    chain: pd.DataFrame,
    model: str,
    dte_lo: float,
    dte_hi: float,
    tenor_count: int,
    k_lo: float,
    k_hi: float,
    k_points: int,
    fengler_mode: str,
    fengler_max_maturities: int,
    fengler_max_strikes: int,
    fengler_calendar_grid: int,
):
    return build_surface_explorer(
        chain,
        model=model,
        dte_range=(float(dte_lo), float(dte_hi)),
        tenor_count=int(tenor_count),
        k_range=(float(k_lo), float(k_hi)),
        k_points=int(k_points),
        fengler_mode=fengler_mode,
        fengler_max_maturities=int(fengler_max_maturities),
        fengler_max_strikes=(None if int(fengler_max_strikes) <= 0 else int(fengler_max_strikes)),
        fengler_calendar_grid=int(fengler_calendar_grid),
    )


def _read_table_from_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    payload = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(payload))
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(BytesIO(payload))
    raise ValueError("upload must be CSV or Parquet")


def _read_table_from_path(path_text: str) -> pd.DataFrame:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("local file must be CSV or Parquet")


def _render_surface_explorer_page():
    st.title("VolForge · Surface Explorer")
    st.caption("Inspect the fitted volatility surface, ATM/MFIV term structures, and individual smile curves from one chain snapshot.")

    with st.sidebar:
        st.header("Surface Explorer")
        sx_symbol = st.text_input("Symbol", "SPY", key="surface_symbol").strip().upper()
        sx_provider = st.selectbox("Provider", available_providers(), key="surface_provider")
        sx_source = st.radio("Chain source", ("Fetch current", "Latest saved snapshot", "Local snapshot path"), key="surface_source")
        sx_chain_root = st.text_input("Chain archive root", "data/chains", key="surface_chain_root")
        sx_local_path = ""
        if sx_source == "Local snapshot path":
            sx_local_path = st.text_input("chain.parquet path", "", key="surface_local_path")
        sx_dte_lo, sx_dte_hi = st.slider("DTE range", 1, 365, (7, 180), key="surface_dte")
        sx_max_exp = st.number_input("Max expiries when fetching", 2, 40, 16, key="surface_max_exp")
        sx_model = st.selectbox("Surface model", ("SVI", "SSVI", "eSSVI", "Fengler"), index=0, key="surface_model")
        sx_tenors = st.slider("Display tenor rows", 5, 24, 12, key="surface_tenors")
        sx_k_lo, sx_k_hi = st.slider("Log-moneyness range", -0.60, 0.60, (-0.25, 0.25), step=0.01, key="surface_k_range")
        sx_k_points = st.slider("Curve grid points", 17, 101, 41, step=2, key="surface_k_points")
        sx_delta_target = st.number_input(
            "Delta-surface constant tenor", 7, 180, 30, step=1, key="surface_delta_target",
            help="The RW-style delta smile is interpolated in total-variance time at this maturity.",
        )

        sx_fengler_mode = "fast"
        sx_fengler_mats, sx_fengler_strikes, sx_fengler_grid = 5, 60, 61
        if sx_model == "Fengler":
            scope = st.selectbox("Fengler scope", ("Fast", "Expanded", "Full research"), key="surface_fengler_scope")
            if scope == "Expanded":
                sx_fengler_mode, sx_fengler_mats, sx_fengler_strikes, sx_fengler_grid = "expanded", 9, 90, 101
            elif scope == "Full research":
                sx_fengler_mode, sx_fengler_mats, sx_fengler_strikes, sx_fengler_grid = "full", 999, 0, 181
        sx_run = st.button("Load / fit surface", type="primary", use_container_width=True, key="surface_run")

    if not sx_run and "surface_has_run" not in st.session_state:
        st.info("Choose a chain source and model, then click **Load / fit surface**.")
        return
    if sx_run:
        st.session_state["surface_has_run"] = True

    try:
        with st.spinner(f"Loading chain and fitting {sx_model}…"):
            if sx_source == "Fetch current":
                sx_chain = _fetch_chain_cached(sx_symbol, sx_provider, int(sx_max_exp), int(sx_dte_lo), int(sx_dte_hi))
            elif sx_source == "Latest saved snapshot":
                refs = list_chain_snapshots(sx_symbol, provider=sx_provider, root=sx_chain_root, include_legacy_yahoo=True)
                refs = select_daily_snapshots(refs, policy="latest")
                if not refs:
                    raise FileNotFoundError(f"No saved {sx_provider} snapshots found for {sx_symbol} under {sx_chain_root}")
                sx_chain = load_chain_snapshot(refs[-1])
            else:
                if not sx_local_path.strip():
                    raise ValueError("Enter a local chain.parquet path")
                sx_chain = load_chain_snapshot(Path(sx_local_path).expanduser())

            explorer = _build_surface_explorer_cached(
                sx_chain, sx_model, float(sx_dte_lo), float(sx_dte_hi), int(sx_tenors),
                float(sx_k_lo), float(sx_k_hi), int(sx_k_points), sx_fengler_mode,
                int(sx_fengler_mats), int(sx_fengler_strikes), int(sx_fengler_grid),
            )
            delta_surface, delta_target, delta_ratios, delta_lumps = _build_delta_surface_cached(
                sx_chain, float(sx_dte_lo), float(sx_dte_hi), float(sx_delta_target)
            )
    except Exception as exc:
        st.error(f"Could not build surface: {exc}")
        return

    quote = pd.to_datetime(sx_chain["quote_time"], utc=True, errors="coerce").max().tz_convert("America/New_York")
    st.caption(f"{sx_symbol} · {quote:%Y-%m-%d %H:%M %Z} · {sx_provider} · {explorer.model}")
    a, b, c, d = st.columns(4)
    a.metric("Model", explorer.model)
    b.metric("Reliable", "Yes" if explorer.reliable else "Diagnostic")
    c.metric("Fit RMSE", _pct(explorer.rmse_iv, 2))
    d.metric("Surface tenors", str(len(explorer.surface.tenor_days)))
    st.caption(explorer.detail)

    surface_tab, term_tab, curve_tab, delta_tab, ratio_tab, points_tab = st.tabs((
        "Surface", "Term structure", "Smile / curve", "Delta surface", "Delta ratios", "Raw IV points"
    ))

    with surface_tab:
        st.subheader(f"{explorer.model} implied-volatility surface")
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Surface(
                x=explorer.surface.k_grid,
                y=explorer.surface.tenor_days,
                z=100 * explorer.surface.iv,
            )])
            fig.update_layout(
                scene={"xaxis_title": "Log-moneyness k", "yaxis_title": "DTE", "zaxis_title": "IV (%)"},
                margin={"l": 0, "r": 0, "t": 25, "b": 0},
                height=620,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info('Install the `viz` extra for the interactive 3D surface: `pip install -e ".[dashboard,data,viz]"`.')
            surface_frame = explorer.surface.to_frame() * 100
            st.dataframe(surface_frame.round(2), use_container_width=True)
        st.caption("Rows are tenor/DTE; columns are log-moneyness. Values are implied volatility in percent.")

    with term_tab:
        st.subheader("ATM volatility term structure")
        model_term = pd.DataFrame({
            "dte": explorer.surface.tenor_days,
            f"{explorer.model} ATM": 100 * explorer.surface.iv[:, int(np.argmin(np.abs(explorer.surface.k_grid)))],
        }).set_index("dte")
        raw_atm = explorer.raw_atm_term.set_index("dte")[["raw_atm_iv"]].rename(columns={"raw_atm_iv": "Observed near-ATM"}) * 100
        atm_plot = model_term.join(raw_atm, how="outer").sort_index()
        st.line_chart(atm_plot)
        st.dataframe(atm_plot.round(2), use_container_width=True)

        st.subheader("Model-free implied-volatility term structure")
        mfiv = explorer.mfiv_curve.set_index("dte")[["mfiv"]].rename(columns={"mfiv": "Raw-strip MFIV"}) * 100
        st.line_chart(mfiv)
        st.dataframe(mfiv.round(2), use_container_width=True)
        st.caption("ATM term structure describes the center of the fitted surface; MFIV integrates the full OTM strip and is the variance measure used by the VRP workflow.")

    with curve_tab:
        st.subheader("Smile / skew curve")
        tenors = [float(x) for x in explorer.surface.tenor_days]
        selected_tenor = st.selectbox("Display tenor", tenors, index=int(np.argmin(np.abs(np.asarray(tenors) - 30.0))), format_func=lambda x: f"{x:.1f} DTE")
        i = int(np.argmin(np.abs(explorer.surface.tenor_days - float(selected_tenor))))
        model_curve = pd.DataFrame({"k": explorer.surface.k_grid, explorer.model: 100 * explorer.surface.iv[i]}).set_index("k")
        raw_dtes = explorer.raw_points["dte"].drop_duplicates().to_numpy(float)
        nearest_raw = float(raw_dtes[np.argmin(np.abs(raw_dtes - float(selected_tenor)))])
        raw_curve = explorer.raw_points[np.isclose(explorer.raw_points["dte"], nearest_raw)][["k", "iv"]].copy()
        raw_curve["Observed IV"] = 100 * raw_curve.pop("iv")
        raw_curve = raw_curve.set_index("k")
        st.line_chart(model_curve.join(raw_curve, how="outer").sort_index())
        st.caption(f"Observed points shown from the nearest actual expiry: {nearest_raw:.1f} DTE.")
        curve_table = explorer.raw_points[np.isclose(explorer.raw_points["dte"], nearest_raw)][["strike", "k", "iv"]].copy()
        curve_table["iv"] *= 100
        st.dataframe(curve_table.rename(columns={"iv": "observed_iv_pct"}), use_container_width=True, hide_index=True)

    with delta_tab:
        st.subheader("Delta volatility surface")
        st.write(
            "Model-light view: VolForge computes its own IVs and spot deltas, interpolates to standard "
            "10Δ / 15Δ / 25Δ put and call buckets, and keeps ATM as the center. No SVI-family fit is required."
        )
        delta_frame = delta_surface.display_frame() * 100
        pretty = {
            "iv_10p": "10Δ put", "iv_15p": "15Δ put", "iv_25p": "25Δ put",
            "atm_iv": "ATM", "iv_25c": "25Δ call", "iv_15c": "15Δ call", "iv_10c": "10Δ call",
        }
        shown_delta = delta_frame.rename(columns=pretty)
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Heatmap(
                x=list(shown_delta.columns),
                y=[float(x) for x in shown_delta.index],
                z=shown_delta.to_numpy(float),
                colorbar={"title": "IV (%)"},
            ))
            fig.update_layout(xaxis_title="Delta bucket", yaxis_title="DTE", height=520, margin={"l": 0, "r": 0, "t": 25, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(shown_delta.round(2), use_container_width=True)

        target_rows = []
        for key, label in pretty.items():
            value = float(delta_target.get(key, np.nan))
            ratio_key = key.replace("iv_", "delta_ratio_") if key != "atm_iv" else None
            target_rows.append({
                "bucket": label,
                "iv_pct": 100 * value if np.isfinite(value) else np.nan,
                "ratio_to_atm": float(delta_target.get(ratio_key, np.nan)) if ratio_key else 1.0,
            })
        st.markdown(f"**{float(sx_delta_target):.0f}-day constant-maturity delta smile**")
        st.dataframe(pd.DataFrame(target_rows), use_container_width=True, hide_index=True)
        st.caption("Maturity interpolation is linear in total variance (IV² × time), not directly in IV.")

    with ratio_tab:
        st.subheader("Delta ratios · the volatility compass")
        ratio_cols = [c for c in delta_ratios.columns if c.startswith("delta_ratio_")]
        if ratio_cols:
            plot = delta_ratios.set_index("dte")[ratio_cols].rename(columns={
                "delta_ratio_10p": "10Δ put / ATM",
                "delta_ratio_15p": "15Δ put / ATM",
                "delta_ratio_25p": "25Δ put / ATM",
                "delta_ratio_25c": "25Δ call / ATM",
                "delta_ratio_15c": "15Δ call / ATM",
                "delta_ratio_10c": "10Δ call / ATM",
            })
            st.line_chart(plot)
            st.dataframe(delta_ratios.round(4), use_container_width=True, hide_index=True)

        st.markdown("**Local term-structure lumps**")
        st.write(
            "Each interior expiry is compared with the straight line through its immediate neighbors. "
            "Large residuals flag a local shape dislocation; historical z-scores are tracked separately in VRP history."
        )
        st.dataframe(delta_lumps.round(4), use_container_width=True, hide_index=True)

    with points_tab:
        pts = explorer.raw_points.copy()
        pts["iv"] *= 100
        st.dataframe(pts.rename(columns={"iv": "iv_pct"}), use_container_width=True, hide_index=True)


page = st.sidebar.radio("Page", ("Forward VRP dashboard", "Surface explorer", "How to use"), index=0)
if page == "How to use":
    _render_demo_page()
    st.stop()
if page == "Surface explorer":
    _render_surface_explorer_page()
    st.stop()


st.title("VolForge · Forward VRP")
st.caption(
    "Measurement dashboard: full-strip model-free implied variance versus high-frequency "
    "integrated realized variance. The live VRP value below is a trailing feature, not the "
    "future 30-day training label."
)

with st.sidebar:
    st.header("Snapshot")
    symbol = st.text_input("Symbol", "SPY").strip().upper()
    provider = st.selectbox("Option provider", available_providers(), index=0)
    price_side = st.radio("Raw MFIV quote side", ("mid", "bid"), horizontal=True)
    target_days = st.number_input("Constant tenor (days)", min_value=7, max_value=180, value=30, step=1)
    dte_lo, dte_hi = st.slider("Option DTE range", min_value=1, max_value=365, value=(7, 180))
    max_expiries = st.number_input("Max expiries", min_value=2, max_value=40, value=16, step=1)

    st.divider()
    st.header("Integrated RV")
    bar_source = st.radio(
        "Intraday source",
        ("Yahoo recent bars (preview)", "Local CSV / Parquet"),
    )
    uploaded_bars = None
    local_bar_path = ""
    if bar_source.startswith("Yahoo"):
        bar_interval = st.selectbox("Bar interval", ("5m", "15m"), index=0)
        bar_period = st.selectbox("History", ("60d",), index=0)
        st.caption("Useful for live diagnostics; not a substitute for research-grade historical HF data.")
    else:
        uploaded_bars = st.file_uploader("Upload intraday bars", type=["csv", "parquet", "pq"])
        local_bar_path = st.text_input(
            "…or local path",
            value=f"data/intraday/{symbol}.parquet",
        )
        st.caption("Expected: timestamp + close, or a DatetimeIndex + Close column.")

    run = st.button("Run / refresh", type="primary", use_container_width=True)

if not run and "vrp_has_run" not in st.session_state:
    st.info("Choose the data sources in the sidebar and click **Run / refresh**.")
    st.stop()
if run:
    st.session_state["vrp_has_run"] = True

try:
    with st.spinner("Building MFIV and integrated-RV snapshot…"):
        chain = _fetch_chain_cached(symbol, provider, int(max_expiries), int(dte_lo), int(dte_hi))
        if bar_source.startswith("Yahoo"):
            bars = _fetch_yahoo_intraday(symbol, bar_interval, bar_period)
        elif uploaded_bars is not None:
            bars = normalise_intraday_bars(_read_table_from_upload(uploaded_bars))
        else:
            bars = normalise_intraday_bars(_read_table_from_path(local_bar_path))

        snapshot = build_dashboard_snapshot(
            chain,
            bars,
            target_days=float(target_days),
            price_side=price_side,
        )

except Exception as exc:
    st.error(f"Could not build the snapshot: {exc}")
    if provider == "orats":
        st.caption("ORATS requires ORATS_API_TOKEN in the environment and the appropriate data entitlement.")
    st.stop()

local_fit_lo, local_fit_hi = _surface_focus_dte_range(
    chain,
    target_days=float(target_days),
    user_range=(float(dte_lo), float(dte_hi)),
)

with st.sidebar:
    st.divider()
    st.header("Surface confirmation")
    st.caption(
        "Optional and expensive. Raw-strip MFIV remains the fast default; "
        "SSVI/eSSVI/Fengler only run when you ask for confirmation."
    )
    confirmation_models = tuple(st.multiselect(
        "Models",
        ("SSVI", "eSSVI", "Fengler"),
        default=("SSVI", "eSSVI"),
        help="Fengler is opt-in because it is the most expensive real-chain confirmation.",
    ))
    fengler_scope = "Fast"
    if "Fengler" in confirmation_models:
        fengler_scope = st.selectbox(
            "Fengler scope",
            ("Fast", "Expanded", "Full research"),
            index=0,
            help=(
                "Fast: ~5 maturities × 60 strikes near target tenor. Expanded: ~9 maturities × 90 strikes "
                "across the selected DTE range. Full research: all cleaned maturities/strikes; can take much longer."
            ),
        )

    if fengler_scope == "Fast":
        fit_lo, fit_hi = local_fit_lo, local_fit_hi
        fengler_mode = "fast"
        fengler_max_maturities, fengler_max_strikes, fengler_calendar_grid = 5, 60, 61
    elif fengler_scope == "Expanded":
        fit_lo, fit_hi = float(dte_lo), float(dte_hi)
        fengler_mode = "expanded"
        fengler_max_maturities, fengler_max_strikes, fengler_calendar_grid = 9, 90, 101
    else:
        fit_lo, fit_hi = float(dte_lo), float(dte_hi)
        fengler_mode = "full"
        fengler_max_maturities, fengler_max_strikes, fengler_calendar_grid = 999, 0, 181

    surface_key = (
        str(symbol), str(provider), str(snapshot.quote_time), float(target_days),
        float(fit_lo), float(fit_hi), str(price_side), tuple(confirmation_models),
        str(fengler_mode), int(fengler_max_maturities), int(fengler_max_strikes),
        int(fengler_calendar_grid),
    )
    confirmations = st.session_state.setdefault("surface_confirmations", {})
    surface_results = confirmations.get(surface_key, {})

    run_surface_confirmation = st.button(
        "Run selected confirmation",
        use_container_width=True,
        disabled=not confirmation_models,
        help="Runs only the selected models and caches the result for this exact chain snapshot/configuration.",
    )
    if run_surface_confirmation:
        try:
            model_text = " + ".join(confirmation_models)
            with st.spinner(f"Fitting {model_text} once for this chain snapshot…"):
                surface_results = _build_surface_comparison_cached(
                    chain, float(target_days), float(fit_lo), float(fit_hi),
                    tuple(confirmation_models), str(fengler_mode),
                    int(fengler_max_maturities), int(fengler_max_strikes), int(fengler_calendar_grid),
                )
            confirmations[surface_key] = surface_results
            st.session_state["surface_confirmations"] = confirmations
        except Exception as exc:
            st.warning(f"Surface confirmation failed: {exc}")
            surface_results = {}

    if surface_results:
        scope_note = f" · Fengler {fengler_scope.lower()}" if "Fengler" in confirmation_models else ""
        st.success(f"Cached confirmation: {fit_lo:.1f}–{fit_hi:.1f} DTE{scope_note}")
        source_options = ["Raw strip"] + [name for name in ("SSVI", "eSSVI", "Fengler") if name in surface_results]
    else:
        st.caption(
            f"Not run for this configuration. Planned fit window: {fit_lo:.1f}–{fit_hi:.1f} DTE."
        )
        source_options = ["Raw strip"]

    if st.session_state.get("mfiv_source") not in source_options:
        st.session_state["mfiv_source"] = "Raw strip"
    mfiv_source = st.selectbox(
        "Headline MFIV source",
        source_options,
        key="mfiv_source",
        help="SSVI/eSSVI/Fengler become selectable only after that model has been run for the current snapshot.",
    )

selected_name = mfiv_source
selected_target = snapshot.target_mfiv
if mfiv_source != "Raw strip":
    result = surface_results.get(mfiv_source)
    if result is None:
        st.warning(f"{mfiv_source} MFIV was unavailable; falling back to the raw strip.")
        selected_name = "Raw strip"
    else:
        selected_target = result.target
        if not result.reliable:
            st.warning(f"{mfiv_source} fit is not marked reliable. Use it as a diagnostic, not confirmation.")

selected_vrp_variance = float(selected_target.implied_variance - snapshot.trailing_target_variance)
selected_vol_spread = float(selected_target.implied_volatility - snapshot.trailing_target_volatility)
context = classify_vrp_context(snapshot, mfiv_variance=selected_target.implied_variance)
candidate = classify_vrp_candidate(
    snapshot,
    context,
    mfiv_variance=selected_target.implied_variance,
    mfiv_volatility=selected_target.implied_volatility,
)

quote_local = snapshot.quote_time.tz_convert("America/New_York")
st.caption(
    f"{snapshot.symbol} · option quote {quote_local:%Y-%m-%d %H:%M %Z} · "
    f"provider {provider} · headline {selected_name} MFIV"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{selected_name} MFIV {target_days}d", _pct(selected_target.implied_volatility))
c2.metric(f"Trailing integrated RV {target_days}d", _pct(snapshot.trailing_target_volatility))
c3.metric("Vol spread", _pct(selected_vol_spread))
c4.metric("VRP variance pts", _var_pts(selected_vrp_variance))
rv_3 = snapshot.realized_curve.loc[snapshot.realized_curve["days"] == 3, "realized_volatility"]
rv_30 = snapshot.realized_curve.loc[snapshot.realized_curve["days"] == 30, "realized_volatility"]
rv_slope = float(rv_3.iloc[0] - rv_30.iloc[0]) if len(rv_3) and len(rv_30) else np.nan
c5.metric("RV 3d − 30d", _pct(rv_slope))

if candidate.level == "strong":
    st.success(f"**{candidate.label}** — {candidate.explanation}")
elif candidate.level in {"candidate", "watch"}:
    st.info(f"**{candidate.label}** — {candidate.explanation}")
else:
    st.warning(f"**{candidate.label}** — {candidate.explanation}")
with st.expander("Why VolForge assigned this candidate label"):
    for reason in candidate.reasons:
        st.write(f"• {reason}")
    st.caption("Candidate ≠ trade instruction. Surface quality, tail risk, execution and forward-RV forecasting remain separate checks.")

st.subheader(f"Regime: {context.state}")
st.write(context.explanation)
ctx1, ctx2, ctx3 = st.columns(3)
ctx1.metric("Current RV3 − RV30", _pct(context.rv_slope))
ctx2.metric("Recent slope peak", _pct(context.recent_slope_peak))
ctx3.metric("Cooling from recent shock?", "Yes" if context.cooling_from_shock else "No")

if selected_vrp_variance > 0:
    st.success("Selected implied variance is above trailing integrated realized variance.")
else:
    st.warning("Selected implied variance is not above trailing integrated realized variance.")

current_tab, surface_tab, history_tab, diagnostics_tab, raw_tab = st.tabs(
    ["Current snapshot", "Surface models", "History / features", "Diagnostics", "Raw chain"]
)

with current_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Model-free implied-vol term structure")
        implied_plot = snapshot.mfiv_curve[["dte", "implied_volatility"]].copy()
        implied_plot["MFIV (%)"] = 100 * implied_plot.pop("implied_volatility")
        st.line_chart(implied_plot, x="dte", y="MFIV (%)")
        display = snapshot.mfiv_curve.copy()
        display["implied_volatility"] *= 100
        display["implied_variance"] *= 100
        display["parity_r2"] = display["parity_r2"].round(6)
        st.dataframe(
            display[["expiry", "dte", "implied_volatility", "implied_variance", "forward", "parity_r2", "n_strikes"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Integrated realized-vol term structure")
        if snapshot.realized_curve.empty:
            st.info("Not enough intraday history for the requested realized-vol windows.")
        else:
            realized_plot = snapshot.realized_curve.copy()
            realized_plot["Integrated RV (%)"] = 100 * realized_plot.pop("realized_volatility")
            st.line_chart(realized_plot, x="days", y="Integrated RV (%)")
            st.dataframe(realized_plot, use_container_width=True, hide_index=True)

    st.subheader("Realized-vol history")
    rv_cols = [c for c in ("rv_3", "rv_9", "rv_30", "rv_60", "rv_180") if c in snapshot.realized_history]
    if rv_cols:
        hist_plot = snapshot.realized_history[rv_cols].dropna(how="all") * 100
        st.line_chart(hist_plot)

with surface_tab:
    st.subheader("Raw strip vs surface-smoothed MFIV")
    st.write(
        "SSVI, eSSVI and Fengler are fit to the cleaned current chain, repriced on the same observed strike support, "
        "and then integrated with the same MFIV formula. This is a diagnostic comparison, not a model vote."
    )
    if not surface_results:
        st.info(
            "Raw-strip MFIV is the fast default. Choose models and click **Run selected confirmation** in the sidebar only when "
            "you want SSVI/eSSVI/Fengler as a quality-control check."
        )
        st.caption(f"Current confirmation window: {fit_lo:.1f}–{fit_hi:.1f} DTE.")
    else:
        rows = [{
            "model": "Raw strip",
            "mfiv": snapshot.target_mfiv.implied_volatility,
            "variance": snapshot.target_mfiv.implied_variance,
            "reliable": True,
            "rmse_iv": np.nan,
            "detail": f"{price_side} observed quotes",
        }]
        for name, result in surface_results.items():
            rows.append({
                "model": name, "mfiv": result.target.implied_volatility,
                "variance": result.target.implied_variance, "reliable": result.reliable,
                "rmse_iv": result.rmse_iv, "detail": result.detail,
            })
        comp = pd.DataFrame(rows)
        comp["difference_vs_raw"] = comp["mfiv"] - snapshot.target_mfiv.implied_volatility
        display_comp = comp.copy()
        for col in ("mfiv", "difference_vs_raw", "rmse_iv"):
            display_comp[col] = 100 * display_comp[col]
        st.dataframe(display_comp, use_container_width=True, hide_index=True)

        plot = pd.DataFrame({"Raw strip": snapshot.mfiv_curve.set_index("dte")["implied_volatility"] * 100})
        for name, result in surface_results.items():
            if not result.curve.empty:
                plot = plot.join(result.curve.set_index("dte")[["implied_volatility"]].rename(columns={"implied_volatility": name}) * 100, how="outer")
        st.line_chart(plot.sort_index())
        st.caption("A large Raw-vs-model gap is a reason to inspect chain quality, strike coverage, and model reliability before interpreting VRP.")

with history_tab:
    st.subheader("Daily VRP history")
    st.write(
        "Build or update the compact research table from saved chain snapshots plus integrated realized variance. "
        "The builder reads local data first; it does not issue hundreds of historical option API calls."
    )

    with st.expander("Build / update VRP history", expanded=False):
        st.markdown("**1. Archive the current chain (optional)**")
        hist_chain_root = st.text_input("Chain archive root", "data/chains", key="history_chain_root")
        save_col, saved_info = st.columns([1, 2])
        with save_col:
            if st.button("Save current chain", use_container_width=True, key="history_save_chain"):
                try:
                    ref = save_chain_snapshot(chain, provider=provider, root=hist_chain_root)
                    st.session_state["history_saved_chain"] = str(ref.path)
                except Exception as exc:
                    st.error(f"Could not save chain: {exc}")
        with saved_info:
            if st.session_state.get("history_saved_chain"):
                st.success(f"Saved: {st.session_state['history_saved_chain']}")
            else:
                st.caption("The history builder only uses chain snapshots already saved under the archive root.")

        st.markdown("**2. Choose realized-variance history**")
        history_rv_source = st.radio(
            "RV source",
            ("Current dashboard bars", "Local intraday bars", "Daily integrated variance"),
            horizontal=True,
            key="history_rv_source",
        )
        history_rv_path = ""
        if history_rv_source == "Local intraday bars":
            history_rv_path = st.text_input(
                "Intraday bars path", f"data/intraday/{symbol}.parquet", key="history_bars_path"
            )
            st.caption("CSV/Parquet with timestamp + close (or common equivalents).")
        elif history_rv_source == "Daily integrated variance":
            history_rv_path = st.text_input(
                "Daily variance path", f"data/realized/{symbol}.parquet", key="history_daily_var_path"
            )
            st.caption("CSV/Parquet with date + integrated_variance.")
        else:
            st.caption("Uses the bars already loaded for the live dashboard. Useful for a quick update; long research history should come from your local archive.")

        st.markdown("**3. Build / update**")
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            history_policy = st.selectbox("Daily snapshot", ("latest", "earliest", "closest"), key="history_snapshot_policy")
        with hc2:
            history_target_time = st.text_input("Target time ET", "15:30", key="history_target_time", disabled=history_policy != "closest")
        with hc3:
            history_rv_asof = st.selectbox("RV as-of", ("previous_session", "same_session"), key="history_rv_asof")
        history_output_root = st.text_input("Derived history root", "data/derived/vrp", key="history_output_root")

        if st.button("Build / update VRP history", type="primary", use_container_width=True, key="history_build"):
            try:
                with st.spinner("Reading saved chains and rebuilding VRP history…"):
                    if history_rv_source == "Current dashboard bars":
                        daily_for_history = daily_integrated_variance(bars)
                    elif history_rv_source == "Local intraday bars":
                        hist_bars = normalise_intraday_bars(_read_table_from_path(history_rv_path))
                        daily_for_history = daily_integrated_variance(hist_bars)
                    else:
                        daily_for_history = load_daily_variance(history_rv_path)

                    hcfg = VRPHistoryConfig(
                        target_days=float(target_days),
                        price_side=price_side,
                        rv_asof=history_rv_asof,
                        snapshot_policy=history_policy,
                        target_time=(history_target_time if history_policy == "closest" else None),
                    )
                    built_history = build_vrp_history(
                        symbol,
                        daily_for_history,
                        provider=provider,
                        chain_root=hist_chain_root,
                        config=hcfg,
                    )
                    target_path = save_vrp_history(
                        built_history, symbol=symbol, provider=provider, root=history_output_root
                    )
                st.session_state["vrp_history_path"] = str(target_path)
                st.session_state["vrp_history_build_rows"] = int(len(built_history))
                st.session_state["vrp_history_build_labels"] = int(built_history.get("forward_rv_var", pd.Series(dtype=float)).notna().sum())
                st.success(
                    f"Updated {len(built_history)} rows · "
                    f"{st.session_state['vrp_history_build_labels']} forward labels · {target_path}"
                )
            except Exception as exc:
                st.error(f"Could not build VRP history: {exc}")

    st.write(
        "Load the compact derived-history file below. Required columns: `date`, `mfiv_var`, `trailing_rv_var`. "
        "Optional: `forward_rv_var` for the ex-post forward VRP label."
    )
    hist_upload = st.file_uploader("VRP history CSV / Parquet", type=["csv", "parquet", "pq"], key="vrp_history")
    if "vrp_history_path" not in st.session_state:
        st.session_state["vrp_history_path"] = f"data/derived/vrp/provider={provider}/symbol={symbol}/history.parquet"
    hist_path = st.text_input(
        "…or history path",
        key="vrp_history_path",
    )

    hist_frame = None
    try:
        if hist_upload is not None:
            hist_frame = _read_table_from_upload(hist_upload)
        elif Path(hist_path).expanduser().exists():
            hist_frame = _read_table_from_path(hist_path)
    except Exception as exc:
        st.error(f"Could not load VRP history: {exc}")

    if hist_frame is None:
        st.info("No derived VRP history loaded yet. Run `scripts/build_vrp_history.py` after capturing chains and realized-vol data.")
    else:
        try:
            enriched = prepare_vrp_history(hist_frame)
        except Exception as exc:
            st.error(f"History format error: {exc}")
        else:
            h1, h2 = st.columns(2)
            with h1:
                st.markdown("**MFIV vs trailing integrated RV**")
                vols = enriched[["mfiv_vol", "trailing_rv_vol"]].rename(
                    columns={"mfiv_vol": "MFIV", "trailing_rv_vol": "Integrated RV"}
                ) * 100
                st.line_chart(vols)
            with h2:
                st.markdown("**VRP z-score**")
                st.line_chart(enriched[["vrp_z"]])

            h3, h4 = st.columns(2)
            with h3:
                st.markdown("**VRP percentile**")
                st.line_chart(100 * enriched[["vrp_percentile"]])
            with h4:
                st.markdown("**Vol-of-vol**")
                st.line_chart(enriched[["vol_of_vol"]])

            delta_ratio_cols = [
                c for c in (
                    "delta_ratio_10p", "delta_ratio_15p", "delta_ratio_25p",
                    "delta_ratio_25c", "delta_ratio_15c", "delta_ratio_10c",
                ) if c in enriched
            ]
            if delta_ratio_cols:
                st.markdown("### Delta-ratio history")
                st.line_chart(enriched[delta_ratio_cols])
                z_cols = [f"{c}_z" for c in delta_ratio_cols if f"{c}_z" in enriched]
                if z_cols:
                    st.markdown("**Delta-ratio historical z-scores**")
                    st.line_chart(enriched[z_cols])

            decomp_cols = [
                c for c in (
                    "surface_parallel_shift", "surface_put_skew_change", "surface_call_skew_change",
                    "surface_downside_convexity_change", "surface_upside_convexity_change",
                ) if c in enriched
            ]
            if decomp_cols:
                st.markdown("### Delta-surface change decomposition")
                st.line_chart(100 * enriched[decomp_cols])
                latest = enriched[decomp_cols].dropna(how="all").tail(1)
                if not latest.empty:
                    latest_display = (100 * latest.T).rename(columns={latest.index[0]: "latest_change_vol_pts"})
                    st.dataframe(latest_display, use_container_width=True)
                st.caption(
                    "Observable delta-space decomposition: ATM parallel shift, 25Δ put/call skew changes, and wing convexity changes. "
                    "This is not presented as an exact Vanna-Volga replication."
                )

            if "forward_vrp" in enriched:
                st.markdown("**Ex-post forward VRP label**")
                st.line_chart(100 * enriched[["forward_vrp"]])
            st.dataframe(enriched.tail(100), use_container_width=True)

with diagnostics_tab:
    st.subheader("Chain quality")
    q = snapshot.chain_quality
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{q['rows']:,}")
    d2.metric("Expiries", str(q["expiries"]))
    d3.metric("Unique strikes", str(q["strikes"]))
    d4.metric("Spot", f"{q['spot']:.2f}")
    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Zero-bid rows", str(q["zero_bid_rows"]))
    d6.metric("Crossed rows", str(q["crossed_rows"]))
    d7.metric("Median spread", f"{q['median_spread']:.4f}")
    d8.metric("Median rel. spread", _pct(float(q["median_rel_spread"]), 2))

    st.subheader("Constant-tenor interpolation")
    ct = snapshot.target_mfiv
    st.code(
        f"target={ct.target_days:.0f}d  lower={ct.lower_days:.2f}d  upper={ct.upper_days:.2f}d\n"
        f"weight={ct.interpolation_weight:.4f}  variance={ct.implied_variance:.6f}  "
        f"vol={ct.implied_volatility:.4f}"
    )
    st.caption(
        "Constant-tenor MFIV is interpolated in total variance (variance × time), not directly in volatility."
    )

with raw_tab:
    st.subheader("Canonical option chain")
    cols = [
        "quote_time", "expiry", "dte", "strike", "right", "bid", "ask", "mid", "spread",
        "rel_spread", "underlying_price", "volume", "open_interest", "source",
    ]
    cols = [c for c in cols if c in chain.columns]
    st.dataframe(chain[cols], use_container_width=True, hide_index=True)

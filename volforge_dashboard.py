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
    build_surface_explorer,
    classify_vrp_candidate,
    classify_vrp_context,
    normalise_intraday_bars,
    prepare_vrp_history,
)
from volforge.data.provider import available_providers, fetch_chain
from volforge.data.intraday import load_realized_archive, realized_archive_path
from volforge.data.storage import (
    list_chain_snapshots,
    load_chain_snapshot,
    select_daily_snapshots,
)
from volforge.delta_surface import (
    build_delta_surface, constant_tenor_delta_slice, delta_lump_scores,
    delta_ratio_term_structure,
)
from volforge.vix_curve import load_vix_curve_history
from volforge.forecasting import (
    latest_realized_model_forecasts, walk_forward_realized_forecasts, forecast_metrics,
    latest_xgboost_forecasts, walk_forward_xgboost, xgboost_available,
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
5. **Surface quality (optional):** If a candidate deserves deeper inspection, open **Advanced Surface Diagnostics** and compare the raw strip with SVI/SSVI/eSSVI/Fengler. The calibrated models no longer gate the primary VRP workflow.
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

    st.subheader("Where the calibrated surfaces fit now")
    st.write(
        "The primary VRP path now uses raw-strip MFIV and model-light delta features. SVI/SSVI/eSSVI/Fengler live under "
        "**Advanced Surface Diagnostics** for smoothing, arbitrage checks, sparse-strike interpolation, and raw-vs-smoothed validation. "
        "They are useful quality-control tools, but they no longer need to run for ordinary VRP candidate screening."
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


@st.cache_data(ttl=1800, show_spinner=False)
def _load_vix_curve_cached() -> pd.DataFrame:
    return load_vix_curve_history()


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


@st.cache_data(show_spinner=False, max_entries=8)
def _walk_forward_model_lab_cached(
    daily_variance: pd.Series,
    target_days: int,
    min_train: int,
    refit_every: int,
):
    predictions = walk_forward_realized_forecasts(
        daily_variance,
        target_days=int(target_days),
        min_train=int(min_train),
        refit_every=int(refit_every),
    )
    return predictions, forecast_metrics(predictions)


@st.cache_data(show_spinner=False, max_entries=6)
def _walk_forward_xgb_cached(
    history: pd.DataFrame,
    target_days: int,
    min_train: int,
    refit_every: int,
):
    predictions = walk_forward_xgboost(
        history,
        target_days=int(target_days),
        min_train=int(min_train),
        refit_every=int(refit_every),
        quantiles=(0.70,),
    )
    return predictions, forecast_metrics(predictions)


def _render_model_lab_page():
    st.title("VolForge · Model Lab")
    st.caption(
        "Realized-volatility forecasting and VRP machine learning are separate datasets. "
        "HAR/HEAVY use the long Alpaca realized archive; XGBoost uses the scarcer option-chain/VRP history."
    )

    with st.sidebar:
        st.header("Model Lab")
        ml_symbol = st.text_input("Symbol", "SPY", key="model_lab_symbol").strip().upper()
        ml_target_days = st.number_input("Forecast horizon (days)", 7, 180, 30, step=1, key="model_lab_target")
        ml_rv_provider = st.selectbox("Realized-vol provider", ["alpaca"], key="model_lab_rv_provider")
        ml_rv_feed = st.selectbox("Realized-vol feed", ["iex", "sip"], key="model_lab_rv_feed")
        default_rv = realized_archive_path(ml_symbol, provider=ml_rv_provider, feed=ml_rv_feed)
        ml_realized_path = st.text_input("Realized archive path", str(default_rv), key="model_lab_realized_path")
        ml_provider = st.selectbox("Option provider (XGBoost only)", available_providers(), key="model_lab_provider")
        ml_history_path = st.text_input(
            "VRP history path (XGBoost only)",
            f"data/derived/vrp/provider={ml_provider}/symbol={ml_symbol}/history.parquet",
            key="model_lab_history_path",
        )
        ml_min_train = st.number_input("Minimum training rows", 20, 1000, 80, step=10, key="model_lab_min_train")
        ml_refit = st.number_input("Refit every N observations", 1, 100, 20, step=1, key="model_lab_refit")

    st.subheader("Realized-volatility models")
    st.caption("Persistence, HAR and HEAVY do not require option-chain history or MFIV.")
    realized_path = Path(ml_realized_path).expanduser()
    try:
        daily_rm = load_realized_archive(realized_path)
    except Exception as exc:
        st.warning(f"Could not load realized archive: {exc}")
        daily_rm = pd.Series(dtype="float64", name="integrated_variance")

    if daily_rm.empty:
        st.info(
            "No realized archive found. Build it with "
            f"`python scripts/update_intraday.py --symbol {ml_symbol}`. "
            "HAR and HEAVY will then use the full Alpaca history, independent of VRP history."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Realized sessions", f"{len(daily_rm):,}")
        c2.metric("From", daily_rm.index.min().strftime("%Y-%m-%d"))
        c3.metric("Through", daily_rm.index.max().strftime("%Y-%m-%d"))

        try:
            latest = latest_realized_model_forecasts(daily_rm, target_days=int(ml_target_days))
        except Exception as exc:
            st.warning(f"Could not produce realized-vol forecasts: {exc}")
            latest = pd.DataFrame()
        if latest.empty:
            st.info("Need enough realized sessions for HAR/HEAVY. HEAVY requires at least 30 daily realized measures.")
        else:
            display = latest.copy()
            display["forecast_rv_vol_pct"] = 100 * display["forecast_rv_vol"]
            st.dataframe(
                display[["model", "forecast_rv_vol_pct", "detail"]].round(3),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Run realized-model walk-forward", type="primary", key="model_lab_run"):
            try:
                with st.spinner("Running purged HAR/HEAVY forecasts on the realized archive…"):
                    predictions, metrics = _walk_forward_model_lab_cached(
                        daily_rm, int(ml_target_days), int(ml_min_train), int(ml_refit)
                    )
            except Exception as exc:
                st.error(f"Realized-model walk-forward failed: {exc}")
            else:
                st.dataframe(metrics.round(8), use_container_width=True, hide_index=True)
                if not predictions.empty:
                    with st.expander("Forecast chart", expanded=False):
                        chart = predictions.pivot_table(index="date", columns="model", values="forecast_rv_var", aggfunc="last")
                        actual = predictions.drop_duplicates("date").set_index("date")["actual_rv_var"].rename("Actual forward RV")
                        st.line_chart(100 * np.sqrt(chart.join(actual, how="outer").sort_index().clip(lower=0.0)))

    st.divider()
    st.subheader("VRP / XGBoost")
    st.caption(
        "This section uses option-chain history because MFIV, delta ratios and VRP features only exist on saved chain dates. "
        "A sparse VRP history does not block HAR or HEAVY above."
    )
    upload = st.file_uploader("Optional VRP history CSV / Parquet", type=["csv", "parquet", "pq"], key="model_lab_upload")
    try:
        history = _read_table_from_upload(upload) if upload is not None else _read_table_from_path(ml_history_path)
    except Exception as exc:
        st.info(f"VRP history not available yet: {exc}")
        history = pd.DataFrame()

    if history.empty:
        st.info("XGBoost will become useful after option-chain/VRP history accumulates. It is not required for HAR/HEAVY.")
    else:
        dates = pd.to_datetime(history["date"], errors="coerce") if "date" in history else pd.Series(pd.DatetimeIndex(history.index))
        labeled = int(pd.to_numeric(history["forward_rv_var"], errors="coerce").notna().sum()) if "forward_rv_var" in history else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("VRP rows", f"{len(history):,}")
        c2.metric("Forward labels", f"{labeled:,}")
        c3.metric("Through", dates.dropna().max().strftime("%Y-%m-%d") if dates.notna().any() else "—")

        if not xgboost_available():
            st.info("Install the ML extra to enable XGBoost: `pip install -e .[ml]` (or `uv sync --extra ml`).")
        elif labeled < int(ml_min_train):
            st.info(f"XGBoost needs labeled VRP history. Current labels: {labeled}; requested minimum: {int(ml_min_train)}.")
        elif st.button("Fit latest XGBoost + q70", key="model_lab_xgb_latest"):
            try:
                with st.spinner("Fitting experimental XGBoost forecasts…"):
                    xgb_latest, importance = latest_xgboost_forecasts(history, quantiles=(0.70,))
            except Exception as exc:
                st.error(f"XGBoost fit failed: {exc}")
            else:
                xgb_display = xgb_latest.copy()
                xgb_display["forecast_rv_vol_pct"] = 100 * xgb_display["forecast_rv_vol"]
                xgb_display["mfiv_vol_pct"] = 100 * np.sqrt(xgb_display["mfiv_var"].clip(lower=0.0))
                xgb_display["vol_spread_pct"] = xgb_display["mfiv_vol_pct"] - xgb_display["forecast_rv_vol_pct"]
                st.dataframe(
                    xgb_display[["model", "forecast_rv_vol_pct", "mfiv_vol_pct", "vol_spread_pct", "detail"]].round(3),
                    use_container_width=True,
                    hide_index=True,
                )
                with st.expander("XGBoost feature importance", expanded=False):
                    st.dataframe(importance.head(20), use_container_width=True, hide_index=True)

        if xgboost_available() and labeled >= int(ml_min_train) + 5 and st.button("Run XGBoost walk-forward", key="model_lab_xgb_walk"):
            try:
                with st.spinner("Running purged XGBoost + q70 forecasts…"):
                    xpred, xmetrics = _walk_forward_xgb_cached(
                        history, int(ml_target_days), int(ml_min_train), int(ml_refit)
                    )
            except Exception as exc:
                st.error(f"XGBoost walk-forward failed: {exc}")
            else:
                st.dataframe(xmetrics.round(8), use_container_width=True, hide_index=True)
                if not xpred.empty:
                    with st.expander("XGBoost forecast chart", expanded=False):
                        chart = xpred.pivot_table(index="date", columns="model", values="forecast_rv_var", aggfunc="last")
                        actual = xpred.drop_duplicates("date").set_index("date")["actual_rv_var"].rename("Actual forward RV")
                        st.line_chart(100 * np.sqrt(chart.join(actual, how="outer").sort_index().clip(lower=0.0)))

    with st.expander("Model rules", expanded=False):
        st.markdown(
            """
- **Persistence:** trailing target-horizon realized variance.
- **HAR 3/9/30:** realized-only direct regression using the long Alpaca archive.
- **HEAVY-RM:** realized-measure dynamics using the long Alpaca archive.
- **XGBoost:** experimental VRP model using MFIV/RV/delta/regime features on saved option-chain dates.
- **q70 XGBoost:** conservative 70th-percentile forward-RV estimate.
- **Promotion rule:** XGBoost does not become a headline forecast until it beats the realized-vol benchmarks out of sample.
"""
        )


def _render_surface_explorer_page():
    st.title("VolForge · Advanced Surface Diagnostics")
    st.caption("Optional calibrated-surface diagnostics. The core VRP workflow uses raw MFIV, realized variance, delta ratios, and term structure without requiring SVI-family or Fengler calibration.")

    with st.sidebar:
        st.header("Advanced Surface Diagnostics")
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

            if st.checkbox("Show 3D delta surface", value=False, key="surface_explorer_delta_3d"):
                st.markdown("**3D delta volatility surface**")
                fig3d = go.Figure(data=[go.Surface(
                    x=list(shown_delta.columns),
                    y=[float(x) for x in shown_delta.index],
                    z=shown_delta.to_numpy(float),
                    colorbar={"title": "IV (%)"},
                )])
                fig3d.update_layout(
                    scene={
                        "xaxis_title": "Delta bucket",
                        "yaxis_title": "DTE",
                        "zaxis_title": "IV (%)",
                    },
                    margin={"l": 0, "r": 0, "t": 25, "b": 0},
                    height=620,
                )
                st.plotly_chart(fig3d, use_container_width=True)
                st.caption("Same delta-volatility matrix as the heatmap, shown in 3D; no additional surface fit is run.")
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


page = st.sidebar.radio(
    "Page",
    ("Forward VRP dashboard", "Model lab", "Advanced surface diagnostics", "How to use"),
    index=0,
)
if page == "How to use":
    _render_demo_page()
    st.stop()
if page == "Model lab":
    _render_model_lab_page()
    st.stop()
if page == "Advanced surface diagnostics":
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
    alpaca_default_path = Path(
        f"data/intraday/provider=alpaca/feed=iex/symbol={symbol}/bars_5min.parquet"
    )
    bar_source = st.radio(
        "Intraday source",
        ("Local Alpaca/IEX archive", "Yahoo recent bars (preview)", "Local CSV / Parquet"),
        index=0 if alpaca_default_path.exists() else 1,
    )
    uploaded_bars = None
    local_bar_path = ""
    if bar_source.startswith("Yahoo"):
        bar_interval = st.selectbox("Bar interval", ("5m", "15m"), index=0)
        bar_period = st.selectbox("History", ("60d",), index=0)
        st.caption("Preview only. The research path is the local Alpaca/IEX archive.")
    elif bar_source.startswith("Local Alpaca"):
        local_bar_path = st.text_input(
            "Alpaca archive path",
            value=str(alpaca_default_path),
        )
        st.caption("Update it with `python scripts/update_intraday.py --symbol SYMBOL`.")
    else:
        uploaded_bars = st.file_uploader("Upload intraday bars", type=["csv", "parquet", "pq"])
        local_bar_path = st.text_input("…or local path", value=f"data/intraday/{symbol}.parquet")
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

# The primary VRP workflow uses the observable raw option strip. Calibrated
# SVI/SSVI/eSSVI/Fengler surfaces live under Advanced Surface Diagnostics so
# they cannot slow down or gate normal candidate screening.
selected_name = "Raw strip"
selected_target = snapshot.target_mfiv

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

current_tab, history_tab, diagnostics_tab, raw_tab = st.tabs(
    ["Current snapshot", "History / features", "Diagnostics", "Raw chain"]
)

with current_tab:
    st.subheader("Volatility term structure")
    implied = snapshot.mfiv_curve[["dte", "implied_volatility"]].copy()
    implied["days"] = implied["dte"].round().astype(int)
    implied = implied.groupby("days", as_index=True)["implied_volatility"].mean().rename("MFIV")
    if snapshot.realized_curve.empty:
        curve = implied.to_frame() * 100
    else:
        realized = snapshot.realized_curve.set_index("days")["realized_volatility"].rename("Integrated RV")
        curve = pd.concat([implied, realized], axis=1).sort_index() * 100
    st.line_chart(curve)
    with st.expander("Term-structure tables", expanded=False):
        display = snapshot.mfiv_curve.copy()
        display["implied_volatility"] *= 100
        display["implied_variance"] *= 100
        display["parity_r2"] = display["parity_r2"].round(6)
        st.markdown("**MFIV**")
        st.dataframe(
            display[["expiry", "dte", "implied_volatility", "implied_variance", "forward", "parity_r2", "n_strikes"]],
            use_container_width=True,
            hide_index=True,
        )
        if not snapshot.realized_curve.empty:
            realized_display = snapshot.realized_curve.copy()
            realized_display["realized_volatility"] *= 100
            st.markdown("**Integrated RV**")
            st.dataframe(realized_display, use_container_width=True, hide_index=True)
    with st.expander("Realized-vol history", expanded=False):
        rv_cols = [c for c in ("rv_3", "rv_9", "rv_30", "rv_60", "rv_180") if c in snapshot.realized_history]
        if rv_cols:
            st.line_chart(snapshot.realized_history[rv_cols].dropna(how="all") * 100)
        else:
            st.caption("Not enough realized history yet.")

with history_tab:
    st.subheader("Daily VRP history")
    st.caption("Read-only here. Update data with the scripts; the dashboard reads the finished research table.")

    try:
        vix_curve = _load_vix_curve_cached()
        latest_curve = vix_curve.iloc[-1]
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("VIX", f"{latest_curve['VIX']:.2f}")
        vc2.metric("VIX3M", f"{latest_curve['VIX3M']:.2f}")
        vc3.metric("VIX3M − VIX", f"{latest_curve['vix3m_minus_vix']:+.2f}")
        z10 = latest_curve["vix_curve_z_10d"]
        vc4.metric("Curve z-score · 10d", "—" if pd.isna(z10) else f"{z10:+.2f}")
        with st.expander("VIX curve charts", expanded=False):
            st.caption("VIX3M − VIX")
            st.line_chart(vix_curve[["vix3m_minus_vix"]].tail(126))
            st.caption("Prior-only 10-session z-score")
            st.line_chart(vix_curve[["vix_curve_z_10d"]].tail(126))
    except Exception as exc:
        st.caption(f"Cboe VIX curve history unavailable: {exc}")

    refs = list_chain_snapshots(symbol, provider=provider, root="data/chains", include_legacy_yahoo=True)
    history_path = Path(f"data/derived/vrp/provider={provider}/symbol={symbol}/history.parquet")
    with st.expander("History source and rebuild commands", expanded=False):
        override = st.text_input("Optional history path override", str(history_path), key=f"history_path_{provider}_{symbol}")
        history_path = Path(override).expanduser()
        st.code(
            f"python scripts/update_intraday.py --symbol {symbol}\n"
            f"python scripts/build_vrp_history.py --symbol {symbol} --provider {provider}",
            language="powershell",
        )
        st.caption("The default realized source is Alpaca/IEX. Use --daily-variance or --bars to override it.")

    hist_frame = None
    try:
        if history_path.exists():
            hist_frame = _read_table_from_path(str(history_path))
    except Exception as exc:
        st.error(f"Could not load VRP history: {exc}")

    if hist_frame is None:
        h1, h2 = st.columns(2)
        h1.metric("Saved chain snapshots", f"{len(refs):,}")
        h2.metric("History file", "Missing")
        st.info(
            "No derived VRP history yet. Run `scripts/update_intraday.py`, then `scripts/build_vrp_history.py`. "
            "The scripts print coverage and row counts so stale history is visible immediately."
        )
    else:
        try:
            enriched = prepare_vrp_history(hist_frame)
        except Exception as exc:
            st.error(f"History format error: {exc}")
        else:
            latest_hist = enriched.tail(1)
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("History rows", f"{len(enriched):,}")
            h2.metric("Through", enriched.index.max().strftime("%Y-%m-%d"))
            h3.metric("Latest MFIV", _pct(float(latest_hist["mfiv_vol"].iloc[0])))
            h4.metric("Latest RV", _pct(float(latest_hist["trailing_rv_vol"].iloc[0])))
            st.markdown("**MFIV vs trailing integrated RV**")
            vols = enriched[["mfiv_vol", "trailing_rv_vol"]].rename(
                columns={"mfiv_vol": "MFIV", "trailing_rv_vol": "Integrated RV"}
            ) * 100
            st.line_chart(vols)

            with st.expander("More history diagnostics", expanded=False):
                metric_cols = [c for c in ("vrp_z", "vrp_percentile", "vol_of_vol") if c in enriched]
                if metric_cols:
                    st.line_chart(enriched[metric_cols])
                delta_ratio_cols = [
                    c for c in (
                        "delta_ratio_10p", "delta_ratio_15p", "delta_ratio_25p",
                        "delta_ratio_25c", "delta_ratio_15c", "delta_ratio_10c",
                    ) if c in enriched
                ]
                if delta_ratio_cols:
                    st.markdown("**Delta ratios**")
                    st.line_chart(enriched[delta_ratio_cols])
                decomp_cols = [
                    c for c in (
                        "surface_parallel_shift", "surface_put_skew_change", "surface_call_skew_change",
                        "surface_downside_convexity_change", "surface_upside_convexity_change",
                    ) if c in enriched
                ]
                if decomp_cols:
                    st.markdown("**Delta-space change decomposition**")
                    st.line_chart(100 * enriched[decomp_cols])
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

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
    classify_vrp_context,
    normalise_intraday_bars,
    prepare_vrp_history,
)
from volforge.data.provider import available_providers, fetch_chain


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

    st.subheader("RV3 / RV30 context matrix")
    matrix = pd.DataFrame([
        {"RV state": "RV3 > RV30", "MFIV state": "High", "Read": "Shock active; premium may be forming, but realized risk is still hot."},
        {"RV state": "RV3 > RV30", "MFIV state": "Not high", "Read": "Poor compensation for an active shock."},
        {"RV state": "RV3 < RV30 after recent positive slope", "MFIV state": "Still high", "Read": "Post-shock / lingering-premium candidate; especially worth researching."},
        {"RV state": "RV3 < RV30", "MFIV state": "Low", "Read": "Calm market, but probably little premium to harvest."},
    ])
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.subheader("Why compare Raw, SSVI, and Fengler?")
    st.write(
        "Raw-strip MFIV uses the observed option quotes directly. SSVI gives a global parametric, static-arbitrage-aware "
        "surface; Fengler smooths in option-price space under convexity/monotonicity/calendar constraints. VolForge integrates "
        "each model over the **same observed strike support** so differences reflect smoothing/model quality rather than a different wing domain."
    )
    st.caption("If a model is flagged unreliable, do not use its smoothed MFIV as confirmation. Treat the disagreement itself as a diagnostic.")


page = st.sidebar.radio("Page", ("Forward VRP dashboard", "How to use"), index=0)
if page == "How to use":
    _render_demo_page()
    st.stop()


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
):
    """Cache expensive SSVI/Fengler fits by option-chain snapshot."""
    return build_surface_mfiv_comparison(
        chain,
        target_days=float(target_days),
        dte_range=(float(fit_lo), float(fit_hi)),
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

fit_lo, fit_hi = _surface_focus_dte_range(
    chain,
    target_days=float(target_days),
    user_range=(float(dte_lo), float(dte_hi)),
)
surface_key = (
    str(symbol), str(provider), str(snapshot.quote_time), float(target_days),
    float(fit_lo), float(fit_hi), str(price_side),
)
confirmations = st.session_state.setdefault("surface_confirmations", {})
surface_results = confirmations.get(surface_key, {})

with st.sidebar:
    st.divider()
    st.header("Surface confirmation")
    st.caption(
        "Optional and expensive. Raw-strip MFIV remains the fast default; "
        "SSVI/Fengler only run when you ask for confirmation."
    )
    run_surface_confirmation = st.button(
        "Run surface confirmation",
        use_container_width=True,
        help="Fits only nearby expiries around the target tenor, then caches the result for this chain snapshot.",
    )
    if run_surface_confirmation:
        try:
            with st.spinner("Fitting SSVI + Fengler once for this chain snapshot…"):
                surface_results = _build_surface_comparison_cached(
                    chain, float(target_days), float(fit_lo), float(fit_hi)
                )
            confirmations[surface_key] = surface_results
            st.session_state["surface_confirmations"] = confirmations
        except Exception as exc:
            st.warning(f"Surface confirmation failed: {exc}")
            surface_results = {}

    if surface_results:
        st.success(f"Cached confirmation: {fit_lo:.1f}–{fit_hi:.1f} DTE")
        source_options = ["Raw strip"] + [name for name in ("SSVI", "Fengler") if name in surface_results]
    else:
        st.caption(f"Not run for this snapshot. Planned fit window: {fit_lo:.1f}–{fit_hi:.1f} DTE.")
        source_options = ["Raw strip"]

    if st.session_state.get("mfiv_source") not in source_options:
        st.session_state["mfiv_source"] = "Raw strip"
    mfiv_source = st.selectbox(
        "Headline MFIV source",
        source_options,
        key="mfiv_source",
        help="SSVI/Fengler become selectable only after confirmation has been run for the current snapshot.",
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

st.subheader(context.state)
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
        "SSVI and Fengler are fit to the cleaned current chain, repriced on the same observed strike support, "
        "and then integrated with the same MFIV formula. This is a diagnostic comparison, not a model vote."
    )
    if not surface_results:
        st.info(
            "Raw-strip MFIV is the fast default. Click **Run surface confirmation** in the sidebar only when "
            "you want SSVI/Fengler as a quality-control check."
        )
        st.caption(f"The confirmation fit will focus on roughly {fit_lo:.1f}–{fit_hi:.1f} DTE instead of the full option range.")
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
        "Load a compact derived-history file instead of making the dashboard issue hundreds of historical "
        "option-chain API calls. Required columns: `date`, `mfiv_var`, `trailing_rv_var`. Optional: "
        "`forward_rv_var` for the ex-post forward VRP label."
    )
    hist_upload = st.file_uploader("VRP history CSV / Parquet", type=["csv", "parquet", "pq"], key="vrp_history")
    hist_path = st.text_input("…or history path", value=f"data/vrp/{symbol}.csv")

    hist_frame = None
    try:
        if hist_upload is not None:
            hist_frame = _read_table_from_upload(hist_upload)
        elif Path(hist_path).expanduser().exists():
            hist_frame = _read_table_from_path(hist_path)
    except Exception as exc:
        st.error(f"Could not load VRP history: {exc}")

    if hist_frame is None:
        st.info("No derived VRP history loaded yet. The upcoming batch-history builder will write this format.")
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

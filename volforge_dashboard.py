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

from volforge.dashboard import build_dashboard_snapshot, normalise_intraday_bars, prepare_vrp_history
from volforge.data.provider import available_providers, fetch_chain


st.set_page_config(page_title="VolForge · Forward VRP", page_icon="〽", layout="wide")


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
    price_side = st.radio("MFIV quote side", ("mid", "bid"), horizontal=True)
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

quote_local = snapshot.quote_time.tz_convert("America/New_York")
st.caption(
    f"{snapshot.symbol} · option quote {quote_local:%Y-%m-%d %H:%M %Z} · "
    f"provider {provider} · {price_side} MFIV"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"MFIV {target_days}d", _pct(snapshot.target_mfiv.implied_volatility))
c2.metric(f"Trailing integrated RV {target_days}d", _pct(snapshot.trailing_target_volatility))
c3.metric("Vol spread", _pct(snapshot.current_vol_spread))
c4.metric("VRP variance pts", _var_pts(snapshot.current_vrp_variance))
rv_3 = snapshot.realized_curve.loc[snapshot.realized_curve["days"] == 3, "realized_volatility"]
rv_30 = snapshot.realized_curve.loc[snapshot.realized_curve["days"] == 30, "realized_volatility"]
rv_slope = float(rv_3.iloc[0] - rv_30.iloc[0]) if len(rv_3) and len(rv_30) else np.nan
c5.metric("RV 3d − 30d", _pct(rv_slope))

if snapshot.current_vrp_variance > 0:
    st.success("Current implied variance is above trailing integrated realized variance.")
else:
    st.warning("Current implied variance is not above trailing integrated realized variance.")

current_tab, history_tab, diagnostics_tab, raw_tab = st.tabs(
    ["Current snapshot", "History / features", "Diagnostics", "Raw chain"]
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

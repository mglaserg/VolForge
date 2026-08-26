"""Historical Forward-VRP dataset construction from saved chain snapshots.

This module deliberately separates expensive/raw market data from the compact
research table used by the dashboard and future ML models.  The first version
uses raw-strip MFIV; SSVI/eSSVI/Fengler can be added later as enrichment passes
without redownloading option data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data.storage import list_chain_snapshots, load_chain_snapshot, select_daily_snapshots
from .mfiv import constant_tenor_mfiv, mfiv_term_structure
from .realized import forward_integrated_variance, integrated_volatility, rolling_integrated_variance
from .vrp import forward_vrp_label, realized_term_structure, vol_of_vol, vrp_features

__all__ = [
    "VRPHistoryConfig",
    "build_vrp_history",
    "save_vrp_history",
    "load_daily_variance",
]


@dataclass(frozen=True)
class VRPHistoryConfig:
    target_days: float = 30.0
    mfiv_tenors: tuple[int, ...] = (7, 30, 60, 180)
    rv_windows: tuple[int, ...] = (3, 9, 30, 60, 180)
    price_side: str = "mid"
    rv_term_basis: str = "trading"
    target_rv_basis: str = "calendar"
    # With midday chain snapshots a full same-day RV observation would contain
    # returns that occur after the quote. Previous-session is the safe default.
    rv_asof: str = "previous_session"
    z_window: int = 252
    min_periods: int | None = None
    snapshot_policy: str = "latest"
    target_time: str | None = None


def _prepare_daily(series: pd.Series) -> pd.Series:
    out = pd.Series(series, dtype="float64", copy=True)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.isna()].sort_index()
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")]


def _reindex_daily(frame: pd.DataFrame | pd.Series, dates: pd.DatetimeIndex):
    # Research dates are trading dates, while realized inputs can have sparse
    # holidays. Forward-fill is only used for *trailing* information.
    return frame.reindex(dates, method="ffill")


def build_vrp_history(
    symbol: str,
    daily_variance: pd.Series,
    *,
    provider: str = "yahoo",
    chain_root: str | Path = "data/chains",
    config: VRPHistoryConfig | None = None,
) -> pd.DataFrame:
    """Build one compact daily VRP research table from saved chain snapshots.

    Forward labels are only populated when enough future realized data exists;
    rerunning the builder later naturally fills previously unavailable labels.
    """
    cfg = config or VRPHistoryConfig()
    if cfg.rv_asof not in {"previous_session", "same_session"}:
        raise ValueError("rv_asof must be 'previous_session' or 'same_session'")

    daily = _prepare_daily(daily_variance)
    refs = list_chain_snapshots(symbol, provider=provider, root=chain_root, include_legacy_yahoo=True)
    refs = select_daily_snapshots(refs, policy=cfg.snapshot_policy, target_time=cfg.target_time)
    if not refs:
        raise FileNotFoundError(f"no saved {provider} chain snapshots found for {symbol} under {chain_root}")

    rows: list[dict] = []
    for ref in refs:
        chain = load_chain_snapshot(ref)
        quote_time = pd.to_datetime(chain["quote_time"], utc=True).median()
        quote_date = quote_time.tz_convert("America/New_York").tz_localize(None).normalize()
        mfiv_slices = mfiv_term_structure(chain, price_side=cfg.price_side)
        row = {
            "date": quote_date,
            "symbol": symbol.upper(),
            "provider": provider.lower(),
            "quote_time": quote_time,
            "chain_path": str(ref.path),
            "option_rows": int(len(chain)),
            "expiries": int(chain["expiry"].nunique()),
            "underlying_price": float(pd.to_numeric(chain["underlying_price"], errors="coerce").median()),
        }
        for tenor in cfg.mfiv_tenors:
            try:
                ct = constant_tenor_mfiv(mfiv_slices, float(tenor))
                row[f"mfiv_var_{tenor}"] = float(ct.implied_variance)
                row[f"mfiv_vol_{tenor}"] = float(ct.implied_volatility)
            except ValueError:
                row[f"mfiv_var_{tenor}"] = np.nan
                row[f"mfiv_vol_{tenor}"] = np.nan
        target_col = f"mfiv_var_{int(cfg.target_days)}"
        row["mfiv_var"] = row.get(target_col, np.nan)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    out = out.set_index(pd.DatetimeIndex(out.pop("date"), name="date"))
    dates = out.index

    # Term-structure diagnostics use trading-session windows, matching the
    # existing dashboard convention for RV3/RV9/RV30.
    rv_term = realized_term_structure(daily, windows=cfg.rv_windows, basis=cfg.rv_term_basis)
    target_trailing = rolling_integrated_variance(daily, int(cfg.target_days), basis=cfg.target_rv_basis)
    if cfg.rv_asof == "previous_session":
        rv_term = rv_term.shift(1)
        target_trailing = target_trailing.shift(1)
    rv_term = _reindex_daily(rv_term, dates)
    target_trailing = _reindex_daily(target_trailing, dates)

    for col in rv_term.columns:
        out[col] = rv_term[col]
    out["trailing_rv_var"] = target_trailing
    out["trailing_rv_vol"] = integrated_volatility(out["trailing_rv_var"])

    # Daily-aligned ex-post label: realized variance in sessions strictly after
    # the snapshot date, annualized to the target calendar horizon.
    fwd = forward_integrated_variance(daily, int(cfg.target_days), basis=cfg.target_rv_basis)
    out["forward_rv_var"] = fwd.reindex(dates)
    out["forward_rv_vol"] = integrated_volatility(out["forward_rv_var"])

    features = vrp_features(
        out["mfiv_var"],
        out["trailing_rv_var"],
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
    )
    for col in ("vrp", "vrp_z", "vrp_percentile", "mfiv_z", "mfiv_percentile"):
        out[col] = features[col]
    out["forward_vrp"] = forward_vrp_label(out["mfiv_var"], out["forward_rv_var"])
    out["vol_of_vol"] = vol_of_vol(out["mfiv_var"], window=20)

    if "rv_slope_3_30" in out:
        out["rv_slope_3_30_delta1"] = out["rv_slope_3_30"].diff()
        out["rv_slope_3_30_recent_peak_10"] = out["rv_slope_3_30"].shift(1).rolling(10, min_periods=1).max()
        out["rv_cooling_from_recent_shock"] = (
            (out["rv_slope_3_30"] < 0)
            & (out["rv_slope_3_30_recent_peak_10"] > 0)
        )

    return out.reset_index()


def save_vrp_history(
    history: pd.DataFrame,
    *,
    symbol: str,
    provider: str,
    root: str | Path = "data/derived/vrp",
) -> Path:
    path = Path(root) / f"provider={provider.lower()}" / f"symbol={symbol.upper()}"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "history.parquet"
    history.to_parquet(target, index=False)
    return target


def load_daily_variance(path: str | Path, *, date_col: str = "date", value_col: str = "integrated_variance") -> pd.Series:
    """Load a saved daily integrated-variance series from CSV or Parquet."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("daily variance file must be CSV or Parquet")
    if date_col not in frame or value_col not in frame:
        raise ValueError(f"daily variance file needs columns {date_col!r} and {value_col!r}")
    idx = pd.to_datetime(frame[date_col], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    return _prepare_daily(pd.Series(values.to_numpy(), index=idx, name=value_col))

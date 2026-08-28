import numpy as np
import pandas as pd
from scipy.stats import norm

from volforge.data.schema import add_derived_columns
from volforge.data.storage import save_chain_snapshot
from volforge.history import VRPHistoryConfig, build_vrp_history, save_vrp_history



def _parquet_fallback(monkeypatch):
    try:
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass

    def to_parquet(self, path, index=False, **kwargs):
        frame = self.reset_index(drop=True) if not index else self
        frame.to_pickle(path)

    def read_parquet(path, columns=None, **kwargs):
        frame = pd.read_pickle(path)
        return frame if columns is None else frame.loc[:, columns]

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet, raising=True)
    monkeypatch.setattr(pd, "read_parquet", read_parquet, raising=True)

def _bs_price(S, K, sigma, T, r, call):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _two_expiry_chain(day, sigma=0.20, spot=100.0, r=0.03):
    quote = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=12, minutes=30)
    quote = quote.tz_convert("UTC")
    rows = []
    for days in (20, 40):
        expiry = quote + pd.Timedelta(days=days)
        T = days / 365.25
        for K in np.arange(55.0, 151.0, 2.0):
            for right, call in (("C", True), ("P", False)):
                p = _bs_price(spot, K, sigma, T, r, call)
                rows.append({
                    "symbol": "SPY", "quote_time": quote, "expiry": expiry,
                    "strike": K, "right": right,
                    "bid": max(p - 0.001, 1e-6), "ask": p + 0.001,
                    "underlying_price": spot, "source": "test",
                })
    return add_derived_columns(pd.DataFrame(rows))


def test_history_builder_reads_saved_chains_and_adds_forward_labels(tmp_path, monkeypatch):
    _parquet_fallback(monkeypatch)
    chain_root = tmp_path / "chains"
    snapshot_days = pd.to_datetime(["2026-06-15", "2026-07-15", "2026-08-14"])
    for day in snapshot_days:
        save_chain_snapshot(_two_expiry_chain(day), provider="yahoo", root=chain_root)

    rv_idx = pd.date_range("2026-05-01", "2026-09-30", freq="B")
    daily = pd.Series(0.20**2 / 252.0, index=rv_idx, name="integrated_variance")
    cfg = VRPHistoryConfig(
        mfiv_tenors=(30,),
        rv_windows=(3, 9, 30),
        z_window=2,
        min_periods=1,
    )
    hist = build_vrp_history("SPY", daily, provider="yahoo", chain_root=chain_root, config=cfg)

    assert len(hist) == 3
    assert {
        "mfiv_var", "trailing_rv_var", "daily_rm", "forward_rv_var", "forward_vrp", "rv_slope_3_30",
        "atm_iv", "delta_ratio_25p", "delta_ratio_25c",
        "surface_parallel_shift", "surface_put_skew_change", "surface_downside_convexity_change",
    } <= set(hist.columns)
    assert hist["mfiv_var"].notna().all()
    assert hist["delta_ratio_25p"].notna().all()
    assert np.allclose(hist["delta_ratio_25p"], 1.0, atol=0.03)
    assert hist["forward_rv_var"].notna().all()
    assert np.allclose(hist["mfiv_var"], 0.20**2, atol=0.004)

    target = save_vrp_history(hist, symbol="SPY", provider="yahoo", root=tmp_path / "derived")
    assert target.exists()
    loaded = pd.read_parquet(target)
    assert len(loaded) == 3

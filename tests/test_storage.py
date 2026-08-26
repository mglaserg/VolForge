import json

import pandas as pd

from volforge.data.schema import add_derived_columns, expiry_datetime
from volforge.data.storage import (
    list_chain_snapshots,
    load_chain_snapshot,
    save_chain_snapshot,
    select_daily_snapshots,
)



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

def _tiny_chain(quote="2026-08-26 16:30:00+00:00", symbol="SPY"):
    q = pd.Timestamp(quote)
    exp = expiry_datetime("2026-09-18")
    rows = []
    for strike in (440.0, 450.0):
        for right in ("C", "P"):
            rows.append({
                "symbol": symbol,
                "quote_time": q,
                "expiry": exp,
                "strike": strike,
                "right": right,
                "bid": 2.0,
                "ask": 2.2,
                "underlying_price": 450.0,
                "source": "test",
            })
    return add_derived_columns(pd.DataFrame(rows))


def test_provider_partitioned_snapshot_roundtrip(tmp_path, monkeypatch):
    _parquet_fallback(monkeypatch)
    ref = save_chain_snapshot(_tiny_chain(), provider="yahoo", root=tmp_path)
    text = str(ref.path).replace("\\", "/")
    assert "provider=yahoo/symbol=SPY/date=2026-08-26/time=123000/chain.parquet" in text
    assert ref.path.exists()
    meta = json.loads((ref.path.parent / "metadata.json").read_text())
    assert meta["provider"] == "yahoo"
    assert meta["rows"] == 4

    loaded = load_chain_snapshot(ref)
    assert {"mid", "T", "dte"} <= set(loaded.columns)
    assert len(loaded) == 4


def test_list_and_select_daily_snapshots(tmp_path, monkeypatch):
    _parquet_fallback(monkeypatch)
    save_chain_snapshot(_tiny_chain("2026-08-26 14:00:00+00:00"), provider="yahoo", root=tmp_path)
    save_chain_snapshot(_tiny_chain("2026-08-26 19:30:00+00:00"), provider="yahoo", root=tmp_path)
    refs = list_chain_snapshots("SPY", provider="yahoo", root=tmp_path)
    assert len(refs) == 2
    latest = select_daily_snapshots(refs, policy="latest")
    assert len(latest) == 1
    assert latest[0].time == "153000"
    closest = select_daily_snapshots(refs, policy="closest", target_time="10:15")
    assert closest[0].time == "100000"

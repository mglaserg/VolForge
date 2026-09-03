from urllib.parse import parse_qs, urlparse

import pandas as pd

from volforge.data.alpaca import fetch_alpaca_bars
from volforge.data.intraday import (
    intraday_archive_path,
    load_intraday_archive,
    realized_archive_path,
    load_realized_archive,
    save_intraday_archive,
    save_realized_archive,
)
from volforge.realized import daily_integrated_variance, regular_session_bars


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


def test_alpaca_fetch_follows_pagination_and_normalises():
    calls = []

    def fake_request(url, headers):
        calls.append((url, headers))
        query = parse_qs(urlparse(url).query)
        if "page_token" not in query:
            return {
                "bars": [{"t": "2026-08-31T13:30:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10, "n": 2, "vw": 100.2}],
                "next_page_token": "next",
            }
        return {
            "bars": [{"t": "2026-08-31T13:35:00Z", "o": 100.5, "h": 101, "l": 100, "c": 100.8, "v": 12, "n": 3, "vw": 100.7}],
            "next_page_token": None,
        }

    bars = fetch_alpaca_bars(
        "spy",
        start="2026-08-31",
        timeframe="5Min",
        feed="iex",
        key_id="key",
        secret_key="secret",
        request_json=fake_request,
    )
    assert len(calls) == 2
    assert len(bars) == 2
    assert bars["symbol"].unique().tolist() == ["SPY"]
    assert bars["feed"].unique().tolist() == ["iex"]
    assert str(bars["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_intraday_archive_is_feed_aware_and_idempotent(tmp_path, monkeypatch):
    _parquet_fallback(monkeypatch)
    path = intraday_archive_path("SPY", provider="alpaca", feed="iex", root=tmp_path)
    assert "provider=alpaca" in str(path)
    assert "feed=iex" in str(path)

    a = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-31T13:30Z", "2026-08-31T13:35Z"], utc=True),
        "close": [100.0, 100.2],
    })
    b = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-31T13:35Z", "2026-08-31T13:40Z"], utc=True),
        "close": [100.25, 100.3],
    })
    save_intraday_archive(a, path)
    save_intraday_archive(b, path)
    got = load_intraday_archive(path)
    assert len(got) == 3
    assert got.loc[got["timestamp"] == pd.Timestamp("2026-08-31T13:35Z"), "close"].iloc[0] == 100.25


def test_regular_session_and_realized_archive(tmp_path, monkeypatch):
    _parquet_fallback(monkeypatch)
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-08-31T12:00Z",  # premarket ET
            "2026-08-31T13:30Z",
            "2026-08-31T19:55Z",
            "2026-09-01T13:30Z",
            "2026-09-01T19:55Z",
            "2026-09-01T22:00Z",  # after hours ET
        ], utc=True),
        "close": [99.0, 100.0, 101.0, 102.0, 102.5, 103.0],
    })
    rth = regular_session_bars(bars)
    assert len(rth) == 4
    daily = daily_integrated_variance(rth)
    assert len(daily) == 2
    path = realized_archive_path("SPY", provider="alpaca", feed="iex", root=tmp_path)
    save_realized_archive(daily, path)
    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == ["date", "integrated_variance"]
    assert len(loaded) == 2
    series = load_realized_archive(path)
    assert len(series) == 2
    assert series.name == "integrated_variance"

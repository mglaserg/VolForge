"""Alpaca historical-equity bars for VolForge realized-volatility research.

The default feed is IEX because it is available on Alpaca's free Basic plan.
The feed is always carried into the archive path so IEX and later SIP research
cannot be mixed accidentally.
"""

from __future__ import annotations

import json
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from volforge.config import get_env

__all__ = ["fetch_alpaca_bars"]

_BASE_URL = "https://data.alpaca.markets/v2/stocks"


def _credentials(key_id: str | None, secret_key: str | None) -> tuple[str, str]:
    key = key_id or get_env("APCA_API_KEY_ID")
    secret = secret_key or get_env("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials missing. Add APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY to .env or set them in the process environment."
        )
    return key, secret


def _request_json(url: str, headers: dict[str, str]) -> dict:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed Alpaca HTTPS URL
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Alpaca HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Alpaca request failed: {exc.reason}") from exc


def _rfc3339(value: str | pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def fetch_alpaca_bars(
    symbol: str,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    timeframe: str = "5Min",
    feed: str = "iex",
    adjustment: str = "raw",
    limit: int = 10_000,
    key_id: str | None = None,
    secret_key: str | None = None,
    request_json: Callable[[str, dict[str, str]], dict] | None = None,
) -> pd.DataFrame:
    """Download one symbol's historical bars, following Alpaca pagination."""
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    feed = str(feed).strip().lower()
    if feed not in {"iex", "sip", "boats", "otc"}:
        raise ValueError("feed must be one of iex, sip, boats, otc")
    if not (1 <= int(limit) <= 10_000):
        raise ValueError("limit must be between 1 and 10000")

    key, secret = _credentials(key_id, secret_key)
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    request_fn = request_json or _request_json
    params: dict[str, str | int] = {
        "timeframe": str(timeframe),
        "start": _rfc3339(start),
        "feed": feed,
        "adjustment": adjustment,
        "sort": "asc",
        "limit": int(limit),
    }
    if end is not None:
        params["end"] = _rfc3339(end)

    records: list[dict] = []
    page_token: str | None = None
    while True:
        page_params = dict(params)
        if page_token:
            page_params["page_token"] = page_token
        url = f"{_BASE_URL}/{symbol}/bars?{urlencode(page_params)}"
        payload = request_fn(url, headers)
        for bar in payload.get("bars") or []:
            records.append({
                "timestamp": bar.get("t"),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "trade_count": bar.get("n"),
                "vwap": bar.get("vw"),
                "symbol": symbol,
                "provider": "alpaca",
                "feed": feed,
                "timeframe": str(timeframe),
            })
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    columns = [
        "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap",
        "symbol", "provider", "feed", "timeframe",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame.from_records(records, columns=columns)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "close"])
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

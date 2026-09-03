"""ORATS option-chain adapter.

The adapter normalizes ORATS strike rows into VolForge's canonical chain schema.
It supports delayed/live end-of-day style snapshots and one-minute intraday
snapshots without leaking ORATS-specific fields into downstream analytics.

Authentication is read from ``ORATS_API_TOKEN`` unless ``token=...`` is passed.
The implementation uses the Python standard library so ORATS support does not
add another mandatory runtime dependency.
"""

from __future__ import annotations

from io import StringIO
import json
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from volforge.config import get_env

from .schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, add_derived_columns, expiry_datetime, validate_chain

__all__ = ["ORATSProvider", "fetch_chain"]

_COLUMN_ORDER = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)
_BASE = "https://api.orats.io/datav2"


class ORATSProvider:
    """Provider implementation for ORATS strike-chain endpoints."""

    name = "orats"

    # Imported lazily to avoid a provider.py <-> orats.py import cycle.
    @property
    def capabilities(self):
        from .provider import ProviderCapabilities

        return ProviderCapabilities(
            historical_chains=True,
            intraday_history=True,
            live_quotes=True,
            full_bid_ask=True,
        )

    def __init__(self, token: str | None = None, transport: Callable | None = None):
        self.token = token
        self._transport = transport or _http_get

    def fetch_chain(
        self,
        symbol: str,
        max_expiries: int | None = None,
        dte_range: tuple[float, float] | None = (7.0, 120.0),
        settlement: str = "default",
        *,
        trade_date=None,
        intraday: bool = False,
        live: bool = False,
        **_: object,
    ) -> pd.DataFrame:
        """Fetch an ORATS chain and return canonical VolForge rows.

        Parameters
        ----------
        trade_date:
            For EOD history, a date accepted by ORATS (YYYY-MM-DD). For
            intraday history, a timestamp formatted as YYYYMMDDHHMM or any
            value coercible by pandas; it is converted to US/Eastern.
        intraday:
            Use the one-minute chain endpoint. Historical intraday data is
            selected automatically when ``trade_date`` is supplied.
        live:
            Use ORATS live endpoints. Live access requires the applicable ORATS
            agreements/entitlements. Ignored for historical requests.
        """
        token = self.token or get_env("ORATS_API_TOKEN")
        if not token:
            raise RuntimeError(
                "ORATS API token missing; add ORATS_API_TOKEN to .env, "
                "set it in the process environment, or pass token=..."
            )

        params: dict[str, str] = {"token": token, "ticker": symbol.upper()}
        if dte_range is not None and not intraday:
            params["dte"] = f"{dte_range[0]:g},{dte_range[1]:g}"

        if intraday:
            if trade_date is None:
                path = "/live/one-minute/strikes/chain" if live else "/one-minute/strikes/chain"
            else:
                path = "/hist/live/one-minute/strikes/chain" if live else "/hist/one-minute/strikes/chain"
                params["tradeDate"] = _format_intraday_time(trade_date)
            payload, content_type = self._transport(_BASE + path, params)
            raw = _parse_csv(payload)
        else:
            if trade_date is not None:
                path = "/hist/strikes"
                params["tradeDate"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
            else:
                path = "/live/strikes" if live else "/strikes"
            payload, content_type = self._transport(_BASE + path, params)
            raw = _parse_json_rows(payload)

        if raw.empty:
            raise RuntimeError(f"ORATS returned no rows for {symbol}")

        frame = _normalise_orats(raw, symbol=symbol, settlement=settlement)
        if dte_range is not None:
            frame = frame[(frame["dte"] >= dte_range[0]) & (frame["dte"] <= dte_range[1])]
        if max_expiries is not None and not frame.empty:
            keep = sorted(frame["expiry"].unique())[:max_expiries]
            frame = frame[frame["expiry"].isin(keep)]
        if frame.empty:
            raise RuntimeError(f"ORATS returned no usable expiries for {symbol}")
        return frame.sort_values(["expiry", "strike", "right"]).reset_index(drop=True)


def fetch_chain(symbol: str, **kwargs) -> pd.DataFrame:
    """Convenience wrapper mirroring the legacy Yahoo adapter API."""
    return ORATSProvider(token=kwargs.pop("token", None)).fetch_chain(symbol, **kwargs)


def _http_get(url: str, params: dict[str, str]) -> tuple[bytes, str]:
    request_url = f"{url}?{urlencode(params)}"
    with urlopen(request_url, timeout=60) as response:  # nosec B310 - fixed HTTPS ORATS host
        return response.read(), response.headers.get("content-type", "")


def _parse_json_rows(payload: bytes | str) -> pd.DataFrame:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    obj = json.loads(text)
    if isinstance(obj, dict):
        rows = obj.get("data", [])
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    return pd.DataFrame(rows)


def _parse_csv(payload: bytes | str) -> pd.DataFrame:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    return pd.read_csv(StringIO(text))


def _format_intraday_time(value) -> str:
    text = str(value)
    if len(text) == 12 and text.isdigit():
        return text
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return ts.strftime("%Y%m%d%H%M")


def _common_quote_time(raw: pd.DataFrame) -> pd.Timestamp:
    for col in ("snapShotDate", "updatedAt", "quoteDate"):
        if col in raw:
            values = pd.to_datetime(raw[col], errors="coerce", utc=True).dropna()
            if len(values):
                return pd.Timestamp(values.max())

    if "tradeDate" in raw and len(raw):
        # EOD ORATS rows are represented at the US close when no finer timestamp
        # is supplied. This keeps expiry T on the same clock as Yahoo snapshots.
        d = pd.Timestamp(str(raw["tradeDate"].iloc[0])[:10])
        return (d + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")
    return pd.Timestamp.now(tz="UTC")


def _num(raw: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name not in raw:
        return pd.Series(default, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[name], errors="coerce").astype("float64")


def _normalise_orats(raw: pd.DataFrame, symbol: str, settlement: str) -> pd.DataFrame:
    quote_time = _common_quote_time(raw)
    expiry = pd.to_datetime(raw["expirDate"], errors="coerce").map(
        lambda x: expiry_datetime(x, settlement) if pd.notna(x) else pd.NaT
    )
    strike = _num(raw, "strike")
    spot = _num(raw, "spotPrice")
    if spot.isna().all() or (spot <= 0).all():
        spot = _num(raw, "stockPrice")
    else:
        fallback = _num(raw, "stockPrice")
        spot = spot.where((spot > 0) & spot.notna(), fallback)

    parts = []
    for right, prefix in (("C", "call"), ("P", "put")):
        out = pd.DataFrame(index=raw.index)
        out["symbol"] = symbol.upper()
        out["quote_time"] = quote_time
        out["expiry"] = expiry
        out["strike"] = strike
        out["right"] = right
        out["bid"] = _num(raw, f"{prefix}BidPrice")
        out["ask"] = _num(raw, f"{prefix}AskPrice")
        out["underlying_price"] = spot
        out["last"] = np.nan
        out["volume"] = _num(raw, f"{prefix}Volume")
        out["open_interest"] = _num(raw, f"{prefix}OpenInterest")
        out["last_trade_time"] = pd.NaT
        out["vendor_iv"] = _num(raw, f"{prefix}MidIv")
        out["source"] = "orats"
        parts.append(out[_COLUMN_ORDER])

    frame = pd.concat(parts, ignore_index=True)
    frame = frame.dropna(subset=["expiry", "strike", "bid", "ask", "underlying_price"])
    frame = frame[frame["underlying_price"] > 0].copy()
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True)
    frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True)
    frame["last_trade_time"] = pd.to_datetime(frame["last_trade_time"], utc=True)
    validate_chain(frame)
    return add_derived_columns(frame)

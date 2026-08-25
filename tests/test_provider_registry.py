"""Tests for the additive provider abstraction boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from volforge.data.provider import (
    ProviderCapabilities,
    available_providers,
    fetch_chain,
    get_provider,
    register_provider,
)


class DummyProvider:
    name = "dummy"
    capabilities = ProviderCapabilities(historical_chains=True)

    def fetch_chain(
        self,
        symbol,
        max_expiries=None,
        dte_range=(7.0, 120.0),
        settlement="default",
        **kwargs,
    ):
        quote_time = pd.Timestamp("2026-08-25 14:00", tz="UTC")
        expiry = pd.Timestamp("2026-09-25 20:00", tz="UTC")
        return pd.DataFrame(
            {
                "symbol": [symbol, symbol],
                "quote_time": [quote_time, quote_time],
                "expiry": [expiry, expiry],
                "strike": [100.0, 100.0],
                "right": ["C", "P"],
                "bid": [4.8, 4.6],
                "ask": [5.0, 4.8],
                "underlying_price": [100.0, 100.0],
                "source": ["dummy", "dummy"],
            }
        )


def test_yahoo_is_available_without_modifying_yahoo_module():
    assert "yahoo" in available_providers()
    yahoo = get_provider("YAHOO")
    assert yahoo.name == "yahoo"
    assert yahoo.capabilities.historical_chains is False


def test_yahoo_wrapper_delegates_to_existing_adapter(monkeypatch):
    import volforge.data.yahoo as yahoo_module

    def fake_yahoo_fetch(symbol, **kwargs):
        return DummyProvider().fetch_chain(symbol, **kwargs)

    monkeypatch.setattr(yahoo_module, "fetch_chain", fake_yahoo_fetch)
    chain = fetch_chain("TEST", provider="yahoo")

    assert chain["source"].eq("dummy").all()
    assert {"mid", "spread", "rel_spread", "T", "dte"} <= set(chain.columns)


def test_custom_provider_is_validated_at_boundary():
    register_provider(DummyProvider(), replace=True)
    chain = fetch_chain("TEST", provider="dummy")

    assert set(chain["right"]) == {"C", "P"}
    assert chain["source"].eq("dummy").all()
    assert chain["quote_time"].dt.tz is not None
    assert chain["expiry"].dt.tz is not None


def test_unknown_provider_has_clear_error():
    with pytest.raises(ValueError, match="unknown option-data provider"):
        get_provider("not-a-provider")

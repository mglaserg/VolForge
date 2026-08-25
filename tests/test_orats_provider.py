import json

import pandas as pd

from volforge.data.orats import ORATSProvider
from volforge.data.provider import available_providers, fetch_chain
from volforge.data.schema import REQUIRED_COLUMNS


def _rows():
    return {
        "data": [
            {
                "ticker": "SPY",
                "tradeDate": "2026-08-25",
                "expirDate": "2026-09-18",
                "strike": 600,
                "spotPrice": 610.0,
                "callBidPrice": 18.0,
                "callAskPrice": 18.2,
                "putBidPrice": 7.8,
                "putAskPrice": 8.0,
                "callVolume": 100,
                "putVolume": 90,
                "callOpenInterest": 1000,
                "putOpenInterest": 900,
                "callMidIv": 0.21,
                "putMidIv": 0.22,
                "updatedAt": "2026-08-25T19:45:00Z",
            },
            {
                "ticker": "SPY",
                "tradeDate": "2026-08-25",
                "expirDate": "2026-09-18",
                "strike": 610,
                "spotPrice": 610.0,
                "callBidPrice": 12.0,
                "callAskPrice": 12.2,
                "putBidPrice": 11.7,
                "putAskPrice": 11.9,
                "callVolume": 120,
                "putVolume": 110,
                "callOpenInterest": 1200,
                "putOpenInterest": 1100,
                "callMidIv": 0.20,
                "putMidIv": 0.20,
                "updatedAt": "2026-08-25T19:45:00Z",
            },
        ]
    }


def test_orats_normalises_to_canonical_chain():
    def transport(url, params):
        assert url.endswith("/strikes")
        assert params["ticker"] == "SPY"
        return json.dumps(_rows()).encode(), "application/json"

    provider = ORATSProvider(token="test-token", transport=transport)
    df = provider.fetch_chain("SPY", dte_range=None)
    assert len(df) == 4
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert set(df["right"]) == {"C", "P"}
    assert set(df["source"]) == {"orats"}
    assert str(df["quote_time"].dtype) == "datetime64[ns, UTC]"


def test_orats_is_registered_builtin():
    assert "orats" in available_providers()


def test_generic_fetch_accepts_orats_instance():
    def transport(url, params):
        return json.dumps(_rows()).encode(), "application/json"

    provider = ORATSProvider(token="x", transport=transport)
    df = fetch_chain("SPY", provider=provider, dte_range=None)
    assert len(df) == 4

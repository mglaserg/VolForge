import numpy as np
import pandas as pd
from scipy.stats import norm

from volforge.data.schema import add_derived_columns
from volforge.mfiv import constant_tenor_mfiv, mfiv_for_expiry


def _bs_price(S, K, sigma, T, r, call):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _chain(days, sigma=0.20, spot=100.0, r=0.03):
    quote = pd.Timestamp("2026-08-25 16:00", tz="America/New_York").tz_convert("UTC")
    expiry = quote + pd.Timedelta(days=days)
    T = days / 365.25
    # Wide, dense grid is important because the model-free formula integrates tails.
    strikes = np.arange(40.0, 181.0, 1.0)
    rows = []
    for K in strikes:
        for right, call in (("C", True), ("P", False)):
            p = _bs_price(spot, K, sigma, T, r, call)
            rows.append({
                "symbol": "TEST", "quote_time": quote, "expiry": expiry,
                "strike": K, "right": right,
                "bid": max(p - 0.0005, 1e-6), "ask": p + 0.0005,
                "underlying_price": spot,
            })
    return add_derived_columns(pd.DataFrame(rows))


def test_mfiv_recovers_flat_black_scholes_variance():
    c = _chain(30, sigma=0.20)
    s = mfiv_for_expiry(c, c["expiry"].iloc[0], min_strikes=20)
    assert abs(s.implied_variance - 0.20**2) < 0.003


def test_constant_tenor_interpolates_total_variance():
    c1 = _chain(20, sigma=0.18)
    c2 = _chain(40, sigma=0.22)
    s1 = mfiv_for_expiry(c1, c1["expiry"].iloc[0], min_strikes=20)
    s2 = mfiv_for_expiry(c2, c2["expiry"].iloc[0], min_strikes=20)
    ct = constant_tenor_mfiv([s1, s2], 30)
    expected_theta = s1.total_variance + 0.5 * (s2.total_variance - s1.total_variance)
    assert np.isclose(ct.implied_variance, expected_theta / (30 / 365.25), rtol=1e-10)

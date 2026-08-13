"""End-to-end test of the data layer using a mock that mimics yfinance's
exact output shape (column names, dtypes, quirks). No network required.

The mock deliberately injects the pathologies real Yahoo chains contain:
zero bids, crossed quotes, stale untraded strikes, absurd wing spreads, and
a nonsense vendor IV column.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, "../volforge/src")

from volforge import SVIParams, calibrate_svi, svi_total_variance, residual_report
from volforge.blackscholes import black_price
from volforge.data.clean import CleanConfig, clean_chain, matched_pairs
from volforge.data.pipeline import build_all_slices, build_slice
from volforge.data.yahoo import fetch_chain

rng = np.random.default_rng(11)
SPOT = 612.40
TRUE = {  # dte -> SVI params. Tuned to realistic SPY levels: ATM ~13-16.5%
    9:  SVIParams(a=0.000086, b=0.0050, rho=-0.82, m=0.012, sigma=0.055),
    23: SVIParams(a=0.000266, b=0.0100, rho=-0.78, m=0.018, sigma=0.090),
    44: SVIParams(a=0.000376, b=0.0170, rho=-0.74, m=0.022, sigma=0.130),
    72: SVIParams(a=0.000605, b=0.0250, rho=-0.71, m=0.026, sigma=0.170),
}


class MockTicker:
    """Mimics yfinance.Ticker for the surface we actually use."""

    def __init__(self):
        today = pd.Timestamp.now(tz="UTC").tz_convert("America/New_York").normalize()
        self._map = {(today + pd.Timedelta(days=d)).strftime("%Y-%m-%d"): d
                     for d in TRUE}
        self.options = tuple(sorted(self._map))
        self.fast_info = {"last_price": SPOT}

    def option_chain(self, exp_str):
        dte = self._map[exp_str]
        T = dte / 365.25
        p = TRUE[dte]
        K = np.round(np.arange(SPOT * 0.72, SPOT * 1.28, 5.0) / 5.0) * 5.0
        F = SPOT * np.exp(0.043 * T)              # ~4.3% carry
        k = np.log(K / F)
        iv = np.sqrt(np.maximum(svi_total_variance(k, p), 1e-8) / T)
        D = np.exp(-0.043 * T)

        frames = {}
        for right, is_call in (("calls", True), ("puts", False)):
            mid = np.asarray(black_price(F, K, iv, T, is_call, D), float)
            # spread widens with distance from spot and shrinks with price
            half = np.maximum(0.01, 0.02 + 0.9 * np.abs(k) ** 1.5) * np.maximum(mid, 0.3) ** 0.35
            mid = mid + rng.normal(0, half / 4)
            bid = np.maximum(mid - half, 0.0).round(2)
            ask = (mid + half).round(2)

            vol = rng.poisson(np.maximum(2000 * np.exp(-8 * np.abs(k)), 0.05))
            oi = rng.poisson(np.maximum(9000 * np.exp(-5 * np.abs(k)), 0.2))

            # --- inject real-world garbage ---
            deep = np.abs(k) > 0.22
            bid = np.where(deep & (rng.random(len(K)) < 0.45), 0.0, bid)   # zero bid
            crossed = rng.random(len(K)) < 0.02
            ask = np.where(crossed, bid * 0.9, ask)                        # crossed
            ask = np.where(rng.random(len(K)) < 0.03, ask * 6, ask)        # absurd spread

            now = pd.Timestamp.now(tz="UTC")
            last_trade = np.where(vol > 0, now, now - pd.Timedelta(days=40))

            frames[right] = pd.DataFrame({
                "contractSymbol": [f"SPY{i}" for i in range(len(K))],
                "strike": K,
                "lastPrice": mid.round(2),
                "bid": bid,
                "ask": ask,
                "volume": vol.astype(float),
                "openInterest": oi.astype(float),
                "impliedVolatility": iv * rng.uniform(0.85, 1.15, len(K)),  # Yahoo junk
                "lastTradeDate": pd.to_datetime(last_trade, utc=True),
                "inTheMoney": (K < SPOT) if is_call else (K > SPOT),
            })
        return SimpleNamespace(calls=frames["calls"], puts=frames["puts"])


# ---------------------------------------------------------------- fetch
raw = fetch_chain("SPY", dte_range=(5, 120), ticker=MockTicker())
print(f"[fetch]  {len(raw)} rows, {raw['expiry'].nunique()} expiries, "
      f"cols ok={set(['mid','spread','T','dte']) <= set(raw.columns)}")

# ---------------------------------------------------------------- clean
print("\n[clean]")
clean, rep = clean_chain(raw, CleanConfig(dte_range=(5, 120)))
assert rep.total_kept > 100, "cleaning is too aggressive"

# ---------------------------------------------------------------- slices
print("\n[slices]")
slices = build_all_slices(clean)
for s in slices:
    print(f"  {s}  spot={s.spot:.2f}  implied r={s.forward_fit.rate:.4f}")
assert len(slices) == len(TRUE)
for s in slices:
    assert s.forward_fit.is_sane, f"parity fit not sane for dte={s.dte:.0f}"

# ------------------------------------------------------- calibrate + recover
print("\n[calibrate]")
for s in slices:
    fit = calibrate_svi(s.k, s.w, s.T, weights=s.weights)
    dte_key = min(TRUE, key=lambda d: abs(d - s.dte))
    t = TRUE[dte_key]
    atm_fit = np.sqrt(svi_total_variance(0.0, fit.params) / s.T)
    atm_true = np.sqrt(svi_total_variance(0.0, t) / s.T)
    print(f"  dte={s.dte:5.1f}  n={fit.n_obs:3d}  rmse={fit.rmse_iv*100:5.2f}vp  "
          f"ATM {atm_fit*100:5.2f}% vs true {atm_true*100:5.2f}%  "
          f"rho={fit.params.rho:+.3f} (true {t.rho:+.3f})  "
          f"reliable={fit.is_reliable} {list(fit.boundary_flags)}")
    assert abs(atm_fit - atm_true) < 0.01, "ATM vol off by >1 vol point"

    rr = residual_report(s.k, s.iv, fit.params, s.T,
                         iv_half_spread=s.half_spread / np.maximum(
                             np.asarray(__import__("volforge").black_vega(
                                 s.forward, s.strikes, s.iv, s.T,
                                 s.forward_fit.discount), float), 1e-12))
    print(f"        residuals: {rr.summary()}")

# ---------------------------------------------------------------- pairs sanity
pairs = matched_pairs(clean, slices[0].expiry)
print(f"\n[pairs] {len(pairs)} matched strikes on nearest expiry, "
      f"cols include combined_spread={('combined_spread' in pairs.columns)}")

print("\nALL DATA-LAYER CHECKS PASSED")

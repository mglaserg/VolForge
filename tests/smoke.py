import sys
import numpy as np

# sys.path.insert(0, "../volforge/src")

from volforge.blackscholes import black_price, implied_vol, implied_vol_vec
from volforge.forward import fit_forward, log_moneyness
from volforge.svi import SVIParams, svi_total_variance, calibrate_svi, is_butterfly_free

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- 1. IV round trip
F, T, D = 500.0, 45 / 365.25, np.exp(-0.045 * 45 / 365.25)
errs, dropped = [], 0
for K in [400, 450, 490, 500, 510, 550, 620]:
    for s in [0.08, 0.15, 0.35, 0.9]:
        cp = K >= F                      # OTM leg only, as in production
        p = float(black_price(F, K, s, T, cp, D))
        iv = implied_vol(p, F, K, T, cp, D)
        if np.isnan(iv):
            dropped += 1                 # time value below float resolution
        else:
            errs.append(abs(iv - s))
print(f"[1] IV inversion max abs error: {max(errs):.2e}  (dropped {dropped} degenerate)")
assert max(errs) < 1e-8

# ------------------------------------------------- 2. forward from parity regression
true = SVIParams(a=0.012, b=0.14, rho=-0.72, m=0.021, sigma=0.16)
K = np.arange(380.0, 641.0, 5.0)
k = np.log(K / F)
iv_true = np.sqrt(svi_total_variance(k, true) / T)

C = np.asarray(black_price(F, K, iv_true, T, True, D), dtype=float)
P = np.asarray(black_price(F, K, iv_true, T, False, D), dtype=float)
# quote noise: half-spread grows in the wings
half = 0.02 + 0.15 * np.abs(k)
C_mid = C + rng.normal(0, half / 3)
P_mid = P + rng.normal(0, half / 3)

ff = fit_forward(K, C_mid, P_mid, T, spot=F, moneyness_window=0.10,
                 weights=1.0 / half)
print(f"[2] F={ff.forward:.4f} (true {F})  D={ff.discount:.6f} (true {D:.6f})  "
      f"R2={ff.r_squared:.6f}  n={ff.n_pairs}  sane={ff.is_sane}")
assert abs(ff.forward - F) < 0.5
assert abs(ff.discount - D) < 5e-4

# --------------------------------------------- 3. OTM-only IVs from our own forward
is_call = K >= ff.forward
mid = np.where(is_call, C_mid, P_mid)
iv = implied_vol_vec(mid, ff.forward, K, T, is_call, ff.discount)
kk = log_moneyness(K, ff.forward)
ok = np.isfinite(iv)
w_obs = iv[ok] ** 2 * T
print(f"[3] inverted {ok.sum()}/{len(K)} quotes, "
      f"mean |iv - true| = {np.mean(np.abs(iv[ok] - iv_true[ok])):.5f}")

# --------------------------------------------------------------- 4. SVI calibration
weights = 1.0 / half[ok]
fit = calibrate_svi(kk[ok], w_obs, T, weights=weights)
p = fit.params
print(f"[4] fitted  a={p.a:.5f} b={p.b:.5f} rho={p.rho:.4f} m={p.m:.5f} sigma={p.sigma:.5f}")
print(f"    true    a={true.a:.5f} b={true.b:.5f} rho={true.rho:.4f} m={true.m:.5f} sigma={true.sigma:.5f}")
print(f"    rmse_iv={fit.rmse_iv*100:.3f} vol pts   max_err={fit.max_abs_err_iv*100:.3f} pts")
print(f"    butterfly_free={fit.butterfly_free}  min_g={fit.min_durrleman_g:.5f}  "
      f"slopes={fit.wing_slopes[0]:.3f}/{fit.wing_slopes[1]:.3f}")
assert fit.rmse_iv < 0.01
assert fit.butterfly_free

# ------------------------------------------------- 5. stability across noise draws
params = []
for s in range(15):
    r = np.random.default_rng(100 + s)
    Cm = C + r.normal(0, half / 3)
    Pm = P + r.normal(0, half / 3)
    f2 = fit_forward(K, Cm, Pm, T, spot=F, weights=1.0 / half)
    ic = K >= f2.forward
    m2 = np.where(ic, Cm, Pm)
    iv2 = implied_vol_vec(m2, f2.forward, K, T, ic, f2.discount)
    g = np.isfinite(iv2)
    fit2 = calibrate_svi(log_moneyness(K, f2.forward)[g], iv2[g] ** 2 * T, T,
                         weights=1.0 / half[g], x0=None)
    params.append(fit2.params.as_array())
P_ = np.array(params)
print("[5] param std across 15 noise draws (a,b,rho,m,sigma):",
      np.array2string(P_.std(axis=0), precision=5))
print("    ATM iv std:", f"{np.std([np.sqrt(svi_total_variance(0.0, SVIParams(*q))/T) for q in P_])*100:.4f} vol pts")

# ---------------------------------------------------- 6. arbitrage detector catches bad params
bad = SVIParams(a=0.02, b=0.9, rho=-0.95, m=0.0, sigma=0.02)
okb, ming, kb = is_butterfly_free(bad)
print(f"[6] deliberately bad slice: butterfly_free={okb} min_g={ming:.4f} at k={kb:.3f}")
assert not okb

print("\nALL CHECKS PASSED")

"""Plotting.

Every function takes an optional `ax` and returns it, so plots compose into
whatever figure you want. Nothing here calls plt.show(); that stays the
caller's decision so the same functions work in notebooks and scripts.

The one plot worth making first is `plot_residuals_vs_spread`. It answers the
question the whole project depends on, and it answers it visually in a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "plot_smile",
    "plot_residuals_vs_spread",
    "plot_slice_diagnostics",
    "plot_surface_3d",
    "plot_surface_heatmap",
    "plot_term_structure",
    "plot_feature_history",
    "plot_pca_loadings",
    "plot_pca_scores",
    "plot_pca_residual_heatmap",
    "plot_signal_buckets",
]


def _mpl():
    import matplotlib.pyplot as plt
    return plt


def plot_smile(slc, fit=None, ax=None, show_spread=True):
    """Market IV against the fitted SVI curve, with bid/ask error bars."""
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(9, 5))[1]

    if show_spread:
        from .blackscholes import black_vega
        vega = np.asarray(black_vega(slc.forward, slc.strikes, slc.iv, slc.T,
                                     slc.forward_fit.discount), float)
        hs_iv = np.where(vega > 1e-10, slc.half_spread / np.maximum(vega, 1e-10), np.nan)
        ax.errorbar(slc.k, slc.iv * 100, yerr=hs_iv * 100, fmt="none",
                    ecolor="0.75", elinewidth=1.2, capsize=2, zorder=1,
                    label="bid/ask")

    calls, puts = slc.is_call, ~slc.is_call
    ax.scatter(slc.k[calls], slc.iv[calls] * 100, s=26, marker="o",
               facecolors="none", edgecolors="tab:blue", label="calls (OTM)", zorder=3)
    ax.scatter(slc.k[puts], slc.iv[puts] * 100, s=26, marker="s",
               facecolors="none", edgecolors="tab:red", label="puts (OTM)", zorder=3)

    if fit is not None:
        from .svi import svi_iv
        kk = np.linspace(slc.k.min() - 0.02, slc.k.max() + 0.02, 300)
        ax.plot(kk, svi_iv(kk, fit.params, slc.T) * 100, "k-", lw=1.8,
                label="SVI fit", zorder=2)
        flag = "" if fit.is_reliable else f"  [{','.join(fit.boundary_flags)}]"
        ax.set_title(f"{slc.dte:.0f}d smile   F={slc.forward:.2f}   "
                     f"RMSE={fit.rmse_iv*100:.2f}vp   n={fit.n_obs}{flag}")
    else:
        ax.set_title(f"{slc.dte:.0f}d smile   F={slc.forward:.2f}")

    ax.axvline(0, color="0.85", lw=1, zorder=0)
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("implied vol (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_residuals_vs_spread(slc, fit, ax=None):
    """Fit residuals against the bid/ask band. The plot that decides the project.

    Points inside the shaded band are indistinguishable from quote noise: there
    is no way to trade them, because crossing the spread costs more than the
    apparent edge. If essentially everything sits inside the band, the residual
    strategy has no room to work and the phases built on it will not survive
    transaction costs.
    """
    plt = _mpl()
    from .blackscholes import black_vega
    from .svi import svi_iv

    ax = ax or plt.subplots(figsize=(9, 4))[1]

    vega = np.asarray(black_vega(slc.forward, slc.strikes, slc.iv, slc.T,
                                 slc.forward_fit.discount), float)
    hs_iv = np.where(vega > 1e-10, slc.half_spread / np.maximum(vega, 1e-10), np.nan)
    resid = (slc.iv - svi_iv(slc.k, fit.params, slc.T)) * 100
    band = hs_iv * 100

    order = np.argsort(slc.k)
    ax.fill_between(slc.k[order], -band[order], band[order], color="0.85",
                    label="±half-spread", zorder=0)
    inside = np.abs(resid) <= band
    ax.scatter(slc.k[inside], resid[inside], s=26, color="0.45",
               label="within spread", zorder=2)
    ax.scatter(slc.k[~inside], resid[~inside], s=38, color="tab:orange",
               edgecolors="k", linewidths=0.5, label="outside spread", zorder=3)

    ax.axhline(0, color="k", lw=1, zorder=1)
    frac = float(np.mean(~inside[np.isfinite(band)])) if np.isfinite(band).any() else np.nan
    ax.set_title(f"{slc.dte:.0f}d residuals vs market width   "
                 f"{frac:.0%} outside the spread")
    ax.set_xlabel("log-moneyness  k")
    ax.set_ylabel("market − SVI (vol points)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_slice_diagnostics(slc, fit, figsize=(11, 7)):
    """Smile, residuals-vs-spread, parity fit and Durrleman g in one figure."""
    plt = _mpl()
    from .forward import log_moneyness  # noqa: F401
    from .svi import durrleman_g

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_smile(slc, fit, ax=axes[0, 0])
    plot_residuals_vs_spread(slc, fit, ax=axes[0, 1])

    ax = axes[1, 0]
    ff = slc.forward_fit
    ax.text(0.05, 0.75, f"F = {ff.forward:.3f}\nD = {ff.discount:.6f}\n"
                        f"implied r = {ff.rate:.4f}\nparity R² = {ff.r_squared:.6f}\n"
                        f"pairs = {ff.n_pairs}\nsane = {ff.is_sane}",
            transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    ax.text(0.05, 0.30, f"a={fit.params.a:.5f}  b={fit.params.b:.5f}\n"
                        f"ρ={fit.params.rho:+.4f}  m={fit.params.m:+.5f}\n"
                        f"σ={fit.params.sigma:.5f}\n"
                        f"slopes={fit.wing_slopes[0]:.3f}/{fit.wing_slopes[1]:.3f}",
            transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    ax.set_title("parity & SVI parameters")
    ax.axis("off")

    ax = axes[1, 1]
    kk = np.linspace(slc.k.min() - 0.1, slc.k.max() + 0.1, 400)
    g = durrleman_g(kk, fit.params)
    ax.plot(kk, g, "k-", lw=1.5)
    ax.axhline(0, color="tab:red", lw=1, ls="--")
    ax.fill_between(kk, np.minimum(g, 0), 0, color="tab:red", alpha=0.3)
    ax.set_title(f"Durrleman g(k)   min={fit.min_durrleman_g:.4f}   "
                 f"butterfly-free={fit.butterfly_free}")
    ax.set_xlabel("k")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def plot_surface_3d(surface, figsize=(10, 7), cmap="viridis"):
    """3D implied-vol surface for one trade date."""
    plt = _mpl()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    K, T = np.meshgrid(surface.k_grid, surface.tenor_days)
    ax.plot_surface(K, T, surface.iv * 100, cmap=cmap, edgecolor="none", alpha=0.9)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("tenor (days)")
    ax.set_zlabel("implied vol (%)")
    ax.set_title(f"{surface.symbol} {surface.trade_date:%Y-%m-%d}   "
                 f"clean={surface.is_clean}")
    ax.view_init(elev=22, azim=-125)
    return fig


def plot_surface_heatmap(matrix, tenors, ks, ax=None, cmap="RdBu_r",
                         title="", center_zero=True, cbar_label=""):
    """Generic (tenor x k) heatmap, used for surfaces and residuals alike."""
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    M = np.asarray(matrix, float)
    vmax = np.nanmax(np.abs(M)) if center_zero else None
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap,
                   vmin=-vmax if center_zero else None,
                   vmax=vmax if center_zero else None,
                   extent=[ks[0], ks[-1], 0, len(tenors)])
    ax.set_yticks(np.arange(len(tenors)) + 0.5)
    ax.set_yticklabels([f"{int(t)}d" for t in tenors])
    ax.set_xlabel("log-moneyness k")
    ax.set_title(title)
    cb = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    return ax


def plot_term_structure(surfaces, ax=None):
    """ATM vol by tenor through time."""
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    surfaces = sorted(surfaces, key=lambda s: s.trade_date)
    dates = [s.trade_date for s in surfaces]
    kz = np.argmin(np.abs(surfaces[0].k_grid))
    for i, t in enumerate(surfaces[0].tenor_days):
        ax.plot(dates, [s.iv[i, kz] * 100 for s in surfaces], lw=1.3, label=f"{int(t)}d")
    ax.set_ylabel("ATM implied vol (%)")
    ax.set_title("ATM term structure through time")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    ax.grid(alpha=0.25)
    return ax


def plot_feature_history(features: pd.DataFrame, cols=None, ax=None):
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    cols = cols or [c for c in features.columns if c.startswith(("atm_", "skew_"))][:6]
    for c in cols:
        ax.plot(features.index, features[c], lw=1.2, label=c)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    ax.set_title("surface features through time")
    return ax


def plot_pca_loadings(model, n=4, figsize=(11, 3.0)):
    """One heatmap per component, reshaped back onto the (tenor x k) grid."""
    plt = _mpl()
    tenors = sorted({t for t, _ in model.node_index})
    ks = sorted({k for _, k in model.node_index})
    n = min(n, model.n_components)
    desc = model.describe()

    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), squeeze=False)
    for i in range(n):
        L = model.loadings[i].reshape(len(tenors), len(ks))
        plot_surface_heatmap(L, tenors, ks, ax=axes[0, i],
                             title=f"PC{i+1} · {desc.loc[i,'label']} · "
                                   f"{model.explained_variance_ratio[i]:.1%}")
    fig.tight_layout()
    return fig


def plot_pca_scores(scores: pd.DataFrame, n=3, ax=None):
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    for c in list(scores.columns)[:n]:
        ax.plot(scores.index, scores[c], lw=1.1, label=f"PC{c}")
    ax.axhline(0, color="0.7", lw=1)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("PC scores through time")
    return ax


def plot_pca_residual_heatmap(residuals: pd.DataFrame, date=None, ax=None):
    """Localised anomalies for one date: what the dominant modes did not explain."""
    date = date or residuals.index[-1]
    row = residuals.loc[date]
    tenors = sorted({t for t, _ in residuals.columns})
    ks = sorted({k for _, k in residuals.columns})
    M = row.to_numpy().reshape(len(tenors), len(ks))
    return plot_surface_heatmap(M, tenors, ks, ax=ax,
                                title=f"PCA residual  {pd.Timestamp(date):%Y-%m-%d}",
                                cbar_label="total variance")


def plot_signal_buckets(buckets: pd.DataFrame, ax=None):
    """Convergence by signal bucket. Look for monotonicity, not a single spike."""
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    x = np.arange(len(buckets))
    ax.bar(x, buckets["mean_net"], color=np.where(buckets["mean_net"] > 0,
                                                  "tab:green", "tab:red"), alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in buckets.index], rotation=45, ha="right", fontsize=8)
    for i, n in enumerate(buckets["n"]):
        ax.annotate(f"n={n}", (i, 0), ha="center", va="bottom", fontsize=7, color="0.3")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("mean net convergence (half-spreads)")
    ax.set_title("convergence by signal bucket, net of costs")
    ax.grid(alpha=0.25, axis="y")
    return ax

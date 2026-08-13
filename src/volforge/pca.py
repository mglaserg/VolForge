"""PCA on the volatility surface.

Three decisions here materially change what the components mean.

**PCA runs on daily *changes* in total variance, not levels.** Surface levels
are strongly non-stationary and highly autocorrelated; PCA on levels mostly
recovers the sample mean plus a slow trend, and the "explained variance" is
close to meaningless. Changes are far closer to stationary.

**Spot returns are regressed out first, by default.** Equity vol surfaces are
dominated by the spot-vol relationship. Without this step, PC1 is reliably just
"the index went down", it will absorb 70-90% of the variance, and every
subsequent component is polluted by whatever the linear fit left behind. What
remains after neutralising is the part of surface motion not explained by
direction, which is where relative-value signal lives.

**A caveat you should hold onto.** The surface grid is evaluated from SVI fits,
so with n expiries the matrix has rank at most 5n regardless of how fine the
grid is. PCA cannot discover structure the parameterisation cannot express, and
some of what you recover is SVI's own geometry rather than the market's. To
check how much, refit on raw inverted IVs interpolated to the grid and compare
loadings. `notes` on the fitted model is there to record which input you used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PCAModel", "fit_surface_pca", "reconstruct", "pca_residuals"]


@dataclass
class PCAModel:
    loadings: np.ndarray                  # (n_components, n_nodes)
    explained_variance_ratio: np.ndarray
    center: np.ndarray                    # per-node mean of the input changes
    scale: np.ndarray                     # per-node std used to standardise
    node_index: list                      # [(tenor_days, k), ...]
    n_days: int
    notes: str = ""
    beta_spot: np.ndarray | None = None   # per-node loading on spot return

    @property
    def n_components(self) -> int:
        return self.loadings.shape[0]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project standardised changes onto the components."""
        Z = (X.to_numpy() - self.center) / self.scale
        return pd.DataFrame(Z @ self.loadings.T, index=X.index,
                            columns=range(1, self.n_components + 1))

    def describe(self) -> pd.DataFrame:
        """Heuristic labels for each component from its loading pattern.

        These are hints, not conclusions. Always look at the loading plot before
        believing a label -- 'level' and 'term structure' in particular are easy
        to confuse when the tenor range is narrow.
        """
        tenors = np.array([t for t, _ in self.node_index])
        ks = np.array([k for _, k in self.node_index])
        rows = []
        for i, L in enumerate(self.loadings):
            same_sign = np.abs(np.sign(L).sum()) / len(L)
            corr_k = np.corrcoef(L, ks)[0, 1] if np.std(L) > 0 else 0.0
            corr_t = np.corrcoef(L, tenors)[0, 1] if np.std(L) > 0 else 0.0
            curv = np.corrcoef(L, ks ** 2)[0, 1] if np.std(L) > 0 else 0.0

            if same_sign > 0.85 and abs(corr_t) < 0.5:
                label = "level"
            elif abs(corr_k) > 0.7:
                label = "skew"
            elif abs(corr_t) > 0.6:
                label = "term structure"
            elif abs(curv) > 0.6:
                label = "curvature"
            else:
                label = "mixed"
            rows.append({"component": i + 1, "label": label,
                         "explained": self.explained_variance_ratio[i],
                         "sign_uniformity": same_sign, "corr_k": corr_k,
                         "corr_tenor": corr_t, "corr_k2": curv})
        return pd.DataFrame(rows)


def fit_surface_pca(
    panel: pd.DataFrame,
    n_components: int = 5,
    use_changes: bool = True,
    spot: pd.Series | None = None,
    standardise: bool = True,
    min_days: int = 60,
) -> tuple[PCAModel, pd.DataFrame]:
    """Fit PCA on a surface panel. Returns (model, scores).

    Parameters
    ----------
    panel : rows = trade_date, columns = MultiIndex (tenor_days, k), total variance
    spot : underlying price series indexed like `panel`. If given, each node's
        daily change is regressed on the spot log-return and the residual is
        used. See module docstring -- this is on by default for a reason.
    min_days : PCA on a short sample is not measurement, it is overfitting.
        With n_nodes columns you want at least a few hundred days.
    """
    panel = panel.sort_index()
    X = panel.diff().dropna() if use_changes else panel.copy()

    if len(X) < min_days:
        raise ValueError(
            f"only {len(X)} usable days; need >= {min_days}. With {panel.shape[1]} "
            f"grid nodes, PCA on a sample this short will fit noise. Keep "
            f"snapshotting and revisit."
        )

    beta = None
    if spot is not None:
        r = np.log(spot.sort_index()).diff().reindex(X.index)
        ok = r.notna()
        X, r = X.loc[ok], r.loc[ok]
        rv = r.to_numpy()
        # Per-node OLS on the spot return, intercept included; keep residuals.
        A = np.column_stack([np.ones_like(rv), rv])
        coef, *_ = np.linalg.lstsq(A, X.to_numpy(), rcond=None)
        beta = coef[1]
        X = pd.DataFrame(X.to_numpy() - A @ coef, index=X.index, columns=X.columns)

    center = X.to_numpy().mean(axis=0)
    scale = X.to_numpy().std(axis=0, ddof=1) if standardise else np.ones(X.shape[1])
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X.to_numpy() - center) / scale

    # SVD rather than an eigendecomposition of the covariance: better
    # conditioned when nodes are near-collinear, which grid surfaces always are.
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n_components = min(n_components, Vt.shape[0])
    var = S ** 2 / max(len(Z) - 1, 1)
    ratio = var / var.sum()

    # Sign convention: make each component's largest-magnitude loading positive,
    # so that scores are comparable across refits.
    L = Vt[:n_components].copy()
    for i in range(n_components):
        if L[i][np.argmax(np.abs(L[i]))] < 0:
            L[i] *= -1

    model = PCAModel(
        loadings=L,
        explained_variance_ratio=ratio[:n_components],
        center=center,
        scale=scale,
        node_index=list(panel.columns),
        n_days=len(Z),
        notes=("changes" if use_changes else "levels")
              + ("; spot-neutralised" if spot is not None else "")
              + ("; standardised" if standardise else ""),
        beta_spot=beta,
    )
    scores = pd.DataFrame(Z @ L.T, index=X.index, columns=range(1, n_components + 1))
    return model, scores


def reconstruct(model: PCAModel, scores: pd.DataFrame, n_pcs: int | None = None) -> pd.DataFrame:
    """Rebuild the (standardised, de-meaned) input from the first n_pcs."""
    n = n_pcs or model.n_components
    Z_hat = scores.iloc[:, :n].to_numpy() @ model.loadings[:n]
    cols = pd.MultiIndex.from_tuples(model.node_index, names=["tenor_days", "k"])
    return pd.DataFrame(Z_hat * model.scale + model.center, index=scores.index, columns=cols)


def pca_residuals(model: PCAModel, X: pd.DataFrame, scores: pd.DataFrame,
                  n_pcs: int | None = None) -> pd.DataFrame:
    """Actual minus reconstruction: localised surface anomalies.

    A large residual at one node means that part of the surface moved in a way
    the dominant modes do not explain. That is the candidate signal -- and also
    exactly what a single bad slice looks like, so check `Surface.is_clean` for
    the day before trading it.
    """
    fitted = reconstruct(model, scores, n_pcs)
    return X.reindex(fitted.index) - fitted

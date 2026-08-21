"""Persistence.

SQLite via the standard library, so there is no dependency to install and the
whole research history is one portable file. Every table has a natural key and
every write is an upsert on that key, which means re-running a day's fit
corrects it rather than duplicating it. That property matters more than it
sounds: research pipelines get re-run constantly, and silent duplicate rows
will quietly corrupt every downstream average you compute.

Tables
------
svi_parameters         one row per (symbol, trade_date, expiry)
ssvi_parameters        one global coupled-surface fit per (symbol, trade_date)
essvi_parameters       one extended-SSVI fit per (symbol, trade_date)
fengler_runs           one nonparametric Fengler surface run per (symbol, trade_date)
surface_grid           legacy raw-SVI fixed grid (kept for backward compatibility)
modeled_surface_grid   model-tagged fixed grid for SVI/SSVI/eSSVI/Fengler comparisons
features               one row per (symbol, trade_date)
pca_scores             one row per (symbol, trade_date, model_id, component)
pca_models             one row per fitted PCA, with loadings stored as JSON
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["VolDB"]


SCHEMA = """
CREATE TABLE IF NOT EXISTS svi_parameters (
    symbol           TEXT    NOT NULL,
    trade_date       TEXT    NOT NULL,
    expiry           TEXT    NOT NULL,
    dte              REAL,
    t_years          REAL,
    a                REAL,
    b                REAL,
    rho              REAL,
    m                REAL,
    sigma            REAL,
    forward          REAL,
    spot             REAL,
    discount         REAL,
    implied_rate     REAL,
    parity_r2        REAL,
    rmse             REAL,
    rmse_iv          REAL,
    max_abs_err_iv   REAL,
    n_obs            INTEGER,
    butterfly_free   INTEGER,
    min_durrleman_g  REAL,
    slope_left       REAL,
    slope_right      REAL,
    boundary_flags   TEXT,
    is_reliable      INTEGER,
    created_at       TEXT    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date, expiry)
);

CREATE INDEX IF NOT EXISTS ix_svi_symbol_date ON svi_parameters (symbol, trade_date);

CREATE TABLE IF NOT EXISTS ssvi_parameters (
    symbol                    TEXT NOT NULL,
    trade_date                TEXT NOT NULL,
    rho                       REAL,
    eta                       REAL,
    gamma                     REAL,
    phi_form                  TEXT,
    rmse                      REAL,
    rmse_iv                   REAL,
    max_abs_err_iv            REAL,
    n_obs                     INTEGER,
    n_slices                  INTEGER,
    n_theta_slices            INTEGER,
    success                   INTEGER,
    butterfly_free            INTEGER,
    calendar_free             INTEGER,
    min_durrleman_g           REAL,
    max_bfly_condition1       REAL,
    max_bfly_condition2       REAL,
    calendar_ratio_min        REAL,
    calendar_ratio_max        REAL,
    calendar_ratio_upper      REAL,
    theta_repair              REAL,
    theta_repair_fraction     REAL,
    theta_curve               TEXT,
    slice_rmse_iv             TEXT,
    created_at                TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS ix_ssvi_symbol_date ON ssvi_parameters (symbol, trade_date);

CREATE TABLE IF NOT EXISTS essvi_parameters (
    symbol                    TEXT NOT NULL,
    trade_date                TEXT NOT NULL,
    rho0                      REAL,
    rho_m                     REAL,
    a                         REAL,
    eta                       REAL,
    gamma                     REAL,
    theta_max                 REAL,
    phi_form                  TEXT,
    rmse                      REAL,
    rmse_iv                   REAL,
    max_abs_err_iv            REAL,
    n_obs                     INTEGER,
    n_slices                  INTEGER,
    n_theta_slices            INTEGER,
    success                   INTEGER,
    butterfly_free            INTEGER,
    calendar_free             INTEGER,
    min_durrleman_g           REAL,
    max_bfly_condition1       REAL,
    max_bfly_condition2       REAL,
    calendar_margin_min       REAL,
    calendar_lhs_max          REAL,
    calendar_gamma_min        REAL,
    calendar_gamma_max        REAL,
    theta_repair              REAL,
    theta_repair_fraction     REAL,
    theta_curve               TEXT,
    slice_rmse_iv             TEXT,
    created_at                TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS ix_essvi_symbol_date ON essvi_parameters (symbol, trade_date);

CREATE TABLE IF NOT EXISTS fengler_runs (
    symbol                    TEXT NOT NULL,
    trade_date                TEXT NOT NULL,
    smoothing_lambda          REAL,
    success                   INTEGER,
    butterfly_free            INTEGER,
    calendar_free             INTEGER,
    rmse_iv                   REAL,
    max_abs_err_iv            REAL,
    n_obs                     INTEGER,
    n_slices                  INTEGER,
    calendar_margin_min       REAL,
    slice_diagnostics         TEXT,
    created_at                TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS ix_fengler_symbol_date ON fengler_runs (symbol, trade_date);

CREATE TABLE IF NOT EXISTS surface_grid (
    symbol        TEXT NOT NULL,
    trade_date    TEXT NOT NULL,
    tenor_days    REAL NOT NULL,
    k             REAL NOT NULL,
    total_var     REAL,
    iv            REAL,
    extrapolated  INTEGER,
    PRIMARY KEY (symbol, trade_date, tenor_days, k)
);

CREATE INDEX IF NOT EXISTS ix_surf_symbol_date ON surface_grid (symbol, trade_date);

-- Model-aware grid.  The legacy surface_grid table remains the raw-SVI fixed
-- grid for backward compatibility; this table lets SVI/SSVI/eSSVI/Fengler
-- coexist without overwriting each other.
CREATE TABLE IF NOT EXISTS modeled_surface_grid (
    symbol        TEXT NOT NULL,
    trade_date    TEXT NOT NULL,
    model         TEXT NOT NULL,
    tenor_days    REAL NOT NULL,
    k             REAL NOT NULL,
    total_var     REAL,
    iv            REAL,
    extrapolated  INTEGER,
    PRIMARY KEY (symbol, trade_date, model, tenor_days, k)
);

CREATE INDEX IF NOT EXISTS ix_model_surf_symbol_date
ON modeled_surface_grid (symbol, trade_date, model);

CREATE TABLE IF NOT EXISTS features (
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS pca_models (
    model_id       TEXT PRIMARY KEY,
    symbol         TEXT,
    fitted_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    n_components   INTEGER,
    n_days         INTEGER,
    grid           TEXT,
    loadings       TEXT,
    explained_var  TEXT,
    center         TEXT,
    scale          TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS pca_scores (
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    component   INTEGER NOT NULL,
    score       REAL,
    PRIMARY KEY (symbol, trade_date, model_id, component)
);
"""


def _d(x) -> str:
    """Normalise any date-ish input to an ISO date string."""
    return pd.Timestamp(x).strftime("%Y-%m-%d")


class VolDB:
    """Thin wrapper over a SQLite file. Safe to construct repeatedly."""

    def __init__(self, path="volforge.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:") if self.path == ":memory:" else None
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = self._mem or sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            if self._mem is None:
                con.close()

    # ------------------------------------------------------------------ SVI
    def save_svi_fit(self, symbol, trade_date, fit, slc=None, **extra):
        """Upsert one calibrated slice. `fit` is an SVIFit, `slc` an optional Slice."""
        row = {
            "symbol": symbol,
            "trade_date": _d(trade_date),
            "expiry": _d(extra.pop("expiry", None) or slc.expiry),
            "dte": getattr(slc, "dte", None),
            "t_years": getattr(slc, "T", None),
            "forward": getattr(slc, "forward", None),
            "spot": getattr(slc, "spot", None),
            "discount": getattr(getattr(slc, "forward_fit", None), "discount", None),
            "implied_rate": getattr(getattr(slc, "forward_fit", None), "rate", None),
            "parity_r2": getattr(getattr(slc, "forward_fit", None), "r_squared", None),
        }
        f = fit.as_row()
        row.update(
            a=f["a"], b=f["b"], rho=f["rho"], m=f["m"], sigma=f["sigma"],
            rmse=f["rmse"], rmse_iv=f["rmse_iv"], max_abs_err_iv=f["max_abs_err_iv"],
            n_obs=f["n_obs"], butterfly_free=int(f["butterfly_free"]),
            min_durrleman_g=f["min_durrleman_g"], slope_left=f["slope_left"],
            slope_right=f["slope_right"], boundary_flags=f["boundary_flags"],
            is_reliable=int(f["is_reliable"]),
        )
        row.update(extra)
        self._upsert("svi_parameters", [row])
        return row

    def save_svi_fits(self, symbol, trade_date, fits_and_slices):
        return [self.save_svi_fit(symbol, trade_date, f, s) for f, s in fits_and_slices]

    def load_svi_fits(self, symbol, trade_date=None, reliable_only=False) -> pd.DataFrame:
        q = "SELECT * FROM svi_parameters WHERE symbol = ?"
        args = [symbol]
        if trade_date is not None:
            q += " AND trade_date = ?"
            args.append(_d(trade_date))
        if reliable_only:
            q += " AND is_reliable = 1"
        q += " ORDER BY trade_date, dte"
        with self.connect() as con:
            return pd.read_sql_query(q, con, params=args)

    # ----------------------------------------------------------------- SSVI
    def save_ssvi_fit(self, symbol, trade_date, fit):
        """Upsert one global SSVI calibration for a trade date."""
        row = fit.as_row()
        payload = {
            "symbol": symbol,
            "trade_date": _d(trade_date),
            "rho": row["rho"],
            "eta": row["eta"],
            "gamma": row["gamma"],
            "phi_form": row["phi_form"],
            "rmse": row["rmse"],
            "rmse_iv": row["rmse_iv"],
            "max_abs_err_iv": row["max_abs_err_iv"],
            "n_obs": row["n_obs"],
            "n_slices": row["n_slices"],
            "n_theta_slices": row["n_theta_slices"],
            "success": int(row["success"]),
            "butterfly_free": int(row["butterfly_free"]),
            "calendar_free": int(row["calendar_free"]),
            "min_durrleman_g": row["min_durrleman_g"],
            "max_bfly_condition1": row["max_bfly_condition1"],
            "max_bfly_condition2": row["max_bfly_condition2"],
            "calendar_ratio_min": row["calendar_ratio_min"],
            "calendar_ratio_max": row["calendar_ratio_max"],
            "calendar_ratio_upper": row["calendar_ratio_upper"],
            "theta_repair": row["theta_repair"],
            "theta_repair_fraction": row["theta_repair_fraction"],
            "theta_curve": json.dumps(fit.theta_curve.to_dict()),
            "slice_rmse_iv": json.dumps({str(k): float(v) for k, v in fit.slice_rmse_iv.items()}),
        }
        self._upsert("ssvi_parameters", [payload])
        return payload

    def load_ssvi_fits(self, symbol, trade_date=None) -> pd.DataFrame:
        q = "SELECT * FROM ssvi_parameters WHERE symbol = ?"
        args = [symbol]
        if trade_date is not None:
            q += " AND trade_date = ?"
            args.append(_d(trade_date))
        q += " ORDER BY trade_date"
        with self.connect() as con:
            return pd.read_sql_query(q, con, params=args)

    # ---------------------------------------------------------------- eSSVI
    def save_essvi_fit(self, symbol, trade_date, fit):
        """Upsert one global eSSVI calibration for a trade date."""
        row = fit.as_row()
        payload = {
            "symbol": symbol,
            "trade_date": _d(trade_date),
            "rho0": row["rho0"],
            "rho_m": row["rho_m"],
            "a": row["a"],
            "eta": row["eta"],
            "gamma": row["gamma"],
            "theta_max": row["theta_max"],
            "phi_form": row["phi_form"],
            "rmse": row["rmse"],
            "rmse_iv": row["rmse_iv"],
            "max_abs_err_iv": row["max_abs_err_iv"],
            "n_obs": row["n_obs"],
            "n_slices": row["n_slices"],
            "n_theta_slices": row["n_theta_slices"],
            "success": int(row["success"]),
            "butterfly_free": int(row["butterfly_free"]),
            "calendar_free": int(row["calendar_free"]),
            "min_durrleman_g": row["min_durrleman_g"],
            "max_bfly_condition1": row["max_bfly_condition1"],
            "max_bfly_condition2": row["max_bfly_condition2"],
            "calendar_margin_min": row["calendar_margin_min"],
            "calendar_lhs_max": row["calendar_lhs_max"],
            "calendar_gamma_min": row["calendar_gamma_min"],
            "calendar_gamma_max": row["calendar_gamma_max"],
            "theta_repair": row["theta_repair"],
            "theta_repair_fraction": row["theta_repair_fraction"],
            "theta_curve": json.dumps(fit.theta_curve.to_dict()),
            "slice_rmse_iv": json.dumps({str(k): float(v) for k, v in fit.slice_rmse_iv.items()}),
        }
        self._upsert("essvi_parameters", [payload])
        return payload

    def load_essvi_fits(self, symbol, trade_date=None) -> pd.DataFrame:
        q = "SELECT * FROM essvi_parameters WHERE symbol = ?"
        args = [symbol]
        if trade_date is not None:
            q += " AND trade_date = ?"
            args.append(_d(trade_date))
        q += " ORDER BY trade_date"
        with self.connect() as con:
            return pd.read_sql_query(q, con, params=args)

    # -------------------------------------------------------------- Fengler
    def save_fengler_fit(self, symbol, trade_date, fit):
        row = fit.as_row()
        payload = {
            "symbol": symbol,
            "trade_date": _d(trade_date),
            "smoothing_lambda": row["smoothing_lambda"],
            "success": int(row["success"]),
            "butterfly_free": int(row["butterfly_free"]),
            "calendar_free": int(row["calendar_free"]),
            "rmse_iv": row["rmse_iv"],
            "max_abs_err_iv": row["max_abs_err_iv"],
            "n_obs": row["n_obs"],
            "n_slices": row["n_slices"],
            "calendar_margin_min": row["calendar_margin_min"],
            "slice_diagnostics": json.dumps([s.as_dict() for s in fit.slices]),
        }
        self._upsert("fengler_runs", [payload])
        return payload

    def load_fengler_fits(self, symbol, trade_date=None) -> pd.DataFrame:
        q = "SELECT * FROM fengler_runs WHERE symbol = ?"
        args = [symbol]
        if trade_date is not None:
            q += " AND trade_date = ?"
            args.append(_d(trade_date))
        q += " ORDER BY trade_date"
        with self.connect() as con:
            return pd.read_sql_query(q, con, params=args)

    # -------------------------------------------------------------- surfaces
    def save_surface(self, symbol, trade_date, surface):
        """Persist a Surface object onto the fixed grid."""
        rows = []
        for i, tenor in enumerate(surface.tenor_days):
            for j, k in enumerate(surface.k_grid):
                rows.append({
                    "symbol": symbol,
                    "trade_date": _d(trade_date),
                    "tenor_days": float(tenor),
                    "k": float(k),
                    "total_var": float(surface.total_var[i, j]),
                    "iv": float(surface.iv[i, j]),
                    "extrapolated": int(surface.extrapolated[i]),
                })
        self._upsert("surface_grid", rows)
        return len(rows)

    def save_model_surface(self, symbol, trade_date, model, surface):
        """Persist any model on the common grid without overwriting other models."""
        rows = []
        model = str(model).lower()
        for i, tenor in enumerate(surface.tenor_days):
            for j, k in enumerate(surface.k_grid):
                rows.append({
                    "symbol": symbol,
                    "trade_date": _d(trade_date),
                    "model": model,
                    "tenor_days": float(tenor),
                    "k": float(k),
                    "total_var": float(surface.total_var[i, j]),
                    "iv": float(surface.iv[i, j]),
                    "extrapolated": int(surface.extrapolated[i]),
                })
        self._upsert("modeled_surface_grid", rows)
        return len(rows)

    def load_model_surface_panel(self, symbol, model) -> pd.DataFrame:
        """Wide total-variance panel for one named surface model."""
        with self.connect() as con:
            df = pd.read_sql_query(
                "SELECT trade_date, tenor_days, k, total_var FROM modeled_surface_grid "
                "WHERE symbol = ? AND model = ? ORDER BY trade_date",
                con, params=[symbol, str(model).lower()])
        if df.empty:
            return df
        panel = df.pivot_table(index="trade_date", columns=["tenor_days", "k"],
                               values="total_var")
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()

    def load_surface_panel(self, symbol) -> pd.DataFrame:
        """Wide panel: rows = trade_date, columns = (tenor_days, k) nodes."""
        with self.connect() as con:
            df = pd.read_sql_query(
                "SELECT trade_date, tenor_days, k, total_var FROM surface_grid "
                "WHERE symbol = ? ORDER BY trade_date", con, params=[symbol])
        if df.empty:
            return df
        panel = df.pivot_table(index="trade_date", columns=["tenor_days", "k"],
                               values="total_var")
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()

    # -------------------------------------------------------------- features
    def save_features(self, symbol, trade_date, features: dict):
        clean = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                     else float(v)) for k, v in features.items()}
        self._upsert("features", [{"symbol": symbol, "trade_date": _d(trade_date),
                                   "payload": json.dumps(clean)}])

    def load_features(self, symbol) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query(
                "SELECT trade_date, payload FROM features WHERE symbol = ? "
                "ORDER BY trade_date", con, params=[symbol])
        if df.empty:
            return df
        out = pd.DataFrame([json.loads(p) for p in df["payload"]])
        out.index = pd.to_datetime(df["trade_date"])
        return out

    # ------------------------------------------------------------------ PCA
    def save_pca_model(self, model_id, symbol, model):
        self._upsert("pca_models", [{
            "model_id": model_id,
            "symbol": symbol,
            "n_components": int(model.loadings.shape[0]),
            "n_days": int(model.n_days),
            "grid": json.dumps([list(map(float, t)) for t in model.node_index]),
            "loadings": json.dumps(model.loadings.tolist()),
            "explained_var": json.dumps(model.explained_variance_ratio.tolist()),
            "center": json.dumps(model.center.tolist()),
            "scale": json.dumps(model.scale.tolist()),
            "notes": model.notes,
        }])

    def save_pca_scores(self, symbol, model_id, scores: pd.DataFrame):
        rows = [{"symbol": symbol, "trade_date": _d(d), "model_id": model_id,
                 "component": int(c), "score": float(scores.loc[d, c])}
                for d in scores.index for c in scores.columns
                if np.isfinite(scores.loc[d, c])]
        self._upsert("pca_scores", rows)
        return len(rows)

    def load_pca_scores(self, symbol, model_id) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query(
                "SELECT trade_date, component, score FROM pca_scores "
                "WHERE symbol = ? AND model_id = ? ORDER BY trade_date, component",
                con, params=[symbol, model_id])
        if df.empty:
            return df
        out = df.pivot(index="trade_date", columns="component", values="score")
        out.index = pd.to_datetime(out.index)
        return out

    # ---------------------------------------------------------------- helper
    def _upsert(self, table, rows):
        if not rows:
            return
        cols = list(rows[0])
        placeholders = ",".join("?" * len(cols))
        collist = ",".join(cols)
        sql = f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})"
        vals = [tuple(None if (isinstance(r[c], float) and not np.isfinite(r[c]))
                      else r[c] for c in cols) for r in rows]
        with self.connect() as con:
            con.executemany(sql, vals)

    def table_counts(self) -> dict:
        with self.connect() as con:
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("svi_parameters", "ssvi_parameters", "essvi_parameters", "fengler_runs", "surface_grid",
                              "modeled_surface_grid", "features",
                              "pca_models", "pca_scores")}

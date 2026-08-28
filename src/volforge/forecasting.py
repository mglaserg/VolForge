"""Forward realized-variance forecasting for the VRP research workflow.

The first forecasting layer deliberately uses interpretable benchmarks before
machine learning:

* Persistence: future 30-day variance ~= current trailing 30-day variance.
* HAR-style direct regression: forward variance on RV3/RV9/RV30.
* HEAVY-RM: Shephard-Sheppard realized-measure dynamics estimated from daily
  integrated variance and recursively projected over the target horizon.

All forecasts are returned in annualized variance units so they can be compared
directly with VolForge MFIV.  No function in this module emits a trade signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .realized import CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR

__all__ = [
    "HARFit",
    "HEAVYRMFit",
    "fit_har",
    "fit_heavy_rm",
    "latest_model_forecasts",
    "walk_forward_forecasts",
    "forecast_metrics",
]


DEFAULT_HAR_FEATURES = ("rv_var_3", "rv_var_9", "rv_var_30")


def _history_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    if "date" not in frame:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        else:
            raise ValueError("history needs a 'date' column or DatetimeIndex")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class HARFit:
    """Direct HAR-style forecast of annualized forward variance."""

    intercept: float
    coefficients: tuple[float, ...]
    feature_cols: tuple[str, ...]
    nobs: int

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in frame]
        if missing:
            raise ValueError(f"HAR input missing columns: {missing}")
        x = frame.loc[:, self.feature_cols].apply(pd.to_numeric, errors="coerce")
        values = self.intercept + x.to_numpy(float) @ np.asarray(self.coefficients, dtype=float)
        values = np.maximum(values, 0.0)
        return pd.Series(values, index=frame.index, name="har_forecast")


def fit_har(
    history: pd.DataFrame,
    *,
    target_col: str = "forward_rv_var",
    feature_cols: tuple[str, ...] = DEFAULT_HAR_FEATURES,
) -> HARFit:
    """Fit a direct 30-day HAR-style OLS benchmark.

    VolForge uses the already-computed short/medium/long annualized realized
    variance features (RV3/RV9/RV30 by default) and predicts the ex-post
    forward realized-variance label directly.  The caller is responsible for
    passing only training rows that were knowable at the forecast date.
    """
    frame = _history_frame(history)
    required = [target_col, *feature_cols]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"HAR history missing columns: {missing}")
    work = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < len(feature_cols) + 5:
        raise ValueError("not enough labeled observations to fit HAR")
    x = work.loc[:, feature_cols].to_numpy(float)
    y = work[target_col].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return HARFit(
        intercept=float(beta[0]),
        coefficients=tuple(float(v) for v in beta[1:]),
        feature_cols=tuple(feature_cols),
        nobs=int(len(work)),
    )


@dataclass(frozen=True)
class HEAVYRMFit:
    """HEAVY realized-measure equation.

    mu_t = omega + alpha * RM_{t-1} + beta * mu_{t-1}

    Parameters are estimated by positive quasi-likelihood on the daily realized
    measure.  This is the HEAVY-RM component, which is the piece directly useful
    for forecasting the realized variance that will be compared with MFIV.
    """

    omega: float
    alpha: float
    beta: float
    nobs: int
    last_rm: float
    last_mu: float
    objective: float
    success: bool

    @property
    def persistence(self) -> float:
        return float(self.alpha + self.beta)

    def forecast_daily(self, steps: int) -> np.ndarray:
        if steps <= 0:
            raise ValueError("steps must be positive")
        out = np.empty(int(steps), dtype=float)
        out[0] = self.omega + self.alpha * self.last_rm + self.beta * self.last_mu
        for i in range(1, len(out)):
            # E[RM_{t+i-1}] = mu_{t+i-1} under the HEAVY-RM recursion.
            out[i] = self.omega + (self.alpha + self.beta) * out[i - 1]
        return np.maximum(out, 0.0)

    def forecast_horizon_variance(
        self,
        target_days: float = 30.0,
        *,
        realized_measure: pd.Series | np.ndarray | None = None,
    ) -> float:
        """Project daily RM and convert the sum to option-clock annualized variance.

        ``realized_measure`` can extend the filtered state without re-estimating
        the parameters. This is used by walk-forward evaluation between scheduled
        parameter refits.
        """
        if target_days <= 0:
            raise ValueError("target_days must be positive")
        last_rm, last_mu = self.last_rm, self.last_mu
        if realized_measure is not None:
            rm = pd.to_numeric(pd.Series(realized_measure), errors="coerce").dropna().to_numpy(float)
            rm = rm[np.isfinite(rm) & (rm >= 0.0)]
            if len(rm):
                mu = _heavy_filter(rm, self.omega, self.alpha, self.beta)
                last_rm, last_mu = float(rm[-1]), float(mu[-1])

        trading_steps = max(1, int(round(float(target_days) * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)))
        daily = np.empty(trading_steps, dtype=float)
        daily[0] = self.omega + self.alpha * last_rm + self.beta * last_mu
        for i in range(1, trading_steps):
            daily[i] = self.omega + (self.alpha + self.beta) * daily[i - 1]
        total = float(np.sum(np.maximum(daily, 0.0)))
        return total * (CALENDAR_DAYS_PER_YEAR / float(target_days))


def _heavy_filter(rm: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    mu = np.empty(len(rm), dtype=float)
    unconditional = omega / max(1.0 - alpha - beta, 1e-6)
    mu[0] = max(float(np.nanmean(rm[: min(len(rm), 20)])), unconditional, 1e-10)
    for t in range(1, len(rm)):
        mu[t] = omega + alpha * rm[t - 1] + beta * mu[t - 1]
        if not np.isfinite(mu[t]) or mu[t] <= 0:
            mu[t] = 1e-10
    return mu


def fit_heavy_rm(realized_measure: pd.Series | np.ndarray) -> HEAVYRMFit:
    """Estimate the HEAVY realized-measure equation with positivity/stability constraints."""
    rm = pd.to_numeric(pd.Series(realized_measure), errors="coerce").dropna().to_numpy(float)
    rm = rm[np.isfinite(rm) & (rm >= 0.0)]
    if len(rm) < 30:
        raise ValueError("HEAVY-RM needs at least 30 daily realized-measure observations")
    positive_mean = float(np.mean(rm))
    if positive_mean <= 0:
        raise ValueError("HEAVY-RM needs positive realized variance")

    # Scale the tiny variance numbers to O(1) for a better conditioned solve.
    scale = positive_mean
    y = np.maximum(rm / scale, 1e-10)

    def objective(params: np.ndarray) -> float:
        omega, alpha, beta = map(float, params)
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        mu = _heavy_filter(y, omega, alpha, beta)
        # Exponential/MEM quasi-likelihood, constant terms omitted.
        return float(np.sum(np.log(mu[1:]) + y[1:] / mu[1:]))

    a0, b0 = 0.10, 0.80
    w0 = max(1e-5, (1.0 - a0 - b0) * float(np.mean(y)))
    result = minimize(
        objective,
        x0=np.array([w0, a0, b0], dtype=float),
        method="SLSQP",
        bounds=((1e-8, 10.0), (0.0, 0.998), (0.0, 0.998)),
        constraints=({"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]},),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"HEAVY-RM calibration failed: {result.message}")

    omega_s, alpha, beta = map(float, result.x)
    mu_scaled = _heavy_filter(y, omega_s, alpha, beta)
    return HEAVYRMFit(
        omega=float(omega_s * scale),
        alpha=float(alpha),
        beta=float(beta),
        nobs=int(len(rm)),
        last_rm=float(rm[-1]),
        last_mu=float(mu_scaled[-1] * scale),
        objective=float(result.fun),
        success=bool(result.success),
    )


def latest_model_forecasts(
    history: pd.DataFrame,
    *,
    target_days: float = 30.0,
    har_features: tuple[str, ...] = DEFAULT_HAR_FEATURES,
) -> pd.DataFrame:
    """Fit all available benchmarks and forecast the latest row.

    The returned table is intentionally descriptive: it reports forecast RV and
    expected VRP, never a trade instruction.
    """
    frame = _history_frame(history)
    if frame.empty:
        raise ValueError("history is empty")
    latest = frame.iloc[[-1]]
    if "mfiv_var" not in latest:
        raise ValueError("history needs mfiv_var")
    mfiv = float(pd.to_numeric(latest["mfiv_var"], errors="coerce").iloc[0])
    rows: list[dict] = []

    if "trailing_rv_var" in latest:
        persistence = float(pd.to_numeric(latest["trailing_rv_var"], errors="coerce").iloc[0])
        if np.isfinite(persistence):
            rows.append({"model": "Persistence", "forecast_rv_var": persistence, "detail": "Current trailing target-horizon RV"})

    try:
        har = fit_har(frame, feature_cols=har_features)
        pred = float(har.predict(latest).iloc[0])
        rows.append({"model": "HAR 3/9/30", "forecast_rv_var": pred, "detail": f"Direct OLS · {har.nobs} labeled rows"})
    except (ValueError, np.linalg.LinAlgError):
        pass

    if "daily_rm" in frame:
        rm = pd.to_numeric(frame["daily_rm"], errors="coerce").dropna()
        try:
            heavy = fit_heavy_rm(rm)
            pred = float(heavy.forecast_horizon_variance(target_days))
            rows.append({
                "model": "HEAVY-RM",
                "forecast_rv_var": pred,
                "detail": f"α={heavy.alpha:.3f} β={heavy.beta:.3f} · {heavy.nobs} RM rows",
            })
        except (ValueError, RuntimeError):
            pass

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["forecast_rv_vol"] = np.sqrt(out["forecast_rv_var"].clip(lower=0.0))
    out["mfiv_var"] = mfiv
    out["expected_vrp"] = mfiv - out["forecast_rv_var"]
    return out


def walk_forward_forecasts(
    history: pd.DataFrame,
    *,
    target_days: int = 30,
    min_train: int = 80,
    refit_every: int = 20,
    har_features: tuple[str, ...] = DEFAULT_HAR_FEATURES,
) -> pd.DataFrame:
    """Purged expanding-window comparison of Persistence, HAR and HEAVY-RM.

    For HAR, a training label is eligible only when its entire forward horizon
    ends before the forecast date.  This prevents overlapping 30-day labels
    from leaking information into the test row.  HEAVY-RM is estimated only
    from realized measures observable by each forecast date.
    """
    frame = _history_frame(history)
    required = {"forward_rv_var", "trailing_rv_var", *har_features}
    missing = sorted(c for c in required if c not in frame)
    if missing:
        raise ValueError(f"history missing forecast columns: {missing}")

    target = pd.to_numeric(frame["forward_rv_var"], errors="coerce")
    predictions: list[dict] = []
    last_har_train_count = -1
    last_har: HARFit | None = None
    last_heavy_obs = -1
    last_heavy: HEAVYRMFit | None = None

    for i, row in frame.iterrows():
        y = float(target.iloc[i]) if np.isfinite(target.iloc[i]) else np.nan
        if not np.isfinite(y):
            continue
        forecast_date = pd.Timestamp(row["date"])
        purge_cutoff = forecast_date - pd.Timedelta(days=int(target_days))
        train = frame[(frame["date"] < purge_cutoff) & pd.to_numeric(frame["forward_rv_var"], errors="coerce").notna()]
        if len(train) < int(min_train):
            continue

        persistence = pd.to_numeric(pd.Series([row["trailing_rv_var"]]), errors="coerce").iloc[0]
        if np.isfinite(persistence):
            predictions.append({"date": forecast_date, "model": "Persistence", "forecast_rv_var": float(persistence), "actual_rv_var": y})

        if last_har is None or len(train) - last_har_train_count >= int(refit_every):
            try:
                last_har = fit_har(train, feature_cols=har_features)
                last_har_train_count = len(train)
            except ValueError:
                last_har = None
        if last_har is not None:
            try:
                pred = float(last_har.predict(pd.DataFrame([row])).iloc[0])
                if np.isfinite(pred):
                    predictions.append({"date": forecast_date, "model": "HAR 3/9/30", "forecast_rv_var": pred, "actual_rv_var": y})
            except ValueError:
                pass

        if "daily_rm" in frame:
            rm_history = pd.to_numeric(frame.loc[frame["date"] <= forecast_date, "daily_rm"], errors="coerce").dropna()
            if len(rm_history) >= 30:
                if last_heavy is None or len(rm_history) - last_heavy_obs >= int(refit_every):
                    try:
                        last_heavy = fit_heavy_rm(rm_history)
                        last_heavy_obs = len(rm_history)
                    except (ValueError, RuntimeError):
                        last_heavy = None
                if last_heavy is not None:
                    pred = float(last_heavy.forecast_horizon_variance(float(target_days), realized_measure=rm_history))
                    if np.isfinite(pred):
                        predictions.append({"date": forecast_date, "model": "HEAVY-RM", "forecast_rv_var": pred, "actual_rv_var": y})

    out = pd.DataFrame(predictions)
    if out.empty:
        return out
    out["error"] = out["forecast_rv_var"] - out["actual_rv_var"]
    return out.sort_values(["date", "model"]).reset_index(drop=True)


def forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-sample variance-forecast accuracy by model."""
    if predictions.empty:
        return pd.DataFrame(columns=["model", "n", "mse", "mae", "qlike", "bias"])
    rows = []
    eps = 1e-12
    for model, group in predictions.groupby("model", sort=False):
        y = pd.to_numeric(group["actual_rv_var"], errors="coerce").to_numpy(float)
        f = pd.to_numeric(group["forecast_rv_var"], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(f) & (y >= 0) & (f > 0)
        y, f = y[ok], f[ok]
        if len(y) == 0:
            continue
        err = f - y
        ratio = np.maximum(y, eps) / np.maximum(f, eps)
        rows.append({
            "model": model,
            "n": int(len(y)),
            "mse": float(np.mean(err ** 2)),
            "mae": float(np.mean(np.abs(err))),
            "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
            "bias": float(np.mean(err)),
        })
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)

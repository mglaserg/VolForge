"""Quote cleaning.

Every filter reports what it removed. This is not decoration: when a slice
fails to calibrate, the first question is always "what did I throw away, and
why", and a pipeline that silently drops 60% of the chain will waste days of
your time. `CleanReport.to_frame()` is meant to be printed every run.

Filters are ordered cheapest-and-most-obvious first, so the report reads as a
funnel rather than a pile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import add_derived_columns

__all__ = ["CleanReport", "CleanConfig", "clean_chain", "matched_pairs"]


@dataclass
class CleanConfig:
    min_price: float = 0.05          # below the minimum tick, mid is meaningless
    max_rel_spread: float = 0.50     # spread as fraction of mid
    max_abs_spread: float | None = 0.10  # absolute escape hatch for cheap wings
    moneyness_range: tuple[float, float] = (0.70, 1.30)  # K / spot
    dte_range: tuple[float, float] | None = (7.0, 120.0)
    require_activity: bool = True    # drop quotes with zero volume AND zero OI
    max_stale_days: float | None = 5.0   # by last_trade_time, when available
    min_strikes_per_expiry: int = 8
    min_pairs_per_expiry: int = 5    # matched call/put strikes, for parity fit


@dataclass
class CleanReport:
    steps: list = field(default_factory=list)

    def record(self, name, before, after):
        self.steps.append({"step": name, "kept": after,
                           "dropped": before - after,
                           "pct_dropped": (before - after) / before if before else 0.0})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def __str__(self):
        f = self.to_frame()
        if f.empty:
            return "CleanReport: no steps"
        lines = [f"{r['step']:<28} kept {r['kept']:>6}   dropped {r['dropped']:>6} "
                 f"({r['pct_dropped']:>5.1%})" for _, r in f.iterrows()]
        return "\n".join(lines)

    @property
    def total_kept(self):
        return self.steps[-1]["kept"] if self.steps else 0


def clean_chain(df: pd.DataFrame, config: CleanConfig | None = None,
                verbose: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    """Apply the standard quote filters. Returns (clean_df, report).

    Note what is *not* done here: no-arbitrage price bounds. Those require a
    forward, which we only get after the parity regression, so they are applied
    per-slice downstream.
    """
    cfg = config or CleanConfig()
    rep = CleanReport()
    df = add_derived_columns(df)
    rep.record("raw", len(df), len(df))

    def step(name, mask):
        nonlocal df
        before = len(df)
        df = df.loc[mask].copy()
        rep.record(name, before, len(df))

    step("finite bid/ask", np.isfinite(df["bid"]) & np.isfinite(df["ask"]))
    step("two-sided (bid>0, ask>0)", (df["bid"] > 0) & (df["ask"] > 0))
    step("not crossed (ask>=bid)", df["ask"] >= df["bid"])
    step("mid >= min_price", df["mid"] >= cfg.min_price)

    spread_ok = df["rel_spread"] <= cfg.max_rel_spread
    if cfg.max_abs_spread is not None:
        spread_ok |= df["spread"] <= cfg.max_abs_spread
    step("spread within limits", spread_ok)

    if cfg.require_activity:
        vol = df["volume"].fillna(0) if "volume" in df else 0
        oi = df["open_interest"].fillna(0) if "open_interest" in df else 0
        step("has volume or open interest", (vol > 0) | (oi > 0))

    if cfg.max_stale_days is not None and "last_trade_time" in df.columns:
        age = (df["quote_time"] - df["last_trade_time"]).dt.total_seconds() / 86400.0
        step("not stale", ~(age > cfg.max_stale_days).fillna(False))

    lo, hi = cfg.moneyness_range
    mny = df["strike"] / df["underlying_price"]
    step("moneyness range", (mny >= lo) & (mny <= hi))

    if cfg.dte_range is not None:
        d0, d1 = cfg.dte_range
        step("dte range", (df["dte"] >= d0) & (df["dte"] <= d1))

    before = len(df)
    df = df.drop_duplicates(subset=["expiry", "strike", "right"], keep="first")
    rep.record("dedupe", before, len(df))

    # Expiry-level gates: a slice with too few strikes cannot support a
    # 5-parameter fit, and one with too few matched pairs cannot support a
    # parity regression.
    counts = df.groupby("expiry")["strike"].nunique()
    keep_exp = set(counts[counts >= cfg.min_strikes_per_expiry].index)
    step("expiry: enough strikes", df["expiry"].isin(keep_exp))

    pair_counts = (
        df.groupby(["expiry", "strike"])["right"].nunique()
        .eq(2).groupby("expiry").sum()
    )
    keep_exp = set(pair_counts[pair_counts >= cfg.min_pairs_per_expiry].index)
    step("expiry: enough C/P pairs", df["expiry"].isin(keep_exp))

    if verbose:
        print(rep)
        n_exp = df["expiry"].nunique() if len(df) else 0
        print(f"{'':<28} -> {n_exp} usable expiries")

    return df.reset_index(drop=True), rep


def matched_pairs(df: pd.DataFrame, expiry=None) -> pd.DataFrame:
    """Pivot to one row per (expiry, strike) with call and put columns.

    Only strikes where both legs survived cleaning are returned -- these are
    what the put-call parity regression consumes.
    """
    if expiry is not None:
        df = df[df["expiry"] == expiry]

    wide = df.pivot_table(
        index=["expiry", "strike", "underlying_price", "T", "dte"],
        columns="right",
        values=["bid", "ask", "mid", "spread", "volume", "open_interest"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b.lower()}" for a, b in wide.columns]
    wide = wide.reset_index()

    need = ["mid_c", "mid_p"]
    missing = [c for c in need if c not in wide.columns]
    if missing:
        return wide.iloc[0:0]

    wide = wide.dropna(subset=need)
    wide["combined_spread"] = wide.get("spread_c", 0) + wide.get("spread_p", 0)
    return wide.sort_values(["expiry", "strike"]).reset_index(drop=True)

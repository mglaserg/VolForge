"""Provider-neutral local persistence for option-chain snapshots.

Snapshots use Hive-style partitions so the same archive can be queried later
with DuckDB/Arrow without changing the on-disk layout::

    data/chains/
      provider=yahoo/
        symbol=SPY/
          date=2026-08-26/
            time=123000/
              chain.parquet
              metadata.json

The archive stores VolForge's canonical vendor-neutral columns.  Derived
columns such as ``mid``/``T``/``dte`` are recomputed on load so improvements to
those formulas do not require rewriting historical raw snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, add_derived_columns, validate_chain

__all__ = [
    "ChainSnapshotRef",
    "save_chain_snapshot",
    "load_chain_snapshot",
    "list_chain_snapshots",
    "select_daily_snapshots",
]

_ARCHIVE_TZ = "America/New_York"
_SCHEMA_VERSION = 1
_CANONICAL_COLUMNS = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)


@dataclass(frozen=True)
class ChainSnapshotRef:
    provider: str
    symbol: str
    date: date_type
    time: str | None
    path: Path
    quote_time: pd.Timestamp | None = None
    legacy: bool = False


def _normalise_provider(provider: str) -> str:
    value = str(provider).strip().lower()
    if not value:
        raise ValueError("provider cannot be empty")
    return value


def _snapshot_clock(chain: pd.DataFrame) -> pd.Timestamp:
    q = pd.to_datetime(chain["quote_time"], utc=True, errors="coerce").dropna()
    if q.empty:
        raise ValueError("chain has no valid quote_time")
    # Providers normally stamp a chain with one time. Median is robust to a few
    # rows carrying slightly different timestamps.
    ns = q.astype("int64")
    return pd.Timestamp(int(np.median(ns)), tz="UTC")


def save_chain_snapshot(
    chain: pd.DataFrame,
    *,
    provider: str,
    root: str | Path = "data/chains",
    metadata: dict | None = None,
    overwrite: bool = False,
) -> ChainSnapshotRef:
    """Persist one canonical chain snapshot without overwriting other times.

    A second capture on the same day gets a separate ``time=HHMMSS`` partition,
    which is important once ORATS/intraday snapshots are available.
    """
    validate_chain(chain)
    provider = _normalise_provider(provider)
    symbols = chain["symbol"].dropna().astype(str).str.upper().unique()
    if len(symbols) != 1:
        raise ValueError(f"snapshot must contain exactly one symbol, found {symbols.tolist()}")
    symbol = str(symbols[0])
    quote_time = _snapshot_clock(chain)
    local = quote_time.tz_convert(_ARCHIVE_TZ)
    day = local.date()
    clock = local.strftime("%H%M%S")

    path = (
        Path(root)
        / f"provider={provider}"
        / f"symbol={symbol}"
        / f"date={day.isoformat()}"
        / f"time={clock}"
    )
    path.mkdir(parents=True, exist_ok=True)
    target = path / "chain.parquet"
    if target.exists() and not overwrite:
        return ChainSnapshotRef(provider, symbol, day, clock, target, quote_time, False)

    cols = [c for c in _CANONICAL_COLUMNS if c in chain.columns]
    stored = chain.loc[:, cols].copy()
    stored.to_parquet(target, index=False)

    dte = add_derived_columns(stored)["dte"]
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "provider": provider,
        "symbol": symbol,
        "quote_time_utc": quote_time.isoformat(),
        "quote_time_local": local.isoformat(),
        "rows": int(len(stored)),
        "expiries": int(stored["expiry"].nunique()),
        "strike_count": int(stored["strike"].nunique()),
        "dte_min": float(dte.min()) if len(dte) else None,
        "dte_max": float(dte.max()) if len(dte) else None,
        "underlying_price": float(pd.to_numeric(stored["underlying_price"], errors="coerce").median()),
        "sources": sorted(stored["source"].dropna().astype(str).unique().tolist()) if "source" in stored else [],
    }
    if metadata:
        manifest["extra"] = metadata
    (path / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return ChainSnapshotRef(provider, symbol, day, clock, target, quote_time, False)


def load_chain_snapshot(snapshot: ChainSnapshotRef | str | Path) -> pd.DataFrame:
    path = snapshot.path if isinstance(snapshot, ChainSnapshotRef) else Path(snapshot)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_chain(frame)
    return add_derived_columns(frame)


def _parse_new_ref(path: Path) -> ChainSnapshotRef | None:
    parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
    try:
        provider = parts["provider"]
        symbol = parts["symbol"].upper()
        day = pd.Timestamp(parts["date"]).date()
        clock = parts["time"]
    except (KeyError, ValueError):
        return None
    try:
        local = pd.Timestamp(f"{day.isoformat()} {clock[:2]}:{clock[2:4]}:{clock[4:6]}", tz=_ARCHIVE_TZ)
        quote_time = local.tz_convert("UTC")
    except Exception:
        quote_time = None
    return ChainSnapshotRef(provider, symbol, day, clock, path, quote_time, False)


def _legacy_refs(symbol: str, root: Path) -> list[ChainSnapshotRef]:
    out: list[ChainSnapshotRef] = []
    base = root / f"symbol={symbol.upper()}"
    if not base.exists():
        return out
    for p in sorted(base.glob("date=*/chain.parquet")):
        try:
            day = pd.Timestamp(p.parent.name.split("=", 1)[1]).date()
        except Exception:
            continue
        out.append(ChainSnapshotRef("yahoo", symbol.upper(), day, None, p, None, True))
    return out


def list_chain_snapshots(
    symbol: str,
    *,
    provider: str | None = None,
    root: str | Path = "data/chains",
    include_legacy_yahoo: bool = True,
) -> list[ChainSnapshotRef]:
    """List canonical archive snapshots, optionally including old Yahoo files."""
    root = Path(root)
    symbol = symbol.upper()
    providers: Iterable[str]
    if provider is None:
        providers = [p.name.split("=", 1)[1] for p in root.glob("provider=*") if p.is_dir()]
    else:
        providers = [_normalise_provider(provider)]

    refs: list[ChainSnapshotRef] = []
    for prov in providers:
        for p in root.glob(f"provider={prov}/symbol={symbol}/date=*/time=*/chain.parquet"):
            ref = _parse_new_ref(p)
            if ref is not None:
                refs.append(ref)
    if include_legacy_yahoo and (provider is None or _normalise_provider(provider) == "yahoo"):
        refs.extend(_legacy_refs(symbol, root))
    return sorted(refs, key=lambda r: (r.date, r.time or ""))


def _resolved_quote_time(ref: ChainSnapshotRef) -> pd.Timestamp:
    if ref.quote_time is not None:
        return ref.quote_time
    frame = pd.read_parquet(ref.path, columns=["quote_time"])
    q = pd.to_datetime(frame["quote_time"], utc=True, errors="coerce").dropna()
    if q.empty:
        return pd.Timestamp(ref.date, tz=_ARCHIVE_TZ).tz_convert("UTC")
    return pd.Timestamp(int(np.median(q.astype("int64"))), tz="UTC")


def select_daily_snapshots(
    snapshots: Iterable[ChainSnapshotRef],
    *,
    policy: str = "latest",
    target_time: str | None = None,
) -> list[ChainSnapshotRef]:
    """Choose one snapshot per calendar date.

    ``policy='closest'`` requires ``target_time='HH:MM'`` and is useful once
    intraday ORATS history is available, because research can consistently use
    (for example) the snapshot nearest 15:30 ET rather than mixing quote times.
    """
    policy = str(policy).strip().lower()
    if policy not in {"latest", "earliest", "closest"}:
        raise ValueError("policy must be 'latest', 'earliest', or 'closest'")
    if policy == "closest" and not target_time:
        raise ValueError("target_time='HH:MM' is required for policy='closest'")

    grouped: dict[date_type, list[ChainSnapshotRef]] = {}
    for ref in snapshots:
        grouped.setdefault(ref.date, []).append(ref)

    selected: list[ChainSnapshotRef] = []
    for day, refs in sorted(grouped.items()):
        timed = [(ref, _resolved_quote_time(ref).tz_convert(_ARCHIVE_TZ)) for ref in refs]
        if policy == "latest":
            selected.append(max(timed, key=lambda x: x[1])[0])
        elif policy == "earliest":
            selected.append(min(timed, key=lambda x: x[1])[0])
        else:
            hh, mm = map(int, str(target_time).split(":"))
            target = pd.Timestamp(day) + pd.Timedelta(hours=hh, minutes=mm)
            target = target.tz_localize(_ARCHIVE_TZ)
            selected.append(min(timed, key=lambda x: abs((x[1] - target).total_seconds()))[0])
    return selected

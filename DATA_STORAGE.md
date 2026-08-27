# VolForge data storage

VolForge keeps paid/raw market data separate from compact research outputs. The
layout is intentionally provider-partitioned so Yahoo and ORATS observations are
never silently mixed.

```text
data/
├── chains/
│   ├── provider=yahoo/
│   │   └── symbol=SPY/
│   │       └── date=2026-08-26/
│   │           └── time=123000/
│   │               ├── chain.parquet
│   │               └── metadata.json
│   └── provider=orats/
│       └── symbol=SPY/
│           └── date=2026-08-26/
│               └── time=153000/
│                   ├── chain.parquet
│                   └── metadata.json
│
├── intraday/
│   └── provider=<underlying-data-provider>/
│       └── symbol=SPY/
│           └── interval=5m/
│               └── year=2026/
│                   └── month=08/
│                       └── bars.parquet
│
├── vendor/
│   └── orats/
│       └── ... optional immutable ORATS-native payload archive ...
│
└── derived/
    └── vrp/
        └── provider=yahoo/
            └── symbol=SPY/
                └── history.parquet
```

## Why both `vendor/orats` and `chains/provider=orats`?

When ORATS is enabled, the best long-term practice is to retain the purchased
ORATS-native fields unchanged under `data/vendor/orats/` *and* save the
normalized VolForge canonical chain under `data/chains/provider=orats/`.
Normalization rules will evolve. Keeping the vendor payload means a future
VolForge version can rebuild the canonical archive without purchasing or
redownloading the same history again.

The current ORATS adapter returns the canonical chain. Native-payload archival
can be wired into the ORATS backfill job once the subscription/data product is
known.

## One snapshot per time

The `time=HHMMSS` partition prevents an intraday capture from overwriting an
earlier capture on the same date. For historical research, the VRP builder can
select the earliest/latest snapshot or the snapshot closest to a fixed ET time.
Consistency of snapshot time matters more than accumulating arbitrary snapshots.

## Raw versus derived

Option chains are expensive and should be treated as immutable source data.
Derived MFIV/RV/VRP tables are cheap and disposable: they can always be rebuilt
when VolForge's calculations improve. `history.parquet` also carries the compact
delta-surface research features (10Δ/15Δ/25Δ bucket IVs, delta ratios, historical
z-scores/percentiles, local lump diagnostics, and ATM/skew/convexity changes), so
those features never require storing another large market-data archive.

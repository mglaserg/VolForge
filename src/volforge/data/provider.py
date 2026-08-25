"""Vendor-neutral option-chain provider registry.

This module is intentionally additive: existing vendor adapters (including
``volforge.data.yahoo``) remain unchanged. New code can resolve a provider by
name, while older imports continue to work exactly as before.

A provider's only hard requirement is that it returns VolForge's canonical
option-chain DataFrame defined in :mod:`volforge.data.schema`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .schema import add_derived_columns, validate_chain

__all__ = [
    "ProviderCapabilities",
    "OptionChainProvider",
    "YahooProvider",
    "ORATSProvider",
    "available_providers",
    "fetch_chain",
    "get_provider",
    "register_provider",
]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Descriptive capabilities for a market-data provider."""

    historical_chains: bool = False
    intraday_history: bool = False
    live_quotes: bool = False
    full_bid_ask: bool = True


@runtime_checkable
class OptionChainProvider(Protocol):
    """Contract for adapters that return canonical VolForge option chains."""

    name: str
    capabilities: ProviderCapabilities

    def fetch_chain(
        self,
        symbol: str,
        max_expiries: int | None = None,
        dte_range: tuple[float, float] | None = (7.0, 120.0),
        settlement: str = "default",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return one option-chain snapshot in canonical schema."""
        ...


def _orats_provider_class():
    from .orats import ORATSProvider
    return ORATSProvider


# Public alias is resolved lazily below to avoid import cycles in module loading.
class YahooProvider:
    """Adapter wrapper around the existing ``volforge.data.yahoo`` module.

    Keeping the wrapper here avoids modifying ``yahoo.py`` during the provider
    abstraction step. That makes this migration safe for repositories that have
    local Yahoo changes while preserving the legacy import path.
    """

    name = "yahoo"
    capabilities = ProviderCapabilities(
        historical_chains=False,
        intraday_history=False,
        live_quotes=False,
        full_bid_ask=True,
    )

    def fetch_chain(
        self,
        symbol: str,
        max_expiries: int | None = None,
        dte_range: tuple[float, float] | None = (7.0, 120.0),
        settlement: str = "default",
        **kwargs: Any,
    ) -> pd.DataFrame:
        # Lazy import keeps yfinance optional until Yahoo is actually used and
        # leaves the existing Yahoo module completely untouched.
        from .yahoo import fetch_chain as yahoo_fetch_chain

        return yahoo_fetch_chain(
            symbol=symbol,
            max_expiries=max_expiries,
            dte_range=dte_range,
            settlement=settlement,
            **kwargs,
        )


_PROVIDERS: dict[str, OptionChainProvider] = {}
_BUILTINS_LOADED = False


def _normalise_name(name: str) -> str:
    value = str(name).strip().lower()
    if not value:
        raise ValueError("provider name cannot be empty")
    return value


def register_provider(provider: OptionChainProvider, *, replace: bool = False) -> None:
    """Register a provider instance by its ``name``."""
    if not isinstance(provider, OptionChainProvider):
        raise TypeError("provider must implement the OptionChainProvider protocol")

    name = _normalise_name(provider.name)
    if name in _PROVIDERS and not replace:
        raise ValueError(f"provider {name!r} is already registered")
    _PROVIDERS[name] = provider


def _load_builtin_providers() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    register_provider(YahooProvider(), replace=True)
    register_provider(_orats_provider_class()(), replace=True)
    _BUILTINS_LOADED = True


def available_providers() -> tuple[str, ...]:
    """Return registered provider names."""
    _load_builtin_providers()
    return tuple(sorted(_PROVIDERS))


def get_provider(name: str = "yahoo") -> OptionChainProvider:
    """Resolve a registered provider by name."""
    _load_builtin_providers()
    key = _normalise_name(name)
    try:
        return _PROVIDERS[key]
    except KeyError as exc:
        known = ", ".join(available_providers()) or "<none>"
        raise ValueError(
            f"unknown option-data provider {name!r}; available: {known}"
        ) from exc


def fetch_chain(
    symbol: str,
    *,
    provider: str | OptionChainProvider = "yahoo",
    max_expiries: int | None = None,
    dte_range: tuple[float, float] | None = (7.0, 120.0),
    settlement: str = "default",
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch a canonical option chain through a provider adapter.

    Examples
    --------
    New provider-aware code::

        from volforge.data.provider import fetch_chain
        chain = fetch_chain("SPY", provider="yahoo")

    Existing code may continue to use::

        from volforge.data.yahoo import fetch_chain

    until callers are migrated deliberately.
    """
    adapter = get_provider(provider) if isinstance(provider, str) else provider
    if not isinstance(adapter, OptionChainProvider):
        raise TypeError("provider must be a registered name or OptionChainProvider")

    frame = adapter.fetch_chain(
        symbol=symbol,
        max_expiries=max_expiries,
        dte_range=dte_range,
        settlement=settlement,
        **kwargs,
    )
    validate_chain(frame)
    return add_derived_columns(frame)


def __getattr__(name: str):
    if name == "ORATSProvider":
        return _orats_provider_class()
    raise AttributeError(name)

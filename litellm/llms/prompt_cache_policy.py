from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

from litellm.types.utils import ModelInfo

_TTLS: Final = MappingProxyType({"5m": 300, "30m": 1800, "1h": 3600, "24h": 86400})


@dataclass(frozen=True, slots=True)
class CacheHistoryPolicy:
    lifetime: int
    implicit: bool
    user_boundaries: bool
    assumptions: tuple[str, ...]

    def eligible(self, role: object, next_role: object, initial: bool) -> bool:
        return not self.user_boundaries or role == "user" or (role == "tool" and next_role != "tool") or initial


def cache_ttl(control: object, default: int) -> int:
    value: Final = cast(Mapping[str, object], control).get("ttl") if isinstance(control, Mapping) else None
    return _TTLS.get(value, default) if isinstance(value, str) else default


def cache_history_policy(
    model: str,
    provider: str,
    prices: ModelInfo | None,
    options: Mapping[str, object],
    retention: object,
) -> CacheHistoryPolicy:
    mode: Final = prices.get("prompt_cache_mode") if prices else None
    configured_ttl: Final = prices.get("prompt_cache_default_ttl_seconds") if prices else None
    known_ttl: Final = (
        configured_ttl
        if configured_ttl is not None and configured_ttl > 0
        else 300
        if provider == "anthropic"
        else None
    )
    modern_openai: Final = (
        provider in ("openai", "azure")
        and prices is not None
        and prices.get("supports_prompt_cache_breakpoint") is True
    )
    requested_retention: Final = (
        _TTLS.get(retention) if provider in ("openai", "azure") and isinstance(retention, str) else None
    )
    default: Final = requested_retention or known_ttl or (600 if retention == "in_memory" else 1800)
    specified: Final = isinstance(options.get("ttl"), str) and options.get("ttl") in _TTLS
    return CacheHistoryPolicy(
        lifetime=cache_ttl(options, default),
        implicit=(mode != "explicit" if mode is not None else provider != "anthropic")
        and options.get("mode") != "explicit",
        user_boundaries=modern_openai,
        assumptions=(
            ()
            if specified or known_ttl is not None or requested_retention is not None
            else ("cache_lifetime_assumed_10m",)
            if retention == "in_memory"
            else ("cache_lifetime_assumed_30m",)
        ),
    )

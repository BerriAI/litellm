"""Atomic affinity claims shared by deployment and tier-model selection."""

import json
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.caching._redis_scripts import claim_affinity_pin as claim_affinity_pin_script
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisScriptClient

_PIN_JSON_ADAPTER: Final = TypeAdapter[JsonValue](JsonValue)


def set_local_affinity_pin(cache: DualCache, cache_key: str, value: object, ttl_seconds: int) -> None:
    """Replace the entry because InMemoryCache.set_cache preserves a live key's expiry."""
    cache.in_memory_cache.delete_cache(cache_key)
    cache.in_memory_cache.set_cache(cache_key, value, ttl=ttl_seconds)


def _legacy_pin_matches(stored: object, pin_value: Mapping[str, str]) -> bool:
    if isinstance(stored, dict):
        return all(stored.get(key) is not None and str(stored[key]) == value for key, value in pin_value.items())
    return isinstance(stored, str) and len(pin_value) == 1 and stored in pin_value.values()


def claim_affinity_pin_in_memory(
    cache: DualCache,
    cache_key: str,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
) -> object:
    """No await between read and write, so same-loop claims agree during a Redis outage."""
    existing: Final[object] = cache.in_memory_cache.get_cache(cache_key)
    if existing is not None and eligible_values is None:
        if _legacy_pin_matches(existing, pin_value):
            set_local_affinity_pin(cache, cache_key, pin_value, ttl_seconds)
        return existing
    winner: Final = existing if existing is not None and existing in (eligible_values or ()) else pin_value
    set_local_affinity_pin(cache, cache_key, winner, ttl_seconds)
    return winner


def _decode_pin(value: str) -> object:
    try:
        return _PIN_JSON_ADAPTER.validate_json(value)
    except ValidationError:
        return value


async def claim_affinity_pin(
    cache: DualCache,
    cache_key: str,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
) -> object:
    """Return the authoritative first writer, replacing it only when it becomes ineligible.

    Eligible claims refresh the returned winner. Legacy deployment claims only refresh
    a matching candidate. Resolve Redis per call because the proxy attaches it lazily.
    """
    redis_cache: Final = cache.redis_cache
    if redis_cache is not None:
        try:
            # Typed object, not the script's bytes: a client built with decode_responses returns str.
            raw: Final[object] = await claim_affinity_pin_script(
                RedisScriptClient(redis_cache),
                key=cache_key,
                pin=json.dumps(dict(pin_value)),  # mutable-ok: JSON serialization requires dict, not a generic Mapping
                ttl=int(ttl_seconds),
                eligible_json=(
                    json.dumps(tuple(dict(value) for value in eligible_values))  # mutable-ok: JSON requires dict
                    if eligible_values is not None
                    else ""
                ),
            )
            if not isinstance(raw, bytes | str):
                return pin_value
            winner: Final = _decode_pin(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            set_local_affinity_pin(cache, cache_key, winner, ttl_seconds)
            return winner
        except Exception as error:  # noqa: BLE001  # Redis/Lua faults retain same-pod affinity through local claims
            verbose_router_logger.debug("Affinity cache: Redis claim failed, using pod-local claim. error=%s", error)
    return claim_affinity_pin_in_memory(cache, cache_key, pin_value, ttl_seconds, eligible_values=eligible_values)

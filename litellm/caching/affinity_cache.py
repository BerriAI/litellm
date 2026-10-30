"""Pod-local affinity policy shared by deployment and tier-model selection."""

from collections.abc import Mapping
from typing import Final

from litellm.caching.dual_cache import DualCache

ROUTER_SESSION_PINS_TARGET: Final = "router_session_pins"


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
    preserve_legacy_model_ids: bool = False,
) -> object:
    """No await between read and write, so same-loop claims agree during a Redis outage."""
    existing: Final[object] = cache.in_memory_cache.get_cache(cache_key)
    if existing is not None and eligible_values is None:
        if _legacy_pin_matches(existing, pin_value):
            set_local_affinity_pin(cache, cache_key, pin_value, ttl_seconds)
        return existing
    eligible_existing: Final = existing in (eligible_values or ()) or (
        preserve_legacy_model_ids
        and isinstance(existing, str)
        and any("model_id" in value and _legacy_pin_matches(existing, value) for value in (eligible_values or ()))
    )
    winner: Final = existing if existing is not None and eligible_existing else pin_value
    set_local_affinity_pin(cache, cache_key, winner, ttl_seconds)
    return winner

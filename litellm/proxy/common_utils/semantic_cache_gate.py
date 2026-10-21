from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

SKIP_CACHE_CONTROLS: Final = MappingProxyType({"no-cache": True, "no-store": True})
CACHED_CALL_TYPES_BY_ROUTE: Final = MappingProxyType(
    {
        "aresponses": frozenset({"aresponses", "responses", "acompletion"}),
        "anthropic_messages": frozenset({"anthropic_messages", "aanthropic_messages", "acompletion"}),
        "agenerate_content": frozenset({"acompletion"}),
        "agenerate_content_stream": frozenset({"acompletion"}),
    }
)


def _cache_serves(cache: Cache, call_type: str) -> bool:
    cached_call_types: Final = CACHED_CALL_TYPES_BY_ROUTE.get(call_type, frozenset({call_type}))
    return not cached_call_types.isdisjoint(cache.supported_call_types or ())


async def _gate_allows(
    gate: CustomLogger,
    user_api_key_dict: UserAPIKeyAuth,
    data: Mapping[str, object],
    call_type: str,
) -> bool:
    try:
        return await gate.async_should_semantic_cache(user_api_key_dict, data, call_type) is True
    except Exception:  # noqa: BLE001  # operator hooks can raise anything, and every failure skips the cache
        verbose_proxy_logger.exception(
            "%s.async_should_semantic_cache raised, so this request skips the semantic cache", type(gate).__name__
        )
        return False


async def _every_gate_allows(
    gates: Sequence[CustomLogger],
    user_api_key_dict: UserAPIKeyAuth,
    data: Mapping[str, object],
    call_type: str,
) -> bool:
    for gate in gates:
        if not await _gate_allows(gate, user_api_key_dict, data, call_type):
            return False
    return True


async def gated_cache_controls(
    data: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    call_type: str,
    gates: Sequence[CustomLogger],
    cache: Cache | None,
) -> Mapping[str, object] | None:
    if not gates or cache is None or not cache.is_semantic_cache() or not _cache_serves(cache, call_type):
        return None
    if await _every_gate_allows(gates, user_api_key_dict, data, call_type):
        return None
    controls: Final = data.get("cache")
    return {**(controls if isinstance(controls, Mapping) else {}), **SKIP_CACHE_CONTROLS}

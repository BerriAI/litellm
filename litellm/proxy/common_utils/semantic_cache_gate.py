from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import Cache
from litellm.constants import REQUEST_FALLBACK_KEYS
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

SKIP_CACHE_CONTROLS: Final = MappingProxyType({"no-cache": True, "no-store": True})
SEMANTIC_CACHE_ROUTES: Final = frozenset(
    {
        "acompletion",
        "atext_completion",
        "aembedding",
        "aresponses",
        "_aresponses_websocket",
        "anthropic_messages",
        "agenerate_content",
        "agenerate_content_stream",
        "acreate_interaction",
        "asend_message",
    }
)
_FALLBACK_ENTRIES: Final = TypeAdapter(list[object])
_FIELDS: Final = TypeAdapter(Mapping[str, object])


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


def _skipping(controls: object) -> Mapping[str, object]:
    return {**(controls if isinstance(controls, Mapping) else {}), **SKIP_CACHE_CONTROLS}


def _skipping_fallback_entry(entry: object) -> object:
    if not isinstance(entry, Mapping):
        return entry
    fields: Final = _FIELDS.validate_python(entry)
    if "cache" not in fields:
        return fields
    return {**fields, "cache": _skipping(fields["cache"])}


def _skipping_fallbacks(data: Mapping[str, object]) -> Mapping[str, Sequence[object]]:
    return {
        key: [_skipping_fallback_entry(entry) for entry in _FALLBACK_ENTRIES.validate_python(entries)]
        for key in REQUEST_FALLBACK_KEYS
        if isinstance(entries := data.get(key), list)
    }


def _skipping_router_settings_override(data: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    override: Final = data.get("router_settings_override")
    if not isinstance(override, Mapping):
        return {}
    settings: Final = _FIELDS.validate_python(override)
    return {"router_settings_override": {**settings, **_skipping_fallbacks(settings)}}


async def gated_cache_overrides(
    data: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    call_type: str,
    gates: Sequence[CustomLogger],
    cache: Cache | None,
) -> Mapping[str, object]:
    if not gates or cache is None or not cache.is_semantic_cache() or call_type not in SEMANTIC_CACHE_ROUTES:
        return {}
    if await _every_gate_allows(gates, user_api_key_dict, data, call_type):
        return {}
    return {
        "cache": _skipping(data.get("cache")),
        **_skipping_fallbacks(data),
        **_skipping_router_settings_override(data),
    }

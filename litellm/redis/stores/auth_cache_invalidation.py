import json
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping

AUTH_CACHE_INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_DECODED_JSON: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class CacheInvalidation:
    cache_key: str
    new_value: float | None = None
    ttl: float | None = None


def encode_cache_invalidation(invalidation: CacheInvalidation) -> str:
    fields: Final[tuple[tuple[str, str | float | None], ...]] = (
        ("cache_key", invalidation.cache_key),
        ("new_value", invalidation.new_value),
        ("ttl", invalidation.ttl),
    )
    return json.dumps({field: value for field, value in fields if value is not None})


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def decode_cache_invalidation(data: object) -> CacheInvalidation | None:
    text: Final = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if not isinstance(text, str):
        return None
    try:
        parsed: Final = as_str_mapping(_DECODED_JSON.validate_python(json.loads(text)))
    except json.JSONDecodeError:
        return None
    if parsed is None:
        return None
    cache_key: Final = parsed.get("cache_key")
    if not isinstance(cache_key, str):
        return None
    return CacheInvalidation(
        cache_key=cache_key, new_value=_number_or_none(parsed.get("new_value")), ttl=_number_or_none(parsed.get("ttl"))
    )

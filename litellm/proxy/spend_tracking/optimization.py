from collections.abc import Mapping
from typing import cast


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def _extract_cache_read_tokens_from_usage(usage_obj: object) -> int:
    if not isinstance(usage_obj, Mapping):
        return 0
    typed_usage = cast(Mapping[str, object], usage_obj)  # cast-ok: usage object is checked as a mapping above
    explicit = _positive_int(typed_usage.get("cache_read_input_tokens"))
    if explicit:
        return explicit
    details = typed_usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        typed_details = cast(Mapping[str, object], details)  # cast-ok: details is checked as a mapping above
        return _positive_int(typed_details.get("cached_tokens"))
    return 0


def extract_cache_read_tokens(metadata: Mapping[str, object]) -> int:
    return max(
        _extract_cache_read_tokens_from_usage(metadata.get("usage_object")),
        _extract_cache_read_tokens_from_usage(metadata.get("additional_usage_values")),
    )

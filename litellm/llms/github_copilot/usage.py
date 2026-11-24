from collections.abc import Mapping, Sequence
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _cache_count(rows: Sequence[Mapping[str, object]], token_type: str, *standard_counts: object) -> int | None:
    reported: Final = next(
        (
            count
            for row in rows
            if row.get("token_type") == token_type and (count := _token_count(row.get("token_count"))) is not None
        ),
        None,
    )
    if reported is None:
        return None
    return next((count for value in standard_counts if (count := _token_count(value)) is not None), reported)


def copilot_cache_counts(usage: Mapping[str, object], copilot_usage: object) -> tuple[int | None, int | None]:
    sidecar: Final = as_str_mapping(copilot_usage)
    if sidecar is None:
        return None, None
    rows: Final = tuple(
        row for item in as_object_list(sidecar.get("token_details")) or () if (row := as_str_mapping(item)) is not None
    )
    details: Final = as_str_mapping(usage.get("prompt_tokens_details")) or {}
    return (
        _cache_count(rows, "cache_read", usage.get("cache_read_input_tokens"), details.get("cached_tokens")),
        _cache_count(
            rows,
            "cache_write",
            usage.get("cache_creation_input_tokens"),
            details.get("cache_write_tokens"),
            details.get("cache_creation_tokens"),
        ),
    )


def chat_usage_with_copilot_cache(usage: Mapping[str, object], copilot_usage: object) -> Mapping[str, object]:
    read, write = copilot_cache_counts(usage, copilot_usage)
    if read is None and write is None:
        return usage
    return {
        **usage,
        **({"cache_read_input_tokens": read} if read is not None else {}),
        **({"cache_creation_input_tokens": write} if write is not None else {}),
    }

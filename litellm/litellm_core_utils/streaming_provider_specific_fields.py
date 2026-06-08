from typing import Any, Callable, Dict, Iterable

ProviderSpecificFieldMerger = Callable[[Dict[str, Any], str, Any], bool]


def _get_provider_specific_field_mergers() -> Iterable[ProviderSpecificFieldMerger]:
    from litellm.llms.vertex_ai.gemini.streaming_provider_specific_fields import (
        merge_gemini_streaming_provider_specific_field,
    )

    return (merge_gemini_streaming_provider_specific_field,)


def merge_streaming_provider_specific_field(
    combined_provider_fields: Dict[str, Any], key: str, value: Any
) -> None:
    for merge_field in _get_provider_specific_field_mergers():
        if merge_field(combined_provider_fields, key, value):
            return

    if key not in combined_provider_fields:
        combined_provider_fields[key] = value
    elif isinstance(value, list) and isinstance(combined_provider_fields[key], list):
        # For lists like web_search_results, take the last (most complete) one.
        combined_provider_fields[key] = value
    else:
        combined_provider_fields[key] = value

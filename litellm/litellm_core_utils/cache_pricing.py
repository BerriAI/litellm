from types import MappingProxyType
from typing import Final

from litellm.types.utils import ModelInfo, Usage


def fallback_missing_cache_rates_to_input(usage: Usage, model_info: ModelInfo | None) -> Usage:
    """Move cache buckets without configured rates into ordinary input."""
    details: Final = usage.prompt_tokens_details
    if details is None or model_info is None:
        return usage

    cached_tokens: Final = getattr(details, "cached_tokens", 0) or 0
    cache_creation_tokens: Final = (getattr(details, "cache_creation_tokens", 0) or 0) or (
        getattr(details, "cache_write_tokens", 0) or 0
    )
    prices_reads: Final = model_info.get("cache_read_input_token_cost") is not None
    prices_writes: Final = model_info.get("cache_creation_input_token_cost") is not None
    reads: Final = cached_tokens if prices_reads else 0
    writes: Final = cache_creation_tokens if prices_writes else 0
    if (reads, writes) == (cached_tokens, cache_creation_tokens):
        return usage

    other_modalities: Final = sum(
        (getattr(details, field, 0) or 0) for field in ("audio_tokens", "image_tokens", "video_tokens")
    )
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        completion_tokens_details=usage.completion_tokens_details,
        prompt_tokens_details=details.model_copy(
            update=MappingProxyType(
                {
                    "cached_tokens": reads,
                    "cache_creation_tokens": writes,
                    "cache_write_tokens": writes,
                    "cache_creation_token_details": details.cache_creation_token_details if writes else None,
                    "text_tokens": max(usage.prompt_tokens - reads - writes - other_modalities, 0),
                }
            )
        ),
    )

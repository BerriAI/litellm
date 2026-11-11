from collections.abc import Mapping
from typing import Final


def extract_cache_read_tokens(usage_object: Mapping[str, object] | None) -> int:
    """Cache-read tokens from a logged usage object, whatever shape recorded them.

    Anthropic writes a top-level ``cache_read_input_tokens``; OpenAI-compatible
    providers (moonshotai, openai, deepseek, etc.) write
    ``prompt_tokens_details.cached_tokens``. This is the one owner of that
    normalization: callers hand over the usage object rather than threading a
    count that could disagree with it.
    """
    if not usage_object:
        return 0
    explicit: Final = usage_object.get("cache_read_input_tokens")
    if isinstance(explicit, (int, float)) and explicit:
        return int(explicit)
    details: Final = usage_object.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    cached: Final = details.get("cached_tokens")
    return int(cached) if isinstance(cached, (int, float)) else 0


def extract_cache_creation_tokens(usage_object: Mapping[str, object] | None) -> int:
    """Cache-write tokens from a logged usage object, whatever shape recorded them.

    Anthropic writes a top-level ``cache_creation_input_tokens``; OpenAI-compatible
    providers (kimi-k2 etc.) write ``prompt_tokens_details.cache_write_tokens`` or
    ``prompt_tokens_details.cache_creation_tokens``.
    """
    if not usage_object:
        return 0
    explicit: Final = usage_object.get("cache_creation_input_tokens")
    if isinstance(explicit, (int, float)) and explicit:
        return int(explicit)
    details: Final = usage_object.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    written: Final = next(
        (
            value
            for value in (details.get("cache_write_tokens"), details.get("cache_creation_tokens"))
            if isinstance(value, (int, float)) and value
        ),
        0,
    )
    return int(written)

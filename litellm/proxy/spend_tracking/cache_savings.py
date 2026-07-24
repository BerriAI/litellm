"""
Single chokepoint for reading prompt-cache token counts out of a SpendLog
``usage_object``. Imported by the daily-spend DB writer and by the hourly
savings endpoint, which reads the same fields directly in Postgres via
``CACHE_READ_INPUT_TOKENS_SQL``.
"""

from collections.abc import Mapping


def _token_count_or_zero(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def extract_cache_read_tokens(usage_obj: Mapping[str, object]) -> int:
    """
    Anthropic: top-level cache_read_input_tokens field.
    OpenAI-compatible (moonshotai, openai, deepseek, etc.): prompt_tokens_details.cached_tokens.
    """
    explicit = _token_count_or_zero(usage_obj.get("cache_read_input_tokens"))
    if explicit:
        return explicit
    details = usage_obj.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    return _token_count_or_zero(details.get("cached_tokens"))


def extract_cache_creation_tokens(usage_obj: Mapping[str, object]) -> int:
    """
    Anthropic: top-level cache_creation_input_tokens field.
    OpenAI-compatible (kimi-k2 etc.): prompt_tokens_details.cache_write_tokens
    or prompt_tokens_details.cache_creation_tokens.
    """
    explicit = _token_count_or_zero(usage_obj.get("cache_creation_input_tokens"))
    if explicit:
        return explicit
    details = usage_obj.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    return _token_count_or_zero(details.get("cache_write_tokens")) or _token_count_or_zero(
        details.get("cache_creation_tokens")
    )


# Postgres twin of ``extract_cache_read_tokens``, for aggregating cache-read
# tokens without pulling every spend log into Python. Assumes the enclosing
# query aliases "LiteLLM_SpendLogs" as ``sl``. The ``jsonb_typeof(...) =
# 'number'`` guards mirror ``_token_count_or_zero``: anything that is not a JSON
# number reads as absent rather than raising a cast error. Any change to the
# Python reader has to land here in the same commit, or hourly savings will
# disagree with the daily rollup they are drawn against.
CACHE_READ_INPUT_TOKENS_SQL = """
COALESCE(
    NULLIF(
        CASE
            WHEN jsonb_typeof(sl.metadata -> 'usage_object' -> 'cache_read_input_tokens') = 'number'
            THEN trunc((sl.metadata -> 'usage_object' ->> 'cache_read_input_tokens')::numeric)
        END,
        0
    ),
    CASE
        WHEN jsonb_typeof(sl.metadata -> 'usage_object' -> 'prompt_tokens_details' -> 'cached_tokens') = 'number'
        THEN trunc((sl.metadata -> 'usage_object' -> 'prompt_tokens_details' ->> 'cached_tokens')::numeric)
    END,
    0
)
"""

"""What a ``Usage`` reports beyond its totals: cached, modality and reasoning token counts and web search requests."""

from typing import Final, TypedDict, cast

from typing_extensions import ReadOnly

from litellm.types.utils import (
    CacheCreationTokenDetails,
    Usage,
    text_tokens_without_nested_reasoning,
)


# Pre-resolved CallTypes enum values for fast membership checks
def token_detail_value(details: object, key: str) -> int | None:
    if isinstance(details, dict):
        value = details.get(key)
    else:
        value = getattr(details, key, None)
    return value if isinstance(value, int) else None


def coerce_token_count(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def get_web_search_requests(server_tool_use: object) -> int | None:
    """
    Tolerantly read ``web_search_requests`` from a ``server_tool_use`` value
    that may be ``None``, a ``dict``, a ``ServerToolUse`` pydantic instance,
    or any other object supporting attribute access.

    Returns ``None`` when the value cannot be resolved — callers can
    distinguish "absent" from "zero" using ``is None``.

    See https://github.com/BerriAI/litellm/issues/26153 — ``stream_chunk_builder``
    historically left this as a plain ``dict``, which broke direct attribute
    access in cost calculation.
    """
    if server_tool_use is None:
        return None
    if isinstance(server_tool_use, dict):
        return server_tool_use.get("web_search_requests")
    return getattr(server_tool_use, "web_search_requests", None)


def get_web_search_requests_from_usage(usage: Usage) -> int | None:
    """Read ``web_search_requests`` from a ``Usage``'s ``server_tool_use``.

    ``Usage`` deletes unset optional fields from ``__dict__`` (see
    ``SafeAttributeModel``), so direct attribute access can raise
    ``AttributeError``; ``getattr`` with a default is required here.
    """
    return get_web_search_requests(getattr(usage, "server_tool_use", None))


class PromptTokensDetailsResult(TypedDict):
    cache_hit_tokens: int
    cache_hit_audio_tokens: ReadOnly[int]

    cached_text_tokens: ReadOnly[int]
    cached_audio_tokens: ReadOnly[int]
    cached_image_tokens: ReadOnly[int]
    has_cached_tokens_details: ReadOnly[bool]
    cache_creation_tokens: int
    cache_creation_token_details: CacheCreationTokenDetails | None
    text_tokens: int
    audio_tokens: int
    image_tokens: int
    video_tokens: int
    character_count: int
    image_count: int
    video_length_seconds: float
    audio_length_seconds: float
    query_count: int


def parse_prompt_tokens_details(usage: Usage) -> PromptTokensDetailsResult:
    cache_hit_tokens: Final = cast(int | None, getattr(usage.prompt_tokens_details, "cached_tokens", 0)) or 0
    cache_creation_tokens: Final = (
        cast(
            int | None,
            getattr(usage.prompt_tokens_details, "cache_write_tokens", 0)
            or getattr(usage.prompt_tokens_details, "cache_creation_tokens", 0),
        )
        or 0
    )
    cache_creation_token_details: Final = (
        cast(
            CacheCreationTokenDetails | None,
            getattr(usage.prompt_tokens_details, "cache_creation_token_details", None),
        )
        or None
    )
    cached_tokens_details: Final = getattr(usage.prompt_tokens_details, "cached_tokens_details", None)
    cached_audio_tokens: Final = min(token_detail_value(cached_tokens_details, "audio_tokens") or 0, cache_hit_tokens)
    cached_text_tokens: Final = min(
        token_detail_value(cached_tokens_details, "text_tokens") or 0,
        cache_hit_tokens - cached_audio_tokens,
    )
    cached_image_tokens: Final = min(
        token_detail_value(cached_tokens_details, "image_tokens") or 0,
        cache_hit_tokens - cached_audio_tokens - cached_text_tokens,
    )
    text_tokens: Final = max(
        (
            cast(int | None, getattr(usage.prompt_tokens_details, "text_tokens", None))
            or 0  # default to prompt tokens, if this field is not set
        )
        - cached_text_tokens,
        0,
    )
    audio_tokens: Final = max(
        (cast(int | None, getattr(usage.prompt_tokens_details, "audio_tokens", 0)) or 0) - cached_audio_tokens,
        0,
    )
    image_tokens: Final = max(
        (cast(int | None, getattr(usage.prompt_tokens_details, "image_tokens", 0)) or 0) - cached_image_tokens,
        0,
    )
    video_tokens: Final = coerce_token_count(getattr(usage.prompt_tokens_details, "video_tokens", 0))
    character_count: Final = (
        cast(
            int | None,
            getattr(usage.prompt_tokens_details, "character_count", 0),
        )
        or 0
    )
    image_count: Final = cast(int | None, getattr(usage.prompt_tokens_details, "image_count", 0)) or 0
    video_length_seconds: Final = (
        cast(
            float | None,
            getattr(usage.prompt_tokens_details, "video_length_seconds", 0),
        )
        or 0.0
    )
    audio_length_seconds: Final = (
        cast(
            float | None,
            getattr(usage.prompt_tokens_details, "audio_length_seconds", 0),
        )
        or 0.0
    )
    query_count: Final = coerce_token_count(getattr(usage.prompt_tokens_details, "query_count", 0))

    return PromptTokensDetailsResult(
        cache_hit_tokens=cache_hit_tokens,
        cache_hit_audio_tokens=cached_audio_tokens,
        cached_text_tokens=cached_text_tokens,
        cached_audio_tokens=cached_audio_tokens,
        cached_image_tokens=cached_image_tokens,
        has_cached_tokens_details=cached_tokens_details is not None,
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
        character_count=character_count,
        image_count=image_count,
        video_length_seconds=float(video_length_seconds),
        audio_length_seconds=float(audio_length_seconds),
        query_count=query_count,
    )


class CompletionTokensDetailsResult(TypedDict):
    audio_tokens: int
    text_tokens: int
    reasoning_tokens: int
    image_tokens: int
    video_tokens: int


def parse_completion_tokens_details(usage: Usage) -> CompletionTokensDetailsResult:
    audio_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "audio_tokens", 0),
        )
        or 0
    )
    reported_text_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "text_tokens", None),
        )
        or 0  # default to completion tokens, if this field is not set
    )
    reasoning_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "reasoning_tokens", 0),
        )
        or 0
    )
    image_tokens: Final = (
        cast(
            int | None,
            getattr(usage.completion_tokens_details, "image_tokens", 0),
        )
        or 0
    )
    video_tokens: Final = coerce_token_count(getattr(usage.completion_tokens_details, "video_tokens", 0))
    text_tokens: Final = text_tokens_without_nested_reasoning(
        completion_tokens=usage.completion_tokens,
        text_tokens=reported_text_tokens,
        reasoning_tokens=reasoning_tokens,
        other_modality_tokens=audio_tokens + image_tokens + video_tokens,
    )

    return CompletionTokensDetailsResult(
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        reasoning_tokens=reasoning_tokens,
        image_tokens=image_tokens,
        video_tokens=video_tokens,
    )

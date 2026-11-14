"""
Transformation logic for context caching.

Why separate file? Make it easy to see how transformation works
"""

import re
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from litellm.exceptions import ImageFetchError
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.prompt_templates.image_handling import async_inline_remote_media, inline_remote_media
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    gemini_convert_messages_with_history,
    inlines_unless_gemini_file,
    transform_system_message,
)


def get_first_continuous_block_idx(
    filtered_messages: list[tuple[int, AllMessageValues]],  # (idx, message)
) -> int:
    """
    Find the array index that ends the first continuous sequence of message blocks.

    Args:
        filtered_messages: List of tuples containing (index, message) pairs

    Returns:
        int: The array index where the first continuous sequence ends
    """
    if not filtered_messages:
        return -1

    if len(filtered_messages) == 1:
        return 0

    current_value = filtered_messages[0][0]

    # Search forward through the array indices
    for i in range(1, len(filtered_messages)):
        if filtered_messages[i][0] != current_value + 1:
            return i - 1
        current_value = filtered_messages[i][0]

    # If we made it through the whole list, return the last index
    return len(filtered_messages) - 1


def extract_ttl_from_cached_messages(messages: list[AllMessageValues]) -> str | None:
    """
    Returns the longest valid TTL on the cached parts, normalized to Gemini's "<seconds>s" form, since one
    Gemini cache holds every marked part and must outlive the longest-lived of them.
    """
    ttls: Final = chain.from_iterable(
        _cache_control_ttls(message) for message in messages if is_cached_message(message)
    )
    return max(ttls, key=lambda ttl: float(ttl[:-1]), default=None)


def _cache_control_ttls(message: AllMessageValues) -> Iterator[str]:
    fields: Final = as_str_mapping(message)
    parts: Final = None if fields is None else as_object_list(fields.get("content"))
    return (ttl for part in parts or () if (ttl := _ephemeral_cache_control_ttl(part)) is not None)


def _ephemeral_cache_control_ttl(part: object) -> str | None:
    fields: Final = as_str_mapping(part)
    control: Final = None if fields is None else as_str_mapping(fields.get("cache_control"))
    if control is None or control.get("type") != "ephemeral":
        return None
    return _normalize_ttl_to_seconds(control.get("ttl"))


_TTL_PATTERN: Final = re.compile(r"^([0-9]*\.?[0-9]+)([smh])$")
_TTL_UNIT_SECONDS: Final = MappingProxyType({"s": 1, "m": 60, "h": 3600})
_LAST_EXPIRY_GOOGLE_ACCEPTS: Final = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def _normalize_ttl_to_seconds(ttl: object) -> str | None:
    if not isinstance(ttl, str):
        return None
    match: Final = _TTL_PATTERN.match(ttl)
    if match is None:
        return None
    seconds: Final = round(float(match.group(1)) * _TTL_UNIT_SECONDS[match.group(2)], 9)
    longest_ttl: Final = (_LAST_EXPIRY_GOOGLE_ACCEPTS - datetime.now(timezone.utc)).total_seconds()
    if not 0 < seconds <= longest_ttl:
        return None
    return f"{seconds:.9f}".rstrip("0").rstrip(".") + "s"


def separate_cached_messages(
    messages: list[AllMessageValues],
) -> tuple[list[AllMessageValues], list[AllMessageValues]]:
    """
    Returns the conversation prefix that ends at the last cached part of the first continuous block of
    cached messages, plus every later system message, since generateContent rejects a system_instruction
    beside a cachedContent, and the remaining messages in their original order.
    """
    filtered_messages: Final = [(idx, message) for idx, message in enumerate(messages) if is_cached_message(message)]
    if not filtered_messages:
        return [], messages

    prefix_end: Final = filtered_messages[get_first_continuous_block_idx(filtered_messages)][0]
    cached_head, uncached_tail = _split_after_last_cached_part(messages[prefix_end])
    later_messages: Final = messages[prefix_end + 1 :]
    return (
        [
            *messages[:prefix_end],
            cached_head,
            *(message for message in later_messages if message["role"] == "system"),
        ],
        [*uncached_tail, *(message for message in later_messages if message["role"] != "system")],
    )


def _split_after_last_cached_part(
    message: AllMessageValues,
) -> tuple[AllMessageValues, tuple[AllMessageValues, ...]]:
    if message["role"] != "user" or message.get("cache_control") is not None:
        return message, ()
    content: Final = message["content"]
    if not isinstance(content, list):
        return message, ()
    cut: Final = max(
        (idx + 1 for idx, part in enumerate(content) if part.get("cache_control") is not None),
        default=len(content),
    )
    if cut in (0, len(content)):
        return message, ()
    head: Final[ChatCompletionUserMessage] = {**message, "content": content[:cut]}
    tail: Final[ChatCompletionUserMessage] = {**message, "content": content[cut:]}
    return head, (tail,)


def cached_messages_with_media_inlined(
    cached_messages: Sequence[AllMessageValues],
) -> Sequence[AllMessageValues] | ImageFetchError:
    try:
        return inline_remote_media(list(cached_messages), should_inline=inlines_unless_gemini_file)
    except ImageFetchError as error:
        return error


async def async_cached_messages_with_media_inlined(
    cached_messages: Sequence[AllMessageValues],
) -> Sequence[AllMessageValues] | ImageFetchError:
    try:
        return await async_inline_remote_media(list(cached_messages), should_inline=inlines_unless_gemini_file)
    except ImageFetchError as error:
        return error


def cached_messages_end_on_supported_turn(cached_messages: Sequence[AllMessageValues]) -> bool:
    """
    The cachedContents API rejects contents ending on a model turn, which is how it
    classifies both assistant messages and tool results, with HTTP 400
    "Requests ending with a model turn are not supported". System messages are
    extracted into system_instruction before contents are built, so the terminal
    turn is the last non-system message.
    """
    non_system_messages: Final = tuple(message for message in cached_messages if message.get("role") != "system")
    if not non_system_messages:
        return bool(cached_messages)
    return non_system_messages[-1].get("role") not in ("assistant", "tool", "function")


def transform_openai_messages_to_gemini_context_caching(
    model: str,
    messages: list[AllMessageValues],
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    cache_key: str,
    vertex_project: str | None,
    vertex_location: str | None,
) -> CachedContentRequestBody:
    # Extract TTL from cached messages BEFORE system message transformation
    ttl: Final = extract_ttl_from_cached_messages(messages)

    supports_system_message: Final = get_supports_system_message(model=model, custom_llm_provider=custom_llm_provider)

    transformed_system_messages, new_messages = transform_system_message(
        supports_system_message=supports_system_message,
        messages=messages,
    )

    transformed_messages: Final = gemini_convert_messages_with_history(
        messages=new_messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    model_name = f"models/{model}"

    if custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        model_name = f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/{model_name}"

    data: Final = CachedContentRequestBody(
        contents=transformed_messages,
        model=model_name,
        displayName=cache_key,
    )

    # Add TTL if present and valid
    if ttl:
        data["ttl"] = ttl

    if transformed_system_messages is not None:
        data["system_instruction"] = transformed_system_messages

    return data


async def async_transform_openai_messages_to_gemini_context_caching(
    model: str,
    messages: Sequence[AllMessageValues],
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    cache_key: str,
    vertex_project: str | None,
    vertex_location: str | None,
) -> CachedContentRequestBody:
    return await asyncify(transform_openai_messages_to_gemini_context_caching)(
        model=model,
        messages=list(messages),
        custom_llm_provider=custom_llm_provider,
        cache_key=cache_key,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

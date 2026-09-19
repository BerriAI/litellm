"""
Transformation logic for context caching.

Why separate file? Make it easy to see how transformation works
"""

import re
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final, Literal

from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_system_message,
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
    Extract TTL from cached messages. Returns the first valid TTL found.

    Args:
        messages: List of messages to extract TTL from

    Returns:
        Optional[str]: TTL normalized to Gemini's "<seconds>s" form, or None if not found/invalid
    """
    for message in messages:
        if not is_cached_message(message):
            continue

        content = message.get("content")
        if not content or isinstance(content, str):
            continue

        for content_item in content:
            # Type check to ensure content_item is a dictionary before calling .get()
            if not isinstance(content_item, dict):
                continue

            cache_control = content_item.get("cache_control")
            if not cache_control or not isinstance(cache_control, dict):
                continue

            if cache_control.get("type") != "ephemeral":
                continue

            normalized_ttl = _normalize_ttl_to_seconds(cache_control.get("ttl"))
            if normalized_ttl is not None:
                return normalized_ttl

    return None


_TTL_PATTERN: Final = re.compile(r"^([0-9]*\.?[0-9]+)([smh])$")
_TTL_UNIT_SECONDS: Final = MappingProxyType({"s": 1, "m": 60, "h": 3600})
_PROTOBUF_DURATION_MAX_SECONDS: Final = 315_576_000_000


def _normalize_ttl_to_seconds(ttl: object) -> str | None:
    if not isinstance(ttl, str):
        return None
    match: Final = _TTL_PATTERN.match(ttl)
    if match is None:
        return None
    seconds: Final = round(float(match.group(1)) * _TTL_UNIT_SECONDS[match.group(2)], 9)
    if not 0 < seconds <= _PROTOBUF_DURATION_MAX_SECONDS:
        return None
    return f"{seconds:.9f}".rstrip("0").rstrip(".") + "s"


def separate_cached_messages(
    messages: list[AllMessageValues],
) -> tuple[list[AllMessageValues], list[AllMessageValues]]:
    """
    Returns separated cached and non-cached messages.

    Args:
        messages: List of messages to be separated.

    Returns:
        Tuple containing:
        - cached_messages: List of cached messages.
        - non_cached_messages: List of non-cached messages.
    """
    cached_messages: list[AllMessageValues] = []
    non_cached_messages: list[AllMessageValues] = []

    # Extract cached messages and their indices
    filtered_messages: Final[list[tuple[int, AllMessageValues]]] = []
    for idx, message in enumerate(messages):
        if is_cached_message(message=message):
            filtered_messages.append((idx, message))

    # Validate only one block of continuous cached messages
    last_continuous_block_idx: Final = get_first_continuous_block_idx(filtered_messages)
    # Separate messages based on the block of cached messages
    if filtered_messages and last_continuous_block_idx is not None:
        first_cached_idx: Final = filtered_messages[0][0]
        last_cached_idx: Final = filtered_messages[last_continuous_block_idx][0]

        cached_messages = messages[first_cached_idx : last_cached_idx + 1]
        non_cached_messages = messages[:first_cached_idx] + messages[last_cached_idx + 1 :]
    else:
        non_cached_messages = messages

    return cached_messages, non_cached_messages


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

    transformed_system_messages, new_messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )

    transformed_messages: Final = _gemini_convert_messages_with_history(
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

"""
Transformation logic for context caching.

Why separate file? Make it easy to see how transformation works
"""

import re
from collections.abc import Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal

from litellm.litellm_core_utils.prompt_templates.factory import convert_to_gemini_tool_call_invoke
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    is_system_message,
    message_field,
    place_mid_conversation_system,
    split_leading_system_run,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    gemini_convert_messages_with_history,
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


_TOOL_RESULT_ROLES: Final = frozenset({"tool", "function"})
_MODEL_TURN_ROLES: Final = frozenset({"assistant", *_TOOL_RESULT_ROLES})


def _cached_block_end(messages: Sequence[AllMessageValues], block_start: int, block_end: int) -> int:
    next_turn: Final = next(
        (idx for idx in range(block_end, len(messages)) if not is_system_message(messages[idx])), None
    )
    if next_turn is None or message_field(messages[next_turn], "role") not in _TOOL_RESULT_ROLES:
        return block_end
    return next(
        (end for end in range(block_end, block_start, -1) if not is_system_message(messages[end - 1])),
        block_start,
    )


def separate_cached_messages(
    messages: list[AllMessageValues],
    supports_system_message: bool = True,
) -> tuple[list[AllMessageValues], list[AllMessageValues]]:
    """
    Returns the first continuous block of cached messages and the remaining messages.

    The leading run of system messages outside that block stays as it is in the remaining messages, so it still
    becomes the system_instruction. Every later system message comes back already converted to a user turn in
    place, so a reminder added each turn is never hoisted into a system_instruction beside the cachedContent. A
    system run that ends the cached block with only system messages between it and a tool result stays out of
    the cached block, since its user turn goes after that tool result. A model without system message support
    gets its later system messages back as they are, so the request renders them the way it does without a
    cache.
    """
    filtered_messages: Final = [(idx, message) for idx, message in enumerate(messages) if is_cached_message(message)]
    if not filtered_messages:
        return [], messages

    leading_system_run, _ = split_leading_system_run(messages)
    last_continuous_block_idx: Final = get_first_continuous_block_idx(filtered_messages)
    block_start: Final = filtered_messages[0][0]
    cached_block: Final = range(
        block_start,
        _cached_block_end(messages, block_start, filtered_messages[last_continuous_block_idx][0] + 1),
    )
    uncached: Final = tuple((idx, message) for idx, message in enumerate(messages) if idx not in cached_block)
    later_messages: Final = tuple(message for idx, message in uncached if idx >= len(leading_system_run))
    return (
        messages[cached_block.start : cached_block.stop],
        [
            *(message for idx, message in uncached if idx < len(leading_system_run)),
            *(
                place_mid_conversation_system(later_messages, supports_mid_conversation_system=False)
                if supports_system_message
                else later_messages
            ),
        ],
    )


def cached_messages_end_on_supported_turn(cached_messages: Sequence[AllMessageValues]) -> bool:
    """
    The cachedContents API rejects contents ending on a model turn, which is how it
    classifies both assistant messages and tool results, with HTTP 400
    "Requests ending with a model turn are not supported". The leading system messages
    are extracted into system_instruction and every later one becomes a user turn before
    contents are built, so the terminal turn is the last message after that conversion.
    """
    _, rest = split_leading_system_run(cached_messages)
    contents_messages: Final = place_mid_conversation_system(rest, supports_mid_conversation_system=False)
    if not contents_messages:
        return bool(cached_messages)
    return message_field(contents_messages[-1], "role") not in _MODEL_TURN_ROLES


def reject_unusable_tool_call_arguments(
    cached_messages: Sequence[AllMessageValues],
    model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    drop_params: bool,
) -> None:
    if drop_params:
        return
    for message in cached_messages:
        if message["role"] == "assistant":
            convert_to_gemini_tool_call_invoke(message, model=model, custom_llm_provider=custom_llm_provider)


def transform_openai_messages_to_gemini_context_caching(
    model: str,
    messages: list[AllMessageValues],
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    cache_key: str,
    vertex_project: str | None,
    vertex_location: str | None,
    drop_params: bool = False,
) -> CachedContentRequestBody:
    # Extract TTL from cached messages BEFORE system message transformation
    ttl: Final = extract_ttl_from_cached_messages(messages)

    supports_system_message: Final = get_supports_system_message(model=model, custom_llm_provider=custom_llm_provider)

    transformed_system_messages, new_messages = transform_system_message(
        supports_system_message=supports_system_message,
        messages=messages,
        drop_params=drop_params,
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    transformed_messages: Final = gemini_convert_messages_with_history(
        messages=new_messages,
        model=model,
        litellm_params={"drop_params": drop_params},
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

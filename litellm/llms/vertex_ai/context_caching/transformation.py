"""
Transformation logic for context caching.

Why separate file? Make it easy to see how transformation works
"""

import re
from typing import List, Optional, Tuple, Literal

from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_system_message,
)


def get_first_continuous_block_idx(
    filtered_messages: List[Tuple[int, AllMessageValues]],  # (idx, message)
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


def extract_ttl_from_cached_messages(messages: List[AllMessageValues]) -> Optional[str]:
    """
    Extract TTL from cached messages. Returns the first valid TTL found.

    Args:
        messages: List of messages to extract TTL from

    Returns:
        Optional[str]: TTL string in format "3600s" or None if not found/invalid
    """
    for message in messages:
        # Check message-level cache_control first
        msg_cache_control = (
            message.get("cache_control") if isinstance(message, dict) else getattr(message, "cache_control", None)
        )
        if msg_cache_control is not None:
            cc_type = (
                msg_cache_control.get("type")
                if isinstance(msg_cache_control, dict)
                else getattr(msg_cache_control, "type", None)
            )
            if cc_type == "ephemeral":
                ttl = (
                    msg_cache_control.get("ttl")
                    if isinstance(msg_cache_control, dict)
                    else getattr(msg_cache_control, "ttl", None)
                )
                normalized = _normalize_ttl_to_seconds(ttl)
                if normalized is not None:
                    return normalized

        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, list):
            continue

        for content_item in content:
            # Check if content_item is dict or object model
            if isinstance(content_item, dict):
                cache_control = content_item.get("cache_control")
                item_type = content_item.get("type")
            else:
                cache_control = getattr(content_item, "cache_control", None)
                item_type = getattr(content_item, "type", None)

            if item_type == "text" and cache_control is not None:
                cc_type = (
                    cache_control.get("type")
                    if isinstance(cache_control, dict)
                    else getattr(cache_control, "type", None)
                )
                if cc_type == "ephemeral":
                    ttl = (
                        cache_control.get("ttl")
                        if isinstance(cache_control, dict)
                        else getattr(cache_control, "ttl", None)
                    )
                    normalized = _normalize_ttl_to_seconds(ttl)
                    if normalized is not None:
                        return normalized

    return None


def _is_valid_ttl_format(ttl: str) -> bool:
    """
    Validate TTL format. Should be a string ending with 's' for seconds.
    Examples: "3600s", "7200s", "1.5s"

    Args:
        ttl: TTL string to validate

    Returns:
        bool: True if valid format, False otherwise
    """
    if not isinstance(ttl, str):
        return False

    # TTL should end with 's' and contain a valid number before it
    pattern = r"^([0-9]*\.?[0-9]+)s$"
    match = re.match(pattern, ttl)

    if not match:
        return False

    try:
        # Ensure the numeric part is valid and positive
        numeric_part = float(match.group(1))
        return numeric_part > 0
    except ValueError:
        return False


def _normalize_ttl_to_seconds(ttl: object) -> Optional[str]:
    """
    Normalize a cache_control TTL into Gemini's "<seconds>s" format.

    Accepts Gemini-native seconds (e.g. "3600s", "1.5s") and Anthropic-style
    minute/hour units (e.g. "5m", "1h") that Claude Code and the Anthropic
    /v1/messages spec use. Returns None for missing or unparseable values so
    Gemini falls back to its own default TTL.
    """
    if not isinstance(ttl, str):
        return None

    if _is_valid_ttl_format(ttl):
        return ttl

    match = re.match(r"^([0-9]*\.?[0-9]+)(m|h)$", ttl)
    if not match:
        return None

    value = float(match.group(1))

    if value <= 0:
        return None

    seconds = value * (60 if match.group(2) == "m" else 3600)
    return f"{int(seconds)}s" if seconds.is_integer() else f"{seconds}s"


def get_gemini_context_caching_min_tokens(model: str) -> int:
    """
    Minimum input token count required to create an explicit Gemini context cache.

    Gemini rejects a cachedContents create below a per-model floor with a 400, so
    the caller skips caching below this value. Figures from
    https://ai.google.dev/gemini-api/docs/caching (Gemini 1.5 -> 32768, Gemini 2.5
    -> 2048, Gemini 3.x -> 4096). Unknown Gemini models default to the highest
    known floor so a create is never attempted below the real minimum.
    """
    model_lower = model.lower()
    if "gemini-1.5" in model_lower or "gemini-1-5" in model_lower:
        return 32768
    if "gemini-2.5" in model_lower or "gemini-2-5" in model_lower:
        return 2048
    if "gemini-3" in model_lower:
        return 4096
    return 32768


def separate_cached_messages(
    messages: List[AllMessageValues],
) -> Tuple[List[AllMessageValues], List[AllMessageValues]]:
    """
    Returns separated cached and non-cached messages.

    Args:
        messages: List of messages to be separated.

    Returns:
        Tuple containing:
        - cached_messages: List of cached messages.
        - non_cached_messages: List of non-cached messages.
    """
    cached_messages: List[AllMessageValues] = []
    non_cached_messages: List[AllMessageValues] = []

    # Extract cached messages and their indices
    filtered_messages: List[Tuple[int, AllMessageValues]] = []
    for idx, message in enumerate(messages):
        if is_cached_message(message=message):
            filtered_messages.append((idx, message))

    # Validate only one block of continuous cached messages
    last_continuous_block_idx = get_first_continuous_block_idx(filtered_messages)
    # Separate messages based on the block of cached messages
    if filtered_messages and last_continuous_block_idx is not None:
        first_cached_idx = filtered_messages[0][0]
        last_cached_idx = filtered_messages[last_continuous_block_idx][0]

        cached_messages = messages[first_cached_idx : last_cached_idx + 1]
        non_cached_messages = messages[:first_cached_idx] + messages[last_cached_idx + 1 :]
    else:
        non_cached_messages = messages

    return cached_messages, non_cached_messages


def transform_openai_messages_to_gemini_context_caching(
    model: str,
    messages: List[AllMessageValues],
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    cache_key: str,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
) -> CachedContentRequestBody:
    # Extract TTL from cached messages BEFORE system message transformation
    ttl = extract_ttl_from_cached_messages(messages)

    supports_system_message = get_supports_system_message(model=model, custom_llm_provider=custom_llm_provider)

    transformed_system_messages, new_messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )

    transformed_messages = _gemini_convert_messages_with_history(
        messages=new_messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    model_name = "models/{}".format(model)

    if custom_llm_provider == "vertex_ai" or custom_llm_provider == "vertex_ai_beta":
        model_name = f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/{model_name}"

    data = CachedContentRequestBody(
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

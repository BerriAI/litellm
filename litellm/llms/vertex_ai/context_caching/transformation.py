"""
Transformation logic for context caching. 

Why separate file? Make it easy to see how transformation works
"""

import re
from typing import List, Optional, Tuple

from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_system_message,
)


def get_first_continuous_block_idx(
    filtered_messages: List[Tuple[int, AllMessageValues]]  # (idx, message)
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
                
            ttl = cache_control.get("ttl")
            if ttl and _is_valid_ttl_format(ttl):
                return str(ttl)
    
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
    pattern = r'^([0-9]*\.?[0-9]+)s$'
    match = re.match(pattern, ttl)
    
    if not match:
        return False
    
    try:
        # Ensure the numeric part is valid and positive
        numeric_part = float(match.group(1))
        return numeric_part > 0
    except ValueError:
        return False


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
        non_cached_messages = (
            messages[:first_cached_idx] + messages[last_cached_idx + 1 :]
        )
    else:
        non_cached_messages = messages

    return cached_messages, non_cached_messages


def transform_openai_messages_to_gemini_context_caching(
    model: str, messages: List[AllMessageValues], cache_key: str
) -> CachedContentRequestBody:
    # Extract TTL from cached messages BEFORE system message transformation
    ttl = extract_ttl_from_cached_messages(messages)
    
    supports_system_message = get_supports_system_message(
        model=model, custom_llm_provider="gemini"
    )

    transformed_system_messages, new_messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )

    transformed_messages = _gemini_convert_messages_with_history(messages=new_messages)
    
    data = CachedContentRequestBody(
        contents=transformed_messages,
        model="models/{}".format(model),
        displayName=cache_key,
    )
    
    # Add TTL if present and valid
    if ttl:
        data["ttl"] = ttl
    
    if transformed_system_messages is not None:
        data["system_instruction"] = transformed_system_messages

    return data

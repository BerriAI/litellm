"""
Transformation logic for context caching.

Why separate file? Make it easy to see how transformation works
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from litellm.caching.caching import Cache

from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody, ToolConfig, Tools
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
    model: str,
    messages: List[AllMessageValues],
    cache_key: str,
    tools: Optional[List[Tools]] = None,
    tool_choice: Optional[ToolConfig] = None,
) -> CachedContentRequestBody:
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
        tools=tools,
        toolConfig=tool_choice,
    )
    if transformed_system_messages is not None:
        data["system_instruction"] = transformed_system_messages

    return data


local_cache_obj = Cache(type=LiteLLMCacheType.LOCAL)


@dataclass(frozen=True)
class CacheSplitResult:
    """Result of splitting messages into cacheable and non-cacheable parts"""

    remaining_messages: List[
        AllMessageValues
    ]  # Messages that should be sent in actual request
    optional_params: Dict[str, Any]  # Updated params to be sent in actual request
    cache_key: Optional[str]  # Key to use for checking if content is already cached
    cached_content: Optional[
        str
    ]  # cached content ID, no further processing is needed once this is defined
    cache_request_body: Optional[
        CachedContentRequestBody
    ]  # Request body to create new cache if needed

    def with_cached_content(self, cached_content: str) -> "CacheSplitResult":
        """
        Returns an updated CacheSplitResult with the cached content applied.
        """
        updated_params = {**self.optional_params, "cached_content": cached_content}
        return replace(
            self,
            cached_content=cached_content,
            optional_params=updated_params,
            cache_request_body=None,
        )


def extract_cache_configuration(
    model: str,
    messages: List[AllMessageValues],
    optional_params: Dict[str, Any],
) -> CacheSplitResult:
    """
    Checks if a given request should have a cache, and if so, extracts the cache configuration, returning
    a modified version of the messages and optional params.

    - Removes the cached content from the messages
    - Adds the cache key to the optional params
    - If there's cached content, also moves the tool call and tool choice to the optional params, as that is
      required for the cache to work. (The tools are moved into some sort of system prompt on google's side)

      Relevant error:
        "error": {
            "code": 400,
            "message": "CachedContent can not be used with GenerateContent request setting system_instruction, tools or tool_config.\n\nProposed fix: move those values to CachedContent from GenerateContent request.",
            "status": "INVALID_ARGUMENT"
        }

    Returns:
        CacheSplitResult with:
        - remaining_messages: Messages that should be sent in the actual request
        - cache_key: The key to use for checking if content is already cached
        - cached_content: The cached content ID if already provided
        - cache_request_body: The request body to create a new cache entry if needed
    """
    # If cached content is already provided, no need to process messages
    if (
        "cached_content" in optional_params
        and optional_params["cached_content"] is not None
    ):
        return CacheSplitResult(
            remaining_messages=messages,
            optional_params=optional_params,
            cache_key=None,
            cached_content=optional_params["cached_content"],
            cache_request_body=None,
        )

    # Separate messages that can be cached from those that can't
    cached_messages, non_cached_messages = separate_cached_messages(messages=messages)


    # If no messages can be cached, return original messages
    if len(cached_messages) == 0:
        return CacheSplitResult(
            remaining_messages=messages,
            optional_params=optional_params,
            cache_key=None,
            cached_content=None,
            cache_request_body=None,
        )
    if "tools" in optional_params or "tool_choice" in optional_params:
        optional_params = optional_params.copy()
        tools = optional_params.pop("tools", None)
        tool_choice = optional_params.pop("tool_choice", None)
    else:
        tools = None
        tool_choice = None
    key_kwargs = {}
    if tools is not None:
        key_kwargs["tools"] = tools
    if tool_choice is not None:
        key_kwargs["tool_choice"] = tool_choice

    # Generate cache key for the cacheable messages
    cache_key = local_cache_obj.get_cache_key(messages=cached_messages, **key_kwargs)

    # Transform cached messages into request body
    cache_request_body = transform_openai_messages_to_gemini_context_caching(
        model=model, messages=cached_messages, cache_key=cache_key, tools=tools, tool_choice=tool_choice
    )

    return CacheSplitResult(
        remaining_messages=non_cached_messages,
        optional_params=optional_params,
        cache_key=cache_key,
        cached_content=None,
        cache_request_body=cache_request_body,
    )

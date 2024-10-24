"""
Transformation logic for context caching. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List, Tuple

from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody, SystemInstructions
from litellm.utils import is_cached_message

from ..common_utils import VertexAIError, get_supports_system_message
from ..gemini.transformation import _transform_system_message
from ..gemini.vertex_and_google_ai_studio_gemini import (
    _gemini_convert_messages_with_history,
)


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
    if len(filtered_messages) > 1:
        expected_idx = filtered_messages[0][0] + 1
        for idx, _ in filtered_messages[1:]:
            if idx != expected_idx:
                raise VertexAIError(
                    status_code=422,
                    message="Gemini Context Caching only supports 1 message/block of continuous messages. Your idx, messages were - {}".format(
                        filtered_messages
                    ),
                )
            expected_idx += 1

    # Separate messages based on the block of cached messages
    if filtered_messages:
        first_cached_idx = filtered_messages[0][0]
        last_cached_idx = filtered_messages[-1][0]

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
    if transformed_system_messages is not None:
        data["system_instruction"] = transformed_system_messages

    return data

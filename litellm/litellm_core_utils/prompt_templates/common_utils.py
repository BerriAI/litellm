"""
Common utility functions used for translating messages across providers
"""

from typing import Dict, List, Literal, Optional, Union, cast

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from litellm.types.utils import Choices, ModelResponse, StreamingChoices

DEFAULT_USER_CONTINUE_MESSAGE = ChatCompletionUserMessage(
    content="Please continue.", role="user"
)

DEFAULT_ASSISTANT_CONTINUE_MESSAGE = ChatCompletionAssistantMessage(
    content="Please continue.", role="assistant"
)


def handle_messages_with_content_list_to_str_conversion(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    """
    Handles messages with content list conversion
    """
    for message in messages:
        texts = convert_content_list_to_str(message=message)
        if texts:
            message["content"] = texts
    return messages


def strip_name_from_messages(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    """
    Removes 'name' from messages
    """
    new_messages = []
    for message in messages:
        msg_role = message.get("role")
        msg_copy = message.copy()
        if msg_role == "user":
            msg_copy.pop("name", None)  # type: ignore
        new_messages.append(msg_copy)
    return new_messages


def strip_none_values_from_message(message: AllMessageValues) -> AllMessageValues:
    """
    Strips None values from message
    """
    return cast(AllMessageValues, {k: v for k, v in message.items() if v is not None})


def convert_content_list_to_str(message: AllMessageValues) -> str:
    """
    - handles scenario where content is list and not string
    - content list is just text, and no images

    Motivation: mistral api + azure ai don't support content as a list
    """
    texts = ""
    message_content = message.get("content")
    if message_content:
        if message_content is not None and isinstance(message_content, list):
            for c in message_content:
                text_content = c.get("text")
                if text_content:
                    texts += text_content
        elif message_content is not None and isinstance(message_content, str):
            texts = message_content

    return texts


def get_str_from_messages(messages: List[AllMessageValues]) -> str:
    """
    Converts a list of messages to a string
    """
    text = ""
    for message in messages:
        text += convert_content_list_to_str(message=message)
    return text


def is_non_content_values_set(message: AllMessageValues) -> bool:
    ignore_keys = ["content", "role", "name"]
    return any(
        message.get(key, None) is not None for key in message if key not in ignore_keys
    )


def _audio_or_image_in_message_content(message: AllMessageValues) -> bool:
    """
    Checks if message content contains an image or audio
    """
    message_content = message.get("content")
    if message_content:
        if message_content is not None and isinstance(message_content, list):
            for c in message_content:
                if c.get("type") == "image_url" or c.get("type") == "input_audio":
                    return True
    return False


def convert_openai_message_to_only_content_messages(
    messages: List[AllMessageValues],
) -> List[Dict[str, str]]:
    """
    Converts OpenAI messages to only content messages

    Used for calling guardrails integrations which expect string content
    """
    converted_messages = []
    user_roles = ["user", "tool", "function"]
    for message in messages:
        if message.get("role") in user_roles:
            converted_messages.append(
                {"role": "user", "content": convert_content_list_to_str(message)}
            )
        elif message.get("role") == "assistant":
            converted_messages.append(
                {"role": "assistant", "content": convert_content_list_to_str(message)}
            )
    return converted_messages


def get_content_from_model_response(response: Union[ModelResponse, dict]) -> str:
    """
    Gets content from model response
    """
    if isinstance(response, dict):
        new_response = ModelResponse(**response)
    else:
        new_response = response

    content = ""

    for choice in new_response.choices:
        if isinstance(choice, Choices):
            content += choice.message.content if choice.message.content else ""
            if choice.message.function_call:
                content += choice.message.function_call.model_dump_json()
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    content += tc.model_dump_json()
        elif isinstance(choice, StreamingChoices):
            content += getattr(choice, "delta", {}).get("content", "") or ""
    return content


def detect_first_expected_role(
    messages: List[AllMessageValues],
) -> Optional[Literal["user", "assistant"]]:
    """
    Detect the first expected role based on the message sequence.

    Rules:
    1. If messages list is empty, assume 'user' starts
    2. If first message is from assistant, expect 'user' next
    3. If first message is from user, expect 'assistant' next
    4. If first message is system, look at the next non-system message

    Returns:
        str: Either 'user' or 'assistant'
        None: If no 'user' or 'assistant' messages provided
    """
    if not messages:
        return "user"

    for message in messages:
        if message["role"] == "system":
            continue
        return "user" if message["role"] == "assistant" else "assistant"

    return None


def _insert_user_continue_message(
    messages: List[AllMessageValues],
    user_continue_message: Optional[ChatCompletionUserMessage],
    ensure_alternating_roles: bool,
) -> List[AllMessageValues]:
    """
    Inserts a user continue message into the messages list.
    Handles three cases:
    1. Initial assistant message
    2. Final assistant message
    3. Consecutive assistant messages

    Only inserts messages between consecutive assistant messages,
    ignoring all other role types.
    """
    if not messages:
        return messages

    result_messages = messages.copy()  # Don't modify the input list
    continue_message = user_continue_message or DEFAULT_USER_CONTINUE_MESSAGE

    # Handle first message if it's an assistant message
    if result_messages[0]["role"] == "assistant":
        result_messages.insert(0, continue_message)

    # Handle consecutive assistant messages and final message
    i = 1  # Start from second message since we handled first message
    while i < len(result_messages):
        curr_message = result_messages[i]
        prev_message = result_messages[i - 1]

        # Only check for consecutive assistant messages
        # Ignore all other role types
        if curr_message["role"] == "assistant" and prev_message["role"] == "assistant":
            result_messages.insert(i, continue_message)
            i += 2  # Skip over the message we just inserted
        else:
            i += 1

    # Handle final message
    if result_messages[-1]["role"] == "assistant" and ensure_alternating_roles:
        result_messages.append(continue_message)

    return result_messages


def _insert_assistant_continue_message(
    messages: List[AllMessageValues],
    assistant_continue_message: Optional[ChatCompletionAssistantMessage] = None,
    ensure_alternating_roles: bool = True,
) -> List[AllMessageValues]:
    """
    Add assistant continuation messages between consecutive user messages.

    Args:
        messages: List of message dictionaries
        assistant_continue_message: Optional custom assistant message
        ensure_alternating_roles: Whether to enforce alternating roles

    Returns:
        Modified list of messages with inserted assistant messages
    """
    if not ensure_alternating_roles or len(messages) <= 1:
        return messages

    # Create a new list to store modified messages
    modified_messages: List[AllMessageValues] = []

    for i, message in enumerate(messages):
        modified_messages.append(message)

        # Check if we need to insert an assistant message
        if (
            i < len(messages) - 1  # Not the last message
            and message.get("role") == "user"  # Current is user
            and messages[i + 1].get("role") == "user"
        ):  # Next is user

            # Insert assistant message
            continue_message = (
                assistant_continue_message or DEFAULT_ASSISTANT_CONTINUE_MESSAGE
            )
            modified_messages.append(continue_message)

    return modified_messages


def get_completion_messages(
    messages: List[AllMessageValues],
    assistant_continue_message: Optional[ChatCompletionAssistantMessage],
    user_continue_message: Optional[ChatCompletionUserMessage],
    ensure_alternating_roles: bool,
) -> List[AllMessageValues]:
    """
    Ensures messages alternate between user and assistant roles by adding placeholders
    only when there are consecutive messages of the same role.

    1. ensure 'user' message before 1st 'assistant' message
    2. ensure 'user' message after last 'assistant' message
    """
    if not ensure_alternating_roles:
        return messages.copy()

    ## INSERT USER CONTINUE MESSAGE
    messages = _insert_user_continue_message(
        messages, user_continue_message, ensure_alternating_roles
    )

    ## INSERT ASSISTANT CONTINUE MESSAGE
    messages = _insert_assistant_continue_message(
        messages, assistant_continue_message, ensure_alternating_roles
    )
    return messages

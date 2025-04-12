"""
Common utility functions used for translating messages across providers
"""

import io
import mimetypes
import re
from os import PathLike
from typing import Dict, List, Literal, Mapping, Optional, Union, cast

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionFileObject,
    ChatCompletionUserMessage,
)
from litellm.types.utils import (
    Choices,
    ExtractedFileData,
    FileTypes,
    ModelResponse,
    SpecialEnums,
    StreamingChoices,
)

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
    messages: List[AllMessageValues], allowed_name_roles: List[str] = ["user"]
) -> List[AllMessageValues]:
    """
    Removes 'name' from messages
    """
    new_messages = []
    for message in messages:
        msg_role = message.get("role")
        msg_copy = message.copy()
        if msg_role not in allowed_name_roles:
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


def get_format_from_file_id(file_id: Optional[str]) -> Optional[str]:
    """
    Gets format from file id

    unified_file_id = litellm_proxy:{};unified_id,{}
    If not a unified file id, returns 'file' as default format
    """
    from litellm.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

    if not file_id:
        return None
    try:
        transformed_file_id = (
            _PROXY_LiteLLMManagedFiles._convert_b64_uid_to_unified_uid(file_id)
        )
        if transformed_file_id.startswith(
            SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value
        ):
            match = re.match(
                f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}:(.*?);unified_id",
                transformed_file_id,
            )
            if match:
                return match.group(1)

        return None
    except Exception:
        return None


def update_messages_with_model_file_ids(
    messages: List[AllMessageValues],
    model_id: str,
    model_file_id_mapping: Dict[str, Dict[str, str]],
) -> List[AllMessageValues]:
    """
    Updates messages with model file ids.

    model_file_id_mapping: Dict[str, Dict[str, str]] = {
        "litellm_proxy/file_id": {
            "model_id": "provider_file_id"
        }
    }
    """

    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if content:
                if isinstance(content, str):
                    continue
                for c in content:
                    if c["type"] == "file":
                        file_object = cast(ChatCompletionFileObject, c)
                        file_object_file_field = file_object["file"]
                        file_id = file_object_file_field.get("file_id")
                        format = file_object_file_field.get(
                            "format", get_format_from_file_id(file_id)
                        )

                        if file_id:
                            provider_file_id = (
                                model_file_id_mapping.get(file_id, {}).get(model_id)
                                or file_id
                            )
                            file_object_file_field["file_id"] = provider_file_id
                        if format:
                            file_object_file_field["format"] = format
    return messages


def extract_file_data(file_data: FileTypes) -> ExtractedFileData:
    """
    Extracts and processes file data from various input formats.

    Args:
        file_data: Can be a tuple of (filename, content, [content_type], [headers]) or direct file content

    Returns:
        ExtractedFileData containing:
        - filename: Name of the file if provided
        - content: The file content in bytes
        - content_type: MIME type of the file
        - headers: Any additional headers
    """
    # Parse the file_data based on its type
    filename = None
    file_content = None
    content_type = None
    file_headers: Mapping[str, str] = {}

    if isinstance(file_data, tuple):
        if len(file_data) == 2:
            filename, file_content = file_data
        elif len(file_data) == 3:
            filename, file_content, content_type = file_data
        elif len(file_data) == 4:
            filename, file_content, content_type, file_headers = file_data
    else:
        file_content = file_data
    # Convert content to bytes
    if isinstance(file_content, (str, PathLike)):
        # If it's a path, open and read the file
        with open(file_content, "rb") as f:
            content = f.read()
    elif isinstance(file_content, io.IOBase):
        # If it's a file-like object
        content = file_content.read()

        if isinstance(content, str):
            content = content.encode("utf-8")
        # Reset file pointer to beginning
        file_content.seek(0)
    elif isinstance(file_content, bytes):
        content = file_content
    else:
        raise ValueError(f"Unsupported file content type: {type(file_content)}")

    # Use provided content type or guess based on filename
    if not content_type:
        content_type = (
            mimetypes.guess_type(filename)[0]
            if filename
            else "application/octet-stream"
        )

    return ExtractedFileData(
        filename=filename,
        content=content,
        content_type=content_type,
        headers=file_headers,
    )


def unpack_defs(schema, defs):
    properties = schema.get("properties", None)
    if properties is None:
        return

    for name, value in properties.items():
        ref_key = value.get("$ref", None)
        if ref_key is not None:
            ref = defs[ref_key.split("defs/")[-1]]
            unpack_defs(ref, defs)
            properties[name] = ref
            continue

        anyof = value.get("anyOf", None)
        if anyof is not None:
            for i, atype in enumerate(anyof):
                ref_key = atype.get("$ref", None)
                if ref_key is not None:
                    ref = defs[ref_key.split("defs/")[-1]]
                    unpack_defs(ref, defs)
                    anyof[i] = ref
            continue

        items = value.get("items", None)
        if items is not None:
            ref_key = items.get("$ref", None)
            if ref_key is not None:
                ref = defs[ref_key.split("defs/")[-1]]
                unpack_defs(ref, defs)
                value["items"] = ref
                continue

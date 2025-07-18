"""
Common utility functions used for translating messages across providers
"""

import io
import mimetypes
import re
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    cast,
)

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionFileObject,
    ChatCompletionResponseMessage,
    ChatCompletionToolParam,
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

if TYPE_CHECKING:  # newer pattern to avoid importing pydantic objects on __init__.py
    from litellm.types.llms.openai import ChatCompletionImageObject

DEFAULT_USER_CONTINUE_MESSAGE = ChatCompletionUserMessage(
    content="Please continue.", role="user"
)

DEFAULT_ASSISTANT_CONTINUE_MESSAGE = ChatCompletionAssistantMessage(
    content="Please continue.", role="assistant"
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LoggingClass


def handle_any_messages_to_chat_completion_str_messages_conversion(
    messages: Any,
) -> List[Dict[str, str]]:
    """
    Handles any messages to chat completion str messages conversion

    Relevant Issue: https://github.com/BerriAI/litellm/issues/9494
    """
    import json

    if isinstance(messages, list):
        try:
            return cast(
                List[Dict[str, str]],
                handle_messages_with_content_list_to_str_conversion(messages),
            )
        except Exception:
            return [{"input": json.dumps(message, default=str)} for message in messages]
    elif isinstance(messages, dict):
        try:
            return [{"input": json.dumps(messages, default=str)}]
        except Exception:
            return [{"input": str(messages)}]
    elif isinstance(messages, str):
        return [{"input": messages}]
    else:
        return [{"input": str(messages)}]


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


def convert_content_list_to_str(
    message: Union[AllMessageValues, ChatCompletionResponseMessage],
) -> str:
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
    from litellm.proxy.openai_files_endpoints.common_utils import (
        convert_b64_uid_to_unified_uid,
    )

    if not file_id:
        return None
    try:
        transformed_file_id = convert_b64_uid_to_unified_uid(file_id)
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


# ---------------------------------------------------------------------------
# Generic, dependency-free implementation of `unpack_defs`
# ---------------------------------------------------------------------------


def unpack_defs(schema: dict, defs: dict) -> None:
    """Expand *all* ``$ref`` entries pointing into ``$defs`` / ``definitions``.

    This utility walks the entire schema tree (dicts and lists) so it naturally
    resolves references hidden under any keyword – ``items``, ``allOf``,
    ``anyOf``, ``oneOf``, ``additionalProperties``, etc.

    It mutates *schema* in-place and does **not** return anything.  The helper
    keeps memory overhead low by resolving nodes as it encounters them rather
    than materialising a fully dereferenced copy first.
    """

    import copy
    from collections import deque

    # Combine the defs handed down by the caller with defs/definitions found on
    # the current node.  Local keys shadow parent keys to match JSON-schema
    # scoping rules.
    root_defs: dict = {
        **defs,
        **schema.get("$defs", {}),
        **schema.get("definitions", {}),
    }

    # Use iterative approach with queue to avoid recursion
    # Each item in queue is (node, parent_container, key/index, active_defs, seen_ids)
    queue: deque[
        tuple[Any, Union[dict, list, None], Union[str, int, None], dict, set]
    ] = deque([(schema, None, None, root_defs, set())])

    while queue:
        node, parent, key, active_defs, seen = queue.popleft()

        # Avoid infinite loops on self-referential schemas
        if id(node) in seen:
            continue
        seen = seen.copy()  # Create new set for this branch
        seen.add(id(node))

        # ----------------------------- dict -----------------------------
        if isinstance(node, dict):
            # --- Case 1: this node *is* a reference ---
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                target_schema = active_defs.get(ref_name)
                # Unknown reference – leave untouched
                if target_schema is None:
                    continue

                # Merge defs from the target to capture nested definitions
                child_defs = {
                    **active_defs,
                    **target_schema.get("$defs", {}),
                    **target_schema.get("definitions", {}),
                }

                # Replace the reference with resolved copy
                resolved = copy.deepcopy(target_schema)
                if parent is not None and key is not None:
                    if isinstance(parent, dict) and isinstance(key, str):
                        parent[key] = resolved
                    elif isinstance(parent, list) and isinstance(key, int):
                        parent[key] = resolved
                else:
                    # This is the root schema itself
                    schema.clear()
                    schema.update(resolved)
                    resolved = schema

                # Add resolved node to queue for further processing
                queue.append((resolved, parent, key, child_defs, seen))
                continue

            # --- Case 2: regular dict – process its values ---
            # Update defs with any nested $defs/definitions present *here*.
            current_defs = {
                **active_defs,
                **node.get("$defs", {}),
                **node.get("definitions", {}),
            }

            # Add all dict values to queue
            for k, v in node.items():
                queue.append((v, node, k, current_defs, seen))

        # ---------------------------- list ------------------------------
        elif isinstance(node, list):
            # Add all list items to queue
            for idx, item in enumerate(node):
                queue.append((item, node, idx, active_defs, seen))


def _get_image_mime_type_from_url(url: str) -> Optional[str]:
    """
    Get mime type for common image URLs
    See gemini mime types: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#image-requirements

    Supported by Gemini:
     application/pdf
    audio/mpeg
    audio/mp3
    audio/wav
    audio/ogg
    image/png
    image/jpeg
    image/webp
    text/plain
    video/mov
    video/mpeg
    video/mp4
    video/mpg
    video/avi
    video/wmv
    video/mpegps
    video/flv
    """
    url = url.lower()

    # Map file extensions to mime types
    mime_types = {
        # Images
        (".jpg", ".jpeg"): "image/jpeg",
        (".png",): "image/png",
        (".webp",): "image/webp",
        # Videos
        (".mp4",): "video/mp4",
        (".mov",): "video/mov",
        (".mpeg", ".mpg"): "video/mpeg",
        (".avi",): "video/avi",
        (".wmv",): "video/wmv",
        (".mpegps",): "video/mpegps",
        (".flv",): "video/flv",
        # Audio
        (".mp3",): "audio/mp3",
        (".wav",): "audio/wav",
        (".mpeg",): "audio/mpeg",
        (".ogg",): "audio/ogg",
        # Documents
        (".pdf",): "application/pdf",
        (".txt",): "text/plain",
    }

    # Check each extension group against the URL
    for extensions, mime_type in mime_types.items():
        if any(url.endswith(ext) for ext in extensions):
            return mime_type

    return None


def get_tool_call_names(tools: List[ChatCompletionToolParam]) -> List[str]:
    """
    Get tool call names from tools
    """
    tool_call_names: List[str] = []
    for tool in tools:
        if tool.get("type") == "function":
            tool_call_name = tool.get("function", {}).get("name")
            if tool_call_name:
                tool_call_names.append(tool_call_name)
    return tool_call_names


def is_function_call(optional_params: dict) -> bool:
    """
    Checks if the optional params contain the function call
    """
    if "functions" in optional_params and optional_params.get("functions"):
        return True
    return False


def get_file_ids_from_messages(messages: List[AllMessageValues]) -> List[str]:
    """
    Gets file ids from messages
    """
    file_ids = []
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
                        if file_id:
                            file_ids.append(file_id)
    return file_ids


def check_is_function_call(logging_obj: "LoggingClass") -> bool:
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        is_function_call,
    )

    if hasattr(logging_obj, "optional_params") and isinstance(
        logging_obj.optional_params, dict
    ):
        if is_function_call(logging_obj.optional_params):
            return True

    return False


def filter_value_from_dict(dictionary: dict, key: str, depth: int = 0) -> Any:
    """
    Filters a value from a dictionary

    Goes through the nested dict and removes the key if it exists
    """
    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return dictionary

    # Create a copy of keys to avoid modifying dict during iteration
    keys = list(dictionary.keys())
    for k in keys:
        v = dictionary[k]
        if k == key:
            del dictionary[k]
        elif isinstance(v, dict):
            filter_value_from_dict(v, key, depth + 1)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    filter_value_from_dict(item, key, depth + 1)
    return dictionary


def migrate_file_to_image_url(
    message: "ChatCompletionFileObject",
) -> "ChatCompletionImageObject":
    """
    Migrate file to image_url
    """
    from litellm.types.llms.openai import (
        ChatCompletionImageObject,
        ChatCompletionImageUrlObject,
    )

    file_id = message["file"].get("file_id")
    file_data = message["file"].get("file_data")
    format = message["file"].get("format")
    if not file_id and not file_data:
        raise ValueError("file_id and file_data are both None")
    image_url_object = ChatCompletionImageObject(
        type="image_url",
        image_url=ChatCompletionImageUrlObject(
            url=cast(str, file_id or file_data),
        ),
    )
    if format and isinstance(image_url_object["image_url"], dict):
        image_url_object["image_url"]["format"] = format
    return image_url_object


def get_last_user_message(messages: List[AllMessageValues]) -> Optional[str]:
    """
    Get the last consecutive block of messages from the user.

    Example:
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thank you!"},
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ]
    get_user_prompt(messages) -> "What is the weather in Tokyo?"
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        convert_content_list_to_str,
    )

    if not messages:
        return None

    # Iterate from the end to find the last consecutive block of user messages
    user_messages = []
    for message in reversed(messages):
        if message.get("role") == "user":
            user_messages.append(message)
        else:
            # Stop when we hit a non-user message
            break

    if not user_messages:
        return None

    # Reverse to get the messages in chronological order
    user_messages.reverse()

    user_prompt = ""
    for message in user_messages:
        text_content = convert_content_list_to_str(message)
        user_prompt += text_content + "\n"

    result = user_prompt.strip()
    return result if result else None


def set_last_user_message(
    messages: List[AllMessageValues], content: str
) -> List[AllMessageValues]:
    """
    Set the last user message

    1. remove all the last consecutive user messages (FROM THE END)
    2. add the new message
    """
    idx_to_remove = []
    for idx, message in enumerate(reversed(messages)):
        if message.get("role") == "user":
            idx_to_remove.append(idx)
        else:
            # Stop when we hit a non-user message
            break
    if idx_to_remove:
        messages = [
            message
            for idx, message in enumerate(reversed(messages))
            if idx not in idx_to_remove
        ]
        messages.reverse()
    messages.append({"role": "user", "content": content})
    return messages

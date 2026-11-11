import io
import mimetypes
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Final, cast  # noqa: TID251  # Existing casts retained unchanged in this pure move

from litellm import verbose_logger
from litellm.litellm_core_utils.batch_utils import InMemoryFile
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
)
from litellm.types.utils import (
    ExtractedFileData,
    FileTypes,
    SpecialEnums,
)


def get_format_from_file_id(file_id: str | None) -> str | None:
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
        transformed_file_id: Final = convert_b64_uid_to_unified_uid(file_id)
        if transformed_file_id.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            match: Final = re.match(
                f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}:(.*?);unified_id",
                transformed_file_id,
            )
            if match:
                return match.group(1)

        return None
    except Exception:
        return None


def update_messages_with_model_file_ids(
    messages: list[AllMessageValues],
    model_id: str | None,
    model_file_id_mapping: dict[str, dict[str, str]],
) -> list[AllMessageValues]:
    """
    Updates messages with model file ids.

    For managed files (unified file IDs), uses model_file_id_mapping if it
    resolves the id, otherwise decodes the base64-encoded unified file ID
    and extracts the llm_output_file_id directly. Mirrors the Responses-API
    sibling `update_responses_input_with_model_file_ids`.

    model_file_id_mapping: Dict[str, Dict[str, str]] = {
        "litellm_proxy/file_id": {
            "model_id": "provider_file_id"
        }
    }
    """
    from litellm.proxy.openai_files_endpoints.common_utils import (
        convert_b64_uid_to_unified_uid,
        get_original_file_id,
        is_base64_encoded_unified_file_id,
        is_model_embedded_id,
    )

    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if content:
                if isinstance(content, str):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        # Content list items aren't always dicts. e.g.
                        # text_completion forwards a token-ids list/list-of-
                        # lists through this path. Skip non-dict items
                        # instead of indexing into them.
                        continue
                    if c.get("type") == "file":
                        file_object = cast(ChatCompletionFileObject, c)
                        file_object_file_field = file_object.get("file")
                        if not isinstance(file_object_file_field, dict):
                            # Content block has `type: "file"` but not the
                            # OpenAI Chat Completions shape (e.g. a LangChain
                            # v1 standardized file block, or a provider-native
                            # shape that also uses `type: "file"`). Nothing to
                            # remap here, so skip instead of crashing.
                            continue
                        file_id = file_object_file_field.get("file_id")
                        format = file_object_file_field.get("format", get_format_from_file_id(file_id))

                        if file_id:
                            provider_file_id = (
                                model_file_id_mapping.get(file_id, {}).get(model_id)
                                if model_file_id_mapping and model_id is not None
                                else None
                            )
                            if not provider_file_id and is_base64_encoded_unified_file_id(file_id):
                                unified_file_id = convert_b64_uid_to_unified_uid(file_id)
                                if "llm_output_file_id," in unified_file_id:
                                    provider_file_id = unified_file_id.split("llm_output_file_id,")[1].split(";")[0]
                            if not provider_file_id and is_model_embedded_id(file_id):
                                provider_file_id = get_original_file_id(file_id)
                            file_object_file_field["file_id"] = provider_file_id or file_id
                        if format:
                            file_object_file_field["format"] = format
    return messages


def update_responses_input_with_model_file_ids(
    input: object,
    model_id: str | None = None,
    model_file_id_mapping: Mapping[str, Mapping[str, str]] | None = None,
) -> object:
    """
    Updates responses API input with provider-specific file IDs.
    File IDs are always inside the content array, not as direct input_file items.

    For managed files (unified file IDs), uses model_file_id_mapping if provided,
    otherwise decodes the base64-encoded unified file ID and extracts the llm_output_file_id directly.

    Args:
        input: The responses API input parameter
        model_id: The model ID to use for looking up provider-specific file IDs
        model_file_id_mapping: Dictionary mapping litellm file IDs to provider file IDs
                               Format: {"litellm_file_id": {"model_id": "provider_file_id"}}
    """
    from litellm.proxy.openai_files_endpoints.common_utils import (
        convert_b64_uid_to_unified_uid,
        get_original_file_id,
        is_base64_encoded_unified_file_id,
        is_model_embedded_id,
    )

    if isinstance(input, str):
        return input

    if not isinstance(input, list):
        return input

    updated_input: Final = []
    for item in input:
        if not isinstance(item, dict):
            updated_input.append(item)
            continue

        updated_item = item.copy()
        content = item.get("content")
        if isinstance(content, list):
            updated_content = []
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "input_file":
                    file_id = content_item.get("file_id")
                    if isinstance(file_id, str) and file_id:
                        provider_file_id = file_id  # Default to original

                        # Check if we have a mapping for this file ID
                        if model_file_id_mapping and model_id and file_id in model_file_id_mapping:
                            # Use the model-specific file ID from mapping
                            provider_file_id = model_file_id_mapping.get(file_id, {}).get(model_id) or file_id
                            updated_content_item = content_item.copy()
                            updated_content_item["file_id"] = provider_file_id
                            updated_content.append(updated_content_item)
                        else:
                            # Check if this is a base64-encoded unified file ID without mapping
                            is_unified_file_id = is_base64_encoded_unified_file_id(file_id)
                            if is_unified_file_id:
                                # Fallback: decode unified file ID
                                unified_file_id = convert_b64_uid_to_unified_uid(file_id)
                                if "llm_output_file_id," in unified_file_id:
                                    provider_file_id = unified_file_id.split("llm_output_file_id,")[1].split(";")[0]

                                updated_content_item = content_item.copy()
                                updated_content_item["file_id"] = provider_file_id
                                updated_content.append(updated_content_item)
                            elif is_model_embedded_id(file_id):
                                updated_content_item = content_item.copy()
                                updated_content_item["file_id"] = get_original_file_id(file_id)
                                updated_content.append(updated_content_item)
                            else:
                                # Not a managed file, keep as-is
                                updated_content.append(content_item)
                    else:
                        updated_content.append(content_item)
                else:
                    updated_content.append(content_item)
            updated_item["content"] = updated_content

        updated_input.append(updated_item)

    return updated_input


def _decode_vector_store_ids_in_tools(
    tools: list[dict[str, object]] | None,
) -> list[dict[str, object]] | None:
    """
    Decodes unified (LiteLLM-managed) vector_store_ids in file_search tools to
    provider-native IDs.  Non-unified IDs are passed through unchanged.

    This runs unconditionally — no file-ID mapping is required.
    """
    if not tools or not isinstance(tools, list):
        return tools

    from litellm.llms.base_llm.managed_resources.utils import (
        is_base64_encoded_unified_id,
        parse_unified_id,
    )

    updated_tools: Final = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "file_search":
            updated_tools.append(tool)
            continue

        vector_store_ids = tool.get("vector_store_ids")
        if not isinstance(vector_store_ids, list):
            updated_tools.append(tool)
            continue

        decoded_ids = []
        for vs_id in vector_store_ids:
            if not isinstance(vs_id, str) or not is_base64_encoded_unified_id(vs_id):
                decoded_ids.append(vs_id)
                continue

            parsed = parse_unified_id(vs_id)
            provider_resource_id = parsed.get("provider_resource_id") if parsed else None

            if not provider_resource_id:
                verbose_logger.warning(
                    "file_search tool contains unified vector_store_id '%s' that could "
                    "not be decoded to a provider resource ID — passing original ID. "
                    "Ensure the vector store was created via LiteLLM.",
                    vs_id,
                )
                decoded_ids.append(vs_id)
            else:
                decoded_ids.append(provider_resource_id)

        updated_tools.append({**tool, "vector_store_ids": decoded_ids})

    return updated_tools


def update_responses_tools_with_model_file_ids(
    tools: list[dict[str, object]] | None,
    model_id: str | None = None,
    model_file_id_mapping: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, object]] | None:
    """
    Updates responses API tools with provider-specific file IDs.

    Pass 1 (always): decode unified vector_store_ids in file_search tools.
    Pass 2 (needs mapping): map code_interpreter container file_ids to provider IDs.

    Args:
        tools: The responses API tools parameter
        model_id: The model ID to use for looking up provider-specific file IDs
        model_file_id_mapping: Dictionary mapping litellm file IDs to provider file IDs
                               Format: {"litellm_file_id": {"model_id": "provider_file_id"}}
    """
    if not tools or not isinstance(tools, list):
        return tools

    # Pass 1: decode unified vector_store_ids (no mapping needed)
    tools = _decode_vector_store_ids_in_tools(tools) or tools

    # Pass 2: map code_interpreter file IDs (requires mapping)
    if not model_file_id_mapping or not model_id:
        return tools

    updated_tools: Final = []
    for tool in tools:
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue

        updated_tool = tool.copy()

        # Handle code_interpreter with container file_ids
        if tool.get("type") == "code_interpreter":
            container = tool.get("container")
            if isinstance(container, dict):
                container_file_ids = container.get("file_ids")
                if isinstance(container_file_ids, list):
                    updated_file_ids = []
                    for file_id in container_file_ids:
                        if isinstance(file_id, str):
                            # Check if we have a mapping for this file ID
                            if file_id in model_file_id_mapping:
                                # Map to provider-specific file ID
                                provider_file_id = model_file_id_mapping.get(file_id, {}).get(model_id) or file_id
                                updated_file_ids.append(provider_file_id)
                            else:
                                updated_file_ids.append(file_id)
                        else:
                            updated_file_ids.append(file_id)

                    # Update the tool with new file IDs
                    updated_container = container.copy()
                    updated_container["file_ids"] = updated_file_ids
                    updated_tool["container"] = updated_container

        updated_tools.append(updated_tool)

    return updated_tools


def extract_file_metadata(file_data: FileTypes) -> tuple[str | None, str | None]:
    """
    Resolve (filename, content_type) without reading the file body.

    Mirrors extract_file_data's metadata resolution but never calls .read(), so
    it stays O(1) on large uploads. Use this when only metadata is needed (batch
    detection, GCS object naming) and the body must remain a streamable Path/handle.
    """
    filename: str | None = None
    content_type: str | None = None
    file_content: object = None

    if isinstance(file_data, tuple):
        if len(file_data) == 2:
            filename, file_content = file_data
        elif len(file_data) == 3:
            filename, file_content, content_type = file_data
        elif len(file_data) == 4:
            filename, file_content, content_type, _ = file_data
    elif isinstance(file_data, InMemoryFile):
        filename = file_data.name
        content_type = file_data.content_type
    else:
        file_content = file_data

    if filename is None:
        if isinstance(file_content, PathLike):
            filename = Path(file_content).name
        elif isinstance(file_content, io.IOBase):
            name_attr: Final = getattr(file_content, "name", None)
            if isinstance(name_attr, str):
                filename = Path(name_attr).name

    if not content_type:
        guessed: Final = mimetypes.guess_type(filename)[0] if filename else None
        content_type = guessed or "application/octet-stream"

    return filename, content_type


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
    elif isinstance(file_data, InMemoryFile):
        filename = file_data.name
        file_content = file_data
        content_type = file_data.content_type
    else:
        file_content = file_data
    # Convert content to bytes
    if isinstance(file_content, str):
        # Bare string inputs are rejected: when this helper runs in a proxy
        # request handler the string came from an attacker-controlled form
        # field, and opening it as a path is an arbitrary file read on the
        # proxy host. SDK callers who want to upload from a path should
        # either pass a pathlib.Path (a PathLike instance — see the branch
        # below) or open the file themselves and pass the handle / bytes.
        raise ValueError(
            "extract_file_data does not accept bare str inputs. Pass bytes, "
            "an open file handle, a (filename, content) tuple, or a "
            "pathlib.Path. To upload a local file from a path, call "
            "open(path, 'rb') yourself."
        )
    if isinstance(file_content, PathLike):
        # PathLike (pathlib.Path) is a Python-level type that HTTP form
        # values can't fabricate. Treat as a local file path for SDK
        # convenience.
        if filename is None:
            filename = Path(file_content).name
        with open(file_content, "rb") as f:
            content = f.read()
    elif isinstance(file_content, io.IOBase):
        # If it's a file-like object
        # Try to get filename from file handle if not already set
        if not filename and hasattr(file_content, "name"):
            filename = Path(file_content.name).name

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
        if filename:
            guessed_type: Final = mimetypes.guess_type(filename)[0]
            content_type = guessed_type if guessed_type else "application/octet-stream"
        else:
            content_type = "application/octet-stream"

    return ExtractedFileData(
        filename=filename,
        content=content,
        content_type=content_type,
        headers=file_headers,
    )


def get_image_mime_type_from_url(url: str) -> str | None:
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
    from urllib.parse import urlparse

    url = url.lower()

    # Parse URL to extract path without query parameters
    # This handles URLs like: https://example.com/image.jpg?signature=...
    parsed: Final = urlparse(url)
    path: Final = parsed.path

    # Map file extensions to mime types
    mime_types: Final = {
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
        if any(path.endswith(ext) for ext in extensions):
            return mime_type

    return None


_get_image_mime_type_from_url = get_image_mime_type_from_url


def infer_content_type_from_url_and_content(
    url: str,
    content: bytes,
    current_content_type: str | None = None,
) -> str:
    """
    Infer content type from URL extension and binary content when content-type header is missing or generic.

    This helper implements a fallback strategy for determining MIME types when HTTP headers
    are missing or provide generic values (like binary/octet-stream). It's commonly used
    when processing images and documents from various sources (S3, URLs, etc.).

    Fallback Strategy:
    1. If current_content_type is valid (not None and not generic octet-stream), return it
    2. Try to infer from URL extension (handles query parameters)
    3. Try to detect from binary content signature (magic bytes)
    4. Raise ValueError if all methods fail

    Args:
        url: The URL of the content (used to extract file extension)
        content: The binary content (first ~100 bytes are sufficient for detection)
        current_content_type: The current content-type from headers (may be None or generic)

    Returns:
        str: The inferred MIME type (e.g., "image/png", "application/pdf")

    Raises:
        ValueError: If content type cannot be determined by any method

    Example:
        >>> content_type = infer_content_type_from_url_and_content(
        ...     url="https://s3.amazonaws.com/bucket/image.png?AWSAccessKeyId=123",
        ...     content=png_binary_data,
        ...     current_content_type="binary/octet-stream"
        ... )
        >>> print(content_type)
        "image/png"
    """
    from litellm.litellm_core_utils.token_counter import get_image_type

    media_type: Final = current_content_type.split(";", 1)[0].strip().lower() if current_content_type else None

    # If we have a valid content type that's not generic, use it
    if media_type and media_type not in [
        "binary/octet-stream",
        "application/octet-stream",
    ]:
        return media_type

    # Extension to MIME type mapping
    # Supports images, documents, and other common file types
    extension_to_mime: Final = {
        # Image formats
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        # Document formats
        "pdf": "application/pdf",
        "csv": "text/csv",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "html": "text/html",
        "txt": "text/plain",
        "md": "text/markdown",
    }

    # Try to infer from URL extension
    if url:
        extension: Final = url.split(".")[-1].lower().split("?")[0]  # Remove query params
        inferred_type: Final = extension_to_mime.get(extension)
        if inferred_type:
            return inferred_type

    # Try to detect from binary content signature (magic bytes)
    if content:
        detected_type: Final = get_image_type(content[:100])
        if detected_type:
            type_to_mime: Final = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "heic": "image/heic",
            }
            if detected_type in type_to_mime:
                return type_to_mime[detected_type]

    # If all fallbacks failed, raise error
    raise ValueError(f"Unable to determine content type from URL: {url}. Response content-type: {current_content_type}")


def get_file_ids_from_messages(messages: list[AllMessageValues]) -> list[str]:
    """
    Gets file ids from messages
    """
    file_ids: Final = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if content:
                if isinstance(content, str):
                    continue
                for c in content:
                    if isinstance(c, dict) and c["type"] == "file":
                        file_object = cast(ChatCompletionFileObject, c)
                        file_object_file_field = file_object.get("file")
                        if not isinstance(file_object_file_field, dict):
                            # Content block has `type: "file"` but not the
                            # OpenAI Chat Completions shape. No file_id to
                            # extract, so skip instead of raising KeyError.
                            continue
                        file_id = file_object_file_field.get("file_id")
                        if file_id:
                            file_ids.append(file_id)
    return file_ids

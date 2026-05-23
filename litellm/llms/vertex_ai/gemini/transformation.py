"""
Transformation logic from OpenAI format to Gemini format.

Why separate file? Make it easy to see how transformation works
"""

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_generic_image_chunk_to_openai_image_obj,
    convert_to_anthropic_image_obj,
    convert_to_gemini_tool_call_invoke,
    convert_to_gemini_tool_call_result,
    response_schema_prompt,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import pop_vertex_request_labels
from litellm.types.files import (
    get_file_mime_type_for_file_type,
    get_file_type_from_extension,
    is_gemini_1_5_accepted_file_type,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAudioObject,
    ChatCompletionFileObject,
    ChatCompletionImageObject,
    ChatCompletionTextObject,
    ChatCompletionUserMessage,
)
from litellm.types.llms.vertex_ai import *
from litellm.types.llms.vertex_ai import (
    GenerationConfig,
    PartType,
    RequestBody,
    SafetSettingsConfig,
    SystemInstructions,
    ToolConfig,
    Tools,
)
from litellm.types.utils import GenericImageParsingChunk, LlmProviders

from ..common_utils import (
    _check_text_in_content,
    get_supports_response_schema,
    get_supports_system_message,
)

# Typed as Any to avoid introducing a module-load-time cyclic import to
# vertex_llm_base. The instance is lazily constructed by _get_vertex_base()
# the first time GCS metadata needs to be fetched.
_GCS_METADATA_VERTEX_BASE: Optional[Any] = None
# Shared sync client for GCS JSON API metadata reads so proxy/SSL settings
# from litellm's HTTP stack apply (see Greptile review on PR #27278).
_GCS_METADATA_HTTP_HANDLER: Optional[HTTPHandler] = None
_GEMINI_MIME_TYPE_ALIASES: Dict[str, str] = {
    "image/jpg": "image/jpeg",
}


def _apply_gemini_mime_type_aliases(mime_type: str) -> str:
    """Normalize known MIME aliases only; does not consult the file-type registry.

    Also strips MIME parameters (e.g. ``; charset=utf-8``) so that values
    sourced from GCS object metadata (``contentType``) validate correctly.
    """
    normalized = mime_type.split(";", 1)[0].strip().lower()
    return _GEMINI_MIME_TYPE_ALIASES.get(normalized, normalized)


def _get_vertex_base() -> Any:
    """Lazily return the shared VertexBase instance to avoid a module-load-time cyclic import."""
    global _GCS_METADATA_VERTEX_BASE
    if _GCS_METADATA_VERTEX_BASE is None:
        from ..vertex_llm_base import VertexBase

        _GCS_METADATA_VERTEX_BASE = VertexBase()
    return _GCS_METADATA_VERTEX_BASE


def _get_gcs_metadata_http_handler() -> HTTPHandler:
    global _GCS_METADATA_HTTP_HANDLER
    if _GCS_METADATA_HTTP_HANDLER is None:
        _GCS_METADATA_HTTP_HANDLER = HTTPHandler(timeout=5.0)
    return _GCS_METADATA_HTTP_HANDLER


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _convert_detail_to_media_resolution_enum(
    detail: Optional[str],
) -> Optional[Dict[str, str]]:
    if detail == "low":
        return {"level": "MEDIA_RESOLUTION_LOW"}
    elif detail == "medium":
        return {"level": "MEDIA_RESOLUTION_MEDIUM"}
    elif detail == "high":
        return {"level": "MEDIA_RESOLUTION_HIGH"}
    elif detail == "ultra_high":
        return {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}
    return None


def _get_highest_media_resolution(
    current: Optional[str], new_detail: Optional[str]
) -> Optional[str]:
    """
    Compare two media resolution values and return the highest one.
    Resolution hierarchy: ultra_high > high > medium > low > None
    """
    resolution_priority = {"ultra_high": 4, "high": 3, "medium": 2, "low": 1}
    current_priority = resolution_priority.get(current, 0) if current else 0
    new_priority = resolution_priority.get(new_detail, 0) if new_detail else 0

    if new_priority > current_priority:
        return new_detail
    return current


def _extract_max_media_resolution_from_messages(
    messages: List[AllMessageValues],
) -> Optional[str]:
    """
    Extract the highest media resolution (detail) from image content in messages.

    This is used to set the global media_resolution in generation_config for
    Gemini 2.x models which don't support per-part media resolution.

    Args:
        messages: List of messages in OpenAI format

    Returns:
        The highest detail level found ("high", "low", or None)
    """
    max_resolution: Optional[str] = None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                detail: Optional[str] = None
                if item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        detail = image_url.get("detail")
                elif item.get("type") == "file":
                    file_obj = item.get("file")
                    if isinstance(file_obj, dict):
                        detail = file_obj.get("detail")
                if detail:
                    max_resolution = _get_highest_media_resolution(
                        max_resolution, detail
                    )
    return max_resolution


def _apply_gemini_metadata(
    part: PartType,
    model: Optional[str],
    media_resolution_enum: Optional[Dict[str, str]],
    video_metadata: Optional[Dict[str, Any]],
) -> PartType:
    """
    Apply media_resolution and video_metadata parameters to a Gemini part.

    - Per-part media_resolution: Gemini 3+ only (2.x uses generation_config global).
    - video_metadata (fps, startOffset, endOffset): all Gemini models (1.x, 2.x, 3+).
    """
    if model is None:
        return part

    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    part_dict = dict(part)

    if media_resolution_enum is not None and VertexGeminiConfig._is_gemini_3_or_newer(
        model
    ):
        part_dict["media_resolution"] = media_resolution_enum

    if video_metadata is not None:
        gemini_video_metadata = {}
        if "fps" in video_metadata:
            gemini_video_metadata["fps"] = video_metadata["fps"]
        if "start_offset" in video_metadata:
            gemini_video_metadata["startOffset"] = video_metadata["start_offset"]
        if "end_offset" in video_metadata:
            gemini_video_metadata["endOffset"] = video_metadata["end_offset"]
        if gemini_video_metadata:
            part_dict["video_metadata"] = gemini_video_metadata

    return cast(PartType, part_dict)


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid gs URI: {gs_uri}")
    uri_without_scheme = gs_uri[5:]  # drop gs://
    uri_parts = uri_without_scheme.split("/", 1)
    if len(uri_parts) != 2 or not uri_parts[0] or not uri_parts[1]:
        raise ValueError(f"Invalid gs URI: {gs_uri}")
    return uri_parts[0], uri_parts[1]


def _is_valid_gcs_bucket_name(bucket: str) -> bool:
    """
    Validate bucket name against core GCS naming constraints.
    """
    bucket_length = len(bucket)
    max_bucket_length = 222 if "." in bucket else 63
    if bucket_length < 3 or bucket_length > max_bucket_length:
        return False
    if "." in bucket and any(
        len(label) == 0 or len(label) > 63 for label in bucket.split(".")
    ):
        return False
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*[a-z0-9]", bucket):
        return False
    if ".." in bucket:
        return False
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", bucket):
        return False
    return True


def _gs_uri_requires_content_type_metadata(url: str) -> bool:
    """
    True when _process_gemini_media would call _get_gcs_object_content_type
    (extension-less gs:// and no explicit format passed into that helper).
    """
    if "gs://" not in url:
        return False
    extension_with_dot = os.path.splitext(url)[-1]
    extension = extension_with_dot[1:] if extension_with_dot else ""
    return len(extension) == 0


def _image_url_payload_may_need_sync_gcs_metadata_fetch(
    raw_image_url: Any,
) -> bool:
    """
    True when this image_url value (content-part image_url or assistant ``images[]``
    entry) can trigger a blocking GCS metadata read for MIME resolution.
    """
    fmt: Optional[str] = None
    url: Optional[str] = None
    if isinstance(raw_image_url, dict):
        url = raw_image_url.get("url")  # type: ignore[assignment]
        if not isinstance(url, str):
            return False
        fmt = (
            raw_image_url.get("format")
            or raw_image_url.get("mime_type")
            or raw_image_url.get("content_type")
        )
    elif isinstance(raw_image_url, str):
        url = raw_image_url
    else:
        return False
    if "gs://" not in url or fmt:
        return False
    return _gs_uri_requires_content_type_metadata(url)


def _openai_messages_may_need_sync_gcs_metadata_fetch(
    messages: List[AllMessageValues],
) -> bool:
    """
    Heuristic: True if any message part can trigger a blocking GCS JSON
    metadata read inside _transform_request_body (extension-less gs:// without
    explicit MIME hints). Covers user/system ``content`` parts and assistant
    ``images`` (same paths as ``_gemini_convert_messages_with_history``). Used
    to decide whether ``async_transform_request_body`` should offload the sync
    transform via ``asyncify``.
    """
    for raw in messages:
        msg: Any = raw
        if not isinstance(msg, dict) and hasattr(msg, "model_dump"):
            msg = msg.model_dump(exclude_none=False)
        if not isinstance(msg, dict):
            continue
        images_field = msg.get("images")
        if isinstance(images_field, list):
            for image_item in images_field:
                if not isinstance(image_item, dict):
                    continue
                if _image_url_payload_may_need_sync_gcs_metadata_fetch(
                    image_item.get("image_url")
                ):
                    return True

        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "image_url":
                if _image_url_payload_may_need_sync_gcs_metadata_fetch(
                    item.get("image_url")
                ):
                    return True
            elif itype == "file":
                file_obj = item.get("file")
                if not isinstance(file_obj, dict):
                    continue
                fmt = (
                    file_obj.get("format")
                    or file_obj.get("mime_type")
                    or file_obj.get("content_type")
                )
                passed = file_obj.get("file_id") or file_obj.get("file_data")
                if (
                    isinstance(passed, str)
                    and "gs://" in passed
                    and not fmt
                    and _gs_uri_requires_content_type_metadata(passed)
                ):
                    return True
    return False


def _get_gcs_object_content_type(
    image_url: str,
    vertex_project: Optional[str] = None,
    vertex_credentials: Optional[Any] = None,
) -> Optional[str]:
    """
    Resolve content type from GCS object metadata.

    Only attaches a Bearer token when the caller explicitly supplies Vertex
    credentials, to avoid using the server's default Google credentials on
    the Gemini API-key (Google AI Studio) path and being used as an oracle
    for private GCS object metadata. Without explicit credentials we only
    issue an anonymous request, which only succeeds for publicly-readable
    objects.
    """
    try:
        bucket, object_name = _parse_gs_uri(image_url)
    except ValueError:
        return None
    if not _is_valid_gcs_bucket_name(bucket):
        return None

    headers: Dict[str, str] = {}
    explicit_vertex_auth_provided = (
        vertex_project is not None or vertex_credentials is not None
    )
    if explicit_vertex_auth_provided:
        try:
            access_token, _ = _get_vertex_base().get_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
            )
            headers["Authorization"] = f"Bearer {access_token}"
        except Exception as e:
            raise litellm.BadRequestError(
                message=(
                    "Unable to fetch GCS metadata with provided Vertex credentials/project. "
                    f"Original error: {str(e)}"
                ),
                model=None,
                llm_provider="vertex_ai",
            )

    # Build the URL via httpx.URL with a fixed scheme/host and URL-encode both
    # bucket and object so CodeQL does not flag the interpolation as a
    # potential SSRF that could resolve to an arbitrary host.
    encoded_bucket = quote(bucket, safe="")
    encoded_object = quote(object_name, safe="")
    metadata_url = httpx.URL(
        scheme="https",
        host="storage.googleapis.com",
        path=f"/storage/v1/b/{encoded_bucket}/o/{encoded_object}",
        params={"fields": "contentType"},
    )
    try:
        response = _get_gcs_metadata_http_handler().get(
            url=str(metadata_url),
            headers=headers or None,
        )
    except httpx.RequestError as e:
        if explicit_vertex_auth_provided:
            raise litellm.BadRequestError(
                message=(
                    "Unable to reach GCS JSON API for object metadata with provided "
                    f"Vertex credentials. {type(e).__name__}: {e}"
                ),
                model=None,
                llm_provider="vertex_ai",
            ) from e
        return None

    if response.is_error:
        if explicit_vertex_auth_provided:
            preview = (response.text or "")[:1024]
            raise litellm.BadRequestError(
                message=(
                    "Unable to read GCS object metadata with provided Vertex credentials. "
                    f"HTTP {response.status_code}. Response body (truncated): {preview!r}"
                ),
                model=None,
                llm_provider="vertex_ai",
            )
        return None

    try:
        payload = response.json()
    except ValueError as e:
        if explicit_vertex_auth_provided:
            raise litellm.BadRequestError(
                message=(
                    "GCS metadata response was not valid JSON when using provided "
                    f"Vertex credentials (HTTP {response.status_code}). Error: {e}"
                ),
                model=None,
                llm_provider="vertex_ai",
            ) from e
        return None

    if not isinstance(payload, dict):
        if explicit_vertex_auth_provided:
            raise litellm.BadRequestError(
                message=(
                    "GCS metadata response was not a JSON object when using provided "
                    f"Vertex credentials (HTTP {response.status_code})."
                ),
                model=None,
                llm_provider="vertex_ai",
            )
        return None

    content_type = payload.get("contentType")
    if isinstance(content_type, str) and len(content_type) > 0:
        return content_type

    if explicit_vertex_auth_provided:
        preview = (response.text or "")[:1024]
        raise litellm.BadRequestError(
            message=(
                "GCS metadata JSON did not include a non-empty contentType field when "
                f"using provided Vertex credentials (HTTP {response.status_code}). "
                f"Body (truncated): {preview!r}"
            ),
            model=None,
            llm_provider="vertex_ai",
        )
    return None


def _normalize_and_validate_gemini_mime_type(
    mime_type: str, model: Optional[str]
) -> str:
    # Import lazily to avoid a module-level cyclic-import alert with
    # litellm.types.files.
    from litellm.types.files import get_file_extension_from_mime_type

    normalized_mime_type = _apply_gemini_mime_type_aliases(mime_type)
    try:
        file_extension = get_file_extension_from_mime_type(normalized_mime_type)
        file_type = get_file_type_from_extension(file_extension)
    except ValueError:
        raise litellm.BadRequestError(
            message=f"File type not supported by gemini - {normalized_mime_type}",
            model=model,
            llm_provider="vertex_ai",
        )

    if not is_gemini_1_5_accepted_file_type(file_type):
        raise litellm.BadRequestError(
            message=f"File type not supported by gemini - {file_type}",
            model=model,
            llm_provider="vertex_ai",
        )

    return get_file_mime_type_for_file_type(file_type)


def _process_gemini_media(
    image_url: str,
    format: Optional[str] = None,
    media_resolution_enum: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
    vertex_project: Optional[str] = None,
    vertex_credentials: Optional[Any] = None,
) -> PartType:
    """
    Given a media URL (image, audio, or video), return the appropriate PartType for Gemini
    By the way, actually video_metadata can only be used with videos; it cannot be used with images, audio, or files. However, I haven't made any special handling because vertex returns a parameter error.

    Args:
        image_url: The URL or base64 string of the media (image, audio, or video)
        format: The MIME type of the media
        media_resolution_enum: Media resolution level (for Gemini 3+)
        model: The model name (to check version compatibility)
        video_metadata: Video-specific metadata (fps, start_offset, end_offset)
    """

    try:
        # GCS URIs
        if "gs://" in image_url:
            extension_with_dot = os.path.splitext(image_url)[-1]  # Ex: ".png"
            extension = extension_with_dot[1:]  # Ex: "png"

            explicit_gcs_format = False
            if not format:
                mime_type: Optional[str] = None
                # For extension-less gs:// URIs, we cannot infer from path.
                # If callers pass `format`/`mime_type`, this branch is skipped.
                if extension:
                    file_type = get_file_type_from_extension(extension)

                    # Validate the file type is supported by Gemini
                    if not is_gemini_1_5_accepted_file_type(file_type):
                        raise litellm.BadRequestError(
                            message=f"File type not supported by gemini - {file_type}",
                            model=model,
                            llm_provider="vertex_ai",
                        )

                    mime_type = get_file_mime_type_for_file_type(file_type)
                else:
                    mime_type = _get_gcs_object_content_type(
                        image_url=image_url,
                        vertex_project=vertex_project,
                        vertex_credentials=vertex_credentials,
                    )
                    if mime_type is None:
                        raise litellm.BadRequestError(
                            message=(
                                f"Unable to determine mime type for gs URI: {image_url}. "
                                "This gs:// URI has no file extension and GCS metadata "
                                "lookup failed. Set it explicitly using image_url.format "
                                "(or image_url.mime_type/content_type) or "
                                "message.content[].file.format."
                            ),
                            model=model,
                            llm_provider="vertex_ai",
                        )
            else:
                mime_type = format
                explicit_gcs_format = True
            if mime_type is None:
                raise litellm.BadRequestError(
                    message=f"File type not supported by gemini - {image_url}",
                    model=model,
                    llm_provider="vertex_ai",
                )
            if explicit_gcs_format:
                # Callers who pass format/mime_type explicitly for gs:// URIs
                # rely on pass-through to Gemini (pre-PR behavior). Only apply
                # known MIME aliases; skip litellm's file-type registry.
                mime_type = _apply_gemini_mime_type_aliases(mime_type)
            else:
                mime_type = _normalize_and_validate_gemini_mime_type(
                    mime_type=mime_type,
                    model=model,
                )
            file_data = FileDataType(mime_type=mime_type, file_uri=image_url)
            part: PartType = {"file_data": file_data}
            return _apply_gemini_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        elif image_url.startswith(
            "https://generativelanguage.googleapis.com/v1beta/files/"
        ):
            # Gemini Files API URIs — the file is already uploaded to Google's
            # servers; pass the URI through as file_data without fetching it.
            # These URLs return 403 when accessed directly, so we must not try
            # to resolve their MIME type via HTTP.
            if format:
                file_data = FileDataType(mime_type=format, file_uri=image_url)
            else:
                # Gemini Files API references can be passed through as URI-only.
                file_data = cast(FileDataType, {"file_uri": image_url})
            part = {"file_data": file_data}
            return _apply_gemini_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        elif (
            "https://" in image_url
            and (image_type := format or _get_image_mime_type_from_url(image_url))
            is not None
        ):
            file_data = FileDataType(mime_type=image_type, file_uri=image_url)
            part = {"file_data": file_data}
            return _apply_gemini_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        elif "http://" in image_url or "https://" in image_url or "base64" in image_url:
            image = convert_to_anthropic_image_obj(image_url, format=format)
            _blob: BlobType = {"data": image["data"], "mime_type": image["media_type"]}
            part = {"inline_data": cast(BlobType, _blob)}
            return _apply_gemini_metadata(
                part, model, media_resolution_enum, video_metadata
            )
        raise Exception("Invalid image received - {}".format(image_url))
    except Exception as e:
        raise e


def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split("_")
    return components[0] + "".join(x.capitalize() for x in components[1:])


def _camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


def _get_equivalent_key(key: str, available_keys: set) -> Optional[str]:
    """
    Get the equivalent key from available keys, checking both camelCase and snake_case variants
    """
    if key in available_keys:
        return key

    # Try camelCase version
    camel_key = _snake_to_camel(key)
    if camel_key in available_keys:
        return camel_key

    # Try snake_case version
    snake_key = _camel_to_snake(key)
    if snake_key in available_keys:
        return snake_key

    return None


def check_if_part_exists_in_parts(
    parts: List[PartType], part: PartType, excluded_keys: List[str] = []
) -> bool:
    """
    Check if a part exists in a list of parts
    Handles both camelCase and snake_case key variations (e.g., function_call vs functionCall)
    """
    keys_to_compare = set(part.keys()) - set(excluded_keys)
    for p in parts:
        p_keys = set(p.keys())
        # Check if all keys in part have equivalent values in p
        match_found = True
        for key in keys_to_compare:
            equivalent_key = _get_equivalent_key(key, p_keys)
            if equivalent_key is None or p.get(equivalent_key, None) != part.get(
                key, None
            ):
                match_found = False
                break

        if match_found:
            return True
    return False


def _gemini_convert_messages_with_history(  # noqa: PLR0915
    messages: List[AllMessageValues],
    model: Optional[str] = None,
    litellm_params: Optional[dict] = None,
    custom_llm_provider: Optional[str] = None,
) -> List[ContentType]:
    """
    Converts given messages from OpenAI format to Gemini format

    - Parts must be iterable
    - Roles must alternate b/w 'user' and 'model' (same as anthropic -> merge consecutive roles)
    - Please ensure that function response turn comes immediately after a function call turn
    """
    user_message_types = {"user", "system"}
    contents: List[ContentType] = []

    last_message_with_tool_calls = None

    msg_i = 0
    tool_call_responses = []
    vertex_project = None
    vertex_credentials = None
    if litellm_params:
        vertex_project = litellm_params.get("vertex_project") or litellm_params.get(
            "vertex_ai_project"
        )
        vertex_credentials = litellm_params.get(
            "vertex_credentials"
        ) or litellm_params.get("vertex_ai_credentials")

    try:
        while msg_i < len(messages):
            user_content: List[PartType] = []
            init_msg_i = msg_i
            ## MERGE CONSECUTIVE USER CONTENT ##
            while (
                msg_i < len(messages) and messages[msg_i]["role"] in user_message_types
            ):
                _message_content = messages[msg_i].get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    _parts: List[PartType] = []
                    for element_idx, element in enumerate(_message_content):
                        if (
                            element["type"] == "text"
                            and "text" in element
                            and len(element["text"]) > 0
                        ):
                            element = cast(ChatCompletionTextObject, element)
                            _part = PartType(text=element["text"])
                            _parts.append(_part)
                        elif element["type"] == "image_url":
                            element = cast(ChatCompletionImageObject, element)
                            img_element = element
                            format: Optional[str] = None
                            media_resolution_enum: Optional[Dict[str, str]] = None
                            raw_image_url = img_element.get("image_url")
                            if raw_image_url is None:
                                raise litellm.BadRequestError(
                                    message="Invalid message content: element type is 'image_url' but 'image_url' field is missing ",
                                    model=model,
                                    llm_provider="vertex_ai",
                                )
                            if isinstance(raw_image_url, dict):
                                image_url = raw_image_url.get("url")
                                if image_url is None:
                                    raise litellm.BadRequestError(
                                        message="Invalid message content: element type is 'image_url' but 'url' field is missing inside 'image_url' ",
                                        model=model,
                                        llm_provider="vertex_ai",
                                    )
                                # TypedDict does not declare mime_type/content_type;
                                # read via Dict[str, Any] for caller-provided MIME fields.
                                image_url_dict = cast(Dict[str, Any], raw_image_url)
                                format = (
                                    image_url_dict.get("format")
                                    or image_url_dict.get("mime_type")
                                    or image_url_dict.get("content_type")
                                )
                                detail = image_url_dict.get("detail")
                                media_resolution_enum = (
                                    _convert_detail_to_media_resolution_enum(detail)
                                )
                            else:
                                image_url = raw_image_url
                            _part = _process_gemini_media(
                                image_url=image_url,
                                format=format,
                                media_resolution_enum=media_resolution_enum,
                                model=model,
                                vertex_project=vertex_project,
                                vertex_credentials=vertex_credentials,
                            )
                            _parts.append(_part)
                        elif element["type"] == "input_audio":
                            audio_element = cast(ChatCompletionAudioObject, element)
                            audio_data = audio_element["input_audio"].get("data")
                            audio_format = audio_element["input_audio"].get("format")
                            if audio_data is not None and audio_format is not None:
                                audio_format_modified = (
                                    "audio/" + audio_format
                                    if audio_format.startswith("audio/") is False
                                    else audio_format
                                )  # Gemini expects audio/wav, audio/mp3, etc.
                                openai_image_str = (
                                    convert_generic_image_chunk_to_openai_image_obj(
                                        image_chunk=GenericImageParsingChunk(
                                            type="base64",
                                            media_type=audio_format_modified,
                                            data=audio_data,
                                        )
                                    )
                                )
                                _part = _process_gemini_media(
                                    image_url=openai_image_str,
                                    format=audio_format_modified,
                                    model=model,
                                    vertex_project=vertex_project,
                                    vertex_credentials=vertex_credentials,
                                )
                                _parts.append(_part)
                        elif element["type"] == "file":
                            file_element = cast(ChatCompletionFileObject, element)
                            _file_field = file_element.get("file")
                            if _file_field is None:
                                raise litellm.BadRequestError(
                                    message="Content block has type='file' but is missing the required 'file' field",
                                    model=model,
                                    llm_provider="vertex_ai",
                                )
                            # TypedDict does not declare mime_type/content_type;
                            # read via Dict[str, Any] for caller-provided MIME fields.
                            file_dict = cast(Dict[str, Any], _file_field)
                            file_id = file_dict.get("file_id")
                            format = (
                                file_dict.get("format")
                                or file_dict.get("mime_type")
                                or file_dict.get("content_type")
                            )
                            file_data = file_dict.get("file_data")
                            detail = file_dict.get("detail")
                            video_metadata = file_dict.get("video_metadata")
                            passed_file = file_id or file_data
                            if passed_file is None:
                                raise Exception(
                                    "Unknown file type. Please pass in a file_id or file_data"
                                )

                            # Convert detail to media_resolution_enum
                            media_resolution_enum = (
                                _convert_detail_to_media_resolution_enum(detail)
                            )

                            try:
                                _part = _process_gemini_media(
                                    image_url=passed_file,
                                    format=format,
                                    model=model,
                                    media_resolution_enum=media_resolution_enum,
                                    video_metadata=video_metadata,
                                    vertex_project=vertex_project,
                                    vertex_credentials=vertex_credentials,
                                )
                                _parts.append(_part)
                            except litellm.BadRequestError:
                                raise
                            except Exception as e:
                                raise litellm.BadRequestError(
                                    message=(
                                        f"Unable to determine mime type for file: "
                                        f"{file_id or 'provided data'}, set this explicitly "
                                        f"using message[{msg_i}].content[{element_idx}].file.format "
                                        f"(or file.mime_type/content_type). "
                                        f"Original error: {str(e)}"
                                    ),
                                    model=model,
                                    llm_provider="vertex_ai",
                                )
                    user_content.extend(_parts)
                elif _message_content is not None and isinstance(_message_content, str):
                    _part = PartType(text=_message_content)
                    user_content.append(_part)

                msg_i += 1

            if user_content:
                """
                check that user_content has 'text' parameter.
                    - Known Vertex Error: Unable to submit request because it must have a text parameter.
                    - Relevant Issue: https://github.com/BerriAI/litellm/issues/5515
                """
                has_text_in_content = _check_text_in_content(user_content)
                if has_text_in_content is False:
                    verbose_logger.warning(
                        "No text in user content. Adding a blank text to user content, to ensure Gemini doesn't fail the request. Relevant Issue - https://github.com/BerriAI/litellm/issues/5515"
                    )
                    user_content.append(
                        PartType(text=" ")
                    )  # add a blank text, to ensure Gemini doesn't fail the request.
                contents.append(ContentType(role="user", parts=user_content))
            assistant_content = []
            ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
            while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
                if isinstance(messages[msg_i], BaseModel):
                    msg_dict: Union[ChatCompletionAssistantMessage, dict] = messages[msg_i].model_dump()  # type: ignore
                else:
                    msg_dict = messages[msg_i]  # type: ignore
                assistant_msg = ChatCompletionAssistantMessage(**msg_dict)  # type: ignore
                _message_content = assistant_msg.get("content", None)
                reasoning_content = assistant_msg.get("reasoning_content", None)
                thinking_blocks = assistant_msg.get("thinking_blocks")
                if reasoning_content is not None:
                    assistant_content.append(
                        PartType(thought=True, text=reasoning_content)
                    )
                if thinking_blocks is not None:
                    for block in thinking_blocks:
                        if block["type"] == "thinking":
                            block_thinking_str = block.get("thinking")
                            block_signature = block.get("signature")
                            if (
                                block_thinking_str is not None
                                and block_signature is not None
                            ):
                                try:
                                    assistant_content.append(
                                        PartType(
                                            thoughtSignature=block_signature,
                                            **json.loads(block_thinking_str),
                                        )
                                    )
                                except Exception:
                                    assistant_content.append(
                                        PartType(
                                            thoughtSignature=block_signature,
                                            text=block_thinking_str,
                                        )
                                    )
                if _message_content is not None and isinstance(_message_content, list):
                    _parts = []
                    for element in _message_content:
                        if isinstance(element, dict):
                            if element["type"] == "text":
                                _part = PartType(text=element["text"])
                                _parts.append(_part)

                    assistant_content.extend(_parts)
                elif _message_content is not None and isinstance(_message_content, str):
                    assistant_text = _message_content
                    # Check if message has thought_signatures in provider_specific_fields
                    provider_specific_fields = assistant_msg.get(
                        "provider_specific_fields"
                    )
                    thought_signatures = None
                    if provider_specific_fields and isinstance(
                        provider_specific_fields, dict
                    ):
                        thought_signatures = provider_specific_fields.get(
                            "thought_signatures"
                        )

                    # If we have thought signatures, add them to the part
                    if (
                        thought_signatures
                        and isinstance(thought_signatures, list)
                        and len(thought_signatures) > 0
                    ):
                        # Use the first signature for the text part (Gemini expects one signature per part)
                        assistant_content.append(PartType(text=assistant_text, thoughtSignature=thought_signatures[0]))  # type: ignore
                    else:
                        assistant_content.append(PartType(text=assistant_text))  # type: ignore

                ## HANDLE ASSISTANT IMAGES FIELD
                # Process images field if present (for generated images from assistant)
                assistant_images = assistant_msg.get("images")
                if assistant_images is not None and isinstance(assistant_images, list):
                    for image_item in assistant_images:
                        if isinstance(image_item, dict):
                            image_url_obj = image_item.get("image_url")
                            if isinstance(image_url_obj, dict):
                                assistant_image_url = image_url_obj.get("url")
                                format = (
                                    image_url_obj.get("format")
                                    or image_url_obj.get("mime_type")
                                    or image_url_obj.get("content_type")
                                )
                                detail = image_url_obj.get("detail")
                                media_resolution_enum = (
                                    _convert_detail_to_media_resolution_enum(detail)
                                )
                                if assistant_image_url:
                                    _part = _process_gemini_media(
                                        image_url=assistant_image_url,
                                        format=format,
                                        media_resolution_enum=media_resolution_enum,
                                        model=model,
                                        vertex_project=vertex_project,
                                        vertex_credentials=vertex_credentials,
                                    )
                                    assistant_content.append(_part)

                ## HANDLE ASSISTANT FUNCTION CALL
                if (
                    assistant_msg.get("tool_calls", []) is not None
                    or assistant_msg.get("function_call") is not None
                ):  # support assistant tool invoke conversion
                    gemini_tool_call_parts = convert_to_gemini_tool_call_invoke(
                        assistant_msg,
                        model=model,
                        custom_llm_provider=custom_llm_provider,
                    )
                    ## check if gemini_tool_call already exists in assistant_content
                    for gemini_tool_call_part in gemini_tool_call_parts:
                        if not check_if_part_exists_in_parts(
                            assistant_content,
                            gemini_tool_call_part,
                            excluded_keys=["thoughtSignature"],
                        ):
                            assistant_content.append(gemini_tool_call_part)
                    last_message_with_tool_calls = assistant_msg

                ## HANDLE SERVER-SIDE TOOL INVOCATIONS (context circulation)
                _psf = assistant_msg.get("provider_specific_fields")
                if isinstance(_psf, dict):
                    _ss_invocations = _psf.get("server_side_tool_invocations")
                    if isinstance(_ss_invocations, list):
                        for invocation in _ss_invocations:
                            # Re-inject toolCall part
                            tc_part: Dict[str, Any] = {
                                "toolCall": {
                                    "toolType": invocation.get("tool_type"),
                                    "id": invocation.get("id"),
                                    "args": invocation.get("args"),
                                }
                            }
                            if "thought_signature" in invocation:
                                tc_part["thoughtSignature"] = invocation[
                                    "thought_signature"
                                ]
                            assistant_content.append(tc_part)  # type: ignore

                            # Re-inject toolResponse part if response is present
                            if "response" in invocation:
                                tr_dict: Dict[str, Any] = {
                                    "id": invocation.get("id"),
                                    "response": invocation.get("response"),
                                }
                                if invocation.get("tool_type"):
                                    tr_dict["toolType"] = invocation["tool_type"]
                                tr_part: Dict[str, Any] = {"toolResponse": tr_dict}
                                if "thought_signature" in invocation:
                                    tr_part["thoughtSignature"] = invocation[
                                        "thought_signature"
                                    ]
                                assistant_content.append(tr_part)  # type: ignore

                msg_i += 1

            if assistant_content:
                contents.append(ContentType(role="model", parts=assistant_content))

            ## APPEND TOOL CALL MESSAGES ##
            tool_call_message_roles = ["tool", "function"]
            if (
                msg_i < len(messages)
                and messages[msg_i]["role"] in tool_call_message_roles
            ):
                _part = convert_to_gemini_tool_call_result(
                    messages[msg_i],  # type: ignore
                    last_message_with_tool_calls,  # type: ignore
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
                msg_i += 1
                # Handle both single part and list of parts (for Computer Use with images)
                if isinstance(_part, list):
                    tool_call_responses.extend(_part)
                else:
                    tool_call_responses.append(_part)
            if msg_i < len(messages) and (
                messages[msg_i]["role"] not in tool_call_message_roles
            ):
                if len(tool_call_responses) > 0:
                    contents.append(ContentType(role="user", parts=tool_call_responses))
                    tool_call_responses = []

            if msg_i == init_msg_i:  # prevent infinite loops
                raise Exception(
                    "Invalid Message passed in - {}. File an issue https://github.com/BerriAI/litellm/issues".format(
                        messages[msg_i]
                    )
                )
        if len(tool_call_responses) > 0:
            contents.append(ContentType(role="user", parts=tool_call_responses))

        if len(contents) == 0:
            verbose_logger.warning("""
                No contents in messages. Contents are required. See
                https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent#request-body.
                If the original request did not comply to OpenAI API requirements it should have failed by now,
                but LiteLLM does not check for missing messages.
                Setting an empty content to prevent an 400 error.
                Relevant Issue - https://github.com/BerriAI/litellm/issues/9733
                """)
            contents.append(ContentType(role="user", parts=[PartType(text=" ")]))
        return contents
    except Exception as e:
        raise e


# Keys that LiteLLM consumes internally and must never be forwarded to the
_LITELLM_INTERNAL_EXTRA_BODY_KEYS: frozenset = frozenset({"cache", "tags"})


def _pop_and_merge_extra_body(data: RequestBody, optional_params: dict) -> None:
    """Pop extra_body from optional_params and shallow-merge into data, deep-merging dict values."""
    extra_body: Optional[dict] = optional_params.pop("extra_body", None)
    if extra_body is not None:
        data_dict: dict = data  # type: ignore[assignment]
        for k, v in extra_body.items():
            if k in _LITELLM_INTERNAL_EXTRA_BODY_KEYS:
                continue
            if (
                k in data_dict
                and isinstance(data_dict[k], dict)
                and isinstance(v, dict)
            ):
                data_dict[k].update(v)
            else:
                data_dict[k] = v


def _transform_request_body(  # noqa: PLR0915
    messages: List[AllMessageValues],
    model: str,
    optional_params: dict,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    cached_content: Optional[str],
) -> RequestBody:
    """
    Common transformation logic across sync + async Gemini /generateContent calls.
    """
    # Separate system prompt from rest of message
    supports_system_message = get_supports_system_message(
        model=model, custom_llm_provider=custom_llm_provider
    )
    system_instructions, messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )
    # Checks for 'response_schema' support - if passed in
    if "response_schema" in optional_params:
        supports_response_schema = get_supports_response_schema(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if supports_response_schema is False:
            user_response_schema_message = response_schema_prompt(
                model=model, response_schema=optional_params.get("response_schema")  # type: ignore
            )
            messages.append({"role": "user", "content": user_response_schema_message})
            optional_params.pop("response_schema")

    # Check for any 'litellm_param_*' set during optional param mapping

    remove_keys = []
    for k, v in optional_params.items():
        if k.startswith("litellm_param_"):
            litellm_params.update({k: v})
            remove_keys.append(k)

    optional_params = {k: v for k, v in optional_params.items() if k not in remove_keys}

    try:
        if custom_llm_provider == "gemini":
            content = litellm.GoogleAIStudioGeminiConfig()._transform_messages(
                messages=messages, model=model, litellm_params=litellm_params
            )
        else:
            content = litellm.VertexGeminiConfig()._transform_messages(
                messages=messages, model=model, litellm_params=litellm_params
            )
        tools: Optional[Tools] = optional_params.pop("tools", None)
        tool_choice: Optional[ToolConfig] = optional_params.pop("tool_choice", None)
        include_server_side_tool_invocations: bool = optional_params.pop(
            "include_server_side_tool_invocations", False
        )
        safety_settings: Optional[List[SafetSettingsConfig]] = optional_params.pop(
            "safety_settings", None
        )  # type: ignore
        # Drop output_config as it's not supported by Vertex AI
        optional_params.pop("output_config", None)
        config_fields = GenerationConfig.__annotations__.keys()

        # labels: optional explicit param and/or metadata.requester_metadata (OpenAI metadata)
        labels = pop_vertex_request_labels(optional_params, litellm_params)

        filtered_params = {
            k: v
            for k, v in optional_params.items()
            if _get_equivalent_key(k, set(config_fields))
        }

        generation_config: Optional[GenerationConfig] = GenerationConfig(
            **filtered_params
        )

        # For Gemini 2.x models, also add media_resolution to generation_config (global)
        # as a fallback, since some 2.x versions may not support per-part media_resolution.
        # Gemini 1.x does not support mediaResolution at all.
        if "gemini-2" in model:
            max_media_resolution = _extract_max_media_resolution_from_messages(messages)
            if max_media_resolution:
                media_resolution_value = _convert_detail_to_media_resolution_enum(
                    max_media_resolution
                )
                if media_resolution_value and generation_config is not None:
                    generation_config["mediaResolution"] = media_resolution_value[
                        "level"
                    ]

        data = RequestBody(contents=content)
        # Vertex rejects system_instruction/tools/toolConfig alongside cachedContent.
        # Treat dropping these fields as a request mutation guarded by modify_params.
        can_send_cache_incompatible_fields = (
            cached_content is None or litellm.modify_params is False
        )
        if can_send_cache_incompatible_fields:
            if system_instructions is not None:
                data["system_instruction"] = system_instructions
            if tools is not None:
                data["tools"] = tools
            if tool_choice is not None:
                data["toolConfig"] = tool_choice
            if include_server_side_tool_invocations:
                if "toolConfig" not in data:
                    data["toolConfig"] = {}
                data["toolConfig"]["includeServerSideToolInvocations"] = True
        if safety_settings is not None:
            data["safetySettings"] = safety_settings
        if generation_config is not None and len(generation_config) > 0:
            data["generationConfig"] = generation_config
        if cached_content is not None:
            data["cachedContent"] = cached_content

        if service_tier := optional_params.pop("service_tier", None):
            if isinstance(service_tier, str):
                if service_tier.lower() == "default":
                    data["serviceTier"] = "standard"
                else:
                    data["serviceTier"] = service_tier.lower()
            else:
                data["serviceTier"] = service_tier

        # Only add labels for Vertex AI endpoints (not Google GenAI/AI Studio) and only if non-empty
        if labels and custom_llm_provider != LlmProviders.GEMINI:
            data["labels"] = labels
        _pop_and_merge_extra_body(data, optional_params)
    except Exception as e:
        raise e

    return data


def sync_transform_request_body(
    gemini_api_key: Optional[str],
    messages: List[AllMessageValues],
    api_base: Optional[str],
    model: str,
    client: Optional[HTTPHandler],
    timeout: Optional[Union[float, httpx.Timeout]],
    extra_headers: Optional[dict],
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_auth_header: Optional[str],
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    (
        messages,
        optional_params,
        cached_content,
    ) = context_caching_endpoints.check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key=gemini_api_key or "dummy",
        api_base=api_base,
        model=model,
        client=client,
        timeout=timeout,
        extra_headers=extra_headers,
        cached_content=optional_params.pop("cached_content", None),
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
    )

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


async def async_transform_request_body(
    gemini_api_key: Optional[str],
    messages: List[AllMessageValues],
    api_base: Optional[str],
    model: str,
    client: Optional[AsyncHTTPHandler],
    timeout: Optional[Union[float, httpx.Timeout]],
    extra_headers: Optional[dict],
    optional_params: dict,
    logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,  # type: ignore
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_auth_header: Optional[str],
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    (
        messages,
        optional_params,
        cached_content,
    ) = await context_caching_endpoints.async_check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key=gemini_api_key or "dummy",
        api_base=api_base,
        model=model,
        client=client,
        timeout=timeout,
        extra_headers=extra_headers,
        cached_content=optional_params.pop("cached_content", None),
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
    )

    if _openai_messages_may_need_sync_gcs_metadata_fetch(messages):
        # _transform_request_body may issue a sync httpx.get (up to 5s timeout)
        # via _get_gcs_object_content_type to fetch GCS object metadata. Run the
        # whole sync transformation on a worker thread so it does not block the
        # async event loop.
        return await asyncify(_transform_request_body)(
            messages=messages,
            model=model,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            cached_content=cached_content,
            optional_params=optional_params,
        )

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


def _default_user_message_when_system_message_passed() -> ChatCompletionUserMessage:
    """
    Returns a default user message when a "system" message is passed in gemini fails.

    This adds a blank user message to the messages list, to ensure that gemini doesn't fail the request.
    """
    return ChatCompletionUserMessage(content=".", role="user")


def _transform_system_message(
    supports_system_message: bool, messages: List[AllMessageValues]
) -> Tuple[Optional[SystemInstructions], List[AllMessageValues]]:
    """
    Extracts the system message from the openai message list.

    Converts the system message to Gemini format

    Returns
    - system_content_blocks: Optional[SystemInstructions] - the system message list in Gemini format.
    - messages: List[AllMessageValues] - filtered list of messages in OpenAI format (transformed separately)
    """
    # Separate system prompt from rest of message
    system_prompt_indices = []
    system_content_blocks: List[PartType] = []
    if supports_system_message is True:
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                _system_content_block: Optional[PartType] = None
                if isinstance(message["content"], str):
                    _system_content_block = PartType(text=message["content"])
                elif isinstance(message["content"], list):
                    system_text = ""
                    for content in message["content"]:
                        system_text += content.get("text") or ""
                    _system_content_block = PartType(text=system_text)
                if _system_content_block is not None:
                    system_content_blocks.append(_system_content_block)
                    system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

    if len(system_content_blocks) > 0:
        #########################################################
        # If no messages are passed in, add a blank user message
        # Relevant Issue - https://github.com/BerriAI/litellm/issues/13769
        #########################################################
        if len(messages) == 0:
            messages.append(_default_user_message_when_system_message_passed())
        #########################################################
        return SystemInstructions(parts=system_content_blocks), messages

    return None, messages

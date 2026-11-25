"""
Transformation logic from OpenAI format to Gemini format.

Why separate file? Make it easy to see how transformation works
"""

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_image_mime_type_from_url,
    is_blank_content,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    MAX_VIDEO_DOWNLOAD_BYTES,
    convert_generic_image_chunk_to_openai_image_obj,
    convert_to_anthropic_image_obj,
    convert_to_gemini_tool_call_invoke,
    convert_to_gemini_tool_call_result,
    get_thought_signature_from_tool,
    keeps_gemini_video,
    response_schema_prompt,
    skip_unfetched_video,
)
from litellm.litellm_core_utils.prompt_templates.image_handling import RemoteMedia, async_inline_remote_media
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    is_system_message,
    message_field,
    parts_of,
    place_mid_conversation_system,
    split_leading_system_run,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import pop_vertex_caller_labels, pop_vertex_request_labels
from litellm.llms.vertex_ai.request_labels import vertex_request_label_fields
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
    VERTEX_CREDENTIALS_TYPES,
    GeminiFileInspectionMetadata,
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
    check_text_in_content,
    gemini_fetches_url_itself,
    gemini_media_capability,
    gemini_video_metadata_from_openai,
    get_supports_response_schema,
    get_supports_system_message,
    normalize_youtube_url,
    youtube_video_mime_type,
)

# VertexBase is imported only for type checking to avoid introducing a
# module-load-time cyclic import to vertex_llm_base. The instance is lazily
# constructed by _get_vertex_base() the first time GCS metadata needs to be fetched.
_GCS_METADATA_VERTEX_BASE: "VertexBase | None" = None
# Shared sync client for GCS JSON API metadata reads so proxy/SSL settings
# from litellm's HTTP stack apply (see Greptile review on PR #27278).
_GCS_METADATA_HTTP_HANDLER: HTTPHandler | None = None
_DECODED_JSON: Final = TypeAdapter(object)
_GEMINI_MIME_TYPE_ALIASES: Final[dict[str, str]] = {
    "image/jpg": "image/jpeg",
}


def _apply_gemini_mime_type_aliases(mime_type: str) -> str:
    """Normalize known MIME aliases only; does not consult the file-type registry.

    Also strips MIME parameters (e.g. ``; charset=utf-8``) so that values
    sourced from GCS object metadata (``contentType``) validate correctly.
    """
    normalized: Final = mime_type.split(";", 1)[0].strip().lower()
    return _GEMINI_MIME_TYPE_ALIASES.get(normalized, normalized)


def _get_vertex_base() -> "VertexBase":
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

    from ..vertex_llm_base import VertexBase

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _convert_detail_to_media_resolution_enum(
    detail: str | None,
) -> dict[str, str] | None:
    if detail == "low":
        return {"level": "MEDIA_RESOLUTION_LOW"}
    elif detail == "medium":
        return {"level": "MEDIA_RESOLUTION_MEDIUM"}
    elif detail == "high":
        return {"level": "MEDIA_RESOLUTION_HIGH"}
    elif detail == "ultra_high":
        return {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}
    return None


def changes_gemini_media_settings(detail: str | None, video_metadata: Mapping[str, object] | None) -> bool:
    return _convert_detail_to_media_resolution_enum(detail) is not None or bool(
        gemini_video_metadata_from_openai(video_metadata or {})
    )


_VERTEX_AI_PROVIDERS: Final = ("vertex_ai", "vertex_ai_beta")
_VERTEX_IGNORED_FILE_INSPECTION_FIELDS: Final = ("document_metadata", "spreadsheet_metadata", "audio_metadata")


def refuse_vertex_ignored_media_fields(
    fields: Mapping[str, object],
    *,
    in_tool_result: bool,
    model: str | None,
    custom_llm_provider: str | None,
    drop_params: bool,
) -> None:
    if custom_llm_provider not in _VERTEX_AI_PROVIDERS or drop_params:
        return
    video_metadata: Final = as_str_mapping(fields.get("video_metadata")) or {}
    content_layers: Final = (
        ("video_metadata.content_layers",) if not in_tool_result and video_metadata.get("content_layers") else ()
    )
    ignored: Final = (*content_layers, *(name for name in _VERTEX_IGNORED_FILE_INSPECTION_FIELDS if fields.get(name)))
    if not ignored:
        return
    location: Final = " inside a tool result" if in_tool_result else ""
    raise litellm.BadRequestError(
        message=(
            f"Vertex ignores {', '.join(ignored)} on media{location}. "
            "Use a gemini/ model, remove them, or set drop_params=True."
        ),
        model=model or "",
        llm_provider=custom_llm_provider,
    )


def _vertex_supported_media_fields(
    fields: Mapping[str, object], *, model: str | None, custom_llm_provider: str | None, drop_params: bool
) -> Mapping[str, object]:
    if custom_llm_provider not in _VERTEX_AI_PROVIDERS:
        return fields
    refuse_vertex_ignored_media_fields(
        fields, in_tool_result=False, model=model, custom_llm_provider=custom_llm_provider, drop_params=drop_params
    )
    kept: Final = {key: value for key, value in fields.items() if key not in _VERTEX_IGNORED_FILE_INSPECTION_FIELDS}
    video_metadata: Final = as_str_mapping(fields.get("video_metadata"))
    if video_metadata is None:
        return kept
    return {**kept, "video_metadata": {key: value for key, value in video_metadata.items() if key != "content_layers"}}


def _get_highest_media_resolution(current: str | None, new_detail: str | None) -> str | None:
    """
    Compare two media resolution values and return the highest one.
    Resolution hierarchy: ultra_high > high > medium > low > None
    """
    resolution_priority: Final = {"ultra_high": 4, "high": 3, "medium": 2, "low": 1}
    current_priority: Final = resolution_priority.get(current, 0) if current else 0
    new_priority: Final = resolution_priority.get(new_detail, 0) if new_detail else 0

    if new_priority > current_priority:
        return new_detail
    return current


def gemini_part_media_resolution(
    model: str | None, detail: str | None, *, custom_llm_provider: str | None = None
) -> Mapping[str, str] | None:
    if (
        model is None
        or detail is None
        or not gemini_media_capability(model, "supports_part_media_resolution", custom_llm_provider=custom_llm_provider)
    ):
        return None
    return _convert_detail_to_media_resolution_enum(detail)


def _extract_max_media_resolution_from_messages(
    messages: list[AllMessageValues],
) -> str | None:
    """
    Extract the highest media resolution (detail) from image content in messages.

    This is used to set the global media_resolution in generation_config for
    Gemini 2.x models which don't support per-part media resolution.

    Args:
        messages: List of messages in OpenAI format

    Returns:
        The highest detail level found ("high", "low", or None)
    """
    max_resolution: str | None = None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                detail: str | None = None
                if item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        detail = image_url.get("detail")
                elif item.get("type") == "file":
                    file_obj = item.get("file")
                    if isinstance(file_obj, dict):
                        detail = file_obj.get("resolution") or file_obj.get("detail")
                if detail:
                    max_resolution = _get_highest_media_resolution(max_resolution, detail)
    return max_resolution


def _apply_gemini_metadata(
    part: PartType,
    model: str | None,
    media_resolution_enum: dict[str, str] | None,
    video_metadata: Mapping[str, object] | None,
    custom_llm_provider: str | None,
) -> PartType:
    """
    Apply media_resolution and video_metadata parameters to a Gemini part.

    - Per-part media_resolution: Gemini 3+ only (2.x uses generation_config global).
    - video_metadata (fps, startOffset, endOffset): all Gemini models (1.x, 2.x, 3+).
    """
    if model is None:
        return part

    part_dict: Final = dict(part)

    if media_resolution_enum is not None and gemini_media_capability(
        model, "supports_part_media_resolution", custom_llm_provider=custom_llm_provider
    ):
        part_dict["media_resolution"] = media_resolution_enum

    if video_metadata is not None:
        gemini_video_metadata: Final = gemini_video_metadata_from_openai(video_metadata)
        if gemini_video_metadata:
            part_dict["video_metadata"] = gemini_video_metadata

    return cast(PartType, part_dict)


def _parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid gs URI: {gs_uri}")
    uri_without_scheme: Final = gs_uri[5:]  # drop gs://
    uri_parts: Final = uri_without_scheme.split("/", 1)
    if len(uri_parts) != 2 or not uri_parts[0] or not uri_parts[1]:
        raise ValueError(f"Invalid gs URI: {gs_uri}")
    return uri_parts[0], uri_parts[1]


def _is_valid_gcs_bucket_name(bucket: str) -> bool:
    """
    Validate bucket name against core GCS naming constraints.
    """
    bucket_length: Final = len(bucket)
    max_bucket_length: Final = 222 if "." in bucket else 63
    if bucket_length < 3 or bucket_length > max_bucket_length:
        return False
    if "." in bucket and any(len(label) == 0 or len(label) > 63 for label in bucket.split(".")):
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
    extension_with_dot: Final = os.path.splitext(url)[-1]
    extension: Final = extension_with_dot[1:] if extension_with_dot else ""
    return len(extension) == 0


def _image_url_payload_may_need_sync_gcs_metadata_fetch(
    raw_image_url: object,
) -> bool:
    """
    True when this image_url value (content-part image_url or assistant ``images[]``
    entry) can trigger a blocking GCS metadata read for MIME resolution.
    """
    fmt: str | None = None
    url: str | None = None
    if isinstance(raw_image_url, dict):
        url = raw_image_url.get("url")
        if not isinstance(url, str):
            return False
        fmt = raw_image_url.get("format") or raw_image_url.get("mime_type") or raw_image_url.get("content_type")
    elif isinstance(raw_image_url, str):
        url = raw_image_url
    else:
        return False
    if "gs://" not in url or fmt:
        return False
    return _gs_uri_requires_content_type_metadata(url)


def _message_fields(message: object) -> Mapping[str, object] | None:
    if isinstance(message, BaseModel):
        return as_str_mapping(message.model_dump(exclude_none=False))
    return as_str_mapping(message)


def _video_may_need_download(video_url: object) -> bool:
    url: Final = video_url if isinstance(video_url, str) else (as_str_mapping(video_url) or {}).get("url")
    return isinstance(url, str) and url.startswith(("http://", "https://")) and not gemini_fetches_url_itself(url)


def _openai_messages_may_need_sync_gcs_metadata_fetch(
    messages: list[AllMessageValues],
) -> bool:
    """
    Heuristic: True if any message part can trigger a blocking GCS JSON
    metadata read inside _transform_request_body (extension-less gs:// without
    explicit MIME hints) or a blocking download of a remote ``video_url``.
    Covers user/system/tool ``content`` parts and assistant ``images`` (same
    paths as ``_gemini_convert_messages_with_history``). Used to decide whether
    ``async_transform_request_body`` should offload the sync transform via
    ``asyncify``.
    """
    for raw in messages:
        message_fields = _message_fields(raw)
        if message_fields is None:
            continue
        for image_item in as_object_list(message_fields.get("images")) or ():
            image_fields = as_str_mapping(image_item)
            if image_fields is not None and _image_url_payload_may_need_sync_gcs_metadata_fetch(
                image_fields.get("image_url")
            ):
                return True

        content = as_object_list(message_fields.get("content"))
        if content is None:
            continue
        for item in content:
            fields = as_str_mapping(item)
            if fields is None:
                continue
            itype = fields.get("type")
            if itype == "video_url" and _video_may_need_download(fields.get("video_url")):
                return True
            if itype in ("image_url", "video_url"):
                if _image_url_payload_may_need_sync_gcs_metadata_fetch(fields.get(itype)):
                    return True
            elif itype == "file":
                file_obj = as_str_mapping(fields.get("file"))
                if file_obj is None:
                    continue
                fmt = file_obj.get("format") or file_obj.get("mime_type") or file_obj.get("content_type")
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
    vertex_project: str | None = None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None = None,
) -> str | None:
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

    headers: Final[dict[str, str]] = {}
    explicit_vertex_auth_provided: Final = vertex_project is not None or vertex_credentials is not None
    if explicit_vertex_auth_provided:
        try:
            access_token, _ = _get_vertex_base().get_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
            )
            headers["Authorization"] = f"Bearer {access_token}"
        except Exception as e:
            raise litellm.BadRequestError(
                message=(f"Unable to fetch GCS metadata with provided Vertex credentials/project. Original error: {e}"),
                model=None,
                llm_provider="vertex_ai",
            )

    # Build the URL via httpx.URL with a fixed scheme/host and URL-encode both
    # bucket and object so CodeQL does not flag the interpolation as a
    # potential SSRF that could resolve to an arbitrary host.
    encoded_bucket: Final = quote(bucket, safe="")
    encoded_object: Final = quote(object_name, safe="")
    metadata_url: Final = httpx.URL(
        scheme="https",
        host="storage.googleapis.com",
        path=f"/storage/v1/b/{encoded_bucket}/o/{encoded_object}",
        params={"fields": "contentType"},
    )
    try:
        response: Final = _get_gcs_metadata_http_handler().get(
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
        payload: Final = _DECODED_JSON.validate_python(response.json())
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

    content_type: Final = payload.get("contentType")
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


def _normalize_and_validate_gemini_mime_type(mime_type: str, model: str | None) -> str:
    # Import lazily to avoid a module-level cyclic-import alert with
    # litellm.types.files.
    from litellm.types.files import get_file_extension_from_mime_type

    normalized_mime_type: Final = _apply_gemini_mime_type_aliases(mime_type)
    try:
        file_extension: Final = get_file_extension_from_mime_type(normalized_mime_type)
        file_type: Final = get_file_type_from_extension(file_extension)
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


def process_gemini_media(
    image_url: str,
    format: str | None = None,
    media_resolution_enum: dict[str, str] | None = None,
    model: str | None = None,
    video_metadata: Mapping[str, object] | None = None,
    vertex_project: str | None = None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None = None,
    max_download_bytes: int | None = None,
    *,
    custom_llm_provider: str | None = None,
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
            extension_with_dot: Final = os.path.splitext(image_url)[-1]  # Ex: ".png"
            extension: Final = extension_with_dot[1:]  # Ex: "png"

            explicit_gcs_format = False
            if not format:
                mime_type: str | None = None
                # For extension-less gs:// URIs, we cannot infer from path.
                # If callers pass `format`/`mime_type`, this branch is skipped.
                if extension:
                    file_type: Final = get_file_type_from_extension(extension)

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
            return _apply_gemini_metadata(part, model, media_resolution_enum, video_metadata, custom_llm_provider)
        elif gemini_fetches_url_itself(image_url):
            # Gemini fetches Files API URIs and YouTube links itself. Fetching them here
            # gets a 403 (Files API) or a YouTube HTML page, so pass the URI through.
            # Vertex rejects a fileData without mimeType, so YouTube links default to video/mp4.
            uri_mime_type: Final = format or youtube_video_mime_type(image_url)
            if uri_mime_type:
                file_data = FileDataType(mime_type=uri_mime_type, file_uri=normalize_youtube_url(image_url))
            else:
                # Gemini Files API references can be passed through as URI-only.
                file_data = cast(FileDataType, {"file_uri": image_url})
            part = {"file_data": file_data}
            return _apply_gemini_metadata(part, model, media_resolution_enum, video_metadata, custom_llm_provider)
        elif "https://" in image_url and (image_type := format or get_image_mime_type_from_url(image_url)) is not None:
            file_data = FileDataType(mime_type=image_type, file_uri=image_url)
            part = {"file_data": file_data}
            return _apply_gemini_metadata(part, model, media_resolution_enum, video_metadata, custom_llm_provider)
        elif "http://" in image_url or "https://" in image_url or "base64" in image_url:
            image: Final = convert_to_anthropic_image_obj(image_url, format=format, max_bytes=max_download_bytes)
            _blob: Final[BlobType] = {"data": image["data"], "mime_type": image["media_type"]}
            part = {"inline_data": cast(BlobType, _blob)}
            return _apply_gemini_metadata(part, model, media_resolution_enum, video_metadata, custom_llm_provider)
        raise Exception(f"Invalid image received - {image_url}")
    except Exception as e:
        raise e


_process_gemini_media: Final = process_gemini_media


def _str_field(fields: Mapping[str, object], *keys: str) -> str | None:
    value: Final = next((fields[key] for key in keys if fields.get(key)), None)
    return value if isinstance(value, str) else None


def _process_gemini_video_url(
    video_url: object,
    model: str | None,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    *,
    drop_params: bool,
    custom_llm_provider: str | None,
) -> PartType | None:
    provider: Final = custom_llm_provider or "vertex_ai"
    video_fields: Final = as_str_mapping(video_url) or {}
    url: Final = video_url if isinstance(video_url, str) else _str_field(video_fields, "url")
    if not url and drop_params:
        return None
    if not url:
        raise litellm.BadRequestError(
            message="Invalid message content: element type is 'video_url' but it has no 'url'",
            model=model or "",
            llm_provider=provider,
        )
    sent_fields: Final = _vertex_supported_media_fields(
        video_fields, model=model, custom_llm_provider=custom_llm_provider, drop_params=drop_params
    )
    try:
        part: Final = _process_gemini_media(
            image_url=url,
            format=_str_field(video_fields, "format", "mime_type", "content_type"),
            media_resolution_enum=_convert_detail_to_media_resolution_enum(_str_field(video_fields, "detail")),
            model=model,
            video_metadata=as_str_mapping(sent_fields.get("video_metadata")),
            vertex_project=vertex_project,
            vertex_credentials=vertex_credentials,
            max_download_bytes=MAX_VIDEO_DOWNLOAD_BYTES,
            custom_llm_provider=provider,
        )
    except litellm.ImageFetchError:
        return skip_unfetched_video(
            "The link is unreachable, too large, or not a video.",
            location="a message",
            drop_params=drop_params,
            model=model,
            custom_llm_provider=provider,
        )
    media: Final = part.get("inline_data") or part.get("file_data")
    holds_video: Final = keeps_gemini_video(
        media.get("mime_type") if media is not None else None,
        model=model,
        custom_llm_provider=provider,
        drop_params=drop_params,
    )
    return part if holds_video else None


def _assistant_video_url_part(element: object, video_url_part: Callable[[object], PartType | None]) -> PartType | None:
    fields: Final = as_str_mapping(element)
    if fields is None or fields.get("type") != "video_url":
        return None
    return video_url_part(fields.get("video_url"))


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components: Final = snake_str.split("_")
    return components[0] + "".join(x.capitalize() for x in components[1:])


_snake_to_camel = snake_to_camel


def _camel_case_keys(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return {snake_to_camel(key): value for key, value in metadata.items()}


def _encode_gemini_audio_metadata(fields: Mapping[str, object]) -> GeminiFileInspectionMetadata:
    audio_metadata: Final = as_str_mapping(fields.get("audio_metadata"))
    if audio_metadata is None:
        return GeminiFileInspectionMetadata()
    return GeminiFileInspectionMetadata(audio_metadata=_camel_case_keys(audio_metadata))


def encode_gemini_file_inspection_metadata(
    file: Mapping[str, object], *, model: str | None = None, custom_llm_provider: str | None = None
) -> GeminiFileInspectionMetadata:
    document_metadata: Final = as_str_mapping(file.get("document_metadata"))
    spreadsheet_metadata: Final = as_str_mapping(file.get("spreadsheet_metadata"))
    document: Final = (
        GeminiFileInspectionMetadata(document_metadata=_camel_case_keys(document_metadata))
        if document_metadata is not None
        else GeminiFileInspectionMetadata()
    )
    spreadsheet: Final = (
        GeminiFileInspectionMetadata(spreadsheet_metadata=_camel_case_keys(spreadsheet_metadata))
        if spreadsheet_metadata is not None
        else GeminiFileInspectionMetadata()
    )
    selection: Final = document | spreadsheet | _encode_gemini_audio_metadata(file)
    detail: Final = file.get("resolution") or file.get("detail")
    resolution: Final = gemini_part_media_resolution(
        model, detail if isinstance(detail, str) else None, custom_llm_provider=custom_llm_provider
    )
    if resolution is None:
        return selection
    return {**selection, "media_resolution": resolution}


def camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


_camel_to_snake = camel_to_snake


def _get_equivalent_key(key: str, available_keys: set) -> str | None:
    """
    Get the equivalent key from available keys, checking both camelCase and snake_case variants
    """
    if key in available_keys:
        return key

    # Try camelCase version
    camel_key: Final = snake_to_camel(key)
    if camel_key in available_keys:
        return camel_key

    # Try snake_case version
    snake_key: Final = camel_to_snake(key)
    if snake_key in available_keys:
        return snake_key

    return None


def check_if_part_exists_in_parts(parts: list[PartType], part: PartType, excluded_keys: list[str] = []) -> bool:
    """
    Check if a part exists in a list of parts
    Handles both camelCase and snake_case key variations (e.g., function_call vs functionCall)
    """
    keys_to_compare: Final = set(part.keys()) - set(excluded_keys)
    for p in parts:
        p_keys = set(p.keys())
        # Check if all keys in part have equivalent values in p
        match_found = True
        for key in keys_to_compare:
            equivalent_key = _get_equivalent_key(key, p_keys)
            if equivalent_key is None or p.get(equivalent_key, None) != part.get(key, None):
                match_found = False
                break

        if match_found:
            return True
    return False


def _collect_tool_call_thought_signatures(
    assistant_msg: ChatCompletionAssistantMessage,
) -> frozenset[str]:
    """Thought signatures already carried by this message's tool-call parts.

    Gemini returns each thoughtSignature on exactly one part. When the signed
    part is a function call, the signature is replayed on that tool-call part
    by convert_to_gemini_tool_call_invoke, so attaching the same signature to
    the text part as well would send two copies and double-bill the previous
    turn's reasoning tokens on gemini-3 and newer models.

    Only real signatures count here; a synthesized placeholder must not
    suppress a genuine text-part signature (e.g. replaying gemini-2.5 history
    to a newer model).
    """
    signatures: tuple[str, ...] = ()

    tool_calls: Final = assistant_msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool in tool_calls:
            if not isinstance(tool, dict):
                continue
            signature = get_thought_signature_from_tool(tool)
            if signature:
                signatures += (signature,)

    function_call: Final = assistant_msg.get("function_call")
    if isinstance(function_call, dict):
        signature = get_thought_signature_from_tool({"function": function_call})
        if signature:
            signatures += (signature,)

    provider_specific_fields: Final = assistant_msg.get("provider_specific_fields")
    if not isinstance(provider_specific_fields, dict):
        return frozenset(signatures)

    invocations: Final = provider_specific_fields.get("server_side_tool_invocations")
    if not isinstance(invocations, list):
        return frozenset(signatures)

    for invocation in invocations:
        if not isinstance(invocation, dict):
            continue
        for key in ("thought_signature", "response_thought_signature"):
            invocation_signature = invocation.get(key)
            if isinstance(invocation_signature, str) and invocation_signature:
                signatures += (invocation_signature,)

    return frozenset(signatures)


def _is_blank_user_message(message: AllMessageValues) -> bool:
    fields: Final = as_str_mapping(message)
    return fields is not None and is_blank_content(fields.get("content"))


def _gemini_blank_last_user_turn(
    last_user_span: Sequence[AllMessageValues], contents: Sequence[ContentType]
) -> ContentType | None:
    if not last_user_span or not all(map(_is_blank_user_message, last_user_span)):
        return None
    if contents and contents[-1].get("role") == "user":
        return None
    return ContentType(role="user", parts=[PartType(text=" ")])


@dataclass(frozen=True, slots=True)
class _AssistantImageSource:
    url: str
    format: str | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class _TargetModel:
    model: str | None
    custom_llm_provider: str | None


@dataclass(frozen=True, slots=True)
class _MalformedHistory:
    message: str


def _outputs_images(info: Mapping[str, object]) -> bool:
    modalities: Final = info.get("supported_output_modalities")
    return info.get("mode") == "image_generation" or (isinstance(modalities, list) and "image" in modalities)


def _bundled_model_entry(model: str, custom_llm_provider: str | None) -> Mapping[str, object] | None:
    bundled: Final = litellm.utils.get_bundled_model_cost_map()
    entries: Final = (as_str_mapping(bundled.get(key)) for key in (model, f"{custom_llm_provider}/{model}"))
    return next((entry for entry in entries if entry is not None), None)


def _generates_images(target: _TargetModel) -> bool:
    model: Final = target.model
    if model is None:
        return False
    info: Final = litellm.utils.lookup_model_info_for_registration(model, target.custom_llm_provider)
    registered: Final[Mapping[str, object]] = {} if isinstance(info, litellm.utils.ModelInfoLookupMiss) else info
    bundled: Final = _bundled_model_entry(model, target.custom_llm_provider)
    return _outputs_images(registered) or (bundled is not None and _outputs_images(bundled))


def _message_thought_signatures(
    assistant_msg: ChatCompletionAssistantMessage, *, drop_params: bool
) -> tuple[str, ...] | _MalformedHistory:
    fields: Final = as_str_mapping(assistant_msg.get("provider_specific_fields"))
    entries: Final = (as_object_list(fields.get("thought_signatures")) if fields is not None else None) or ()
    signatures: Final = tuple(entry for entry in entries if isinstance(entry, str))
    if len(signatures) < len(entries) and not drop_params:
        return _MalformedHistory("provider_specific_fields.thought_signatures entries must be strings")
    return signatures


def _own_image_thought_signature(
    image_item: Mapping[str, object], *, drop_params: bool
) -> str | None | _MalformedHistory:
    fields: Final = as_str_mapping(image_item.get("provider_specific_fields"))
    signature: Final = fields.get("thought_signature") if fields is not None else None
    if signature is not None and not isinstance(signature, str) and not drop_params:
        return _MalformedHistory("images[].provider_specific_fields.thought_signature must be a string")
    return signature if isinstance(signature, str) and signature else None


def _assistant_image_source(
    image_item: Mapping[str, object], *, drop_params: bool
) -> _AssistantImageSource | None | _MalformedHistory:
    raw_image_url: Final = image_item.get("image_url")
    image_url: Final = {"url": raw_image_url} if isinstance(raw_image_url, str) else as_str_mapping(raw_image_url)
    if raw_image_url is not None and image_url is None and not drop_params:
        return _MalformedHistory("images[].image_url must be an object or a string")
    url: Final = image_url.get("url") if image_url is not None else None
    if image_url is None or not url or (drop_params and not isinstance(url, str)):
        return None
    if not isinstance(url, str):
        return _MalformedHistory("images[].image_url.url must be a string")
    format_hint: Final = image_url.get("format") or image_url.get("mime_type") or image_url.get("content_type")
    detail: Final = image_url.get("detail")
    return _AssistantImageSource(
        url=url,
        format=format_hint if isinstance(format_hint, str) else None,
        detail=detail if isinstance(detail, str) else None,
    )


def _right_aligned(signatures: Sequence[str], count: int) -> tuple[str | None, ...]:
    claimed: Final = min(count, len(signatures))
    return (None,) * (count - claimed) + tuple(signatures[len(signatures) - claimed :])


def _sourced_assistant_image(
    image_item: Mapping[str, object], *, drop_params: bool
) -> tuple[_AssistantImageSource, str | None] | None | _MalformedHistory:
    source: Final = _assistant_image_source(image_item, drop_params=drop_params)
    if source is None or isinstance(source, _MalformedHistory):
        return source
    signature: Final = _own_image_thought_signature(image_item, drop_params=drop_params)
    if isinstance(signature, _MalformedHistory):
        return signature
    return (source, signature)


def _sourced_assistant_images(
    images: object, *, drop_params: bool
) -> tuple[tuple[_AssistantImageSource, str | None], ...] | _MalformedHistory:
    entries: Final = as_object_list(images) or ()
    items: Final = tuple(item for item in map(as_str_mapping, entries) if item is not None)
    if len(items) < len(entries) and not drop_params:
        return _MalformedHistory("images entries must be objects")
    sourced: Final = tuple(_sourced_assistant_image(item, drop_params=drop_params) for item in items)
    failure: Final = next((result for result in sourced if isinstance(result, _MalformedHistory)), None)
    if failure is not None:
        return failure
    return tuple(result for result in sourced if isinstance(result, tuple))


def _has_text(content: object) -> bool:
    if isinstance(content, str):
        return content != ""
    elements: Final = tuple(filter(None, map(as_str_mapping, as_object_list(content) or ())))
    return any(element.get("type") == "text" and bool(element.get("text")) for element in elements)


def _assistant_text_thought_signature(
    message_signatures: Sequence[str],
    own_image_signatures: frozenset[str],
    tool_call_signatures: frozenset[str],
) -> str | None:
    first: Final = next((signature for signature in message_signatures if signature not in own_image_signatures), None)
    return first if first is not None and first not in tool_call_signatures else None


def _signed_assistant_images(
    sourced_images: tuple[tuple[_AssistantImageSource, str | None], ...],
    message_signatures: Sequence[str],
    claimed_signatures: frozenset[str],
) -> tuple[tuple[_AssistantImageSource, str | None], ...]:
    if any(signature for _, signature in sourced_images):
        return sourced_images
    pool: Final = tuple(signature for signature in message_signatures if signature not in claimed_signatures)
    return tuple(zip((source for source, _ in sourced_images), _right_aligned(pool, len(sourced_images)), strict=True))


def _assistant_image_part(
    source: _AssistantImageSource,
    thought_signature: str | None,
    target: _TargetModel,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
) -> PartType:
    part: Final = _process_gemini_media(
        image_url=source.url,
        format=source.format,
        media_resolution_enum=_convert_detail_to_media_resolution_enum(source.detail),
        model=target.model,
        vertex_project=vertex_project,
        vertex_credentials=vertex_credentials,
        custom_llm_provider=target.custom_llm_provider,
    )
    return part if thought_signature is None else {**part, "thoughtSignature": thought_signature}


def _signed_text_and_image_parts(
    assistant_msg: ChatCompletionAssistantMessage,
    target: _TargetModel,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    *,
    drop_params: bool,
) -> tuple[PartType, ...] | _MalformedHistory:
    tool_call_signatures: Final = _collect_tool_call_thought_signatures(assistant_msg)
    message_signatures: Final = _message_thought_signatures(assistant_msg, drop_params=drop_params)
    if isinstance(message_signatures, _MalformedHistory):
        return message_signatures
    sourced_images: Final = _sourced_assistant_images(assistant_msg.get("images"), drop_params=drop_params)
    if isinstance(sourced_images, _MalformedHistory):
        return sourced_images
    signing_images: Final = sourced_images if sourced_images and _generates_images(target) else ()
    content: Final = assistant_msg.get("content")
    text_signature: Final = (
        _assistant_text_thought_signature(
            message_signatures,
            frozenset(signature for _, signature in signing_images if signature is not None),
            tool_call_signatures,
        )
        if _has_text(content) or not signing_images
        else None
    )
    signed_images: Final = (
        _signed_assistant_images(
            signing_images, message_signatures, tool_call_signatures | frozenset(filter(None, (text_signature,)))
        )
        if signing_images
        else tuple((source, None) for source, _ in sourced_images)
    )
    image_parts: Final = tuple(
        _assistant_image_part(source, signature, target, vertex_project, vertex_credentials)
        for source, signature in signed_images
    )
    if not isinstance(content, str):
        return image_parts
    text_part: Final = (
        PartType(text=content) if text_signature is None else PartType(text=content, thoughtSignature=text_signature)
    )
    return (text_part, *image_parts)


def _assistant_text_and_image_parts(
    assistant_msg: ChatCompletionAssistantMessage,
    target: _TargetModel,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    *,
    drop_params: bool,
) -> tuple[PartType, ...]:
    parts: Final = _signed_text_and_image_parts(
        assistant_msg, target, vertex_project, vertex_credentials, drop_params=drop_params
    )
    if isinstance(parts, _MalformedHistory):
        raise BadRequestError(
            message=f"Invalid assistant message: {parts.message}",
            model=target.model,
            llm_provider=target.custom_llm_provider or "vertex_ai",
        )
    return parts


def gemini_convert_messages_with_history(
    messages: list[AllMessageValues],
    model: str | None = None,
    litellm_params: dict | None = None,
    custom_llm_provider: str | None = None,
    ends_request: bool = True,
) -> list[ContentType]:
    """
    Converts given messages from OpenAI format to Gemini format

    - Parts must be iterable
    - Roles must alternate b/w 'user' and 'model' (same as anthropic -> merge consecutive roles)
    - Please ensure that function response turn comes immediately after a function call turn
    """
    user_message_types: Final = {"user", "system"}
    contents: Final[list[ContentType]] = []

    last_message_with_tool_calls = None

    msg_i = 0
    tool_call_responses = []
    vertex_project = None
    vertex_credentials = None
    request_settings: Final = cast(Mapping[str, object], litellm_params or {})  # cast-ok: untyped shared signature
    drop_params: Final = drops_unsupported_params(request_settings.get("drop_params"))
    if litellm_params:
        vertex_project = litellm_params.get("vertex_project") or litellm_params.get("vertex_ai_project")
        vertex_credentials = litellm_params.get("vertex_credentials") or litellm_params.get("vertex_ai_credentials")
    video_url_part: Final = partial(
        _process_gemini_video_url,
        model=model,
        vertex_project=vertex_project,
        vertex_credentials=vertex_credentials,
        drop_params=drop_params,
        custom_llm_provider=custom_llm_provider,
    )

    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    forward_function_call_id: Final = VertexGeminiConfig.forward_gemini_function_call_id(model or "")

    try:
        while msg_i < len(messages):
            user_content: list[PartType] = []
            init_msg_i = msg_i
            ## MERGE CONSECUTIVE USER CONTENT ##
            while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
                _message_content = messages[msg_i].get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    _parts: list[PartType] = []
                    for element_idx, element in enumerate(_message_content):
                        if element["type"] == "text" and "text" in element and len(element["text"]) > 0:
                            element = cast(ChatCompletionTextObject, element)
                            _part = PartType(text=element["text"])
                            _parts.append(_part)
                        elif element["type"] == "image_url":
                            element = cast(ChatCompletionImageObject, element)
                            img_element = element
                            format: str | None = None
                            media_resolution_enum: dict[str, str] | None = None
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
                                image_url_dict = cast(dict[str, Any], raw_image_url)
                                format = (
                                    image_url_dict.get("format")
                                    or image_url_dict.get("mime_type")
                                    or image_url_dict.get("content_type")
                                )
                                detail = image_url_dict.get("detail")
                                media_resolution_enum = _convert_detail_to_media_resolution_enum(detail)
                            else:
                                image_url = raw_image_url
                            image_fields = _vertex_supported_media_fields(
                                as_str_mapping(raw_image_url) or {},
                                model=model,
                                custom_llm_provider=custom_llm_provider,
                                drop_params=drop_params,
                            )
                            _part = _process_gemini_media(
                                image_url=image_url,
                                format=format,
                                media_resolution_enum=media_resolution_enum,
                                model=model,
                                vertex_project=vertex_project,
                                vertex_credentials=vertex_credentials,
                                custom_llm_provider=custom_llm_provider,
                            )
                            _parts.append({**_part, **_encode_gemini_audio_metadata(image_fields)})
                        elif element["type"] == "video_url":
                            if (video_part := video_url_part(element.get("video_url"))) is not None:
                                _parts.append(video_part)
                        elif element["type"] == "input_audio":
                            audio_element = cast(ChatCompletionAudioObject, element)
                            audio_fields = _vertex_supported_media_fields(
                                {**audio_element, **audio_element["input_audio"]},
                                model=model,
                                custom_llm_provider=custom_llm_provider,
                                drop_params=drop_params,
                            )
                            audio_data = audio_element["input_audio"].get("data")
                            audio_format = audio_element["input_audio"].get("format")
                            if audio_data is not None and audio_format is not None:
                                audio_format_modified = (
                                    "audio/" + audio_format
                                    if audio_format.startswith("audio/") is False
                                    else audio_format
                                )  # Gemini expects audio/wav, audio/mp3, etc.
                                openai_image_str = convert_generic_image_chunk_to_openai_image_obj(
                                    image_chunk=GenericImageParsingChunk(
                                        type="base64",
                                        media_type=audio_format_modified,
                                        data=audio_data,
                                    )
                                )
                                _part = _process_gemini_media(
                                    image_url=openai_image_str,
                                    format=audio_format_modified,
                                    model=model,
                                    vertex_project=vertex_project,
                                    vertex_credentials=vertex_credentials,
                                )
                                _parts.append({**_part, **_encode_gemini_audio_metadata(audio_fields)})
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
                            file_dict = cast(dict[str, Any], _file_field)
                            file_id = file_dict.get("file_id")
                            format = (
                                file_dict.get("format") or file_dict.get("mime_type") or file_dict.get("content_type")
                            )
                            file_data = file_dict.get("file_data")
                            detail = file_dict.get("resolution") or file_dict.get("detail")
                            passed_file = file_id or file_data
                            if passed_file is None:
                                raise Exception("Unknown file type. Please pass in a file_id or file_data")
                            sent_file = _vertex_supported_media_fields(
                                file_dict,
                                model=model,
                                custom_llm_provider=custom_llm_provider,
                                drop_params=drop_params,
                            )

                            # Convert detail to media_resolution_enum
                            media_resolution_enum = _convert_detail_to_media_resolution_enum(detail)

                            try:
                                _part = _process_gemini_media(
                                    image_url=passed_file,
                                    format=format,
                                    model=model,
                                    media_resolution_enum=media_resolution_enum,
                                    video_metadata=as_str_mapping(sent_file.get("video_metadata")),
                                    vertex_project=vertex_project,
                                    vertex_credentials=vertex_credentials,
                                    custom_llm_provider=custom_llm_provider,
                                )
                                _parts.append({**_part, **encode_gemini_file_inspection_metadata(sent_file)})
                            except litellm.BadRequestError:
                                raise
                            except Exception as e:
                                raise litellm.BadRequestError(
                                    message=(
                                        f"Unable to determine mime type for file: "
                                        f"{file_id or 'provided data'}, set this explicitly "
                                        f"using message[{msg_i}].content[{element_idx}].file.format "
                                        f"(or file.mime_type/content_type). "
                                        f"Original error: {e}"
                                    ),
                                    model=model,
                                    llm_provider="vertex_ai",
                                )
                    user_content.extend(_parts)
                elif _message_content is not None and isinstance(_message_content, str):
                    _part = PartType(text=_message_content)
                    user_content.append(_part)

                msg_i += 1

            if (
                ends_request
                and msg_i == len(messages)
                and (blank_user_turn := _gemini_blank_last_user_turn(messages[init_msg_i:], contents)) is not None
            ):
                contents.append(blank_user_turn)
            elif user_content:
                """
                check that user_content has 'text' parameter.
                    - Known Vertex Error: Unable to submit request because it must have a text parameter.
                    - Relevant Issue: https://github.com/BerriAI/litellm/issues/5515
                """
                has_text_in_content = check_text_in_content(user_content)
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
                    msg_dict: ChatCompletionAssistantMessage | dict = messages[msg_i].model_dump()
                else:
                    msg_dict = messages[msg_i]
                assistant_msg = ChatCompletionAssistantMessage(**msg_dict)
                _message_content = assistant_msg.get("content", None)
                reasoning_content = assistant_msg.get("reasoning_content", None)
                if reasoning_content is not None:
                    assistant_content.append(PartType(thought=True, text=reasoning_content))
                if _message_content is not None and isinstance(_message_content, list):
                    _parts = []
                    for element in _message_content:
                        if isinstance(element, dict):
                            if element["type"] == "text":
                                _part = PartType(text=element["text"])
                                _parts.append(_part)
                            elif (video_part := _assistant_video_url_part(element, video_url_part)) is not None:
                                _parts.append(video_part)

                    assistant_content.extend(_parts)

                assistant_content.extend(
                    _assistant_text_and_image_parts(
                        assistant_msg,
                        _TargetModel(model=model, custom_llm_provider=custom_llm_provider),
                        vertex_project,
                        vertex_credentials,
                        drop_params=drop_params,
                    )
                )

                ## HANDLE ASSISTANT FUNCTION CALL
                if (
                    assistant_msg.get("tool_calls", []) is not None or assistant_msg.get("function_call") is not None
                ):  # support assistant tool invoke conversion
                    gemini_tool_call_parts = convert_to_gemini_tool_call_invoke(
                        assistant_msg,
                        model=model,
                        forward_function_call_id=forward_function_call_id,
                        custom_llm_provider=custom_llm_provider,
                        drop_params=drop_params,
                    )
                    ## check if gemini_tool_call already exists in assistant_content
                    for gemini_tool_call_part in gemini_tool_call_parts:
                        if not check_if_part_exists_in_parts(
                            assistant_content,
                            gemini_tool_call_part,
                            excluded_keys=["thoughtSignature"],
                        ):
                            assistant_content.append(gemini_tool_call_part)
                    # Only record this as the active tool-call message when it actually
                    # carries tool calls. The `if` guard above is also entered for a
                    # text-only assistant message (`assistant_msg.get("tool_calls", [])
                    # is not None` is True for an empty list), so without this check a
                    # later assistant message with no tool calls would clobber the
                    # reference. The following tool result would then be matched against
                    # an assistant message that has no tool_calls, raising "Missing
                    # corresponding tool call for tool response message".
                    if assistant_msg.get("tool_calls") or assistant_msg.get("function_call") is not None:
                        last_message_with_tool_calls = assistant_msg

                ## HANDLE SERVER-SIDE TOOL INVOCATIONS (context circulation)
                _psf = assistant_msg.get("provider_specific_fields")
                if isinstance(_psf, dict):
                    _ss_invocations = _psf.get("server_side_tool_invocations")
                    if isinstance(_ss_invocations, list):
                        for invocation in _ss_invocations:
                            # Re-inject toolCall part
                            tc_part: dict[str, object] = {
                                "toolCall": {
                                    "toolType": invocation.get("tool_type"),
                                    "id": invocation.get("id"),
                                    "args": invocation.get("args"),
                                }
                            }
                            if "thought_signature" in invocation:
                                tc_part["thoughtSignature"] = invocation["thought_signature"]
                            assistant_content.append(tc_part)

                            # Re-inject toolResponse part if response is present
                            if "response" in invocation:
                                tr_dict: dict[str, object] = {
                                    "id": invocation.get("id"),
                                    "response": invocation.get("response"),
                                }
                                if invocation.get("tool_type"):
                                    tr_dict["toolType"] = invocation["tool_type"]
                                tr_part: dict[str, object] = {"toolResponse": tr_dict}
                                if "response_thought_signature" in invocation:
                                    tr_part["thoughtSignature"] = invocation["response_thought_signature"]
                                assistant_content.append(tr_part)

                msg_i += 1

            if assistant_content:
                contents.append(ContentType(role="model", parts=assistant_content))

            ## APPEND TOOL CALL MESSAGES ##
            tool_call_message_roles = ["tool", "function"]
            if msg_i < len(messages) and messages[msg_i]["role"] in tool_call_message_roles:
                _part = convert_to_gemini_tool_call_result(
                    messages[msg_i],
                    last_message_with_tool_calls,
                    forward_function_call_id=forward_function_call_id,
                    allow_gemini_file_references=custom_llm_provider == "gemini",
                    supports_media_in_tool_response=model is None
                    or gemini_media_capability(
                        model, "supports_media_in_tool_response", custom_llm_provider=custom_llm_provider
                    ),
                    drop_params=drop_params,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
                msg_i += 1
                # Handle both single part and list of parts (for Computer Use with images)
                if isinstance(_part, list):
                    tool_call_responses.extend(_part)
                else:
                    tool_call_responses.append(_part)
            if msg_i < len(messages) and (messages[msg_i]["role"] not in tool_call_message_roles):
                if len(tool_call_responses) > 0:
                    contents.append(ContentType(role="user", parts=tool_call_responses))
                    tool_call_responses = []

            if msg_i == init_msg_i:  # prevent infinite loops
                raise Exception(
                    f"Invalid Message passed in - {messages[msg_i]}. File an issue https://github.com/BerriAI/litellm/issues"
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


_gemini_convert_messages_with_history = gemini_convert_messages_with_history

# Keys that LiteLLM consumes internally and must never be forwarded to the
_LITELLM_INTERNAL_EXTRA_BODY_KEYS: Final[frozenset] = frozenset({"cache", "tags"})


def _pop_and_merge_extra_body(data: RequestBody, optional_params: dict) -> None:
    """Pop extra_body from optional_params and shallow-merge into data, deep-merging dict values."""
    extra_body: Final[dict | None] = optional_params.pop("extra_body", None)
    if extra_body is not None:
        data_dict: Final[dict] = data
        for k, v in extra_body.items():
            if k in _LITELLM_INTERNAL_EXTRA_BODY_KEYS:
                continue
            if k in data_dict and isinstance(data_dict[k], dict) and isinstance(v, dict):
                data_dict[k].update(v)
            else:
                data_dict[k] = v


class _CallerToolConfig(TypedDict, total=False):
    toolConfig: ReadOnly[JsonValue]
    tool_config: ReadOnly[JsonValue]


_CALLER_TOOL_CONFIG: Final = TypeAdapter(_CallerToolConfig)


def _without_search_tools_next_to_functions(tools: Tools) -> Tools:
    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    kept: Final = VertexGeminiConfig.without_search_tools_next_to_functions(as_object_list(tools) or ())
    return cast(Tools, list(kept))  # cast-ok: a subset of the caller's tools list


def _caller_tool_config(extra_body: object) -> JsonValue | None:
    extra_body_fields: Final = as_str_mapping(extra_body)
    if extra_body_fields is None:
        return None
    try:
        caller: Final = _CALLER_TOOL_CONFIG.validate_python(extra_body_fields)
    except ValidationError:
        return None
    return caller.get("toolConfig", caller.get("tool_config"))


_TOOL_FIELD_KEYS: Final = frozenset(
    ("tools", "tool_choice", "include_server_side_tool_invocations", "includeServerSideToolInvocations")
)


@dataclass(frozen=True, slots=True)
class _ToolFields:
    tools: Tools | None
    tool_config: ToolConfig | None


def _settled_tool_fields(optional_params: Mapping[str, object]) -> _ToolFields:
    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    drops_search_next_to_functions: Final = optional_params.get(
        "includeServerSideToolInvocations"
    ) is False or VertexGeminiConfig.caller_tool_config_keeps_the_old_search_drop(
        _caller_tool_config(optional_params.get("extra_body"))
    )
    tools: Final = cast(Tools | None, optional_params.get("tools"))  # cast-ok: map_openai_params built these tools
    tool_choice: Final = cast(ToolConfig | None, optional_params.get("tool_choice"))  # cast-ok: mapped params
    flags_server_side_tools: Final = (
        optional_params.get("include_server_side_tool_invocations") is True and not drops_search_next_to_functions
    )
    return _ToolFields(
        tools=(
            _without_search_tools_next_to_functions(tools)
            if drops_search_next_to_functions and tools is not None
            else tools
        ),
        tool_config=(
            ToolConfig({**(tool_choice or ToolConfig()), "includeServerSideToolInvocations": True})
            if flags_server_side_tools
            else tool_choice
        ),
    )


def _with_settled_tool_fields(optional_params: Mapping[str, object]) -> Mapping[str, object]:
    fields: Final = _settled_tool_fields(optional_params)
    rest: Final = {k: v for k, v in optional_params.items() if k not in _TOOL_FIELD_KEYS}
    return {
        **rest,
        **({} if fields.tools is None else {"tools": fields.tools}),
        **({} if fields.tool_config is None else {"tool_choice": fields.tool_config}),
    }


def _has_google_maps_tool(tools: object) -> bool:
    """Return True if any tool object in the list has a 'googleMaps' key."""
    if not isinstance(tools, list):
        return False
    return any(isinstance(t, dict) and VertexToolName.GOOGLE_MAPS.value in t for t in tools)


def _rewrite_mime_type_to_response_format(generation_config: GenerationConfig) -> None:
    """
    Convert response_mime_type + response_json_schema/response_schema to the newer
    responseFormat structure when googleMaps is present in tools.

    The Gemini API rejects the combination of googleMaps + response_mime_type:
    'application/json' with the error:
        "Google Maps tool with a response mime type: 'application/json' is unsupported"

    The newer responseFormat field supports this combination on both the Gemini API
    (generativelanguage.googleapis.com) and Vertex AI endpoints.

    Before:
        generationConfig: {
            response_mime_type: "application/json",
            response_json_schema: {...}
        }

    After:
        generationConfig: {
            responseFormat: {
                "text": {"mimeType": "APPLICATION_JSON", "schema": {...}}
            }
        }
    """
    schema = generation_config.pop("response_json_schema", None)
    if schema is None:
        schema = generation_config.pop("response_schema", None)
    generation_config.pop("response_mime_type", None)

    response_format: Final[dict[str, dict[str, object]]] = {"text": {"mimeType": "APPLICATION_JSON"}}
    if schema is not None:
        response_format["text"]["schema"] = schema
    generation_config["responseFormat"] = response_format


def _rewrite_google_maps_response_format(data: RequestBody) -> None:
    generation_config: Final = cast(GenerationConfig | None, data.get("generationConfig"))
    if (
        isinstance(generation_config, dict)
        and _has_google_maps_tool(data.get("tools"))
        and generation_config.get("response_mime_type") == "application/json"
    ):
        _rewrite_mime_type_to_response_format(generation_config)


def transform_request_body(
    messages: list[AllMessageValues],
    model: str,
    optional_params: dict,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    cached_content: str | None,
) -> RequestBody:
    """
    Common transformation logic across sync + async Gemini /generateContent calls.
    """
    request_params: Final = cast(Mapping[str, object], litellm_params)  # cast-ok: untyped shared signature
    # Separate system prompt from rest of message
    supports_system_message: Final = get_supports_system_message(model=model, custom_llm_provider=custom_llm_provider)
    system_instructions, messages = transform_system_message(
        supports_system_message=supports_system_message,
        messages=messages,
        drop_params=drops_unsupported_params(request_params.get("drop_params")),
        model=model,
        custom_llm_provider=custom_llm_provider,
    )
    # Checks for 'response_schema' support - if passed in
    if "response_schema" in optional_params:
        supports_response_schema = get_supports_response_schema(model=model, custom_llm_provider=custom_llm_provider)
        if supports_response_schema is False:
            user_response_schema_message: Final = response_schema_prompt(
                model=model,
                response_schema=optional_params.get("response_schema"),
            )
            messages.append({"role": "user", "content": user_response_schema_message})
            optional_params.pop("response_schema")

    # Check for any 'litellm_param_*' set during optional param mapping

    remove_keys: Final = []
    for k, v in optional_params.items():
        if k.startswith("litellm_param_"):
            litellm_params.update({k: v})
            remove_keys.append(k)

    tool_settings: Final = cast(Mapping[str, object], optional_params)  # cast-ok: untyped shared signature
    tool_fields: Final = _settled_tool_fields(tool_settings)
    optional_params = {k: v for k, v in optional_params.items() if k not in remove_keys and k not in _TOOL_FIELD_KEYS}

    try:
        if custom_llm_provider == "gemini":
            content = litellm.GoogleAIStudioGeminiConfig().transform_messages(
                messages=messages, model=model, litellm_params=litellm_params
            )
        else:
            content = litellm.VertexGeminiConfig().transform_messages(
                messages=messages, model=model, litellm_params=litellm_params
            )
        safety_settings: list[SafetSettingsConfig] | None = optional_params.pop("safety_settings", None)
        # Drop output_config as it's not supported by Vertex AI
        optional_params.pop("output_config", None)
        config_fields: Final = GenerationConfig.__annotations__.keys()

        # labels: optional explicit param and/or metadata.requester_metadata (OpenAI metadata)
        labels: Final = pop_vertex_caller_labels(optional_params, request_params)

        filtered_params = {k: v for k, v in optional_params.items() if _get_equivalent_key(k, set(config_fields))}

        generation_config: Final[GenerationConfig | None] = GenerationConfig(**filtered_params)

        # For Gemini 2.x models, also add media_resolution to generation_config (global)
        # as a fallback, since some 2.x versions may not support per-part media_resolution.
        # Gemini 1.x does not support mediaResolution at all.
        if "gemini-2" in model and not gemini_media_capability(
            model, "supports_part_media_resolution", custom_llm_provider=custom_llm_provider
        ):
            max_media_resolution: Final = _extract_max_media_resolution_from_messages(messages)
            if max_media_resolution:
                media_resolution_value: Final = _convert_detail_to_media_resolution_enum(max_media_resolution)
                if media_resolution_value and generation_config is not None:
                    generation_config["mediaResolution"] = media_resolution_value["level"]

        data: Final = RequestBody(contents=content)
        # Vertex rejects system_instruction/tools/toolConfig alongside cachedContent.
        # Treat dropping these fields as a request mutation guarded by modify_params.
        can_send_cache_incompatible_fields: Final = cached_content is None or litellm.modify_params is False
        if can_send_cache_incompatible_fields:
            if system_instructions is not None:
                data["system_instruction"] = system_instructions
            if tool_fields.tools is not None:
                data["tools"] = tool_fields.tools
            if tool_fields.tool_config is not None:
                data["toolConfig"] = tool_fields.tool_config
        if safety_settings is not None:
            data["safetySettings"] = safety_settings
        if generation_config is not None and len(generation_config) > 0:
            data["generationConfig"] = generation_config
        if cached_content is not None:
            data["cachedContent"] = cached_content

        service_tier: Final[str | None] = optional_params.pop("service_tier", None)
        if service_tier is not None and custom_llm_provider == LlmProviders.GEMINI:
            data["serviceTier"] = service_tier

        if continuation_token := optional_params.pop("continuation_token", None):
            data["continuationToken"] = continuation_token

        # Only add labels for Vertex AI endpoints (not Google GenAI/AI Studio) and only if non-empty
        if labels and custom_llm_provider != LlmProviders.GEMINI:
            data["labels"] = labels
        _pop_and_merge_extra_body(data, optional_params)
        if custom_llm_provider != LlmProviders.GEMINI and vertex_request_label_fields():
            merged_labels: Final = pop_vertex_request_labels({"labels": data.pop("labels", None)}, litellm_params)
            if merged_labels:
                data["labels"] = merged_labels
        _rewrite_google_maps_response_format(data)
    except Exception as e:
        raise e

    return data


_transform_request_body = transform_request_body


def sync_transform_request_body(
    gemini_api_key: str | None,
    messages: list[AllMessageValues],
    api_base: str | None,
    model: str,
    client: HTTPHandler | None,
    timeout: float | httpx.Timeout | None,
    extra_headers: dict | None,
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    vertex_project: str | None,
    vertex_location: str | None,
    vertex_auth_header: str | None,
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints: Final = ContextCachingEndpoints()
    request_settings: Final = cast(Mapping[str, object], litellm_params)  # cast-ok: untyped shared signature

    caller_cache_name: Final = cast(str | None, optional_params.pop("cached_content", None))  # cast-ok: a cache name
    tool_settings: Final = cast(Mapping[str, object], optional_params)  # cast-ok: untyped shared signature
    (
        messages,
        optional_params,
        cached_content,
    ) = context_caching_endpoints.check_and_create_cache(
        messages=messages,
        optional_params=dict(_with_settled_tool_fields(tool_settings)),
        api_key=gemini_api_key or "dummy",
        api_base=api_base,
        model=model,
        client=client,
        timeout=timeout,
        extra_headers=extra_headers,
        cached_content=caller_cache_name,
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
        drop_params=drops_unsupported_params(request_settings.get("drop_params")),
    )

    return transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


def _explicit_mime_type(fields: Mapping[str, object]) -> str | None:
    hint: Final = fields.get("format") or fields.get("mime_type") or fields.get("content_type")
    return hint if isinstance(hint, str) else None


def _ai_studio_inlines(media: RemoteMedia) -> bool:
    return not gemini_fetches_url_itself(media.url)


def _vertex_inlines(media: RemoteMedia) -> bool:
    if gemini_fetches_url_itself(media.url):
        return False
    return media.url.startswith("http://") or (
        _explicit_mime_type(media.fields) is None and get_image_mime_type_from_url(media.url) is None
    )


def should_inline_remote_media(custom_llm_provider: str) -> Callable[[RemoteMedia], bool]:
    return _ai_studio_inlines if custom_llm_provider == "gemini" else _vertex_inlines


async def async_transform_request_body(
    gemini_api_key: str | None,
    messages: list[AllMessageValues],
    api_base: str | None,
    model: str,
    client: AsyncHTTPHandler | None,
    timeout: float | httpx.Timeout | None,
    extra_headers: dict | None,
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    vertex_project: str | None,
    vertex_location: str | None,
    vertex_auth_header: str | None,
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints: Final = ContextCachingEndpoints()
    request_settings: Final = cast(Mapping[str, object], litellm_params)  # cast-ok: untyped shared signature

    caller_cache_name: Final = cast(str | None, optional_params.pop("cached_content", None))  # cast-ok: a cache name
    tool_settings: Final = cast(Mapping[str, object], optional_params)  # cast-ok: untyped shared signature
    (
        messages,
        optional_params,
        cached_content,
    ) = await context_caching_endpoints.async_check_and_create_cache(
        messages=messages,
        optional_params=dict(_with_settled_tool_fields(tool_settings)),
        api_key=gemini_api_key or "dummy",
        api_base=api_base,
        model=model,
        client=client,
        timeout=timeout,
        extra_headers=extra_headers,
        cached_content=caller_cache_name,
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
        drop_params=drops_unsupported_params(request_settings.get("drop_params")),
    )

    inlined_messages: Final = await async_inline_remote_media(
        messages, should_inline=should_inline_remote_media(custom_llm_provider)
    )

    if _openai_messages_may_need_sync_gcs_metadata_fetch(inlined_messages):
        # _transform_request_body may issue a sync httpx.get (up to 5s timeout)
        # via _get_gcs_object_content_type to fetch GCS object metadata. Run the
        # whole sync transformation on a worker thread so it does not block the
        # async event loop.
        return await asyncify(transform_request_body)(
            messages=inlined_messages,
            model=model,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            cached_content=cached_content,
            optional_params=optional_params,
        )

    return transform_request_body(
        messages=inlined_messages,
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


def _part_text(part: object) -> str:
    text: Final = message_field(part, "text")
    return text if isinstance(text, str) else ""


def _system_instruction_part(message: AllMessageValues) -> PartType | None:
    content: Final = message_field(message, "content")
    if isinstance(content, str):
        return PartType(text=content)
    text: Final = "".join(map(_part_text, parts_of(content)))
    return PartType(text=text) if isinstance(content, list) else None


def _has_video_url_part(message: object) -> bool:
    parts: Final = as_object_list((as_str_mapping(message) or {}).get("content")) or ()
    return any((as_str_mapping(part) or {}).get("type") == "video_url" for part in parts)


def reject_system_message_videos(
    messages: Sequence[AllMessageValues], drop_params: bool, model: str | None, custom_llm_provider: str | None
) -> None:
    if drop_params or not any(is_system_message(message) and _has_video_url_part(message) for message in messages):
        return
    raise litellm.BadRequestError(
        message="Gemini system instructions take text only. Move the video to a user message or set drop_params=True.",
        model=model or "",
        llm_provider=custom_llm_provider or "gemini",
    )


def transform_system_message(
    supports_system_message: bool,
    messages: list[AllMessageValues],
    drop_params: bool = False,
    model: str | None = None,
    custom_llm_provider: str | None = None,
) -> tuple[SystemInstructions | None, list[AllMessageValues]]:
    """
    Extracts the leading run of system messages from the openai message list as Gemini system instructions.

    Gemini contents only take the user and model roles, so every later system message becomes a user turn
    in place. Hoisting it into system_instruction would change the head of the prompt every time a client
    adds a reminder, which breaks implicit prompt caching.

    Returns
    - system_content_blocks: Optional[SystemInstructions] - the leading system messages in Gemini format.
    - messages: List[AllMessageValues] - the remaining messages in OpenAI format (transformed separately)
    """
    if supports_system_message is not True:
        return None, messages
    reject_system_message_videos(messages, drop_params, model, custom_llm_provider)
    leading_system_run, rest = split_leading_system_run(messages)
    parts_by_message: Final = tuple((message, _system_instruction_part(message)) for message in leading_system_run)
    system_content_blocks: Final = [part for _, part in parts_by_message if part is not None]
    remaining_messages: Final = [
        *(message for message, part in parts_by_message if part is None),
        *place_mid_conversation_system(rest, supports_mid_conversation_system=False),
    ]
    if not system_content_blocks:
        return None, remaining_messages
    return SystemInstructions(parts=system_content_blocks), remaining_messages or [
        _default_user_message_when_system_message_passed()
    ]


_transform_system_message = transform_system_message

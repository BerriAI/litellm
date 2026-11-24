"""
Transformation logic from OpenAI format to Gemini format.

Why separate file? Make it easy to see how transformation works
"""

import os
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, JsonValue, PositiveInt, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import (
    as_object_list,
    as_str_mapping,
    drops_unsupported_params,
    has_attributes,
)
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.prompt_templates.common_utils import is_blank_content
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
    GeminiMediaSource,
    checked_gemini_video_part,
    convert_to_anthropic_image_obj,
    convert_to_gemini_tool_call_invoke,
    convert_to_gemini_tool_call_result,
    gemini_media_source,
    gemini_tool_media_needs_download,
    get_thought_signature_from_tool,
    response_schema_prompt,
    skip_unfetched_media,
)
from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
    get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.image_handling import RemoteMedia, async_inline_remote_media
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    is_system_message,
    message_field,
    parts_of,
    place_mid_conversation_system,
    split_leading_system_run,
    string_field,
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
    ChatCompletionAssistantToolCall,
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
from litellm.types.utils import LlmProviders

from ..common_utils import (
    check_text_in_content,
    gemini_fetches_url_itself,
    gemini_media_capability,
    gemini_video_metadata_from_openai,
    get_supports_response_schema,
    get_supports_system_message,
    normalize_youtube_url,
    with_forced_bare_root_unions_on_parameters,
    with_integer_enums_that_fit,
    youtube_video_mime_type,
)
from .long_decode import with_default_output_budget

# VertexBase is imported only for type checking to avoid introducing a
# module-load-time cyclic import to vertex_llm_base. The instance is lazily
# constructed by _get_vertex_base() the first time GCS metadata needs to be fetched.
_GCS_METADATA_VERTEX_BASE: "VertexBase | None" = None
# Shared sync client for GCS JSON API metadata reads so proxy/SSL settings
# from litellm's HTTP stack apply (see Greptile review on PR #27278).
_GCS_METADATA_HTTP_HANDLER: HTTPHandler | None = None
_DECODED_JSON: Final = TypeAdapter(object)
_MESSAGE_PARTS: Final = TypeAdapter(Sequence[object])
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
            for item in _MESSAGE_PARTS.validate_python(content):
                if (fields := as_str_mapping(item)) is None:
                    continue
                if (
                    source := gemini_media_source(fields, prefer_file_id=msg.get("role") in ("user", "system"))
                ) is not None and source.detail:
                    max_resolution = _get_highest_media_resolution(max_resolution, source.detail)
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


class _SupportsModelDump(Protocol):
    def model_dump(self, *, exclude_none: bool) -> object: ...


def _message_fields(raw: object) -> Mapping[object, object] | None:
    if isinstance(raw, dict):
        return raw
    if not has_attributes(raw, _SupportsModelDump, "model_dump"):
        return None
    dumped: Final = raw.model_dump(exclude_none=False)
    return dumped if isinstance(dumped, dict) else None


def _openai_messages_may_need_sync_media_fetch(
    messages: Sequence[object],
    custom_llm_provider: str = "vertex_ai",
    model: str | None = None,
) -> bool:
    """Check media left after async prefetch for synchronous downloads or GCS metadata reads."""
    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    nested_inline_only: Final = custom_llm_provider == "gemini" and (
        model is None or VertexGeminiConfig.supports_multimodal_function_responses(model, custom_llm_provider)
    )
    for raw in messages:
        if (msg := _message_fields(raw)) is None:
            continue
        images_field = msg.get("images")
        if isinstance(images_field, list):
            for image_item in images_field:
                if not isinstance(image_item, dict):
                    continue
                if _image_url_payload_may_need_sync_gcs_metadata_fetch(image_item.get("image_url")):
                    return True

        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in _MESSAGE_PARTS.validate_python(content):
            if (fields := as_str_mapping(item)) is None:
                continue
            if (source := gemini_media_source(fields, prefer_file_id=msg.get("role") in ("user", "system"))) is None:
                continue
            if not source.format and _gs_uri_requires_content_type_metadata(source.url):
                return True
            if msg.get("role") in ("tool", "function") and gemini_tool_media_needs_download(source):
                return True
            if source.url.startswith(("http://", "https://")) and (
                _ai_studio_inlines
                if nested_inline_only and msg.get("role") in ("tool", "function")
                else _vertex_inlines
            )(RemoteMedia(source.url, {"format": source.format}, "file")):
                return True
    return False


_openai_messages_may_need_sync_gcs_metadata_fetch: Final = _openai_messages_may_need_sync_media_fetch


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


def encode_gemini_media_source(
    source: GeminiMediaSource,
    model: str | None,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    *,
    custom_llm_provider: str | None = None,
) -> PartType:
    part: Final = process_gemini_media(
        image_url=source.url,
        format=source.format,
        model=model,
        vertex_project=vertex_project,
        vertex_credentials=vertex_credentials,
        media_resolution_enum=_convert_detail_to_media_resolution_enum(source.detail),
        video_metadata=source.video_metadata,
        custom_llm_provider=custom_llm_provider,
        max_download_bytes=source.max_download_bytes,
    )
    return {**part, **encode_gemini_file_inspection_metadata(source.document_metadata, source.spreadsheet_metadata)}


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components: Final = snake_str.split("_")
    return components[0] + "".join(x.capitalize() for x in components[1:])


_snake_to_camel = snake_to_camel


def encode_gemini_file_inspection_metadata(
    document_metadata: Mapping[str, object] | None,
    spreadsheet_metadata: Mapping[str, object] | None,
    media_resolution: Mapping[str, str] | None = None,
) -> GeminiFileInspectionMetadata:
    return gemini_file_inspection_metadata(
        {snake_to_camel(key): value for key, value in document_metadata.items()}
        if document_metadata is not None
        else None,
        {snake_to_camel(key): value for key, value in spreadsheet_metadata.items()}
        if spreadsheet_metadata is not None
        else None,
        media_resolution,
    )


def gemini_file_inspection_metadata(
    document_metadata: Mapping[str, object] | None,
    spreadsheet_metadata: Mapping[str, object] | None,
    media_resolution: Mapping[str, str] | None = None,
) -> GeminiFileInspectionMetadata:
    document: Final = (
        GeminiFileInspectionMetadata(document_metadata=document_metadata)
        if document_metadata is not None
        else GeminiFileInspectionMetadata()
    )
    selection: Final[GeminiFileInspectionMetadata] = (
        {**document, "spreadsheet_metadata": spreadsheet_metadata} if spreadsheet_metadata is not None else document
    )
    if media_resolution is None:
        return selection
    return {**selection, "media_resolution": media_resolution}


def camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


_camel_to_snake = camel_to_snake


def _get_equivalent_key(key: str, available_keys: Collection[object]) -> str | None:
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


def _explicit_tool_call_id(tool_call: ChatCompletionAssistantToolCall) -> str | None:
    tool_call_id: Final = tool_call.get("id")
    if not isinstance(tool_call_id, str):
        return None
    return tool_call_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)[0] or None


def _function_tool_call_ids(tool_calls: Sequence[ChatCompletionAssistantToolCall]) -> tuple[str | None, ...]:
    return tuple(_explicit_tool_call_id(tool_call) for tool_call in tool_calls if "function" in tool_call)


def _is_unseen_tool_call_id(tool_call_id: str | None, seen_ids: Sequence[str | None]) -> bool:
    return tool_call_id is not None and tool_call_id not in seen_ids


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


@dataclass(frozen=True, slots=True)
class _TargetModel:
    model: str | None
    custom_llm_provider: str | None


@dataclass(frozen=True, slots=True)
class _MalformedHistory:
    message: str


def _generates_images(target: _TargetModel) -> bool:
    if target.model is None:
        return False
    info: Final = litellm.utils.lookup_model_info_for_registration(target.model, target.custom_llm_provider)
    if isinstance(info, litellm.utils.ModelInfoLookupMiss):
        return False
    return info.get("mode") == "image_generation" or "image" in (info.get("supported_output_modalities") or ())


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
) -> GeminiMediaSource | None | _MalformedHistory:
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
    return GeminiMediaSource(
        url=url,
        format=format_hint if isinstance(format_hint, str) else None,
        detail=detail if isinstance(detail, str) else None,
    )


def _right_aligned(signatures: Sequence[str], count: int) -> tuple[str | None, ...]:
    claimed: Final = min(count, len(signatures))
    return (None,) * (count - claimed) + tuple(signatures[len(signatures) - claimed :])


def _sourced_assistant_image(
    image_item: Mapping[str, object], *, drop_params: bool
) -> tuple[GeminiMediaSource, str | None] | None | _MalformedHistory:
    source: Final = _assistant_image_source(image_item, drop_params=drop_params)
    if source is None or isinstance(source, _MalformedHistory):
        return source
    signature: Final = _own_image_thought_signature(image_item, drop_params=drop_params)
    if isinstance(signature, _MalformedHistory):
        return signature
    return (source, signature)


def _sourced_assistant_images(
    images: object, *, drop_params: bool
) -> tuple[tuple[GeminiMediaSource, str | None], ...] | _MalformedHistory:
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
    sourced_images: tuple[tuple[GeminiMediaSource, str | None], ...],
    message_signatures: Sequence[str],
    claimed_signatures: frozenset[str],
) -> tuple[tuple[GeminiMediaSource, str | None], ...]:
    if any(signature for _, signature in sourced_images):
        return sourced_images
    pool: Final = tuple(signature for signature in message_signatures if signature not in claimed_signatures)
    return tuple(zip((source for source, _ in sourced_images), _right_aligned(pool, len(sourced_images)), strict=True))


def _assistant_image_part(
    source: GeminiMediaSource,
    thought_signature: str | None,
    target: _TargetModel,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
) -> PartType:
    part: Final = encode_gemini_media_source(
        source, target.model, vertex_project, vertex_credentials, custom_llm_provider=target.custom_llm_provider
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


def _gemini_history_media_part(
    content: Mapping[str, object],
    model: str | None,
    vertex_project: str | None,
    vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    *,
    message_index: int | None = None,
    content_index: int | None = None,
    drop_params: bool = False,
    custom_llm_provider: str | None = None,
) -> PartType | None:
    provider: Final = custom_llm_provider or "vertex_ai"
    source: Final = gemini_media_source(content, prefer_file_id=message_index is not None)
    if source is None:
        if message_index is not None and content.get("type") == "image_url":
            raw_url: Final = content.get("image_url")
            if raw_url is None:
                raise litellm.BadRequestError(
                    message="Invalid message content: element type is 'image_url' but 'image_url' field is missing ",
                    model=model,
                    llm_provider="vertex_ai",
                )
            raw_fields: Final = as_str_mapping(raw_url)
            if raw_fields is not None and raw_fields.get("url") is None:
                raise litellm.BadRequestError(
                    message="Invalid message content: element type is 'image_url' but 'url' field is missing inside 'image_url' ",
                    model=model,
                    llm_provider="vertex_ai",
                )
            raise ValueError(f"Invalid image received - {raw_url}")
        if content.get("type") == "video_url" and not drop_params:
            raise litellm.BadRequestError(
                message="Invalid message content: element type is 'video_url' but it has no 'url'",
                model=model or "",
                llm_provider=provider,
            )
        if message_index is not None and content.get("type") == "file":
            if content.get("file") is None:
                raise litellm.BadRequestError(
                    message="Content block has type='file' but is missing the required 'file' field",
                    model=model,
                    llm_provider="vertex_ai",
                )
            raise Exception("Unknown file type. Please pass in a file_id or file_data")
        return None
    try:
        part: Final = encode_gemini_media_source(
            source, model, vertex_project, vertex_credentials, custom_llm_provider=custom_llm_provider
        )
    except litellm.ImageFetchError:
        if content.get("type") != "video_url":
            raise
        skip_unfetched_media(
            content, location="a message", model=model, custom_llm_provider=provider, drop_params=drop_params
        )
        return None
    except litellm.BadRequestError:
        raise
    except Exception as exc:
        if message_index is None or content.get("type") != "file":
            raise
        file_fields: Final = as_str_mapping(content.get("file")) or {}
        raise litellm.BadRequestError(
            message=(
                f"Unable to determine mime type for file: {file_fields.get('file_id') or 'provided data'}, "
                f"set this explicitly using message[{message_index}].content[{content_index}].file.format "
                f"(or file.mime_type/content_type). Original error: {exc}"
            ),
            model=model,
            llm_provider="vertex_ai",
        ) from exc
    return checked_gemini_video_part(content, part, model=model, custom_llm_provider=provider, drop_params=drop_params)


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
    tool_call_responses: list[PartType] = []
    vertex_credentials = None
    request_settings: Final = cast(Mapping[str, object], litellm_params or {})  # cast-ok: untyped shared signature
    drop_params: Final = drops_unsupported_params(request_settings.get("drop_params"))
    project_setting: Final = request_settings.get("vertex_project") or request_settings.get("vertex_ai_project")
    vertex_project: Final = project_setting if isinstance(project_setting, str) else None
    if litellm_params:
        vertex_credentials = litellm_params.get("vertex_credentials") or litellm_params.get("vertex_ai_credentials")

    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    forward_function_call_id: Final = VertexGeminiConfig.forward_gemini_function_call_id(model or "")
    supports_multimodal_function_responses: Final = model is None or (
        VertexGeminiConfig.supports_multimodal_function_responses(model, custom_llm_provider)
    )

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
                        elif (
                            media_part := _gemini_history_media_part(
                                element,
                                model,
                                vertex_project,
                                vertex_credentials,
                                message_index=msg_i,
                                content_index=element_idx,
                                drop_params=drop_params,
                                custom_llm_provider=custom_llm_provider,
                            )
                        ) is not None:
                            _parts.append(media_part)
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
                    verbose_logger.debug(
                        "No text in user content. Adding a blank text to user content, to ensure Gemini doesn't fail the request. Relevant Issue - https://github.com/BerriAI/litellm/issues/5515"
                    )
                    user_content.append(
                        PartType(text=" ")
                    )  # add a blank text, to ensure Gemini doesn't fail the request.
                contents.append(ContentType(role="user", parts=user_content))
            assistant_content = []
            turn_tool_calls: tuple[ChatCompletionAssistantToolCall, ...] = ()
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
                            elif (
                                media_part := _gemini_history_media_part(
                                    element,
                                    model,
                                    vertex_project,
                                    vertex_credentials,
                                    drop_params=drop_params,
                                    custom_llm_provider=custom_llm_provider,
                                )
                            ) is not None:
                                _parts.append(media_part)

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
                    message_tool_calls = assistant_msg.get("tool_calls")
                    tool_call_ids = (
                        _function_tool_call_ids(message_tool_calls)
                        if message_tool_calls is not None
                        else (None,) * len(gemini_tool_call_parts)
                    )
                    earlier_turn_tool_call_ids = _function_tool_call_ids(turn_tool_calls)
                    for index, (gemini_tool_call_part, tool_call_id) in enumerate(
                        zip(gemini_tool_call_parts, tool_call_ids, strict=True)
                    ):
                        seen_tool_call_ids = (*earlier_turn_tool_call_ids, *tool_call_ids[:index])
                        is_new_tool_call = _is_unseen_tool_call_id(tool_call_id, seen_tool_call_ids)
                        if is_new_tool_call or not check_if_part_exists_in_parts(
                            assistant_content,
                            gemini_tool_call_part,
                            excluded_keys=["thoughtSignature"],
                        ):
                            assistant_content.append(gemini_tool_call_part)
                    # Consecutive assistant messages become one model turn, so the tool
                    # results that follow can answer a call from any of them, not only the last.
                    if assistant_msg.get("tool_calls") or assistant_msg.get("function_call") is not None:
                        turn_tool_calls = (*turn_tool_calls, *(assistant_msg.get("tool_calls") or ()))
                        last_message_with_tool_calls = {**assistant_msg, "tool_calls": list(turn_tool_calls)}

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
                    supports_multimodal_function_responses=supports_multimodal_function_responses,
                    custom_llm_provider=custom_llm_provider,
                    model=model,
                    vertex_project=vertex_project,
                    vertex_credentials=vertex_credentials,
                    drop_params=drop_params,
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
_LITELLM_INTERNAL_EXTRA_BODY_KEYS: Final[frozenset[str]] = frozenset({"cache", "tags"})


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


def _tools_fit_for_caller_tool_config(tools: Tools, tool_config: JsonValue | None, model: str) -> Tools:
    if tool_config is None:
        return tools
    from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    json_tools: Final = cast(JsonValue, tools)  # cast-ok: LiteLLM built these tools as JSON request data
    fitted: Final = with_integer_enums_that_fit(
        partial(with_forced_bare_root_unions_on_parameters, json_tools, tool_config),
        VertexGeminiConfig.is_gemini_3_or_newer(model),
    )
    return cast(Tools, fitted)  # cast-ok: same shape back


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
                llm_provider=custom_llm_provider,
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

        from .vertex_and_google_ai_studio_gemini import VertexGeminiConfig

        drops_top_k: Final = "top_k" in optional_params and not VertexGeminiConfig.supports_sampling_params(model)
        if drops_top_k:
            VertexGeminiConfig.reject_unsupported_sampling_param(
                model=model,
                param="top_k",
                drop_params=drops_unsupported_params(request_params.get("drop_params")),
            )
        caller_tool_config: Final = _caller_tool_config(
            cast(Mapping[str, object], optional_params).get("extra_body")  # cast-ok: untyped shared signature
        )
        config_keys: Final = set(config_fields) - {"top_k"} if drops_top_k else set(config_fields)
        filtered_params = {k: v for k, v in optional_params.items() if _get_equivalent_key(k, config_keys)}

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
                data["tools"] = _tools_fit_for_caller_tool_config(tool_fields.tools, caller_tool_config, model)
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
    extra_headers: Mapping[str, str] | None,
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    vertex_project: str | None,
    vertex_location: str | None,
    vertex_auth_header: str | None,
    context_caching: bool = True,
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
        context_caching=context_caching,
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_auth_header=vertex_auth_header,
        drop_params=drops_unsupported_params(request_settings.get("drop_params")),
    )

    return _with_default_output_budget(
        transform_request_body(
            messages=messages,
            model=model,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            cached_content=cached_content,
            optional_params=optional_params,
        ),
        model,
        custom_llm_provider,
        litellm_params.get("model_info"),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # litellm_params is an untyped dict
    )


class _CostMapOutputModalities(TypedDict):
    supported_output_modalities: NotRequired[ReadOnly[tuple[str, ...]]]


_COST_MAP_OUTPUT_MODALITIES: Final = TypeAdapter(_CostMapOutputModalities)


def _answers_text(cost_map_key: str) -> bool:
    try:
        row: Final = _COST_MAP_OUTPUT_MODALITIES.validate_python(litellm.model_cost.get(cost_map_key))  # pyright: ignore[reportUnknownMemberType]  # cost map rows are untyped JSON
    except ValidationError:
        return False
    return "text" in row.get("supported_output_modalities", ("text",))


class _OutputLimit(TypedDict):
    max_output_tokens: NotRequired[ReadOnly[PositiveInt | None]]


_OUTPUT_LIMIT: Final = TypeAdapter(_OutputLimit)


def _output_limit(fields: object) -> int | None:
    try:
        return _OUTPUT_LIMIT.validate_python(fields).get("max_output_tokens")
    except ValidationError:
        return None


def _catalog_capped_output_limit(cost_map_key: str, override: object) -> int | None:
    limits: Final = (
        _output_limit(GetModelCostMap.loaded_model_cost_map().get(cost_map_key)),
        _output_limit(override),
    )
    return min((limit for limit in limits if limit is not None), default=None)


def _text_chat_model_max_output_tokens(
    model: str, custom_llm_provider: str, deployment_model_info: object
) -> int | None:
    info: Final = litellm.utils.lookup_model_info_for_registration(model, custom_llm_provider)
    if isinstance(info, litellm.utils.ModelInfoLookupMiss) or info.get("mode") != "chat":
        return None
    if not _answers_text(info["key"]):
        return None
    return _catalog_capped_output_limit(info["key"], deployment_model_info or info)


text_chat_model_max_output_tokens: Final = _text_chat_model_max_output_tokens


def _with_default_output_budget(
    request_body: RequestBody, model: str, custom_llm_provider: str, deployment_model_info: object
) -> RequestBody:
    return with_default_output_budget(
        request_body,
        partial(_text_chat_model_max_output_tokens, model, custom_llm_provider, deployment_model_info),
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
    extra_headers: Mapping[str, str] | None,
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    vertex_project: str | None,
    vertex_location: str | None,
    vertex_auth_header: str | None,
    context_caching: bool = True,
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
        context_caching=context_caching,
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

    transform: Final = partial(
        transform_request_body,
        messages=inlined_messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )
    request_body: Final[RequestBody] = (
        await asyncify(transform)()
        if _openai_messages_may_need_sync_media_fetch(inlined_messages, custom_llm_provider, model)
        else transform()
    )
    return _with_default_output_budget(
        request_body,
        model,
        custom_llm_provider,
        litellm_params.get("model_info"),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # litellm_params is an untyped dict
    )


def _default_user_message_when_system_message_passed() -> ChatCompletionUserMessage:
    """
    Returns a default user message when a "system" message is passed in gemini fails.

    This adds a blank user message to the messages list, to ensure that gemini doesn't fail the request.
    """
    return ChatCompletionUserMessage(content=".", role="user")


def _system_instruction_part(message: AllMessageValues) -> PartType | None:
    content: Final = message_field(message, "content")
    if isinstance(content, str):
        return PartType(text=content)
    text: Final = "".join(string_field(part, "text") or "" for part in parts_of(content))
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

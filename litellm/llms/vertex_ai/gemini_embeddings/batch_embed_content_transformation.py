"""
Transformation logic from OpenAI /v1/embeddings format to Google AI Studio /batchEmbedContents format.

Why separate file? Make it easy to see how transformation works
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Final, Literal, cast

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError
from litellm.llms.vertex_ai.common_utils import GEMINI_FILES_API_URI_PREFIX, gemini_video_metadata_from_openai
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.vertex_ai import (
    BlobType,
    ContentType,
    EmbedContentRequest,
    FileDataType,
    GeminiEmbeddingElement,
    GeminiEmbeddingInput,
    PartType,
    PromptTokensDetails,
    UsageMetadata,
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
    VideoMetadataType,
)
from litellm.types.utils import (
    Embedding,
    EmbeddingResponse,
    PromptTokensDetailsWrapper,
    Usage,
)
from litellm.utils import get_formatted_prompt, token_counter

SUPPORTED_EMBEDDING_MIME_TYPES: Final = {
    "image/png",
    "image/jpeg",
    "audio/mpeg",
    "audio/wav",
    "video/mp4",
    "video/quicktime",
    "application/pdf",
}


_GEMINI_API_V1BETA: Final = GEMINI_FILES_API_URI_PREFIX.removesuffix("files/")


def is_file_reference(s: str) -> bool:
    """A `files/...` name or the Files API URI that `/v1/files` returns as the file id."""
    return isinstance(s, str) and (s.startswith("files/") or s.startswith(GEMINI_FILES_API_URI_PREFIX))


def file_reference_name(reference: str) -> str:
    return reference.removeprefix(_GEMINI_API_V1BETA)


_is_file_reference = is_file_reference


def _is_gcs_url(s: str) -> bool:
    """Check if string is a GCS URL (gs://...)."""
    return isinstance(s, str) and s.startswith("gs://")


def _infer_mime_type_from_gcs_url(gcs_url: str) -> str:
    """
    Infer MIME type from GCS URL file extension.

    Args:
        gcs_url: GCS URL like gs://bucket/path/to/file.png

    Returns:
        str: Inferred MIME type

    Raises:
        ValueError: If file extension is not supported
    """
    extension_to_mime: Final = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".pdf": "application/pdf",
    }

    gcs_url_lower: Final = gcs_url.lower()
    for ext, mime_type in extension_to_mime.items():
        if gcs_url_lower.endswith(ext):
            return mime_type

    raise ValueError(
        f"Unable to infer MIME type from GCS URL: {gcs_url}. "
        f"Supported extensions: {', '.join(extension_to_mime.keys())}"
    )


def _parse_data_url(data_url: str, mime_type_override: str | None = None) -> tuple[str, str]:
    """
    Parse a data URL to extract the media type and base64 data.

    Args:
        data_url: Data URL in format: data:image/jpeg;base64,/9j/4AAQ...
        mime_type_override: An explicit media type that replaces the declared one, skipping the allowlist

    Returns:
        tuple: (media_type, base64_data)
            media_type: e.g., "image/jpeg", "video/mp4", "audio/mpeg"
            base64_data: The base64-encoded data without the prefix

    Raises:
        ValueError: If data URL format is invalid or MIME type is unsupported
    """
    if not data_url.startswith("data:"):
        raise ValueError(f"Invalid data URL format: {data_url[:50]}...")

    if "," not in data_url:
        raise ValueError(f"Invalid data URL format (missing comma): {data_url[:50]}...")

    metadata, base64_data = data_url.split(",", 1)
    declared_media_type: Final = metadata[5:].split(";")[0]

    if mime_type_override is not None:
        return mime_type_override, base64_data

    if declared_media_type not in SUPPORTED_EMBEDDING_MIME_TYPES:
        raise ValueError(
            f"Unsupported MIME type for embedding: {declared_media_type}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EMBEDDING_MIME_TYPES))}"
        )

    return declared_media_type, base64_data


def _is_data_url(s: str) -> bool:
    return s.startswith("data:") and ";base64," in s


class _EmbeddingVideoMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fps: float | None = None
    start_offset: str | None = None
    end_offset: str | None = None


class _EmbeddingFile(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str | None = None
    file_data: str | None = None
    filename: str | None = None
    format: Annotated[str, Field(min_length=1)] | None = None
    video_metadata: _EmbeddingVideoMetadata | None = None


class _EmbeddingFileBlock(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["file"]
    file: _EmbeddingFile


_file_block_adapter: Final = TypeAdapter(_EmbeddingFileBlock)
_video_metadata_adapter: Final = TypeAdapter(VideoMetadataType)
_input_shape_adapter: Final[TypeAdapter[str | list[object]]] = TypeAdapter(str | list[object])
_mapping_adapter: Final = TypeAdapter(dict[str, object])
_BLOCK_FIELDS: Final = frozenset(_EmbeddingFileBlock.model_fields)
_FILE_FIELDS: Final = frozenset(_EmbeddingFile.model_fields)
_VIDEO_METADATA_FIELDS: Final = frozenset(_EmbeddingVideoMetadata.model_fields)
_FILE_SOURCE_FORMS: Final = "a data: URI, a gs:// URL, a files/ reference, or a Gemini Files API URI"


def _invalid_input(message: str) -> BadRequestError:
    return BadRequestError(message=message, model=None, llm_provider="gemini")


def _validation_error_summary(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(loc) for loc in detail['loc'])}: {detail['msg']}" for detail in error.errors())


def _as_mapping(value: object) -> dict[str, object] | None:
    try:
        return _mapping_adapter.validate_python(value)
    except ValidationError:
        return None


def _only_fields(mapping: Mapping[str, object], fields: frozenset[str]) -> dict[str, object]:
    return {key: value for key, value in mapping.items() if key in fields}


def _dropping_unsupported_keys(block: Mapping[str, object]) -> dict[str, object]:
    """What `drop_params` keeps of a file content block: the keys this surface understands, at every level."""
    kept_block: Final = _only_fields(block, _BLOCK_FIELDS)
    file: Final = _as_mapping(block.get("file"))
    if file is None:
        return kept_block
    kept_file: Final = _only_fields(file, _FILE_FIELDS)
    video_metadata: Final = _as_mapping(file.get("video_metadata"))
    if video_metadata is None:
        return {**kept_block, "file": kept_file}
    return {
        **kept_block,
        "file": {**kept_file, "video_metadata": _only_fields(video_metadata, _VIDEO_METADATA_FIELDS)},
    }


def _parse_file_block(element: object, drop_params: bool) -> _EmbeddingFileBlock:
    if not isinstance(element, Mapping):
        raise _invalid_input(
            f"Embedding input elements must be strings or file content blocks, got {type(element).__name__}"
        )
    block: Final = cast(Mapping[str, object], element)  # cast-ok: isinstance leaves the key and value types unknown
    try:
        return _file_block_adapter.validate_python(_dropping_unsupported_keys(block) if drop_params else block)
    except ValidationError as error:
        raise _invalid_input(
            f"Invalid file content block in embedding input: {_validation_error_summary(error)}"
        ) from error


def _file_block_source(block: _EmbeddingFileBlock) -> str:
    match (block.file.file_id, block.file.file_data):
        case (str() as file_id, None):
            return file_id
        case (None, str() as file_data):
            return file_data
        case (None, None):
            raise _invalid_input("A file content block in embedding input needs file.file_id or file.file_data")
        case _:
            raise _invalid_input(
                "A file content block in embedding input takes file.file_id or file.file_data, not both"
            )


def _gemini_video_metadata(video_metadata: _EmbeddingVideoMetadata) -> VideoMetadataType:
    return _video_metadata_adapter.validate_python(
        gemini_video_metadata_from_openai(video_metadata.model_dump(exclude_none=True))
    )


def _source_mime_type(
    source: str,
    mime_type_override: str | None,
    resolved_files: Mapping[str, Mapping[str, str]],
) -> str | None:
    if mime_type_override is not None:
        return mime_type_override
    if _is_data_url(source):
        try:
            return _parse_data_url(source)[0]
        except ValueError:
            return None
    if _is_gcs_url(source):
        try:
            return _infer_mime_type_from_gcs_url(source)
        except ValueError:
            return None
    if is_file_reference(source):
        file_info: Final = resolved_files.get(source)
        return None if file_info is None else file_info.get("mime_type")
    return None


def _media_part(
    source: str,
    mime_type_override: str | None,
    resolved_files: Mapping[str, Mapping[str, str]],
) -> PartType:
    if _is_data_url(source):
        mime_type, base64_data = _parse_data_url(source, mime_type_override)
        return PartType(inline_data=BlobType(mime_type=mime_type, data=base64_data))
    if _is_gcs_url(source):
        gcs_mime_type: Final = mime_type_override or _infer_mime_type_from_gcs_url(source)
        return PartType(file_data=FileDataType(mime_type=gcs_mime_type, file_uri=source))
    if is_file_reference(source):
        file_info: Final = resolved_files.get(source)
        if file_info is None:
            raise _invalid_input(
                f"File reference {source!r} could not be resolved: "
                "Gemini Files API references are only supported through the gemini/ provider"
            )
        return PartType(
            file_data=FileDataType(mime_type=mime_type_override or file_info["mime_type"], file_uri=file_info["uri"])
        )
    raise _invalid_input(f"A file content block source must be {_FILE_SOURCE_FORMS}, got {source[:50]!r}")


def _top_level_elements(
    input: GeminiEmbeddingInput,
) -> Sequence[GeminiEmbeddingElement | list[str] | list[GeminiEmbeddingElement]]:
    try:
        _input_shape_adapter.validate_python(input)
    except ValidationError as error:
        raise _invalid_input(
            f"Embedding input must be a string or a list of strings and file content blocks, got {type(input).__name__}"
        ) from error
    return [input] if isinstance(input, str) else input


def _elements(input: GeminiEmbeddingInput) -> Iterator[GeminiEmbeddingElement]:
    for element in _top_level_elements(input):
        if isinstance(element, list):
            yield from element
        else:
            yield element


def _element_source(element: GeminiEmbeddingElement) -> str:
    if isinstance(element, str):
        return element
    return _file_block_source(_parse_file_block(element, drop_params=True))


def flatten_media_sources(input: GeminiEmbeddingInput) -> tuple[str, ...]:
    """Every string element plus every file content block's source, in input order."""
    return tuple(_element_source(element) for element in _elements(input))


def _is_multimodal_input(input: GeminiEmbeddingInput) -> bool:
    """
    Check if the input contains multimodal data (data URIs, file references,
    GCS URLs, file content blocks, or nested lists for combined embeddings).

    Args:
        input: GeminiEmbeddingInput — str, List[element], or List[List[element]] for combined embeddings

    Returns:
        bool: True if any element is multimodal
    """
    return any(_is_multimodal_element(element) for element in _elements(input))


def _is_multimodal_element(element: GeminiEmbeddingElement) -> bool:
    """Check if a single element is multimodal."""
    if not isinstance(element, str):
        return True
    if _is_data_url(element):
        return True
    if is_file_reference(element):
        return True
    if _is_gcs_url(element):
        return True
    return False


def _build_part_for_input(
    element: GeminiEmbeddingElement,
    resolved_files: Mapping[str, Mapping[str, str]] | None = None,
    drop_params: bool = False,
) -> PartType:
    """
    Build a single PartType for an input element, handling text, data URIs,
    file references, GCS URLs, and file content blocks carrying a mime type
    and video_metadata.
    """
    files: Final = resolved_files or {}

    if isinstance(element, str):
        return _media_part(element, None, files) if _is_multimodal_element(element) else PartType(text=element)

    block: Final = _parse_file_block(element, drop_params)
    part: Final = _media_part(_file_block_source(block), block.file.format, files)
    if block.file.video_metadata is None:
        return part
    video_metadata: Final = _gemini_video_metadata(block.file.video_metadata)
    if not video_metadata:
        return part
    part_with_metadata: Final[PartType] = {**part, "video_metadata": video_metadata}
    return part_with_metadata


_SUPPORTED_EMBED_PARAMS: Final = {"outputDimensionality", "taskType", "title"}


def _filter_embed_params(optional_params: dict) -> dict:
    """Map and filter optional_params to only include Gemini embedding fields."""
    gemini_params: Final = optional_params.copy()
    if "dimensions" in gemini_params:
        gemini_params["outputDimensionality"] = gemini_params.pop("dimensions")
    if "task_type" in gemini_params:
        gemini_params["taskType"] = gemini_params.pop("task_type")
    return {k: v for k, v in gemini_params.items() if k in _SUPPORTED_EMBED_PARAMS}


def transform_openai_input_gemini_content(
    input: GeminiEmbeddingInput,
    model: str,
    optional_params: dict,
    resolved_files: dict[str, dict[str, str]] | None = None,
    drop_params: bool = False,
) -> VertexAIBatchEmbeddingsRequestBody:
    """
    Transform OpenAI embedding input to Gemini batchEmbedContents format.

    Each input element becomes a separate EmbedContentRequest, supporting
    text, data URIs, file references, and GCS URLs.

    If an element is a list (nested input), all sub-elements are combined
    into a single content with multiple parts, producing one combined
    embedding for the group.

    Examples:
        input=["text", "image"]         → 2 separate embeddings
        input=[["text", "image"]]       → 1 combined embedding
        input=[["text", "image"], "x"]  → 2 embeddings (1 combined + 1 separate)
    """
    gemini_model_name: Final = f"models/{model}"

    gemini_params: Final = _filter_embed_params(optional_params)

    input_list: Final = _top_level_elements(input)
    requests: Final[list[EmbedContentRequest]] = []

    for element in input_list:
        if isinstance(element, list):
            if not element:
                raise ValueError("Nested input list must not be empty")
            parts = [
                _build_part_for_input(sub, resolved_files=resolved_files, drop_params=drop_params) for sub in element
            ]
        else:
            parts = [_build_part_for_input(element, resolved_files=resolved_files, drop_params=drop_params)]
        request = EmbedContentRequest(
            model=gemini_model_name,
            content=ContentType(parts=parts),
            **gemini_params,
        )
        requests.append(request)

    return VertexAIBatchEmbeddingsRequestBody(requests=requests)


def transform_openai_input_gemini_embed_content(
    input: GeminiEmbeddingInput,
    model: str,
    optional_params: dict,
    resolved_files: dict[str, dict[str, str]] | None = None,
    drop_params: bool = False,
) -> dict:
    """
    Transform OpenAI embedding input to Gemini embedContent format (multimodal).

    Args:
        input: GeminiEmbeddingInput with text, data URIs, or file references
        model: Model name
        optional_params: Additional parameters (taskType, outputDimensionality, etc.)
        resolved_files: Dict mapping file names (files/abc) to {mime_type, uri}

    Returns:
        dict: Gemini embedContent request body with content.parts
    """
    resolved_files = resolved_files or {}

    gemini_params: Final = _filter_embed_params(optional_params)

    input_list: Final = _top_level_elements(input)
    parts: Final[list[PartType]] = []

    for element in input_list:
        if isinstance(element, list):
            raise ValueError(
                "Nested (combined) embeddings are not supported on the embedContent path. "
                "Use the batchEmbedContents path or pass a flat list instead."
            )
        parts.append(_build_part_for_input(element, resolved_files=resolved_files, drop_params=drop_params))

    request_body: Final[dict] = {
        "content": ContentType(parts=parts),
        **gemini_params,
    }

    return request_body


_IMAGE_MIME_TYPES: Final = frozenset({"image/png", "image/jpeg"})
_usage_metadata_adapter: Final = TypeAdapter(UsageMetadata)


def _parse_usage_metadata(raw_usage_metadata: object) -> UsageMetadata | None:
    if not isinstance(raw_usage_metadata, dict):
        return None
    try:
        return _usage_metadata_adapter.validate_python(raw_usage_metadata)
    except ValidationError:
        return None


def _flatten_input(input: GeminiEmbeddingInput) -> tuple[GeminiEmbeddingElement, ...]:
    return tuple(_elements(input))


def _is_image_element(
    element: GeminiEmbeddingElement,
    resolved_files: Mapping[str, Mapping[str, str]],
) -> bool:
    if isinstance(element, str):
        return _source_mime_type(element, None, resolved_files) in _IMAGE_MIME_TYPES
    block: Final = _parse_file_block(element, drop_params=True)
    return _source_mime_type(_file_block_source(block), block.file.format, resolved_files) in _IMAGE_MIME_TYPES


def _is_image_only_input(
    input: GeminiEmbeddingInput,
    resolved_files: Mapping[str, Mapping[str, str]],
) -> bool:
    elements: Final = _flatten_input(input)
    return bool(elements) and all(_is_image_element(element, resolved_files) for element in elements)


def _tokens_for_modality(details: Sequence[PromptTokensDetails], modality: str) -> int:
    return sum(detail["tokenCount"] for detail in details if detail["modality"] == modality)


def _fallback_usage(input: GeminiEmbeddingInput, model: str) -> Usage:
    if _is_multimodal_input(input):
        return Usage(prompt_tokens=0, total_tokens=0)
    input_text: Final = get_formatted_prompt(data={"input": input}, call_type="embedding")
    prompt_tokens: Final = token_counter(model=model, text=input_text)
    return Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)


def _usage_from_embed_content_response(
    input: GeminiEmbeddingInput,
    model: str,
    raw_usage_metadata: object,
    resolved_files: Mapping[str, Mapping[str, str]],
) -> Usage:
    usage_metadata: Final = _parse_usage_metadata(raw_usage_metadata)
    if usage_metadata is None:
        return _fallback_usage(input, model)

    prompt_tokens: Final = usage_metadata.get("promptTokenCount", 0)
    total_tokens: Final = usage_metadata.get("totalTokenCount") or prompt_tokens

    details: Final[Sequence[PromptTokensDetails]] = usage_metadata.get("promptTokensDetails") or ()
    if not details:
        return Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=0,
                image_tokens=prompt_tokens if _is_image_only_input(input, resolved_files) else 0,
            ),
        )

    text_tokens: Final = _tokens_for_modality(details, "TEXT")
    audio_tokens: Final = _tokens_for_modality(details, "AUDIO")
    image_tokens: Final = _tokens_for_modality(details, "IMAGE")
    video_tokens: Final = _tokens_for_modality(details, "VIDEO")

    return Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=text_tokens,
            audio_tokens=audio_tokens,
            image_tokens=image_tokens,
            video_tokens=video_tokens,
        ),
    )


def process_embed_content_response(
    input: GeminiEmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    response_json: dict,
    resolved_files: Mapping[str, Mapping[str, str]] | None = None,
) -> EmbeddingResponse:
    """
    Process Gemini embedContent response (single embedding for multimodal input).

    Args:
        input: Original input
        model_response: EmbeddingResponse to populate
        model: Model name
        response_json: Raw JSON response from embedContent endpoint
        resolved_files: Mapping of file references to resolved metadata

    Returns:
        EmbeddingResponse with single embedding
    """
    if "embedding" not in response_json:
        raise ValueError(f"embedContent response missing 'embedding' field: {response_json}")

    embedding_data: Final = response_json["embedding"]

    openai_embedding: Final = Embedding(
        embedding=embedding_data["values"],
        index=0,
        object="embedding",
    )

    model_response.data = [openai_embedding]
    model_response.model = model
    model_response.usage = _usage_from_embed_content_response(
        input=input,
        model=model,
        raw_usage_metadata=response_json.get("usageMetadata"),
        resolved_files=resolved_files or {},
    )

    return model_response


def process_response(
    input: GeminiEmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    _predictions: VertexAIBatchEmbeddingsResponseObject,
) -> EmbeddingResponse:
    openai_embeddings: Final[list[Embedding]] = []
    for idx, embedding in enumerate(_predictions["embeddings"]):
        openai_embedding = Embedding(
            embedding=embedding["values"],
            index=idx,
            object="embedding",
        )
        openai_embeddings.append(openai_embedding)

    model_response.data = openai_embeddings
    model_response.model = model

    has_nested: Final = isinstance(input, list) and any(isinstance(e, list) for e in input)
    if _is_multimodal_input(input) or has_nested:
        input_list: Final = input if isinstance(input, list) else [input]
        text_elements: Final[list[str]] = []
        for e in input_list:
            if isinstance(e, list):
                text_elements.extend(sub for sub in e if isinstance(sub, str) and not _is_multimodal_element(sub))
            elif isinstance(e, str) and not _is_multimodal_element(e):
                text_elements.append(e)
        if text_elements:
            input_text = get_formatted_prompt(data={"input": text_elements}, call_type="embedding")
            prompt_tokens = token_counter(model=model, text=input_text)
        else:
            prompt_tokens = 0
    else:
        input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
        prompt_tokens = token_counter(model=model, text=input_text)
    model_response.usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)

    return model_response

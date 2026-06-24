import base64
import io
import itertools
import json
import os
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx
from httpx import Headers, Response
from openai.types.file_deleted import FileDeleted

import litellm
from litellm._uuid import uuid
from litellm.files.utils import FilesAPIUtils
from litellm.litellm_core_utils.cloud_storage_security import (
    VERTEX_AI_MANAGED_GCS_PREFIX,
    build_managed_cloud_object_name,
    encode_gcs_object_name_for_url,
    sanitize_cloud_object_path,
    should_allow_legacy_cloud_file_ids,
    split_configured_cloud_bucket_name,
    validate_managed_cloud_file_id,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    extract_file_data,
    extract_file_metadata,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFileUploadStream,
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.vertex_ai.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    FileTypes,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    PathLike,
)
from litellm.types.files import ResumableChunkedUploadConfig
from litellm.types.llms.vertex_ai import GcsBucketResponse
from litellm.types.utils import LlmProviders, ModelResponse

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase

_GCP_LABEL_VALUE_MAX_LEN = 63
_CUSTOM_ID_RAW_LABEL_PREFIX = "b32_"


def _sanitize_gcp_label_value(value: str) -> str:
    """
    Sanitize a string to meet GCP label value constraints.

    GCP label values must:
    - Be lowercase
    - Contain only letters, numbers, underscores, and hyphens
    - Be max 63 characters

    Args:
        value: The string to sanitize

    Returns:
        A sanitized string that meets GCP label constraints
    """
    sanitized = re.sub(r"[^a-z0-9_-]", "_", value.lower())
    return sanitized[:_GCP_LABEL_VALUE_MAX_LEN]


def _encode_gcp_label_value_chunks(value: str) -> List[str]:
    """Encode arbitrary text across one or more GCP-label-safe values."""
    max_encoded_len = _GCP_LABEL_VALUE_MAX_LEN - len(_CUSTOM_ID_RAW_LABEL_PREFIX)
    encoded = (
        base64.b32encode(value.encode("utf-8")).decode("ascii").rstrip("=").lower()
    )
    return [
        f"{_CUSTOM_ID_RAW_LABEL_PREFIX}{encoded[i : i + max_encoded_len]}"
        for i in range(0, len(encoded), max_encoded_len)
    ] or [_CUSTOM_ID_RAW_LABEL_PREFIX]


def _decode_gcp_label_value_chunks(values: List[str]) -> Optional[str]:
    """Decode values produced by _encode_gcp_label_value_chunks."""
    encoded_parts = []
    for value in values:
        if not value.startswith(_CUSTOM_ID_RAW_LABEL_PREFIX):
            return None
        encoded_parts.append(value[len(_CUSTOM_ID_RAW_LABEL_PREFIX) :])
    encoded = "".join(encoded_parts).upper()
    padding = "=" * (-len(encoded) % 8)
    try:
        return base64.b32decode(encoded + padding).decode("utf-8")
    except Exception:
        return None


def _set_litellm_batch_custom_id_labels(labels: Dict[str, str], custom_id: Any) -> None:
    """
    Store OpenAI batch custom_id for Vertex batch correlation.

    ``litellm_custom_id`` is GCP-label-safe (may alter casing and characters).
    ``litellm_custom_id_raw`` encodes the original string for
    round-trip correlation in batch output transforms.
    """
    custom_id_str = str(custom_id)
    labels["litellm_custom_id"] = _sanitize_gcp_label_value(custom_id_str)
    raw_label_chunks = _encode_gcp_label_value_chunks(custom_id_str)
    labels["litellm_custom_id_raw"] = raw_label_chunks[0]
    for index, raw_label_chunk in enumerate(raw_label_chunks[1:], start=1):
        labels[f"litellm_custom_id_raw_{index}"] = raw_label_chunk


def _get_litellm_batch_custom_id_from_labels(labels: Dict[str, Any]) -> str:
    """Prefer encoded custom_id when present (see _set_litellm_batch_custom_id_labels)."""
    raw = labels.get("litellm_custom_id_raw")
    if raw:
        raw_chunks = [str(raw)]
        chunk_prefix = "litellm_custom_id_raw_"
        indexed_chunks = []
        for key, value in labels.items():
            if key.startswith(chunk_prefix) and key[len(chunk_prefix) :].isdigit():
                indexed_chunks.append((int(key[len(chunk_prefix) :]), str(value)))
        raw_chunks.extend(
            raw_label_chunk
            for _, raw_label_chunk in sorted(indexed_chunks, key=lambda item: item[0])
        )
        decoded = _decode_gcp_label_value_chunks(raw_chunks)
        if decoded is not None:
            return decoded
        return str(raw)
    return str(labels.get("litellm_custom_id", "unknown"))


def _openai_batch_jsonl_entry_to_vertex_wrapped_request(
    openai_entry: Dict[str, Any],
    map_openai_to_vertex_params: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Transforms a single OpenAI JSONL batch entry into its Vertex wrapped request.

    jsonl body for vertex is {"request": <request_body>}
    Example Vertex jsonl
    {"request":{"contents": [{"role": "user", "parts": [{"text": "What is the relation between the following video and image samples?"}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/animals.mp4", "mimeType": "video/mp4"}}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/image/cricket.jpeg", "mimeType": "image/jpeg"}}]}]}}
    """
    openai_request_body = openai_entry.get("body") or {}
    vertex_request_body = _transform_request_body(
        messages=openai_request_body.get("messages", []),
        model=openai_request_body.get("model", ""),
        optional_params=map_openai_to_vertex_params(openai_request_body),
        custom_llm_provider="vertex_ai",
        litellm_params={},
        cached_content=None,
    )

    custom_id = openai_entry.get("custom_id")
    if custom_id is not None:
        if "labels" not in vertex_request_body:
            vertex_request_body["labels"] = {}
        _set_litellm_batch_custom_id_labels(vertex_request_body["labels"], custom_id)

    return {"request": vertex_request_body}


def _iter_stripped_lines(raw_lines: Iterable[Union[str, bytes]]) -> Iterator[str]:
    """Decode (when needed), strip, and drop blank lines from an iterable of lines."""
    for raw in raw_lines:
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.strip()
        if line:
            yield line


def _iter_openai_jsonl_lines(openai_file_content: FileTypes) -> Iterator[str]:
    """
    Yield non-empty JSONL lines one at a time without materializing the whole
    payload, so peak memory stays bounded regardless of payload size. Mirrors
    ``str.splitlines()`` + ``line.strip()`` for ``\\n`` / ``\\r\\n`` delimited
    JSONL.
    """
    content: Any = openai_file_content
    if isinstance(content, tuple):
        content = content[1]

    if isinstance(content, (bytes, bytearray)):
        # Scan for newlines in place so a large in-memory payload is not copied
        # into a BytesIO just to iterate it line by line.
        newline = ord("\n")
        start, length = 0, len(content)
        while start < length:
            idx = content.find(newline, start)
            if idx == -1:
                chunk, start = content[start:], length
            else:
                chunk, start = content[start:idx], idx + 1
            line = chunk.decode("utf-8").strip()
            if line:
                yield line
        return

    if isinstance(content, str):
        yield from _iter_stripped_lines(io.StringIO(content))
        return

    if isinstance(content, PathLike):
        with open(str(content), "rb") as handle:
            yield from _iter_stripped_lines(handle)
        return

    if hasattr(content, "read"):
        # The handle is read twice per upload (first-row probe for the GCS
        # object name, then the body stream), so it must rewind to 0. A
        # non-seekable handle would silently resume mid-stream and drop the
        # already-consumed first row, so reject it loudly instead.
        seek = getattr(content, "seek", None)
        if seek is None:
            raise ValueError(
                "Batch upload file handle must be seekable; got a non-seekable "
                "stream. Pass bytes, a path, or a seekable handle."
            )
        try:
            seek(0)
        except (OSError, ValueError) as e:
            raise ValueError(
                "Batch upload file handle must be seekable so it can be re-read "
                "for the GCS object name and the upload body."
            ) from e
        yield from _iter_stripped_lines(content)
        return

    raise ValueError("Unsupported file content type")


def _iter_openai_jsonl_entries(
    openai_file_content: FileTypes,
) -> Iterator[Dict[str, Any]]:
    for line in _iter_openai_jsonl_lines(openai_file_content):
        yield json.loads(line)


class _OpenAIToVertexBatchUploadStream(BaseFileUploadStream):
    """Streams an OpenAI batch JSONL upload as Vertex-wrapped JSONL one row at a
    time, so the transformed payload is never held in full.

    The transform runs lazily as the HTTP client pulls each chunk, which keeps
    peak memory at one row regardless of how large the batch file is.
    """

    def __init__(
        self,
        openai_file_content: FileTypes,
        map_openai_to_vertex_params: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        self._openai_file_content = openai_file_content
        self._map_openai_to_vertex_params = map_openai_to_vertex_params

    def _iter_vertex_jsonl_chunks(self) -> Iterator[bytes]:
        first = True
        for entry in _iter_openai_jsonl_entries(self._openai_file_content):
            wrapped = _openai_batch_jsonl_entry_to_vertex_wrapped_request(
                entry, self._map_openai_to_vertex_params
            )
            prefix = b"" if first else b"\n"
            first = False
            yield prefix + json.dumps(wrapped).encode("utf-8")

    def iter_bytes(self) -> Iterator[bytes]:
        return self._iter_vertex_jsonl_chunks()


class VertexAIFilesConfig(VertexBase, BaseFilesConfig):
    """
    Config for VertexAI Files
    """

    def __init__(self):
        super().__init__()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if not api_key:
            api_key, _ = self.get_access_token(
                credentials=litellm_params.get("vertex_credentials"),
                project_id=litellm_params.get("vertex_project"),
            )
            if not api_key:
                raise ValueError("api_key is required")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get_gcs_object_name_from_batch_jsonl(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique GCS object name for the VertexAI batch prediction job

        named as: litellm-vertex-{model}-{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        if "publishers/google/models" not in _model:
            _model = f"publishers/google/models/{_model}"
        safe_model_path = sanitize_cloud_object_path(_model, fallback="model")
        object_name = f"{VERTEX_AI_MANAGED_GCS_PREFIX}{safe_model_path}/{uuid.uuid4()}"
        return object_name

    def get_object_name(self, file_data: FileTypes, purpose: str) -> str:
        """
        Get the object name for the request.

        Reads only the first JSONL entry (streamed) for batch files, so a large
        upload is never materialized just to derive the GCS object name.
        """
        if purpose == "batch":
            ## 1. If jsonl, derive the object name from the first entry's model
            first_entry = next(_iter_openai_jsonl_entries(file_data), None)
            if first_entry is not None:
                return self._get_gcs_object_name_from_batch_jsonl([first_entry])

        ## 2. If not jsonl, store under a server-generated managed object name
        filename, _ = extract_file_metadata(file_data)
        return build_managed_cloud_object_name(
            prefix=f"{VERTEX_AI_MANAGED_GCS_PREFIX}uploads/",
            filename=filename,
            fallback_filename="file",
        )

    def _get_configured_bucket_name(self, litellm_params: Dict) -> str:
        bucket_name = (
            litellm_params.get("gcs_bucket_name")
            or litellm_params.get("bucket_name")
            or os.getenv("GCS_BUCKET_NAME")
        )
        if not bucket_name:
            raise ValueError("GCS bucket_name is required")
        return bucket_name

    def get_complete_file_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateFileRequest,
    ) -> str:
        """
        Get the complete url for the request
        """
        bucket_name = self._get_configured_bucket_name(litellm_params)
        bucket_name, object_prefix = split_configured_cloud_bucket_name(bucket_name)
        file_data = data.get("file")
        purpose = data.get("purpose")
        if file_data is None:
            raise ValueError("file is required")
        if purpose is None:
            raise ValueError("purpose is required")
        _, content_type = extract_file_metadata(file_data)
        object_name = self.get_object_name(file_data, purpose)
        if object_prefix:
            object_name = f"{object_prefix}/{object_name}"
        encoded_object_name = encode_gcs_object_name_for_url(object_name)
        # Batch jsonl is streamed via a resumable session (bounded memory on
        # large uploads); everything else is a single simple-media upload.
        upload_type = (
            "resumable"
            if FilesAPIUtils.is_batch_jsonl_request(
                create_file_data=data, content_type=content_type
            )
            else "media"
        )
        endpoint = f"upload/storage/v1/b/{bucket_name}/o?uploadType={upload_type}&name={encoded_object_name}"
        api_base = api_base or "https://storage.googleapis.com"
        if not api_base:
            raise ValueError("api_base is required")

        return f"{api_base}/{endpoint}"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def _map_openai_to_vertex_params(
        self,
        openai_request_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        wrapper to call VertexGeminiConfig.map_openai_params
        """
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        config = VertexGeminiConfig()
        _model = openai_request_body.get("model", "")
        vertex_params = config.map_openai_params(
            model=_model,
            non_default_params=openai_request_body,
            optional_params={},
            drop_params=False,
        )
        return vertex_params

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, dict]:
        """
        2 Cases:
        1. Handle basic file upload
        2. Handle batch file upload (.jsonl), streamed to a GCS resumable
           session so large uploads stay memory-bounded.
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("file is required")

        _, content_type = extract_file_metadata(file_data)
        if FilesAPIUtils.is_batch_jsonl_request(
            create_file_data=create_file_data,
            content_type=content_type,
        ):
            return {
                "resumable_chunked_upload": ResumableChunkedUploadConfig(
                    body_stream=_OpenAIToVertexBatchUploadStream(
                        file_data,
                        self._map_openai_to_vertex_params,
                    ),
                    initiate_headers={
                        "X-Upload-Content-Type": "application/json",
                    },
                )
            }

        extracted_file_data_content = extract_file_data(file_data).get("content")
        if isinstance(extracted_file_data_content, bytes):
            return extracted_file_data_content
        raise ValueError("Unsupported file content type")

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform VertexAI File upload response into OpenAI-style FileObject
        """
        response_json = raw_response.json()

        try:
            response_object = GcsBucketResponse(**response_json)  # type: ignore
        except Exception as e:
            raise VertexAIError(
                status_code=raw_response.status_code,
                message=f"Error reading GCS bucket response: {e}",
                headers=raw_response.headers,
            )

        gcs_id = response_object.get("id", "")
        # Remove the last numeric ID from the path
        gcs_id = "/".join(gcs_id.split("/")[:-1]) if gcs_id else ""

        return OpenAIFileObject(
            purpose=response_object.get("purpose", "batch"),
            id=f"gs://{gcs_id}",
            filename=response_object.get("name", ""),
            created_at=_convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=response_object.get("timeCreated", "")
            ),
            status="uploaded",
            bytes=int(response_object.get("size", 0)),
            object="file",
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        return VertexAIError(
            status_code=status_code, message=error_message, headers=headers
        )

    def _parse_gcs_uri(
        self, file_id: str, litellm_params: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Validate a managed GCS file_id and return (bucket, url-encoded-object-path).
        """
        configured_bucket_name = self._get_configured_bucket_name(litellm_params or {})
        bucket_name, object_path = validate_managed_cloud_file_id(
            file_id=file_id,
            scheme="gs://",
            configured_bucket_name=configured_bucket_name,
            allowed_object_prefixes=(VERTEX_AI_MANAGED_GCS_PREFIX,),
            allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(
                litellm_params
            ),
        )
        return bucket_name, encode_gcs_object_name_for_url(object_path)

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        bucket, encoded_object = self._parse_gcs_uri(file_id, litellm_params)
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_object}"
        return url, {}

    def transform_retrieve_file_response(
        self,
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        response_json = raw_response.json()
        gcs_id = response_json.get("id", "")
        gcs_id = "/".join(gcs_id.split("/")[:-1]) if gcs_id else ""
        return OpenAIFileObject(
            id=f"gs://{gcs_id}",
            bytes=int(response_json.get("size", 0)),
            created_at=_convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=response_json.get("timeCreated", "")
            ),
            filename=response_json.get("name", ""),
            object="file",
            purpose=response_json.get("metadata", {}).get("purpose", "batch"),
            status="processed",
            status_details=None,
        )

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        bucket, encoded_object = self._parse_gcs_uri(file_id, litellm_params)
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_object}"
        return url, {}

    def transform_delete_file_response(
        self,
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        file_id = "deleted"
        if hasattr(raw_response, "request") and raw_response.request:
            url = str(raw_response.request.url)
            if "/b/" in url and "/o/" in url:
                import urllib.parse

                bucket_part = url.split("/b/")[-1].split("/o/")[0]
                encoded_name = url.split("/o/")[-1].split("?")[0]
                file_id = f"gs://{bucket_part}/{urllib.parse.unquote(encoded_name)}"
        return FileDeleted(id=file_id, deleted=True, object="file")

    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("VertexAIFilesConfig does not support file listing")

    def transform_list_files_response(
        self,
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        raise NotImplementedError("VertexAIFilesConfig does not support file listing")

    def transform_file_content_request(
        self,
        file_content_request,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        file_id = file_content_request.get("file_id", "")
        bucket, encoded_object = self._parse_gcs_uri(file_id, litellm_params)
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_object}?alt=media"
        return url, {}

    def transform_file_content_response(
        self,
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        """
        Transform file content response, converting Vertex AI batch output to OpenAI format if applicable.

        This method automatically detects and transforms Vertex AI batch prediction outputs
        (predictions.jsonl files) into OpenAI-compatible batch response format.

        If the file is not a batch output or transformation fails, the original content
        is returned as-is to maintain backward compatibility.
        """
        try:
            # Allow users to opt out of automatic Vertex batch output -> OpenAI
            # transformation, e.g. if they consume raw `predictions.jsonl` directly.
            if getattr(litellm, "disable_vertex_batch_output_transformation", False):
                return HttpxBinaryResponseContent(response=raw_response)

            # Try to transform batch output if it's a JSONL file
            content = raw_response.content
            if content:
                transformed_content = self._try_transform_vertex_batch_output_to_openai(
                    content=content,
                    logging_obj=logging_obj,
                )
                if transformed_content != content:
                    # Create a new response with transformed content and updated Content-Length
                    # Update headers with correct Content-Length
                    new_headers = dict(raw_response.headers)
                    new_headers["content-length"] = str(len(transformed_content))

                    mock_response = httpx.Response(
                        status_code=raw_response.status_code,
                        content=transformed_content,
                        headers=new_headers,
                        request=raw_response.request,
                    )
                    return HttpxBinaryResponseContent(response=mock_response)
        except Exception:
            # If transformation fails, return as-is
            pass

        return HttpxBinaryResponseContent(response=raw_response)

    def _try_transform_vertex_batch_output_to_openai(
        self, content: bytes, logging_obj: Optional[LiteLLMLoggingObj] = None
    ) -> bytes:
        """
        Try to transform Vertex AI batch output to OpenAI format.
        If conversion fails at any point, return the original content as-is.

        Vertex AI batch output format (predictions.jsonl):
        {
          "request": {"contents": [...], "labels": {"litellm_custom_id": "request-1", "litellm_custom_id_raw": "..."}},
          "status": "",
          "response": {"candidates": [...], "modelVersion": "gemini-2.5-flash", ...},
          "processed_time": "2026-04-13T10:18:18.102004+00:00"
        }

        OpenAI batch output format:
        {
          "id": "batch_req_...",
          "custom_id": "request-1",
          "response": {
            "status_code": 200,
            "request_id": "chatcmpl-...",
            "body": {<OpenAI chat completion response>}
          },
          "error": null
        }
        """
        try:
            # Read the result file one row at a time. Batch output files can be
            # as large as the (multi-GB) input, so splitting into a list of rows
            # and building a second list of transformed rows peaks at several full
            # copies and OOMs on retrieval.
            lines = _iter_openai_jsonl_lines(content)
            try:
                first_line = next(lines)
            except StopIteration:
                return content

            # Identify a Vertex AI batch output from the first row's
            # discriminating fields. Anything else (e.g. a binary file whose
            # first line is not valid UTF-8/JSON) raises and falls through to the
            # passthrough below, leaving the content untouched.
            first_row = json.loads(first_line)
            is_vertex_batch_output = (
                "request" in first_row
                and "response" in first_row
                and "processed_time" in first_row
                and (
                    "candidates" in first_row.get("response", {})
                    or "promptFeedback" in first_row.get("response", {})
                    or bool(first_row.get("status"))
                )
            )
            if not is_vertex_batch_output:
                return content

            vertex_gemini_config = VertexGeminiConfig()
            # Use a fresh Logging object for the per-row transform so we never
            # mutate the caller's (which already ran pre_call with its own
            # model/start_time/optional_params).
            batch_transform_logging_obj = Logging(
                model="",
                messages=[],
                stream=False,
                call_type="batch_transform",
                start_time=time.time(),
                litellm_call_id="",
                function_id="",
            )
            batch_transform_logging_obj.optional_params = {}
            mock_httpx_response = httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                request=httpx.Request(method="POST", url="https://example.com"),
            )

            # Transform each row straight into the output buffer, so peak memory
            # stays at ~one row plus the output. If any row fails, return the
            # original content unchanged.
            output = bytearray()
            for line in itertools.chain([first_line], lines):
                try:
                    openai_output = (
                        self._transform_single_vertex_batch_output_to_openai(
                            vertex_output=json.loads(line),
                            vertex_gemini_config=vertex_gemini_config,
                            logging_obj=batch_transform_logging_obj,
                            mock_httpx_response=mock_httpx_response,
                        )
                    )
                except Exception:
                    return content
                if output:
                    output += b"\n"
                output += json.dumps(openai_output).encode("utf-8")

            return bytes(output)

        except Exception:
            # If anything fails, return original content
            return content

    def _transform_single_vertex_batch_output_to_openai(
        self,
        vertex_output: Dict[str, Any],
        vertex_gemini_config: VertexGeminiConfig,
        logging_obj: Logging,
        mock_httpx_response: httpx.Response,
    ) -> Dict[str, Any]:
        """
        Transform a single Vertex AI batch output line to OpenAI format.
        Uses the existing VertexGeminiConfig transformation for the response.
        """
        # Extract custom_id from request labels (prefer raw for OpenAI round-trip)
        request_data = vertex_output.get("request", {})
        labels = request_data.get("labels", {}) or {}
        custom_id = _get_litellm_batch_custom_id_from_labels(labels)

        # Check if there's an error
        status = vertex_output.get("status", "")
        has_error = bool(status)

        if has_error:
            return {
                "id": f"batch_req_{uuid.uuid4()}",
                "custom_id": custom_id,
                "response": None,
                "error": {
                    "code": "vertex_ai_error",
                    "message": status,
                },
            }

        # Transform successful response using existing transformation
        vertex_response = vertex_output.get("response", {})

        # Extract model from response
        model = vertex_response.get("modelVersion", "gemini-1.5-flash-001")
        if "@" in model:
            model = model.split("@")[0]

        try:
            # Use existing VertexGeminiConfig transformation
            model_response = ModelResponse()

            transformed_response = vertex_gemini_config._transform_google_generate_content_to_openai_model_response(
                completion_response=vertex_response,
                model_response=model_response,
                model=model,
                logging_obj=logging_obj,
                raw_response=mock_httpx_response,
            )

            # Convert ModelResponse to dict
            response_dict = transformed_response.model_dump()

            # Return in OpenAI batch format
            return {
                "id": f"batch_req_{uuid.uuid4()}",
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": response_dict.get("id", ""),
                    "body": response_dict,
                },
                "error": None,
            }

        except Exception as e:
            return {
                "id": f"batch_req_{uuid.uuid4()}",
                "custom_id": custom_id,
                "response": None,
                "error": {
                    "code": "transformation_error",
                    "message": f"Failed to transform response: {str(e)}",
                },
            }

import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
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
from litellm.types.llms.vertex_ai import GcsBucketResponse
from litellm.types.utils import ExtractedFileData, LlmProviders, ModelResponse

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


def _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
    openai_jsonl_content: List[Dict[str, Any]],
    map_openai_to_vertex_params: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Transforms OpenAI JSONL batch entries to Vertex AI JSONL lines.

    jsonl body for vertex is {"request": <request_body>}
    Example Vertex jsonl
    {"request":{"contents": [{"role": "user", "parts": [{"text": "What is the relation between the following video and image samples?"}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/animals.mp4", "mimeType": "video/mp4"}}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/image/cricket.jpeg", "mimeType": "image/jpeg"}}]}]}}
    {"request":{"contents": [{"role": "user", "parts": [{"text": "Describe what is happening in this video."}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/another_video.mov", "mimeType": "video/mov"}}]}]}}
    """

    vertex_jsonl_content = []
    for _openai_jsonl_content in openai_jsonl_content:
        openai_request_body = _openai_jsonl_content.get("body") or {}
        vertex_request_body = _transform_request_body(
            messages=openai_request_body.get("messages", []),
            model=openai_request_body.get("model", ""),
            optional_params=map_openai_to_vertex_params(openai_request_body),
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
        )

        # Add custom_id as a label for correlation in batch outputs
        custom_id = _openai_jsonl_content.get("custom_id")
        if custom_id is not None:
            if "labels" not in vertex_request_body:
                vertex_request_body["labels"] = {}
            _set_litellm_batch_custom_id_labels(
                vertex_request_body["labels"], custom_id
            )

        vertex_jsonl_content.append({"request": vertex_request_body})
    return vertex_jsonl_content


class VertexAIFilesConfig(VertexBase, BaseFilesConfig):
    """
    Config for VertexAI Files
    """

    def __init__(self):
        self.jsonl_transformation = VertexAIJsonlFilesTransformation()
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

    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

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

    def get_object_name(
        self, extracted_file_data: ExtractedFileData, purpose: str
    ) -> str:
        """
        Get the object name for the request
        """
        extracted_file_data_content = extracted_file_data.get("content")

        if extracted_file_data_content is None:
            raise ValueError("file content is required")

        if purpose == "batch":
            ## 1. If jsonl, check if there's a model name
            file_content = self._get_content_from_openai_file(
                extracted_file_data_content
            )

            # Split into lines and parse each line as JSON
            openai_jsonl_content = [
                json.loads(line) for line in file_content.splitlines() if line.strip()
            ]
            if len(openai_jsonl_content) > 0:
                return self._get_gcs_object_name_from_batch_jsonl(openai_jsonl_content)

        ## 2. If not jsonl, store under a server-generated managed object name
        filename = extracted_file_data.get("filename")
        return build_managed_cloud_object_name(
            prefix=f"{VERTEX_AI_MANAGED_GCS_PREFIX}uploads/",
            filename=filename,
            fallback_filename="file",
        )

    def _get_configured_bucket_name(self, litellm_params: Dict) -> str:
        bucket_name = litellm_params.get("bucket_name") or os.getenv("GCS_BUCKET_NAME")
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
        extracted_file_data = extract_file_data(file_data)
        object_name = self.get_object_name(extracted_file_data, purpose)
        if object_prefix:
            object_name = f"{object_prefix}/{object_name}"
        encoded_object_name = encode_gcs_object_name_for_url(object_name)
        endpoint = f"upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={encoded_object_name}"
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

    def _transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
            openai_jsonl_content=openai_jsonl_content,
            map_openai_to_vertex_params=self._map_openai_to_vertex_params,
        )

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
        2. Handle batch file upload (.jsonl)
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("file is required")
        extracted_file_data = extract_file_data(file_data)
        extracted_file_data_content = extracted_file_data.get("content")

        if extracted_file_data_content is None:
            raise ValueError("file content is required")

        if FilesAPIUtils.is_batch_jsonl_file(
            create_file_data=create_file_data,
            extracted_file_data=extracted_file_data,
        ):
            ## 1. If jsonl, check if there's a model name
            file_content = self._get_content_from_openai_file(
                extracted_file_data_content
            )

            # Split into lines and parse each line as JSON
            openai_jsonl_content = [
                json.loads(line) for line in file_content.splitlines() if line.strip()
            ]
            vertex_jsonl_content = (
                self._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                    openai_jsonl_content
                )
            )
            return "\n".join(json.dumps(item) for item in vertex_jsonl_content)
        elif isinstance(extracted_file_data_content, bytes):
            return extracted_file_data_content
        else:
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
            # Decode content
            content_str = content.decode("utf-8")

            # Check if it's JSONL (multiple lines)
            lines = content_str.strip().split("\n")
            if not lines:
                return content

            # Try to parse the first line to see if it's Vertex AI batch output
            first_line = json.loads(lines[0])

            # Check if it has Vertex AI batch output structure with discriminating fields
            # Must have request, response, and processed_time
            # Plus either candidates (success) or status (error)
            has_base_structure = (
                "response" in first_line
                and "request" in first_line
                and "processed_time" in first_line
            )
            has_success_or_error = (
                "candidates" in first_line.get("response", {})
                or "promptFeedback" in first_line.get("response", {})
                or bool(first_line.get("status"))
            )

            if not (has_base_structure and has_success_or_error):
                # Not a Vertex AI batch output, return as-is
                return content

            vertex_gemini_config = VertexGeminiConfig()
            # Always use a fresh local Logging object for the per-line transformation
            # so we never mutate the caller's logging_obj (which already went through
            # pre_call and has its own model/start_time/optional_params set).
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

            # Transform all lines
            transformed_lines = []
            for line in lines:
                if not line.strip():
                    continue

                try:
                    vertex_output = json.loads(line)
                    openai_output = (
                        self._transform_single_vertex_batch_output_to_openai(
                            vertex_output=vertex_output,
                            vertex_gemini_config=vertex_gemini_config,
                            logging_obj=batch_transform_logging_obj,
                            mock_httpx_response=mock_httpx_response,
                        )
                    )
                    transformed_lines.append(json.dumps(openai_output))
                except Exception:
                    # If any line fails, return original content
                    return content

            # Return transformed content
            return "\n".join(transformed_lines).encode("utf-8")

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


class VertexAIJsonlFilesTransformation(VertexGeminiConfig):
    """
    Transforms OpenAI /v1/files/* requests to VertexAI /v1/files/* requests
    """

    def transform_openai_file_content_to_vertex_ai_file_content(
        self, openai_file_content: Optional[FileTypes] = None
    ) -> Tuple[str, str]:
        """
        Transforms OpenAI FileContentRequest to VertexAI FileContentRequest
        """

        if openai_file_content is None:
            raise ValueError("contents of file are None")
        # Read the content of the file
        file_content = self._get_content_from_openai_file(openai_file_content)

        # Split into lines and parse each line as JSON
        openai_jsonl_content = [
            json.loads(line) for line in file_content.splitlines() if line.strip()
        ]
        vertex_jsonl_content = (
            self._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_jsonl_content
            )
        )
        vertex_jsonl_string = "\n".join(
            json.dumps(item) for item in vertex_jsonl_content
        )
        object_name = self._get_gcs_object_name(
            openai_jsonl_content=openai_jsonl_content
        )
        return vertex_jsonl_string, object_name

    def _transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
            openai_jsonl_content=openai_jsonl_content,
            map_openai_to_vertex_params=self._map_openai_to_vertex_params,
        )

    def _get_gcs_object_name(
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

    def _map_openai_to_vertex_params(
        self,
        openai_request_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        wrapper to call VertexGeminiConfig.map_openai_params
        """
        _model = openai_request_body.get("model", "")
        vertex_params = self.map_openai_params(
            model=_model,
            non_default_params=openai_request_body,
            optional_params={},
            drop_params=False,
        )
        return vertex_params

    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def transform_gcs_bucket_response_to_openai_file_object(
        self, create_file_data: CreateFileRequest, gcs_upload_response: Dict[str, Any]
    ) -> OpenAIFileObject:
        """
        Transforms GCS Bucket upload file response to OpenAI FileObject
        """
        gcs_id = gcs_upload_response.get("id", "")
        # Remove the last numeric ID from the path
        gcs_id = "/".join(gcs_id.split("/")[:-1]) if gcs_id else ""

        return OpenAIFileObject(
            purpose=create_file_data.get("purpose", "batch"),
            id=f"gs://{gcs_id}",
            filename=gcs_upload_response.get("name", ""),
            created_at=_convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=gcs_upload_response.get("timeCreated", "")
            ),
            status="uploaded",
            bytes=gcs_upload_response.get("size", 0),
            object="file",
        )

"""
Supports writing files to Google AI Studio Files API.

For vertex ai, check out the vertex_ai/files/transformation.py file.
"""

import io
import json
import re
import time
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import aclosing
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, Protocol

import httpx
from openai.types.file_deleted import FileDeleted
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.files.types import FileContentStreamingResult
from litellm.files.utils import FilesAPIUtils
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    drops_unsupported_params,
    int_of,
    require_attributes,
    require_get,
    subscript,
)
from litellm.litellm_core_utils.prompt_templates.file_id_utils import (
    extract_file_data,
)
from litellm.litellm_core_utils.url_utils import encode_url_path_segment, join_api_base
from litellm.llms.base_llm.base_utils import map_developer_role_to_system_role
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.gemini.files.utils import gemini_file_id
from litellm.llms.vertex_ai.files.transformation import (
    BatchOutputRowTransformContext,
    OpenAIBatchOutputRow,
    new_batch_output_row_transform_context,
    openai_batch_output_row,
)
from litellm.llms.vertex_ai.gemini.transformation import transform_request_body
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    FileContentRequest,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
)
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import get_optional_params

from ..common_utils import GeminiError, GeminiModelInfo

_DECODED_JSON: Final = TypeAdapter(object)
_RFC3339_PARTS: Final = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:\d{2})")
_JSONL_NEWLINE: Final = b"\n"
_CHAT_COMPLETIONS_BATCH_URL: Final = "/v1/chat/completions"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_ROW_PARAMS_NOT_SENT_TO_GEMINI: Final = frozenset({"model", "messages", "drop_params", "metadata"})


def is_native_gemini_batch_output_row(row: Mapping[str, object]) -> bool:
    """Native rows carry Gemini's ``key`` (and a ``response`` or an ``error``) where OpenAI-shaped rows
    carry ``custom_id``, so the shape is known even when the first row failed."""
    return "key" in row and "custom_id" not in row


def _json_object(line: bytes) -> Mapping[str, object] | None:
    if not line.lstrip().startswith(b"{"):
        return None
    try:
        return _JSON_OBJECT.validate_python(json.loads(line))
    except (ValueError, RecursionError):
        return None


def _gemini_batch_request_row(
    openai_row: Mapping[str, object], batch_model: str | None, deployment_drop_params: object
) -> Mapping[str, object]:
    """The ``{key, request}`` row Gemini runs for one OpenAI ``/v1/chat/completions`` batch row, built with the
    same translation as a chat completion. The deployment model wins over ``body.model``, because Gemini runs
    every row on the model of the batch job."""
    body: Final = _JSON_OBJECT.validate_python(openai_row["body"])
    body_model: Final = body.get("model")
    model: Final = batch_model or (body_model if isinstance(body_model, str) else "")
    messages: Final = map_developer_role_to_system_role(
        body.get("messages", [])  # pyright: ignore[reportArgumentType]  # validating would drop message fields
    )
    optional_params: Final = _JSON_OBJECT.validate_python(
        get_optional_params(
            model=model,
            custom_llm_provider="gemini",
            drop_params=drops_unsupported_params(body.get("drop_params", deployment_drop_params)),
            messages=messages,
            **{  # pyright: ignore[reportArgumentType]  # batch JSONL bodies are untyped; get_optional_params checks each param as chat does
                key: value for key, value in body.items() if key not in _ROW_PARAMS_NOT_SENT_TO_GEMINI
            },
        )
    )
    request: Final = transform_request_body(
        messages=messages,
        model=model,
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params={},
        cached_content=None,
    )
    return {"key": str(openai_row["custom_id"]), "request": request}


def _gemini_batch_input_line(
    lineno: int, line: bytes, batch_model: str | None, deployment_drop_params: object
) -> bytes:
    row: Final = _json_object(line)
    if row is None or row.get("url") != _CHAT_COMPLETIONS_BATCH_URL or "custom_id" not in row:
        return line.rstrip()
    try:
        return json.dumps(_gemini_batch_request_row(row, batch_model, deployment_drop_params)).encode("utf-8")
    except Exception as e:  # noqa: BLE001  # any translation failure is a bad input row, reported with its line
        raise GeminiError(
            status_code=400,
            message=f"Batch input file line {lineno} could not be converted to a Gemini request: {e}",
        ) from e


def _gemini_batch_input_lines(content: bytes, litellm_params: Mapping[str, object]) -> Iterator[bytes]:
    deployment_model: Final = litellm_params.get("model")
    batch_model: Final = (
        deployment_model.removeprefix("gemini/") if isinstance(deployment_model, str) and deployment_model else None
    )
    deployment_drop_params: Final = litellm_params.get("drop_params")
    for lineno, line in enumerate(io.BytesIO(content), start=1):
        if line.strip():
            yield _gemini_batch_input_line(lineno, line, batch_model, deployment_drop_params)


def _openai_batch_output_row_from_native(
    native_row: Mapping[str, object], context: BatchOutputRowTransformContext
) -> OpenAIBatchOutputRow:
    custom_id: Final = str(native_row["key"])
    response: Final = as_str_mapping(native_row.get("response"))
    if response is None:
        failure: Final = as_str_mapping(native_row.get("error")) or {}
        return openai_batch_output_row(
            custom_id=custom_id,
            error_code=str(failure.get("code", "gemini_error")),
            error_message=str(failure.get("message", "")),
        )
    try:
        model_response: Final = context.vertex_gemini_config.transform_google_generate_content_to_openai_model_response(
            completion_response=dict(response),
            model_response=ModelResponse(),
            model=str(response.get("modelVersion", "")),
            logging_obj=context.logging_obj,
            raw_response=context.mock_httpx_response,
        )
    except Exception as e:  # noqa: BLE001  # one malformed row must fail alone, not the whole output file
        return openai_batch_output_row(
            custom_id=custom_id,
            error_code="transformation_error",
            error_message=f"Failed to transform response: {e}",
        )
    return openai_batch_output_row(custom_id=custom_id, body=model_response.model_dump())


def _openai_batch_output_line(line: bytes, context: BatchOutputRowTransformContext) -> bytes:
    """One line of an OpenAI batch output file for a line of a Gemini batch output file. Any other line, such as
    one Gemini already wrote OpenAI-shaped, is returned unchanged."""
    row: Final = _json_object(line)
    if row is None or not is_native_gemini_batch_output_row(row):
        return line
    return json.dumps(_openai_batch_output_row_from_native(row, context)).encode("utf-8")


async def _aiter_openai_batch_output(chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    context: Final = new_batch_output_row_transform_context()
    pending: bytes = b""  # rebind-ok: carries the partial trailing line over to the next chunk
    async with aclosing(chunks):
        async for chunk in chunks:
            *complete_lines, pending = (pending + chunk).split(_JSONL_NEWLINE)
            for line in complete_lines:
                yield _openai_batch_output_line(line, context) + _JSONL_NEWLINE
    if pending:
        yield _openai_batch_output_line(pending, context)


class _ReplaceableText(Protocol):
    def replace(self, old: str, new: str, /) -> str: ...


def _gemini_timestamp(rfc3339: object) -> int:
    """Gemini stamps files with up to nine fractional digits and a Z suffix; Python 3.10's
    fromisoformat takes neither, so trim the fraction to microseconds and spell out UTC."""
    parts: Final = _RFC3339_PARTS.fullmatch(
        require_attributes(rfc3339, _ReplaceableText, "replace").replace("Z", "+00:00")
    )
    if parts is None:
        raise ValueError(f"Unrecognized Gemini timestamp: {rfc3339}")
    seconds, fraction, offset = parts.groups()
    return int(datetime.fromisoformat(f"{seconds}{(fraction or '')[:7]}{offset}").timestamp())


def gemini_file_name(file_id: str) -> str:
    """
    Normalize a file identifier into Gemini's `files/{id}` resource name.

    Supports:
    - `abc123`
    - `files/abc123`
    - `https://generativelanguage.googleapis.com/v1beta/files/abc123`
    """
    bare_id: Final = gemini_file_id(file_id)
    return f"files/{encode_url_path_segment(bare_id, field_name='file_id')}"


class GoogleAIStudioFilesHandler(GeminiModelInfo, BaseFilesConfig):
    def __init__(self):
        pass

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.GEMINI

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        """
        Validate environment and add Gemini API key to headers.
        Google AI Studio uses x-goog-api-key header for authentication.
        """
        resolved_api_key: Final = self.get_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY is required for Google AI Studio file operations")

        headers["x-goog-api-key"] = resolved_api_key
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        endpoint: Final = "upload/v1beta/files"
        api_base = self.get_api_base(api_base)
        if not api_base:
            raise ValueError("api_base is required")

        # Get API key from multiple sources
        final_api_key: Final = api_key or litellm_params.get("api_key") or self.get_api_key()
        if not final_api_key:
            raise ValueError("api_key is required")

        return join_api_base(api_base, endpoint)

    def get_supported_openai_params(self, model: str) -> list[OpenAICreateFileRequestOptionalParams]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: Mapping[str, object],
    ) -> dict:
        """
        Transform the OpenAI-style file creation request into Gemini's format

        Returns:
            dict: Contains both request data and headers for the two-step upload
        """
        # Extract the file information
        file_data: Final = create_file_data.get("file")
        if file_data is None:
            raise ValueError("File data is required")

        # Use the common utility function to extract file data
        extracted_data: Final = extract_file_data(file_data)
        upload_content: Final = (
            _JSONL_NEWLINE.join(_gemini_batch_input_lines(extracted_data["content"], litellm_params))
            if FilesAPIUtils.is_batch_jsonl_request(
                create_file_data=create_file_data, content_type=extracted_data["content_type"]
            )
            else extracted_data["content"]
        )

        # Get file size
        file_size: Final = len(upload_content)

        # Step 1: Initial resumable upload request
        headers: Final = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": extracted_data["content_type"],
            "Content-Type": "application/json",
        }
        headers.update(extracted_data["headers"])  # Add any custom headers

        # Initial metadata request body
        initial_data: Final = {"file": {"display_name": extracted_data["filename"] or str(int(time.time()))}}

        # Step 2: Actual file upload data
        upload_headers: Final = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        return {
            "initial_request": {"headers": headers, "data": initial_data},
            "upload_request": {
                "headers": upload_headers,
                "data": upload_content,
            },
        }

    def transform_create_file_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform Gemini's file upload response into OpenAI-style FileObject
        """
        try:
            response_object: Final = subscript(_DECODED_JSON.validate_python(raw_response.json()), "file")

            # Extract file information from Gemini response

            return OpenAIFileObject.model_validate(
                {
                    "id": subscript(response_object, "uri"),  # Gemini uses URI as identifier
                    "bytes": int_of(subscript(response_object, "sizeBytes")),  # Gemini doesn't return file size
                    "created_at": _gemini_timestamp(subscript(response_object, "createTime")),
                    "filename": subscript(response_object, "displayName"),
                    "object": "file",
                    "purpose": "user_data",  # Default to assistants as that's the main use case
                    "status": "uploaded",
                    "status_details": None,
                }
            )
        except Exception as e:
            verbose_logger.exception("Error parsing file upload response: %s", e)
            raise ValueError(f"Error parsing file upload response: {e}")

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict]:
        """
        Get the URL to retrieve a file from Google AI Studio.

        Endpoint:
        GET https://generativelanguage.googleapis.com/v1beta/{name=files/*}

        The URL should look like:
        https://generativelanguage.googleapis.com/v1beta/files/{file_id}?key=API_KEY

        We expect file_id to be just the file identifier (e.g., files/abc123 or abc123)
        as returned by the upload response. (If it's a full URL, extract the file name.)
        """
        api_key: Final = litellm_params.get("api_key") or self.get_api_key()
        if not api_key:
            raise ValueError("api_key is required")

        file_part: Final = self._normalize_gemini_file_id(file_id)

        configured_api_base: Final = litellm_params.get("api_base")
        api_base: Final = (
            self.get_api_base(configured_api_base if isinstance(configured_api_base, str) else None)
            or "https://generativelanguage.googleapis.com"
        )

        url: Final = join_api_base(api_base, f"v1beta/{file_part}")

        # API key is passed via x-goog-api-key header (set in validate_environment)
        return url, {}

    def _normalize_gemini_file_id(self, file_id: str) -> str:
        return gemini_file_name(file_id)

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform Gemini's file retrieval response into OpenAI-style FileObject
        """
        try:
            verbose_logger.debug("Retrieve file response: %s", raw_response.text)
            decoded_response: Final = _DECODED_JSON.validate_python(raw_response.json())
            verbose_logger.debug("Response JSON: %s", decoded_response)
            response_json: Final = require_get(decoded_response)
            # Map Gemini state to OpenAI status
            gemini_state: Final = response_json.get("state", "STATE_UNSPECIFIED")
            # Explicitly type status as the Literal union
            if gemini_state == "ACTIVE":
                status: Literal["uploaded", "processed", "error"] = "processed"
            elif gemini_state == "FAILED":
                status = "error"
            else:
                status = "uploaded"

            return OpenAIFileObject.model_validate(
                {
                    "id": response_json.get("uri", ""),
                    "bytes": int_of(response_json.get("sizeBytes", 0)),
                    "created_at": _gemini_timestamp(subscript(response_json, "createTime")),
                    "filename": response_json.get("displayName", ""),
                    "object": "file",
                    "purpose": "user_data",
                    "status": status,
                    "status_details": (str(response_json.get("error", "")) if gemini_state == "FAILED" else None),
                }
            )
        except Exception as e:
            verbose_logger.exception("Error parsing file retrieve response: %s", e)
            raise ValueError(f"Error parsing file retrieve response: {e}")

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """
        Transform delete file request for Google AI Studio.

        Args:
            file_id: The file URI (e.g., "files/abc123" or full URI)
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters containing api_key

        Returns:
            tuple[str, dict]: (url, params) for the DELETE request
        """
        api_base: Final = self.get_api_base(litellm_params.get("api_base"))
        if not api_base:
            raise ValueError("api_base is required")

        # Get API key from multiple sources (same pattern as get_complete_url)
        api_key: Final = litellm_params.get("api_key") or self.get_api_key()
        if not api_key:
            raise ValueError("api_key is required")

        # Normalize and encode the file name before interpolating it into the URL.
        file_name: Final = self._normalize_gemini_file_id(file_id)

        # Construct the delete URL
        url: Final = join_api_base(api_base, f"v1beta/{file_name}")

        # Add API key as header (Google AI Studio uses x-goog-api-key header)
        params: Final[dict[str, str]] = {}

        return url, params

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        """
        Transform Gemini's file delete response into OpenAI-style FileDeleted.

        Google AI Studio returns an empty JSON object {} on successful deletion.
        """
        try:
            # Google AI Studio returns {} on successful deletion
            if raw_response.status_code == 200:
                # Extract file ID from the request URL if possible
                file_id = "deleted"
                if hasattr(raw_response, "request") and raw_response.request:
                    url: Final = str(raw_response.request.url)
                    if "/files/" in url:
                        file_id = url.split("/files/")[-1].split("?")[0]
                        # Add the files/ prefix if not present
                        if not file_id.startswith("files/"):
                            file_id = f"files/{file_id}"

                return FileDeleted(id=file_id, deleted=True, object="file")
            else:
                raise ValueError(f"Failed to delete file: {raw_response.text}")
        except Exception as e:
            verbose_logger.exception("Error parsing file delete response: %s", e)
            raise ValueError(f"Error parsing file delete response: {e}")

    def transform_list_files_request(
        self,
        purpose: str | None,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("GoogleAIStudioFilesHandler does not support file listing")

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> list[OpenAIFileObject]:
        raise NotImplementedError("GoogleAIStudioFilesHandler does not support file listing")

    def transform_file_content_request(
        self,
        file_content_request: FileContentRequest,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseFilesConfig signature
        file_id: Final = file_content_request.get("file_id")
        if file_id is None:
            raise ValueError("file_id is required to download file content")
        api_base: Final = litellm_params.get("api_base")
        url: Final = join_api_base(
            self.get_api_base(api_base if isinstance(api_base, str) else None),
            f"download/v1beta/{gemini_file_name(file_id)}:download",
        )
        return url, {"alt": "media"}

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        content: Final = raw_response.content
        context: Final = new_batch_output_row_transform_context()
        transformed: Final = _JSONL_NEWLINE.join(
            _openai_batch_output_line(line, context) for line in content.split(_JSONL_NEWLINE)
        )
        if transformed == content:
            return HttpxBinaryResponseContent(response=raw_response)
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=raw_response.status_code,
                content=transformed,
                headers={**raw_response.headers, "content-length": str(len(transformed))},
                request=raw_response.request,
            )
        )

    async def transform_file_content_stream(
        self,
        *,
        stream_iterator: AsyncGenerator[bytes, None],
        headers: Mapping[str, str],
        request_url: str,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> FileContentStreamingResult:
        return FileContentStreamingResult(
            stream_iterator=_aiter_openai_batch_output(stream_iterator),
            headers=MappingProxyType({key: value for key, value in headers.items() if key.lower() != "content-length"}),
        )

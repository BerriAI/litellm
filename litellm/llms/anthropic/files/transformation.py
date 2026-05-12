"""
Anthropic Files API transformation config.

Implements BaseFilesConfig for Anthropic's Files API (beta).
Reference: https://docs.anthropic.com/en/docs/build-with-claude/files

Anthropic Files API endpoints:
- POST   /v1/files              - Upload a file
- GET    /v1/files              - List files
- GET    /v1/files/{file_id}    - Retrieve file metadata
- DELETE /v1/files/{file_id}    - Delete a file
- GET    /v1/files/{file_id}/content - Download file content
"""

import calendar
import time
from typing import Any, Dict, List, Optional, Union, cast

import httpx
from openai.types.file_deleted import FileDeleted

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import (
    CreateFileRequest,
    FileContentRequest,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
)
from litellm.types.utils import LlmProviders

from ..common_utils import AnthropicError, AnthropicModelInfo

ANTHROPIC_FILES_API_BASE = "https://api.anthropic.com"
ANTHROPIC_FILES_BETA_HEADER = "files-api-2025-04-14"


class AnthropicFilesConfig(BaseFilesConfig):
    """
    Transformation config for Anthropic Files API.

    Anthropic uses:
    - x-api-key header for authentication
    - anthropic-beta: files-api-2025-04-14 header
    - multipart/form-data for file uploads
    - purpose="messages" (Anthropic-specific, not for batches/fine-tuning)
    """

    def __init__(self):
        pass

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ANTHROPIC

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = AnthropicModelInfo.get_api_base(api_base) or ANTHROPIC_FILES_API_BASE
        return f"{api_base.rstrip('/')}/v1/files"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return AnthropicError(
            status_code=status_code,
            message=error_message,
            headers=(
                cast(httpx.Headers, headers) if isinstance(headers, dict) else headers
            ),
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        auth_header = AnthropicModelInfo.get_auth_header(api_key)
        if auth_header is None:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN environment variable or pass api_key parameter."
            )
        headers.update(
            {
                **auth_header,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": ANTHROPIC_FILES_BETA_HEADER,
            }
        )
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        return ["purpose"]

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
        litellm_params: dict,
    ) -> dict:
        """
        Transform to multipart form data for Anthropic file upload.

        Anthropic expects: POST /v1/files with multipart form-data
        - file: the file content
        - purpose: "messages" (defaults to "messages" if not provided)
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("File data is required")

        extracted = extract_file_data(file_data)
        filename = extracted["filename"] or f"file_{int(time.time())}"
        content = extracted["content"]
        content_type = extracted.get("content_type", "application/octet-stream")

        purpose = create_file_data.get("purpose", "messages")

        return {
            "file": (filename, content, content_type),
            "purpose": (None, purpose),
        }

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform Anthropic file response to OpenAI format.

        Anthropic response:
        {
            "id": "file-xxx",
            "type": "file",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 12345,
            "created_at": "2025-01-01T00:00:00Z"
        }
        """
        response_json = raw_response.json()
        return self._parse_anthropic_file(response_json)

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        api_base = (
            AnthropicModelInfo.get_api_base(litellm_params.get("api_base"))
            or ANTHROPIC_FILES_API_BASE
        )
        encoded_file_id = encode_url_path_segment(file_id, field_name="file_id")
        return f"{api_base.rstrip('/')}/v1/files/{encoded_file_id}", {}

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        response_json = raw_response.json()
        return self._parse_anthropic_file(response_json)

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        api_base = (
            AnthropicModelInfo.get_api_base(litellm_params.get("api_base"))
            or ANTHROPIC_FILES_API_BASE
        )
        encoded_file_id = encode_url_path_segment(file_id, field_name="file_id")
        return f"{api_base.rstrip('/')}/v1/files/{encoded_file_id}", {}

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        response_json = raw_response.json()
        file_id = response_json.get("id", "")
        return FileDeleted(
            id=file_id,
            deleted=True,
            object="file",
        )

    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        api_base = (
            AnthropicModelInfo.get_api_base(litellm_params.get("api_base"))
            or ANTHROPIC_FILES_API_BASE
        )
        url = f"{api_base.rstrip('/')}/v1/files"
        params: Dict[str, Any] = {}
        if purpose:
            params["purpose"] = purpose
        return url, params

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        """
        Anthropic list response:
        {
            "data": [...],
            "has_more": false,
            "first_id": "...",
            "last_id": "..."
        }
        """
        response_json = raw_response.json()
        files_data = response_json.get("data", [])
        return [self._parse_anthropic_file(f) for f in files_data]

    def transform_file_content_request(
        self,
        file_content_request: FileContentRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        file_id = file_content_request.get("file_id")
        api_base = (
            AnthropicModelInfo.get_api_base(litellm_params.get("api_base"))
            or ANTHROPIC_FILES_API_BASE
        )
        encoded_file_id = encode_url_path_segment(file_id, field_name="file_id")
        return f"{api_base.rstrip('/')}/v1/files/{encoded_file_id}/content", {}

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        return HttpxBinaryResponseContent(response=raw_response)

    @staticmethod
    def _parse_anthropic_file(file_data: dict) -> OpenAIFileObject:
        """Parse Anthropic file object into OpenAI format."""
        created_at_str = file_data.get("created_at", "")
        if created_at_str:
            try:
                created_at = int(
                    calendar.timegm(
                        time.strptime(
                            created_at_str.replace("Z", "+00:00")[:19],
                            "%Y-%m-%dT%H:%M:%S",
                        )
                    )
                )
            except (ValueError, TypeError):
                created_at = int(time.time())
        else:
            created_at = int(time.time())

        return OpenAIFileObject(
            id=file_data.get("id", ""),
            bytes=file_data.get("size_bytes", file_data.get("bytes", 0)),
            created_at=created_at,
            filename=file_data.get("filename", ""),
            object="file",
            purpose=file_data.get("purpose", "messages"),
            status="uploaded",
            status_details=None,
        )

"""
Manus Files API implementation.

Manus has an OpenAI-compatible Files API with some differences:
- Uses API_KEY header instead of Authorization: Bearer
- File upload is a two-step process:
  1. Create file record to get upload URL
  2. Upload file content to the upload URL

Reference: https://open.manus.im/docs/openai-compatibility#file-management
"""

import time
from typing import Any, Dict, List, Optional, Union

import httpx
from openai.types.file_deleted import FileDeleted

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.files import TwoStepFileUploadConfig, TwoStepFileUploadRequest
from litellm.types.llms.openai import (
    CreateFileRequest,
    FileContentRequest,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
)
from litellm.types.utils import LlmProviders

MANUS_API_BASE = "https://api.manus.im"


class ManusFilesConfig(BaseFilesConfig):
    """
    Configuration for Manus Files API.

    Manus uses:
    - API_KEY header for authentication (not Authorization: Bearer)
    - Two-step file upload process
    - Content-Type: application/json for all requests

    Reference: https://open.manus.im/docs/openai-compatibility#file-management
    """

    def __init__(self):
        pass

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.MANUS

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
        """
        Validate environment and set up headers for Manus API.

        Manus uses API_KEY header instead of Authorization: Bearer.
        For file uploads, don't set Content-Type - httpx will set it for multipart.
        """
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("MANUS_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "Manus API key is required. Set MANUS_API_KEY environment variable or pass api_key parameter."
            )

        # Manus uses API_KEY header, not Authorization: Bearer
        # Manus requires Content-Type: application/json for all requests (even GET)
        headers.update(
            {
                "API_KEY": api_key,
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        """
        Return supported OpenAI file creation parameters for Manus.
        Manus supports the standard 'purpose' parameter.
        """
        return ["purpose"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Manus-specific parameters.
        Manus is OpenAI-compatible, so no special mapping needed.
        """
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for Manus Files API endpoint.

        Returns:
            str: The full URL for the Manus /v1/files endpoint
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("MANUS_API_BASE")
            or MANUS_API_BASE
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Manus API uses /v1/files endpoint
        if api_base.endswith("/v1"):
            return f"{api_base}/files"
        return f"{api_base}/v1/files"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        """
        Return the appropriate error class for Manus API errors.
        Uses OpenAIError since Manus is OpenAI-compatible.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> TwoStepFileUploadConfig:
        """
        Transform OpenAI-style file creation request into Manus's two-step format.

        Manus API spec (https://open.manus.im/docs/openai-compatibility#file-management):
        1. POST /v1/files with JSON {"filename": "..."} → returns {"id": "...", "upload_url": "..."}
        2. PUT to upload_url with raw file content
        """
        # Extract file data
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("File data is required")

        extracted_data = extract_file_data(file_data)
        filename = extracted_data["filename"] or f"file_{int(time.time())}"
        content = extracted_data["content"]

        # Get API base URL
        api_base = self.get_complete_url(
            api_base=litellm_params.get("api_base"),
            api_key=litellm_params.get("api_key"),
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        
        # Get API key
        api_key = (
            litellm_params.get("api_key")
            or litellm.api_key
            or get_secret_str("MANUS_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "Manus API key is required. Set MANUS_API_KEY environment variable or pass api_key parameter."
            )

        # Build typed two-step upload config
        return TwoStepFileUploadConfig(
            initial_request=TwoStepFileUploadRequest(
                method="POST",
                url=api_base,
                headers={
                    "API_KEY": api_key,
                    "Content-Type": "application/json",
                },
                data={"filename": filename},
            ),
            upload_request=TwoStepFileUploadRequest(
                method="PUT",
                url="",  # Will be populated from initial_request response
                headers={},
                data=content,
            ),
            upload_url_location="body",
            upload_url_key="upload_url",
        )

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform Manus's file upload response into OpenAI-style FileObject.

        For two-step uploads, the handler stores the initial response in litellm_params.
        We need to return the file object from the initial POST, not the final PUT.

        Manus initial response format:
        {
            "id": "file-abc123xyz",
            "object": "file",
            "filename": "document.pdf",
            "status": "pending",
            "upload_url": "https://...",
            "upload_expires_at": "...",
            "created_at": "..."
        }
        """
        try:
            # For two-step uploads, get the initial response from litellm_params
            initial_response_data = litellm_params.get("initial_file_response")
            if initial_response_data:
                response_json = initial_response_data
            else:
                # Log raw response for debugging
                verbose_logger.debug(f"Manus raw response text: {raw_response.text}")
                response_json = raw_response.json()

            verbose_logger.debug(f"Manus file response: {response_json}")

            # Parse created_at timestamp
            created_at_str = response_json.get("created_at", "")
            if created_at_str:
                try:
                    # Try parsing ISO format
                    created_at = int(
                        time.mktime(
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
                id=response_json.get("id", ""),
                bytes=response_json.get("bytes", 0),
                created_at=created_at,
                filename=response_json.get("filename", ""),
                object="file",
                purpose=response_json.get("purpose", "assistants"),
                status="uploaded",  # After successful upload, status is uploaded
                status_details=response_json.get("status_details"),
            )
        except Exception as e:
            verbose_logger.exception(f"Error parsing Manus file response: {str(e)}")
            raise ValueError(f"Error parsing Manus file response: {str(e)}")

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Get URL and params for retrieving a file."""
        api_base = self.get_complete_url(
            api_base=litellm_params.get("api_base"),
            api_key=litellm_params.get("api_key"),
            model="",
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        return f"{api_base}/{file_id}", {}

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """Transform retrieve file response."""
        return self.transform_create_file_response(
            model=None,
            raw_response=raw_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Get URL and params for deleting a file."""
        api_base = self.get_complete_url(
            api_base=litellm_params.get("api_base"),
            api_key=litellm_params.get("api_key"),
            model="",
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        return f"{api_base}/{file_id}", {}

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        """Transform delete file response."""
        response_json = raw_response.json()
        return FileDeleted(**response_json)

    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Get URL and params for listing files."""
        api_base = self.get_complete_url(
            api_base=litellm_params.get("api_base"),
            api_key=litellm_params.get("api_key"),
            model="",
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        params = {}
        if purpose:
            params["purpose"] = purpose
        return api_base, params

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        """Transform list files response."""
        response_json = raw_response.json()
        files_data = response_json.get("data", [])
        return [self._parse_file_dict(f) for f in files_data]

    def _parse_file_dict(self, file_dict: Dict[str, Any]) -> OpenAIFileObject:
        """Parse a file dict into OpenAIFileObject."""
        created_at_str = file_dict.get("created_at", "")
        if created_at_str:
            try:
                created_at = int(
                    time.mktime(
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
            id=file_dict.get("id", ""),
            bytes=file_dict.get("bytes", 0),
            created_at=created_at,
            filename=file_dict.get("filename", ""),
            object="file",
            purpose=file_dict.get("purpose", "assistants"),
            status=file_dict.get("status", "uploaded"),
            status_details=file_dict.get("status_details"),
        )

    def transform_file_content_request(
        self,
        file_content_request: FileContentRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """Get URL and params for retrieving file content."""
        file_id = file_content_request.get("file_id")
        api_base = self.get_complete_url(
            api_base=litellm_params.get("api_base"),
            api_key=litellm_params.get("api_key"),
            model="",
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        return f"{api_base}/{file_id}/content", {}

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        """Transform file content response."""
        return HttpxBinaryResponseContent(response=raw_response)


"""
Anthropic Files API Configuration

Handles transformation between OpenAI-style file API and Anthropic's Files API using HTTP requests.
"""

import time
from typing import Dict, List, Optional, Union

from httpx import Headers, Response

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
)
from litellm.types.utils import LlmProviders

from ..common_utils import AnthropicError, AnthropicModelInfo

anthropic_model_info = AnthropicModelInfo()


class AnthropicFilesConfig(BaseFilesConfig):
    """
    Configuration for Anthropic Files API using HTTP requests.

    Transforms OpenAI-style file API requests to Anthropic's Files API format.
    API Reference: https://docs.anthropic.com/en/docs/build-with-claude/files
    """

    def __init__(self):
        super().__init__()

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
        """
        Get the complete URL for Anthropic Files API requests.

        Returns:
            https://api.anthropic.com/v1/files
        """
        # Use anthropic_model_info helper to get the API base
        api_base = anthropic_model_info.get_api_base(api_base)

        # For file operations, use /v1/files endpoint
        url = f"{api_base}/v1/files"

        verbose_logger.debug(f"Anthropic Files API URL: {url}")
        return url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Validate environment and construct Anthropic-specific headers.

        Anthropic Files API requires:
        - x-api-key: API key for authentication
        - anthropic-version: API version (default: 2023-06-01)
        - anthropic-beta: files-api-2025-04-14
        """
        # Get API key
        api_key = anthropic_model_info.get_api_key(api_key)

        if api_key is None:
            raise ValueError(
                "Missing Anthropic API Key. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Construct Anthropic-specific headers for Files API
        anthropic_headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "files-api-2025-04-14",
        }

        # Merge with existing headers
        headers = {**headers, **anthropic_headers}

        verbose_logger.debug(f"Anthropic Files API headers: {headers}")
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        """
        Anthropic Files API doesn't support additional OpenAI parameters.

        Only supports:
        - file: The file to upload (required)
        - No purpose parameter needed
        """
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Anthropic format.

        Anthropic Files API is simple - just takes the file.
        No need to map any additional parameters.
        """
        return optional_params

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[dict, bytes, str]:
        """
        Transform OpenAI-style file creation request to Anthropic format.

        Anthropic expects multipart/form-data with just the file.

        Args:
            model: Model name (not used for file uploads)
            create_file_data: OpenAI-style file creation request
            optional_params: Additional parameters
            litellm_params: LiteLLM-specific parameters

        Returns:
            dict: Formatted request with method, url, headers, and files
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("File data is required for Anthropic Files API")

        # Extract file information using common utility
        extracted_data = extract_file_data(file_data)

        verbose_logger.debug(
            f"Anthropic file upload - filename: {extracted_data.get('filename')}, "
            f"content_type: {extracted_data.get('content_type')}"
        )

        # Get complete URL for upload
        api_base = self.get_complete_url(
            api_base=optional_params.get("api_base"),
            api_key=optional_params.get("api_key"),
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Store for use in response transformation
        litellm_params["upload_url"] = api_base

        # Return dict with method and files for multipart upload
        # This tells the HTTP handler to use httpx's files parameter
        return {
            "method": "POST",
            "url": api_base,
            "headers": {},  # Additional headers already set in validate_environment
            "files": {
                "file": (
                    extracted_data.get("filename", "file"),
                    extracted_data["content"],
                    extracted_data.get("content_type", "application/octet-stream"),
                )
            },
        }

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform Anthropic file upload response to OpenAI format.

        Anthropic response format:
        {
          "id": "file_011CNha8iCJcU1wXNR6q4V8w",
          "type": "file",
          "filename": "document.pdf",
          "mime_type": "application/pdf",
          "size_bytes": 1024000,
          "created_at": "2025-01-01T00:00:00Z",
          "downloadable": false
        }

        OpenAI format:
        {
          "id": "file-abc123",
          "object": "file",
          "bytes": 1024,
          "created_at": 1234567890,
          "filename": "file.pdf",
          "purpose": "assistants",
          "status": "processed"
        }
        """
        try:
            response_json = raw_response.json()

            verbose_logger.debug(f"Anthropic file upload response: {response_json}")

            # Parse created_at timestamp
            created_at_str = response_json.get("created_at", "")
            try:
                # Anthropic returns ISO 8601 format, convert to Unix timestamp
                from dateutil import parser
                created_at = int(parser.parse(created_at_str).timestamp())
            except Exception:
                # Fallback to current time if parsing fails
                created_at = int(time.time())

            # Transform to OpenAI format
            return OpenAIFileObject(
                id=response_json.get("id", ""),
                object="file",
                bytes=response_json.get("size_bytes", 0),
                created_at=created_at,
                filename=response_json.get("filename", ""),
                purpose="assistants",  # Anthropic doesn't have purpose concept
                status="processed",  # Anthropic files are immediately available
            )

        except Exception as e:
            verbose_logger.exception(
                f"Error transforming Anthropic file upload response: {str(e)}"
            )
            raise AnthropicError(
                status_code=raw_response.status_code,
                message=f"Failed to parse Anthropic file upload response: {str(e)}",
                headers=raw_response.headers,
            )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate error class for Anthropic errors.
        """
        return AnthropicError(
            status_code=status_code, message=error_message, headers=headers
        )

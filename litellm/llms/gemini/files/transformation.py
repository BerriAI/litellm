"""
Supports writing files to Google AI Studio Files API.

For vertex ai, check out the vertex_ai/files/handler.py file.
"""
import time
from typing import List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.gemini import GeminiCreateFilesResponseObject
from litellm.types.llms.openai import (
    CreateFileRequest,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
)
from litellm.types.utils import LlmProviders

from ..common_utils import GeminiModelInfo


class GoogleAIStudioFilesHandler(GeminiModelInfo, BaseFilesConfig):
    def __init__(self):
        pass

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.GEMINI

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
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        endpoint = "upload/v1beta/files"
        api_base = self.get_api_base(api_base)
        if not api_base:
            raise ValueError("api_base is required")

        if not api_key:
            raise ValueError("api_key is required")

        url = "{}/{}?key={}".format(api_base, endpoint, api_key)
        return url

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

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict:
        """
        Transform the OpenAI-style file creation request into Gemini's format

        Returns:
            dict: Contains both request data and headers for the two-step upload
        """
        # Extract the file information
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("File data is required")

        # Use the common utility function to extract file data
        extracted_data = extract_file_data(file_data)

        # Get file size
        file_size = len(extracted_data["content"])

        # Step 1: Initial resumable upload request
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": extracted_data["content_type"],
            "Content-Type": "application/json",
        }
        headers.update(extracted_data["headers"])  # Add any custom headers

        # Initial metadata request body
        initial_data = {
            "file": {
                "display_name": extracted_data["filename"] or str(int(time.time()))
            }
        }

        # Step 2: Actual file upload data
        upload_headers = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        return {
            "initial_request": {"headers": headers, "data": initial_data},
            "upload_request": {
                "headers": upload_headers,
                "data": extracted_data["content"],
            },
        }

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform Gemini's file upload response into OpenAI-style FileObject
        """
        try:
            response_json = raw_response.json()

            response_object = GeminiCreateFilesResponseObject(
                **response_json.get("file", {})  # type: ignore
            )

            # Extract file information from Gemini response

            return OpenAIFileObject(
                id=response_object["uri"],  # Gemini uses URI as identifier
                bytes=int(
                    response_object["sizeBytes"]
                ),  # Gemini doesn't return file size
                created_at=int(
                    time.mktime(
                        time.strptime(
                            response_object["createTime"].replace("Z", "+00:00"),
                            "%Y-%m-%dT%H:%M:%S.%f%z",
                        )
                    )
                ),
                filename=response_object["displayName"],
                object="file",
                purpose="user_data",  # Default to assistants as that's the main use case
                status="uploaded",
                status_details=None,
            )
        except Exception as e:
            verbose_logger.exception(f"Error parsing file upload response: {str(e)}")
            raise ValueError(f"Error parsing file upload response: {str(e)}")

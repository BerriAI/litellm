"""
Supports writing files to Google AI Studio Files API.

For vertex ai, check out the vertex_ai/files/handler.py file.
"""
import time
from typing import List, Mapping, Optional

import httpx

from litellm._logging import verbose_logger
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

        # Parse the file_data based on its type
        filename = None
        file_content = None
        content_type = None
        file_headers: Mapping[str, str] = {}

        if isinstance(file_data, tuple):
            if len(file_data) == 2:
                filename, file_content = file_data
            elif len(file_data) == 3:
                filename, file_content, content_type = file_data
            elif len(file_data) == 4:
                filename, file_content, content_type, file_headers = file_data
        else:
            file_content = file_data

        # Handle the file content based on its type
        import io
        from os import PathLike

        # Convert content to bytes
        if isinstance(file_content, (str, PathLike)):
            # If it's a path, open and read the file
            with open(file_content, "rb") as f:
                content = f.read()
        elif isinstance(file_content, io.IOBase):
            # If it's a file-like object
            content = file_content.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
        elif isinstance(file_content, bytes):
            content = file_content
        else:
            raise ValueError(f"Unsupported file content type: {type(file_content)}")

        # Get file size
        file_size = len(content)

        # Use provided content type or guess based on filename
        if not content_type:
            import mimetypes

            content_type = (
                mimetypes.guess_type(filename)[0]
                if filename
                else "application/octet-stream"
            )

        # Step 1: Initial resumable upload request
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": content_type,
            "Content-Type": "application/json",
        }
        headers.update(file_headers)  # Add any custom headers

        # Initial metadata request body
        initial_data = {"file": {"display_name": filename or str(int(time.time()))}}

        # Step 2: Actual file upload data
        upload_headers = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        return {
            "initial_request": {"headers": headers, "data": initial_data},
            "upload_request": {"headers": upload_headers, "data": content},
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

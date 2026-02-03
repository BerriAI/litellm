"""
Supports writing files to Google AI Studio Files API.

For vertex ai, check out the vertex_ai/files/handler.py file.
"""
import time
from typing import Any, List, Literal, Optional

import httpx
from openai.types.file_deleted import FileDeleted

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.gemini import GeminiCreateFilesResponseObject
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    HttpxBinaryResponseContent,
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

    def validate_environment(
        self,
        headers: dict[Any, Any],
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict[Any, Any],
        litellm_params: dict[Any, Any],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict[Any, Any]:
        """
        Validate environment and add Gemini API key to headers.
        Google AI Studio uses x-goog-api-key header for authentication.
        """
        resolved_api_key = self.get_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY is required for Google AI Studio file operations")
        
        headers["x-goog-api-key"] = resolved_api_key
        return headers

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

        # Get API key from multiple sources
        final_api_key = api_key or litellm_params.get("api_key") or self.get_api_key()
        if not final_api_key:
            raise ValueError("api_key is required")

        url = "{}/{}?key={}".format(api_base, endpoint, final_api_key)
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

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        """
        Get the URL to retrieve a file from Google AI Studio.
        
        We expect file_id to be the URI (e.g. https://generativelanguage.googleapis.com/v1beta/files/...)
        as returned by the upload response.
        """
        api_key = litellm_params.get("api_key") or self.get_api_key()
        if not api_key:
            raise ValueError("api_key is required")

        if file_id.startswith("http"):
            url = "{}?key={}".format(file_id, api_key)
        else:
            # Fallback for just file name (files/...)
            api_base = self.get_api_base(litellm_params.get("api_base")) or "https://generativelanguage.googleapis.com"
            api_base = api_base.rstrip("/")
            url = "{}/v1beta/{}?key={}".format(api_base, file_id, api_key)

        # Return empty params dict - API key is already in URL, no query params needed
        return url, {}

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
            response_json = raw_response.json()
            
            # Map Gemini state to OpenAI status
            gemini_state = response_json.get("state", "STATE_UNSPECIFIED")
            # Explicitly type status as the Literal union
            if gemini_state == "ACTIVE":
                status: Literal["uploaded", "processed", "error"] = "processed"
            elif gemini_state == "FAILED":
                status = "error"
            else:
                status = "uploaded"
            
            return OpenAIFileObject(
                id=response_json.get("uri", ""),
                bytes=int(response_json.get("sizeBytes", 0)),
                created_at=int(
                    time.mktime(
                        time.strptime(
                            response_json["createTime"].replace("Z", "+00:00"),
                            "%Y-%m-%dT%H:%M:%S.%f%z",
                        )
                    )
                ),
                filename=response_json.get("displayName", ""),
                object="file",
                purpose="user_data",
                status=status,
                status_details=str(response_json.get("error", "")) if gemini_state == "FAILED" else None,
            )
        except Exception as e:
            verbose_logger.exception(f"Error parsing file retrieve response: {str(e)}")
            raise ValueError(f"Error parsing file retrieve response: {str(e)}")

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
        api_base = self.get_api_base(litellm_params.get("api_base"))
        if not api_base:
            raise ValueError("api_base is required")
        
        # Get API key from multiple sources (same pattern as get_complete_url)
        api_key = litellm_params.get("api_key") or self.get_api_key()
        if not api_key:
            raise ValueError("api_key is required")
        
        # Extract file name from URI if full URI is provided
        # file_id could be "files/abc123" or "https://generativelanguage.googleapis.com/v1beta/files/abc123"
        if file_id.startswith("http"):
            # Extract the file path from full URI
            file_name = file_id.split("/v1beta/")[-1]
        else:
            file_name = file_id if file_id.startswith("files/") else f"files/{file_id}"
        
        # Construct the delete URL
        url = f"{api_base}/v1beta/{file_name}"
        
        # Add API key as header (Google AI Studio uses x-goog-api-key header)
        params: dict = {}
        
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
                    url = str(raw_response.request.url)
                    if "/files/" in url:
                        file_id = url.split("/files/")[-1].split("?")[0]
                        # Add the files/ prefix if not present
                        if not file_id.startswith("files/"):
                            file_id = f"files/{file_id}"
                
                return FileDeleted(
                    id=file_id,
                    deleted=True,
                    object="file"
                )
            else:
                raise ValueError(f"Failed to delete file: {raw_response.text}")
        except Exception as e:
            verbose_logger.exception(f"Error parsing file delete response: {str(e)}")
            raise ValueError(f"Error parsing file delete response: {str(e)}")

    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("GoogleAIStudioFilesHandler does not support file listing")

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        raise NotImplementedError("GoogleAIStudioFilesHandler does not support file listing")

    def transform_file_content_request(
        self,
        file_content_request,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("GoogleAIStudioFilesHandler does not support file content retrieval")

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        raise NotImplementedError("GoogleAIStudioFilesHandler does not support file content retrieval")

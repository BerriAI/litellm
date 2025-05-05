import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from httpx import Headers, Response

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
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    PathLike,
)
from litellm.types.llms.vertex_ai import GcsBucketResponse
from litellm.types.utils import ExtractedFileData, LlmProviders

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase


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
        object_name = f"litellm-vertex-files/{_model}/{uuid.uuid4()}"
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

        ## 2. If not jsonl, return the filename
        filename = extracted_file_data.get("filename")
        if filename:
            return filename
        ## 3. If no file name, return timestamp
        return str(int(time.time()))

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
        bucket_name = litellm_params.get("bucket_name") or os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS bucket_name is required")
        file_data = data.get("file")
        purpose = data.get("purpose")
        if file_data is None:
            raise ValueError("file is required")
        if purpose is None:
            raise ValueError("purpose is required")
        extracted_file_data = extract_file_data(file_data)
        object_name = self.get_object_name(extracted_file_data, purpose)
        endpoint = (
            f"upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={object_name}"
        )
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
        """
        Transforms OpenAI JSONL content to VertexAI JSONL content

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
                optional_params=self._map_openai_to_vertex_params(openai_request_body),
                custom_llm_provider="vertex_ai",
                litellm_params={},
                cached_content=None,
            )
            vertex_jsonl_content.append({"request": vertex_request_body})
        return vertex_jsonl_content

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
        if (
            create_file_data.get("purpose") == "batch"
            and extracted_file_data.get("content_type") == "application/jsonl"
            and extracted_file_data_content is not None
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
            return json.dumps(vertex_jsonl_content)
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
    ):
        """
        Transforms OpenAI JSONL content to VertexAI JSONL content

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
                optional_params=self._map_openai_to_vertex_params(openai_request_body),
                custom_llm_provider="vertex_ai",
                litellm_params={},
                cached_content=None,
            )
            vertex_jsonl_content.append({"request": vertex_request_body})
        return vertex_jsonl_content

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
        object_name = f"litellm-vertex-files/{_model}/{uuid.uuid4()}"
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

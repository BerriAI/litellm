import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm.llms.vertex_ai.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.openai import CreateFileRequest, FileObject, FileTypes, PathLike


class VertexAIFilesTransformation(VertexGeminiConfig):
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
    ) -> FileObject:
        """
        Transforms GCS Bucket upload file response to OpenAI FileObject
        """
        gcs_id = gcs_upload_response.get("id", "")
        # Remove the last numeric ID from the path
        gcs_id = "/".join(gcs_id.split("/")[:-1]) if gcs_id else ""

        return FileObject(
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

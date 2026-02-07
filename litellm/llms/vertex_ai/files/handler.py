import asyncio
import urllib.parse
from typing import Any, Coroutine, Optional, Tuple, Union

import httpx

from litellm import LlmProviders
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import (
    CreateFileRequest,
    FileContentRequest,
    HttpxBinaryResponseContent,
    OpenAIFileObject,
)
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES

from .transformation import VertexAIJsonlFilesTransformation

vertex_ai_files_transformation = VertexAIJsonlFilesTransformation()


class VertexAIFilesHandler(GCSBucketBase):
    """
    Handles Calling VertexAI in OpenAI Files API format v1/files/*

    This implementation uploads files on GCS Buckets
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.VERTEX_AI,
        )

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> OpenAIFileObject:
        gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
            kwargs={}
        )
        headers = await self.construct_request_headers(
            vertex_instance=gcs_logging_config["vertex_instance"],
            service_account_json=gcs_logging_config["path_service_account"],
        )
        bucket_name = gcs_logging_config["bucket_name"]
        (
            logging_payload,
            object_name,
        ) = vertex_ai_files_transformation.transform_openai_file_content_to_vertex_ai_file_content(
            openai_file_content=create_file_data.get("file")
        )
        gcs_upload_response = await self._log_json_data_on_gcs(
            headers=headers,
            bucket_name=bucket_name,
            object_name=object_name,
            logging_payload=logging_payload,
        )

        return vertex_ai_files_transformation.transform_gcs_bucket_response_to_openai_file_object(
            create_file_data=create_file_data,
            gcs_upload_response=gcs_upload_response,
        )

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on VertexAI GCS Bucket

        Only supported for Async litellm.acreate_file
        """

        if _is_async:
            return self.async_create_file(
                create_file_data=create_file_data,
                api_base=api_base,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.async_create_file(
                    create_file_data=create_file_data,
                    api_base=api_base,
                    vertex_credentials=vertex_credentials,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )

    def _extract_bucket_and_object_from_file_id(self, file_id: str) -> Tuple[str, str]:
        """
        Extract bucket name and object path from URL-encoded file_id.

        Expected format: gs%3A%2F%2Fbucket-name%2Fpath%2Fto%2Ffile
        Which decodes to: gs://bucket-name/path/to/file

        Returns:
            tuple: (bucket_name, url_encoded_object_path)
            - bucket_name: "bucket-name"
            - url_encoded_object_path: "path%2Fto%2Ffile"
        """
        decoded_path = urllib.parse.unquote(file_id)

        if decoded_path.startswith("gs://"):
            full_path = decoded_path[5:]  # Remove 'gs://' prefix
        else:
            full_path = decoded_path

        if "/" in full_path:
            bucket_name, object_path = full_path.split("/", 1)
        else:
            bucket_name = full_path
            object_path = ""

        encoded_object_path = urllib.parse.quote(object_path, safe="")

        return bucket_name, encoded_object_path

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> HttpxBinaryResponseContent:
        """
        Download file content from GCS bucket for VertexAI files.

        Args:
            file_content_request: Contains file_id (URL-encoded GCS path)
            vertex_credentials: VertexAI credentials
            vertex_project: VertexAI project ID
            vertex_location: VertexAI location
            timeout: Request timeout
            max_retries: Max retry attempts

        Returns:
            HttpxBinaryResponseContent: Binary content wrapped in compatible response format
        """
        file_id = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required in file_content_request")

        bucket_name, encoded_object_path = self._extract_bucket_and_object_from_file_id(
            file_id
        )

        download_kwargs = {
            "standard_callback_dynamic_params": {"gcs_bucket_name": bucket_name}
        }

        file_content = await self.download_gcs_object(
            object_name=encoded_object_path, **download_kwargs
        )

        if file_content is None:
            decoded_path = urllib.parse.unquote(file_id)
            raise ValueError(f"Failed to download file from GCS: {decoded_path}")

        decoded_path = urllib.parse.unquote(file_id)
        mock_response = httpx.Response(
            status_code=200,
            content=file_content,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request(method="GET", url=decoded_path),
        )

        return HttpxBinaryResponseContent(response=mock_response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        """
        Download file content from GCS bucket for VertexAI files.
        Supports both sync and async operations.

        Args:
            _is_async: Whether to run asynchronously
            file_content_request: Contains file_id (URL-encoded GCS path)
            api_base: API base (unused for GCS operations)
            vertex_credentials: VertexAI credentials
            vertex_project: VertexAI project ID
            vertex_location: VertexAI location
            timeout: Request timeout
            max_retries: Max retry attempts

        Returns:
            HttpxBinaryResponseContent or Coroutine: Binary content wrapped in compatible response format
        """
        if _is_async:
            return self.afile_content(
                file_content_request=file_content_request,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.afile_content(
                    file_content_request=file_content_request,
                    vertex_credentials=vertex_credentials,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )

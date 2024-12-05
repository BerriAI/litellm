import json
import uuid
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai_and_google_ai_studio.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    VertexAIError,
    VertexLLM,
)
from litellm.types.llms.openai import (
    Batch,
    CreateFileRequest,
    FileContentRequest,
    FileObject,
    FileTypes,
    HttpxBinaryResponseContent,
)

from .transformation import VertexAIFilesTransformation

vertex_ai_files_transformation = VertexAIFilesTransformation()


class VertexAIFilesHandler(GCSBucketBase):
    """
    Handles Calling VertexAI in OpenAI Files API format v1/files/*

    This implementation uploads files on GCS Buckets
    """

    pass

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ):
        gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
            kwargs={}
        )
        headers = await self.construct_request_headers(
            vertex_instance=gcs_logging_config["vertex_instance"],
            service_account_json=gcs_logging_config["path_service_account"],
        )
        bucket_name = gcs_logging_config["bucket_name"]
        logging_payload, object_name = (
            vertex_ai_files_transformation.transform_openai_file_content_to_vertex_ai_file_content(
                openai_file_content=create_file_data.get("file")
            )
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
        vertex_credentials: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[FileObject, Coroutine[Any, Any, FileObject]]:
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

        return None  # type: ignore

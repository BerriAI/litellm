import asyncio
import json
import os
import time
from collections.abc import Coroutine, Mapping
from typing import Any, Final
from urllib.parse import unquote

import httpx

from litellm import LlmProviders
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.litellm_core_utils.cloud_storage_security import (
    VERTEX_AI_MANAGED_GCS_PREFIX,
    should_allow_legacy_cloud_file_ids,
    validate_managed_cloud_file_id,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import (
    FileContentRequest,
    HttpxBinaryResponseContent,
)
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import StandardCallbackDynamicParams

from .transformation import VertexAIFilesConfig


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

    def _resolve_read_gcs_config(
        self,
        litellm_params: Mapping[str, object] | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
    ) -> tuple[str | None, str | None]:
        """
        Resolve the GCS bucket and service-account credentials for the read/content path.

        Sources them from the deployment's ``litellm_params`` (``gcs_bucket_name`` /
        ``bucket_name`` and ``vertex_credentials``), mirroring the write path in
        ``VertexAIFilesConfig._get_configured_bucket_name``, and falls back to the global
        ``GCS_BUCKET_NAME`` / ``GCS_PATH_SERVICE_ACCOUNT`` env vars. This lets Vertex batch
        run entirely at the model-group level, so output written to a per-model bucket is
        readable without setting the global env vars.
        """
        params: Final[Mapping[str, object]] = litellm_params or {}
        bucket_candidate: Final = params.get("gcs_bucket_name") or params.get("bucket_name")
        configured_bucket_name = bucket_candidate if isinstance(bucket_candidate, str) else os.getenv("GCS_BUCKET_NAME")

        credentials: Final = params.get("vertex_credentials") or vertex_credentials
        if isinstance(credentials, dict):
            path_service_account: str | None = json.dumps(credentials)
        elif isinstance(credentials, str):
            path_service_account = credentials
        else:
            path_service_account = os.getenv("GCS_PATH_SERVICE_ACCOUNT")

        return configured_bucket_name, path_service_account

    def _extract_bucket_and_object_from_file_id(
        self,
        file_id: str,
        configured_bucket_name: str,
        litellm_params: dict | None = None,
    ) -> tuple[str, str]:
        """
        Validate and extract bucket name and object path from file_id.

        Expected format: gs://bucket-name/litellm-vertex-files/path/to/file

        Returns:
            tuple: (bucket_name, object_path)
            - bucket_name: "bucket-name"
            - object_path: "litellm-vertex-files/path/to/file"
        """
        return validate_managed_cloud_file_id(
            file_id=file_id,
            scheme="gs://",
            configured_bucket_name=configured_bucket_name,
            allowed_object_prefixes=(VERTEX_AI_MANAGED_GCS_PREFIX,),
            allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(litellm_params),
        )

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        vertex_location: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        litellm_params: dict | None = None,
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
        file_id: Final = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required in file_content_request")

        configured_bucket_name, path_service_account = self._resolve_read_gcs_config(
            litellm_params=litellm_params,
            vertex_credentials=vertex_credentials,
        )
        dynamic_params: Final = StandardCallbackDynamicParams(
            gcs_bucket_name=configured_bucket_name,
            gcs_path_service_account=path_service_account,
        )
        gcs_logging_config: Final[GCSLoggingConfig] = await self.get_gcs_logging_config(
            kwargs={"standard_callback_dynamic_params": dynamic_params}
        )
        bucket_name, object_path = self._extract_bucket_and_object_from_file_id(
            file_id=file_id,
            configured_bucket_name=gcs_logging_config["bucket_name"],
            litellm_params=litellm_params,
        )

        download_kwargs: Final = {
            "standard_callback_dynamic_params": {
                "gcs_bucket_name": bucket_name,
                "gcs_path_service_account": gcs_logging_config["path_service_account"],
            }
        }

        file_content: Final = await self.download_gcs_object(object_name=object_path, **download_kwargs)
        decoded_file_id: Final = unquote(file_id)

        if file_content is None:
            raise ValueError(f"Failed to download file from GCS: {decoded_file_id}")

        mock_response: Final = httpx.Response(
            status_code=200,
            content=file_content,
            headers={
                "content-type": "application/octet-stream",
                "content-length": str(len(file_content)),
            },
            request=httpx.Request(method="GET", url=decoded_file_id),
        )

        # Apply transformation to convert Vertex AI batch outputs to OpenAI format
        config: Final = VertexAIFilesConfig()

        # Create a logging object for transformation
        logging_obj: Final = Logging(
            model="",
            messages=[],
            stream=False,
            call_type="afile_content",
            start_time=time.time(),
            litellm_call_id="",
            function_id="",
        )

        return config.transform_file_content_response(
            raw_response=mock_response, logging_obj=logging_obj, litellm_params={}
        )

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        vertex_location: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        litellm_params: dict | None = None,
    ) -> HttpxBinaryResponseContent | Coroutine[Any, Any, HttpxBinaryResponseContent]:
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
                litellm_params=litellm_params,
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
                    litellm_params=litellm_params,
                )
            )

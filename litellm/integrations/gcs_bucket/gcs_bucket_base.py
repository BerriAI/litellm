import json
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

if TYPE_CHECKING:
    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_llm_base import VertexBase
else:
    VertexBase = Any


class GCSBucketBase(CustomLogger):
    def __init__(self, bucket_name: Optional[str] = None) -> None:
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        _path_service_account = os.getenv("GCS_PATH_SERVICE_ACCOUNT")
        _bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self.path_service_account_json: Optional[str] = _path_service_account
        self.BUCKET_NAME: Optional[str] = _bucket_name

    async def construct_request_headers(
        self,
        service_account_json: str,
        vertex_instance: Optional[VertexBase] = None,
    ) -> Dict[str, str]:
        from litellm import vertex_chat_completion

        if vertex_instance is None:
            vertex_instance = vertex_chat_completion

        _auth_header, vertex_project = await vertex_instance._ensure_access_token_async(
            credentials=service_account_json,
            project_id=None,
            custom_llm_provider="vertex_ai",
        )

        auth_header, _ = vertex_instance._get_token_and_url(
            model="gcs-bucket",
            auth_header=_auth_header,
            vertex_credentials=service_account_json,
            vertex_project=vertex_project,
            vertex_location=None,
            gemini_api_key=None,
            stream=None,
            custom_llm_provider="vertex_ai",
            api_base=None,
        )
        verbose_logger.debug("constructed auth_header %s", auth_header)
        headers = {
            "Authorization": f"Bearer {auth_header}",  # auth_header
            "Content-Type": "application/json",
        }

        return headers

    def sync_construct_request_headers(self) -> Dict[str, str]:
        from litellm import vertex_chat_completion

        _auth_header, vertex_project = vertex_chat_completion._ensure_access_token(
            credentials=self.path_service_account_json,
            project_id=None,
            custom_llm_provider="vertex_ai",
        )

        auth_header, _ = vertex_chat_completion._get_token_and_url(
            model="gcs-bucket",
            auth_header=_auth_header,
            vertex_credentials=self.path_service_account_json,
            vertex_project=vertex_project,
            vertex_location=None,
            gemini_api_key=None,
            stream=None,
            custom_llm_provider="vertex_ai",
            api_base=None,
        )
        verbose_logger.debug("constructed auth_header %s", auth_header)
        headers = {
            "Authorization": f"Bearer {auth_header}",  # auth_header
            "Content-Type": "application/json",
        }

        return headers

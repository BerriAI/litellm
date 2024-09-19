import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_dict,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)


class GCSBucketBase(CustomLogger):
    def __init__(self, bucket_name: Optional[str] = None) -> None:
        from litellm.proxy.proxy_server import premium_user

        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.path_service_account_json = os.getenv("GCS_PATH_SERVICE_ACCOUNT", None)
        self.BUCKET_NAME = bucket_name or os.getenv("GCS_BUCKET_NAME", None)

        if self.BUCKET_NAME is None:
            raise ValueError(
                "GCS_BUCKET_NAME is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_BUCKET_NAME' in the environment."
            )

    async def construct_request_headers(self) -> Dict[str, str]:
        from litellm import vertex_chat_completion

        _auth_header, vertex_project = (
            await vertex_chat_completion._ensure_access_token_async(
                credentials=self.path_service_account_json,
                project_id=None,
                custom_llm_provider="vertex_ai",
            )
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

    async def download_gcs_object(self, object_name):
        """
        Download an object from GCS.

        https://cloud.google.com/storage/docs/downloading-objects#download-object-json
        """
        try:
            headers = await self.construct_request_headers()
            url = f"https://storage.googleapis.com/storage/v1/b/{self.BUCKET_NAME}/o/{object_name}?alt=media"

            # Send the GET request to download the object
            response = await self.async_httpx_client.get(url=url, headers=headers)

            if response.status_code != 200:
                verbose_logger.error(
                    "GCS object download error: %s", str(response.text)
                )
                return None

            verbose_logger.debug(
                "GCS object download response status code: %s", response.status_code
            )

            # Return the content of the downloaded object
            return response.content

        except Exception as e:
            verbose_logger.error("GCS object download error: %s", str(e))
            return None

    async def delete_gcs_object(self, object_name):
        """
        Delete an object from GCS.
        """
        try:
            headers = await self.construct_request_headers()
            url = f"https://storage.googleapis.com/storage/v1/b/{self.BUCKET_NAME}/o/{object_name}"

            # Send the DELETE request to delete the object
            response = await self.async_httpx_client.delete(url=url, headers=headers)

            if (response.status_code != 200) or (response.status_code != 204):
                verbose_logger.error(
                    "GCS object delete error: %s, status code: %s",
                    str(response.text),
                    response.status_code,
                )
                return None

            verbose_logger.debug(
                "GCS object delete response status code: %s, response: %s",
                response.status_code,
                response.text,
            )

            # Return the content of the downloaded object
            return response.text

        except Exception as e:
            verbose_logger.error("GCS object download error: %s", str(e))
            return None

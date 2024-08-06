import json
import os
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
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CommonProxyErrors, SpendLogsPayload


class RequestKwargs(TypedDict):
    model: Optional[str]
    messages: Optional[List]
    optional_params: Optional[Dict[str, Any]]


class GCSBucketPayload(TypedDict):
    request_kwargs: Optional[RequestKwargs]
    response_obj: Optional[Dict]
    start_time: str
    end_time: str


class GCSBucketLogger(CustomLogger):
    def __init__(self) -> None:
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )

        self.async_httpx_client = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
        self.path_service_account_json = os.getenv("GCS_PATH_SERVICE_ACCOUNT", None)
        self.BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", None)

        if self.BUCKET_NAME is None:
            raise ValueError(
                "GCS_BUCKET_NAME is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_BUCKET_NAME' in the environment."
            )

        if self.path_service_account_json is None:
            raise ValueError(
                "GCS_PATH_SERVICE_ACCOUNT is not set in the environment, but GCS Bucket is being used as a logging callback. Please set 'GCS_PATH_SERVICE_ACCOUNT' in the environment."
            )
        pass

    #### ASYNC ####
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_success_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )

            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            headers = await self.construct_request_headers()

            logging_payload: GCSBucketPayload = await self.get_gcs_payload(
                kwargs, response_obj, start_time_str, end_time_str
            )

            object_name = response_obj["id"]
            response = await self.async_httpx_client.post(
                headers=headers,
                url=f"https://storage.googleapis.com/upload/storage/v1/b/{self.BUCKET_NAME}/o?uploadType=media&name={object_name}",
                json=logging_payload,
            )

            if response.status_code != 200:
                verbose_logger.error("GCS Bucket logging error: %s", str(response.text))

            verbose_logger.debug("GCS Bucket response %s", response)
            verbose_logger.debug("GCS Bucket status code %s", response.status_code)
            verbose_logger.debug("GCS Bucket response.text %s", response.text)
        except Exception as e:
            verbose_logger.error("GCS Bucket logging error: %s", str(e))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def construct_request_headers(self) -> Dict[str, str]:
        from litellm import vertex_chat_completion

        auth_header, _ = vertex_chat_completion._get_token_and_url(
            model="gcs-bucket",
            vertex_credentials=self.path_service_account_json,
            vertex_project=None,
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

    async def get_gcs_payload(
        self, kwargs, response_obj, start_time, end_time
    ) -> GCSBucketPayload:
        request_kwargs = RequestKwargs(
            model=kwargs.get("model", None),
            messages=kwargs.get("messages", None),
            optional_params=kwargs.get("optional_params", None),
        )
        response_dict = {}
        response_dict = convert_litellm_response_object_to_dict(
            response_obj=response_obj
        )

        gcs_payload: GCSBucketPayload = GCSBucketPayload(
            request_kwargs=request_kwargs,
            response_obj=response_dict,
            start_time=start_time,
            end_time=end_time,
        )

        return gcs_payload

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

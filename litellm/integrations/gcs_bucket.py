import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CommonProxyErrors, SpendLogsPayload


class GCSBucketPayload(SpendLogsPayload):
    messages: Optional[List]
    output: Optional[Union[Dict, str, List]]


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
            headers = await self.construct_request_headers()
            logging_payload: GCSBucketPayload = await self.get_gcs_payload(
                kwargs, response_obj, start_time, end_time
            )

            object_name = logging_payload["request_id"]
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
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            get_logging_payload,
        )

        spend_logs_payload: SpendLogsPayload = get_logging_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            end_user_id=kwargs.get("user"),
        )

        gcs_payload: GCSBucketPayload = GCSBucketPayload(
            **spend_logs_payload, messages=None, output=None
        )
        gcs_payload["messages"] = kwargs.get("messages", None)
        gcs_payload["startTime"] = start_time.isoformat()
        gcs_payload["endTime"] = end_time.isoformat()

        if gcs_payload["completionStartTime"] is not None:
            gcs_payload["completionStartTime"] = gcs_payload[  # type: ignore
                "completionStartTime"  # type: ignore
            ].isoformat()

        output = None
        if response_obj is not None and (
            kwargs.get("call_type", None) == "embedding"
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ):
            output = None
        elif response_obj is not None and isinstance(
            response_obj, litellm.ModelResponse
        ):
            output_list = []
            for choice in response_obj.choices:
                output_list.append(choice.json())
            output = output_list
        elif response_obj is not None and isinstance(
            response_obj, litellm.TextCompletionResponse
        ):
            output_list = []
            for choice in response_obj.choices:
                output_list.append(choice.json())
            output = output_list
        elif response_obj is not None and isinstance(
            response_obj, litellm.ImageResponse
        ):
            output = response_obj["data"]
        elif response_obj is not None and isinstance(
            response_obj, litellm.TranscriptionResponse
        ):
            output = response_obj["text"]

        gcs_payload["output"] = output
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

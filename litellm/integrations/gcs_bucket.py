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
from litellm.proxy._types import SpendLogsPayload


class GCSBucketPayload(SpendLogsPayload):
    messages: Optional[List]
    output: Optional[Union[Dict, str, List]]


class GCSBucketLogger(CustomLogger):
    def __init__(self) -> None:
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

        spend_logs_payload: GCSBucketPayload = get_logging_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            end_user_id=kwargs.get("user"),
        )
        spend_logs_payload["startTime"] = start_time.isoformat()
        spend_logs_payload["endTime"] = end_time.isoformat()
        spend_logs_payload["completionStartTime"] = spend_logs_payload[
            "completionStartTime"
        ].isoformat()

        object_name = spend_logs_payload["request_id"]
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

        spend_logs_payload["output"] = output
        return spend_logs_payload

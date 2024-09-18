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
from litellm.integrations.gcs_bucket_base import GCSBucketBase
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_dict,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CommonProxyErrors, SpendLogsMetadata, SpendLogsPayload
from litellm.types.utils import StandardLoggingMetadata, StandardLoggingPayload


class RequestKwargs(TypedDict):
    model: Optional[str]
    messages: Optional[List]
    optional_params: Optional[Dict[str, Any]]


class GCSBucketPayload(TypedDict):
    request_kwargs: Optional[RequestKwargs]
    response_obj: Optional[Dict]
    start_time: str
    end_time: str
    response_cost: Optional[float]
    metadata: Optional[StandardLoggingMetadata]
    spend_log_metadata: str
    exception: Optional[str]
    log_event_type: Optional[str]


class GCSBucketLogger(GCSBucketBase):
    def __init__(self, bucket_name: Optional[str] = None) -> None:
        from litellm.proxy.proxy_server import premium_user

        super().__init__(bucket_name=bucket_name)
        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
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
            logging_payload["log_event_type"] = "successful_api_call"

            json_logged_payload = json.dumps(logging_payload)

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Modify the object_name to include the date-based folder
            object_name = f"{current_date}/{response_obj['id']}"
            response = await self.async_httpx_client.post(
                headers=headers,
                url=f"https://storage.googleapis.com/upload/storage/v1/b/{self.BUCKET_NAME}/o?uploadType=media&name={object_name}",
                data=json_logged_payload,
            )

            if response.status_code != 200:
                verbose_logger.error("GCS Bucket logging error: %s", str(response.text))

            verbose_logger.debug("GCS Bucket response %s", response)
            verbose_logger.debug("GCS Bucket status code %s", response.status_code)
            verbose_logger.debug("GCS Bucket response.text %s", response.text)
        except Exception as e:
            verbose_logger.error("GCS Bucket logging error: %s", str(e))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.proxy_server import premium_user

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_failure_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )

            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            headers = await self.construct_request_headers()

            logging_payload: GCSBucketPayload = await self.get_gcs_payload(
                kwargs, response_obj, start_time_str, end_time_str
            )
            logging_payload["log_event_type"] = "failed_api_call"

            _litellm_params = kwargs.get("litellm_params") or {}
            metadata = _litellm_params.get("metadata") or {}

            json_logged_payload = json.dumps(logging_payload)

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Modify the object_name to include the date-based folder
            object_name = f"{current_date}/failure-{uuid.uuid4().hex}"

            if "gcs_log_id" in metadata:
                object_name = metadata["gcs_log_id"]

            response = await self.async_httpx_client.post(
                headers=headers,
                url=f"https://storage.googleapis.com/upload/storage/v1/b/{self.BUCKET_NAME}/o?uploadType=media&name={object_name}",
                data=json_logged_payload,
            )

            if response.status_code != 200:
                verbose_logger.error("GCS Bucket logging error: %s", str(response.text))

            verbose_logger.debug("GCS Bucket response %s", response)
            verbose_logger.debug("GCS Bucket status code %s", response.status_code)
            verbose_logger.debug("GCS Bucket response.text %s", response.text)
        except Exception as e:
            verbose_logger.error("GCS Bucket logging error: %s", str(e))

    async def get_gcs_payload(
        self, kwargs, response_obj, start_time, end_time
    ) -> GCSBucketPayload:
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            get_logging_payload,
        )

        request_kwargs = RequestKwargs(
            model=kwargs.get("model", None),
            messages=kwargs.get("messages", None),
            optional_params=kwargs.get("optional_params", None),
        )
        response_dict = {}
        if response_obj:
            response_dict = convert_litellm_response_object_to_dict(
                response_obj=response_obj
            )

        exception_str = None

        # Handle logging exception attributes
        if "exception" in kwargs:
            exception_str = kwargs.get("exception", "")
            if not isinstance(exception_str, str):
                exception_str = str(exception_str)

        _spend_log_payload: SpendLogsPayload = get_logging_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            end_user_id=kwargs.get("end_user_id", None),
        )

        # Ensure everything in the payload is converted to str
        payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        if payload is None:
            raise ValueError("standard_logging_object not found in kwargs")

        gcs_payload: GCSBucketPayload = GCSBucketPayload(
            request_kwargs=request_kwargs,
            response_obj=response_dict,
            start_time=start_time,
            end_time=end_time,
            metadata=payload["metadata"],
            spend_log_metadata=_spend_log_payload.get("metadata", ""),
            response_cost=payload["response_cost"],
            exception=exception_str,
            log_event_type=None,
        )

        return gcs_payload

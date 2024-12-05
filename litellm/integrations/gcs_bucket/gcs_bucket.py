import asyncio
import json
import os
import uuid
from datetime import datetime
from re import S
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict, Union

import httpx
from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CommonProxyErrors, SpendLogsMetadata, SpendLogsPayload
from litellm.types.integrations.gcs_bucket import *
from litellm.types.utils import (
    StandardCallbackDynamicParams,
    StandardLoggingMetadata,
    StandardLoggingPayload,
)

if TYPE_CHECKING:
    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_llm_base import VertexBase
else:
    VertexBase = Any


GCS_DEFAULT_BATCH_SIZE = 2048
GCS_DEFAULT_FLUSH_INTERVAL_SECONDS = 20


class GCSBucketLogger(GCSBucketBase):
    def __init__(self, bucket_name: Optional[str] = None) -> None:
        from litellm.proxy.proxy_server import premium_user

        super().__init__(bucket_name=bucket_name)

        # Init Batch logging settings
        self.log_queue: List[GCSLogQueueItem] = []
        self.batch_size = int(os.getenv("GCS_BATCH_SIZE", GCS_DEFAULT_BATCH_SIZE))
        self.flush_interval = int(
            os.getenv("GCS_FLUSH_INTERVAL", GCS_DEFAULT_FLUSH_INTERVAL_SECONDS)
        )
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()
        super().__init__(
            flush_lock=self.flush_lock,
            batch_size=self.batch_size,
            flush_interval=self.flush_interval,
        )

        if premium_user is not True:
            raise ValueError(
                f"GCS Bucket logging is a premium feature. Please upgrade to use it. {CommonProxyErrors.not_premium_user.value}"
            )

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
            logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )
            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")

            # Add to logging queue - this will be flushed periodically
            self.log_queue.append(
                GCSLogQueueItem(
                    payload=logging_payload, kwargs=kwargs, response_obj=response_obj
                )
            )

        except Exception as e:
            verbose_logger.exception(f"GCS Bucket logging error: {str(e)}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                "GCS Logger: async_log_failure_event logging kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )

            logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )
            if logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")

            # Add to logging queue - this will be flushed periodically
            self.log_queue.append(
                GCSLogQueueItem(
                    payload=logging_payload, kwargs=kwargs, response_obj=response_obj
                )
            )

        except Exception as e:
            verbose_logger.exception(f"GCS Bucket logging error: {str(e)}")

    async def async_send_batch(self):
        """
        Process queued logs in batch - sends logs to GCS Bucket


        GCS Bucket does not have a Batch endpoint to batch upload logs

        Instead, we
            - collect the logs to flush every `GCS_FLUSH_INTERVAL` seconds
            - during async_send_batch, we make 1 POST request per log to GCS Bucket

        """
        if not self.log_queue:
            return

        try:
            for log_item in self.log_queue:
                logging_payload = log_item["payload"]
                kwargs = log_item["kwargs"]
                response_obj = log_item.get("response_obj", None) or {}

                gcs_logging_config: GCSLoggingConfig = (
                    await self.get_gcs_logging_config(kwargs)
                )
                headers = await self.construct_request_headers(
                    vertex_instance=gcs_logging_config["vertex_instance"],
                    service_account_json=gcs_logging_config["path_service_account"],
                )
                bucket_name = gcs_logging_config["bucket_name"]
                object_name = self._get_object_name(
                    kwargs, logging_payload, response_obj
                )
                await self._log_json_data_on_gcs(
                    headers=headers,
                    bucket_name=bucket_name,
                    object_name=object_name,
                    logging_payload=logging_payload,
                )

            # Clear the queue after processing
            self.log_queue.clear()

        except Exception as e:
            verbose_logger.exception(f"GCS Bucket batch logging error: {str(e)}")

    def _get_object_name(
        self, kwargs: Dict, logging_payload: StandardLoggingPayload, response_obj: Any
    ) -> str:
        """
        Get the object name to use for the current payload
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        if logging_payload.get("error_str", None) is not None:
            object_name = f"{current_date}/failure-{uuid.uuid4().hex}"
        else:
            object_name = f"{current_date}/{response_obj.get('id', '')}"

        # used for testing
        _litellm_params = kwargs.get("litellm_params", None) or {}
        _metadata = _litellm_params.get("metadata", None) or {}
        if "gcs_log_id" in _metadata:
            object_name = _metadata["gcs_log_id"]

        return object_name

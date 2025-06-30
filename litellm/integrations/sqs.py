"""SQS Logging Integration

This logger sends ``StandardLoggingPayload`` entries to an AWS SQS queue.

"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import DEFAULT_SQS_BATCH_SIZE, DEFAULT_SQS_FLUSH_INTERVAL_SECONDS
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload

from .custom_batch_logger import CustomBatchLogger


class SQSLogger(CustomBatchLogger, BaseAWSLLM):
    """Batching logger that writes logs to an AWS SQS queue."""

    def __init__(
        self,
        sqs_queue_url: Optional[str] = None,
        sqs_region_name: Optional[str] = None,
        sqs_flush_interval: Optional[int] = DEFAULT_SQS_FLUSH_INTERVAL_SECONDS,
        sqs_batch_size: Optional[int] = DEFAULT_SQS_BATCH_SIZE,
        **kwargs,
    ) -> None:
        try:
            verbose_logger.debug(
                f"in init sqs logger - sqs_callback_params {litellm.aws_sqs_callback_params}"
            )

            self.async_httpx_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback,
            )

            self._init_sqs_params(
                sqs_queue_url=sqs_queue_url,
                sqs_region_name=sqs_region_name,
            )

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()

            verbose_logger.debug(
                f"sqs flush interval: {sqs_flush_interval}, sqs batch size: {sqs_batch_size}"
            )

            CustomBatchLogger.__init__(
                self,
                flush_lock=self.flush_lock,
                flush_interval=sqs_flush_interval,
                batch_size=sqs_batch_size,
            )

            self.log_queue: List[StandardLoggingPayload] = []

            BaseAWSLLM.__init__(self)

        except Exception as e:
            print_verbose(f"Got exception on init sqs client {str(e)}")
            raise e

    def _init_sqs_params(
        self,
        sqs_queue_url: Optional[str] = None,
        sqs_region_name: Optional[str] = None,
    ) -> None:
        litellm.aws_sqs_callback_params = litellm.aws_sqs_callback_params or {}

        for key, value in litellm.aws_sqs_callback_params.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                litellm.aws_sqs_callback_params[key] = litellm.get_secret(value)

        self.sqs_queue_url = (
            litellm.aws_sqs_callback_params.get("sqs_queue_url") or sqs_queue_url
        )
        self.sqs_region_name = (
            litellm.aws_sqs_callback_params.get("sqs_region_name") or sqs_region_name
        )

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        try:
            verbose_logger.debug(
                "SQS Logging - Enters logging function for model %s", kwargs
            )
            standard_logging_payload = kwargs.get("standard_logging_object")
            if standard_logging_payload is None:
                raise ValueError("standard_logging_payload is None")

            self.log_queue.append(standard_logging_payload)
            verbose_logger.debug(
                "sqs logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
        except Exception as e:
            verbose_logger.exception(f"sqs Layer Error - {str(e)}")

    async def async_send_batch(self) -> None:
        verbose_logger.debug(
            f"sqs logger - sending batch of {len(self.log_queue)}"
        )
        if not self.log_queue:
            return

        for payload in self.log_queue:
            asyncio.create_task(self.async_send_message(payload))

    async def async_send_message(self, payload: StandardLoggingPayload) -> None:
        try:
            from urllib.parse import quote

            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            from litellm.litellm_core_utils.asyncify import asyncify

            asyncified_get_credentials = asyncify(self.get_credentials)
            credentials = await asyncified_get_credentials(
                aws_region_name=self.sqs_region_name,
            )

            if self.sqs_queue_url is None:
                raise ValueError("sqs_queue_url not set")

            json_string = safe_dumps(payload)

            body = (
                "Action=SendMessage&Version=2012-11-05&MessageBody="
                + quote(json_string, safe="")
            )

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }

            req = requests.Request(
                "POST", self.sqs_queue_url, data=body, headers=headers
            )
            prepped = req.prepare()

            aws_request = AWSRequest(
                method=prepped.method,
                url=prepped.url,
                data=prepped.body,
                headers=prepped.headers,
            )
            SigV4Auth(credentials, "sqs", self.sqs_region_name).add_auth(
                aws_request
            )

            signed_headers = dict(aws_request.headers.items())

            response = await self.async_httpx_client.post(
                self.sqs_queue_url,
                data=body,
                headers=signed_headers,
            )
            response.raise_for_status()
        except Exception as e:
            verbose_logger.exception(f"Error sending to SQS: {str(e)}")


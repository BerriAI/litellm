"""
Azure Sentinel Integration - sends logs to Azure Log Analytics HTTP Data Collector API

Azure Sentinel uses Log Analytics workspaces for data storage. This integration sends
LiteLLM logs to the Log Analytics workspace using the HTTP Data Collector API.

Reference API: https://learn.microsoft.com/en-us/azure/azure-monitor/logs/data-collector-api

`async_log_success_event` - used by litellm proxy to send logs to Azure Sentinel
`async_log_failure_event` - used by litellm proxy to send failure logs to Azure Sentinel

For batching specific details see CustomBatchLogger class
"""

import asyncio
import base64
import datetime
import hashlib
import hmac
import os
import traceback
from typing import Any, Dict, List, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.azure_sentinel import AzureSentinelInitParams
from litellm.types.utils import StandardLoggingPayload


class AzureSentinelLogger(CustomBatchLogger):
    """
    Logger that sends LiteLLM logs to Azure Sentinel via Azure Log Analytics HTTP Data Collector API
    """

    def __init__(
        self,
        workspace_id: Optional[str] = None,
        shared_key: Optional[str] = None,
        log_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Azure Sentinel logger

        Args:
            workspace_id (str, optional): Azure Log Analytics Workspace ID.
                If not provided, will use AZURE_SENTINEL_WORKSPACE_ID env var.
            shared_key (str, optional): Azure Log Analytics Primary or Secondary Key.
                If not provided, will use AZURE_SENTINEL_SHARED_KEY env var.
            log_type (str, optional): Custom log type name (table name in Log Analytics).
                If not provided, will use AZURE_SENTINEL_LOG_TYPE env var or default to "LiteLLM".
        """
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.workspace_id = workspace_id or os.getenv("AZURE_SENTINEL_WORKSPACE_ID")
        self.shared_key = shared_key or os.getenv("AZURE_SENTINEL_SHARED_KEY")
        self.log_type = log_type or os.getenv("AZURE_SENTINEL_LOG_TYPE", "LiteLLM")

        if not self.workspace_id:
            raise ValueError(
                "AZURE_SENTINEL_WORKSPACE_ID is required. Set it as an environment variable or pass workspace_id parameter."
            )
        if not self.shared_key:
            raise ValueError(
                "AZURE_SENTINEL_SHARED_KEY is required. Set it as an environment variable or pass shared_key parameter."
            )

        self.api_endpoint = (
            f"https://{self.workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
        )

        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: List[StandardLoggingPayload] = []

    def _build_signature(
        self, content_length: int, rfc1123date: str, content_type: str = "application/json"
    ) -> str:
        """
        Build HMAC-SHA256 signature for Azure Log Analytics API authentication

        Args:
            content_length: Length of the request body
            rfc1123date: Current date in RFC 1123 format
            content_type: Content type of the request (default: application/json)

        Returns:
            Base64-encoded signature string
        """
        assert self.shared_key is not None, "shared_key is required but was not set"

        resource = "/api/logs"
        string_to_sign = (
            f"POST\n{content_length}\n{content_type}\nx-ms-date:{rfc1123date}\n{resource}"
        )

        decoded_key = base64.b64decode(self.shared_key)
        encoded_hash = base64.b64encode(
            hmac.new(
                decoded_key, string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
            ).digest()
        ).decode()

        return encoded_hash

    def _build_authorization_header(
        self, content_length: int, rfc1123date: str
    ) -> str:
        """
        Build the Authorization header for Azure Log Analytics API

        Args:
            content_length: Length of the request body
            rfc1123date: Current date in RFC 1123 format

        Returns:
            Authorization header string in format: "SharedKey <workspace_id>:<signature>"
        """
        signature = self._build_signature(
            content_length=content_length, rfc1123date=rfc1123date
        )
        return f"SharedKey {self.workspace_id}:{signature}"

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """
        Async Log success events to Azure Sentinel

        - Gets StandardLoggingPayload from kwargs
        - Adds to batch queue
        - Flushes based on CustomBatchLogger settings

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            verbose_logger.debug(
                "Azure Sentinel: Logging - Enters logging function for model %s", kwargs
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            if standard_logging_payload is None:
                verbose_logger.warning(
                    "Azure Sentinel: standard_logging_object not found in kwargs"
                )
                return

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Azure Sentinel Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """
        Async Log failure events to Azure Sentinel

        - Gets StandardLoggingPayload from kwargs
        - Adds to batch queue
        - Flushes based on CustomBatchLogger settings

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            verbose_logger.debug(
                "Azure Sentinel: Logging - Enters failure logging function for model %s",
                kwargs,
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            if standard_logging_payload is None:
                verbose_logger.warning(
                    "Azure Sentinel: standard_logging_object not found in kwargs"
                )
                return

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Azure Sentinel Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_send_batch(self):
        """
        Sends the batch of logs to Azure Log Analytics API

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                "Azure Sentinel - about to flush %s events", len(self.log_queue)
            )

            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            # Convert log queue to JSON
            body = safe_dumps(self.log_queue)
            content_length = len(body.encode("utf-8"))

            # Get current date in RFC 1123 format
            rfc1123date = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

            # Build authorization header
            authorization = self._build_authorization_header(
                content_length=content_length, rfc1123date=rfc1123date
            )

            # Set headers
            headers = {
                "Authorization": authorization,
                "Content-Type": "application/json",
                "Log-Type": self.log_type,
                "x-ms-date": rfc1123date,
            }

            # Send the request
            response = await self.async_httpx_client.post(
                url=self.api_endpoint, data=body.encode("utf-8"), headers=headers
            )

            if response.status_code not in [200, 202]:
                verbose_logger.error(
                    "Azure Sentinel API error: status_code=%s, response=%s",
                    response.status_code,
                    response.text,
                )
                raise Exception(
                    f"Failed to send logs to Azure Sentinel: {response.status_code} - {response.text}"
                )

            verbose_logger.debug(
                "Azure Sentinel: Response from API status_code: %s",
                response.status_code,
            )

        except Exception as e:
            verbose_logger.exception(
                f"Azure Sentinel Error sending batch API - {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            self.log_queue.clear()


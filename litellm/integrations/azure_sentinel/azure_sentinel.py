"""
Azure Sentinel Integration - sends logs to Azure Log Analytics using Logs Ingestion API

Azure Sentinel uses Log Analytics workspaces for data storage. This integration sends
LiteLLM logs to the Log Analytics workspace using the Azure Monitor Logs Ingestion API.

Reference API: https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview

`async_log_success_event` - used by litellm proxy to send logs to Azure Sentinel
`async_log_failure_event` - used by litellm proxy to send failure logs to Azure Sentinel

For batching specific details see CustomBatchLogger class
"""

import asyncio
import os
import traceback
from typing import List, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


class AzureSentinelLogger(CustomBatchLogger):
    """
    Logger that sends LiteLLM logs to Azure Sentinel via Azure Monitor Logs Ingestion API
    """

    def __init__(
        self,
        dcr_immutable_id: Optional[str] = None,
        stream_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Azure Sentinel logger using Logs Ingestion API

        Args:
            dcr_immutable_id (str, optional): Data Collection Rule (DCR) Immutable ID.
                If not provided, will use AZURE_SENTINEL_DCR_IMMUTABLE_ID env var.
            stream_name (str, optional): Stream name from DCR (e.g., "Custom-LiteLLM").
                If not provided, will use AZURE_SENTINEL_STREAM_NAME env var or default to "Custom-LiteLLM".
            endpoint (str, optional): Data Collection Endpoint (DCE) or DCR ingestion endpoint.
                If not provided, will use AZURE_SENTINEL_ENDPOINT env var.
            tenant_id (str, optional): Azure Tenant ID for OAuth2 authentication.
                If not provided, will use AZURE_SENTINEL_TENANT_ID or AZURE_TENANT_ID env var.
            client_id (str, optional): Azure Client ID (Application ID) for OAuth2 authentication.
                If not provided, will use AZURE_SENTINEL_CLIENT_ID or AZURE_CLIENT_ID env var.
            client_secret (str, optional): Azure Client Secret for OAuth2 authentication.
                If not provided, will use AZURE_SENTINEL_CLIENT_SECRET or AZURE_CLIENT_SECRET env var.
        """
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.dcr_immutable_id = (
            dcr_immutable_id or os.getenv("AZURE_SENTINEL_DCR_IMMUTABLE_ID")
        )
        self.stream_name = stream_name or os.getenv(
            "AZURE_SENTINEL_STREAM_NAME", "Custom-LiteLLM"
        )
        self.endpoint = endpoint or os.getenv("AZURE_SENTINEL_ENDPOINT")
        self.tenant_id = tenant_id or os.getenv("AZURE_SENTINEL_TENANT_ID") or os.getenv(
            "AZURE_TENANT_ID"
        )
        self.client_id = client_id or os.getenv("AZURE_SENTINEL_CLIENT_ID") or os.getenv(
            "AZURE_CLIENT_ID"
        )
        self.client_secret = (
            client_secret
            or os.getenv("AZURE_SENTINEL_CLIENT_SECRET")
            or os.getenv("AZURE_CLIENT_SECRET")
        )

        if not self.dcr_immutable_id:
            raise ValueError(
                "AZURE_SENTINEL_DCR_IMMUTABLE_ID is required. Set it as an environment variable or pass dcr_immutable_id parameter."
            )
        if not self.endpoint:
            raise ValueError(
                "AZURE_SENTINEL_ENDPOINT is required. Set it as an environment variable or pass endpoint parameter."
            )
        if not self.tenant_id:
            raise ValueError(
                "AZURE_SENTINEL_TENANT_ID or AZURE_TENANT_ID is required. Set it as an environment variable or pass tenant_id parameter."
            )
        if not self.client_id:
            raise ValueError(
                "AZURE_SENTINEL_CLIENT_ID or AZURE_CLIENT_ID is required. Set it as an environment variable or pass client_id parameter."
            )
        if not self.client_secret:
            raise ValueError(
                "AZURE_SENTINEL_CLIENT_SECRET or AZURE_CLIENT_SECRET is required. Set it as an environment variable or pass client_secret parameter."
            )

        # Build API endpoint: {Endpoint}/dataCollectionRules/{DCR Immutable ID}/streams/{Stream Name}?api-version=2023-01-01
        self.api_endpoint = (
            f"{self.endpoint.rstrip('/')}/dataCollectionRules/{self.dcr_immutable_id}/streams/{self.stream_name}?api-version=2023-01-01"
        )

        # OAuth2 scope for Azure Monitor
        self.oauth_scope = "https://monitor.azure.com/.default"
        self.oauth_token: Optional[str] = None
        self.oauth_token_expires_at: Optional[float] = None

        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: List[StandardLoggingPayload] = []

    async def _get_oauth_token(self) -> str:
        """
        Get OAuth2 Bearer token for Azure Monitor Logs Ingestion API

        Returns:
            Bearer token string
        """
        # Check if we have a valid cached token
        import time

        if (
            self.oauth_token
            and self.oauth_token_expires_at
            and time.time() < self.oauth_token_expires_at - 60
        ):  # Refresh 60 seconds before expiry
            return self.oauth_token

        # Get new token using client credentials flow
        assert self.tenant_id is not None, "tenant_id is required"
        assert self.client_id is not None, "client_id is required"
        assert self.client_secret is not None, "client_secret is required"

        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        token_data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.oauth_scope,
            "grant_type": "client_credentials",
        }

        response = await self.async_httpx_client.post(
            url=token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to get OAuth2 token: {response.status_code} - {response.text}"
            )

        token_response = response.json()
        self.oauth_token = token_response.get("access_token")
        expires_in = token_response.get("expires_in", 3600)

        if not self.oauth_token:
            raise Exception("OAuth2 token response did not contain access_token")

        # Cache token expiry time
        import time

        self.oauth_token_expires_at = time.time() + expires_in

        return self.oauth_token

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
        Sends the batch of logs to Azure Monitor Logs Ingestion API

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

            # Get OAuth2 token
            bearer_token = await self._get_oauth_token()

            # Convert log queue to JSON array format expected by Logs Ingestion API
            # Each log entry should be a JSON object in the array
            body = safe_dumps(self.log_queue)

            # Set headers for Logs Ingestion API
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            }

            # Send the request
            response = await self.async_httpx_client.post(
                url=self.api_endpoint, data=body.encode("utf-8"), headers=headers
            )

            if response.status_code not in [200, 204]:
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

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
import time
import traceback
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardAuditLogPayload, StandardLoggingPayload

DEFAULT_AZURE_AUTHORITY_HOST: Final = "https://login.microsoftonline.com"
DEFAULT_AZURE_MONITOR_SCOPE: Final = "https://monitor.azure.com/.default"

MONITOR_SCOPE_BY_AUTHORITY_HOST: Final[Mapping[str, str]] = MappingProxyType(
    {
        "login.microsoftonline.com": DEFAULT_AZURE_MONITOR_SCOPE,
        "login.microsoftonline.us": "https://monitor.azure.us/.default",
    }
)


class AzureSentinelLogger(CustomBatchLogger):
    """
    Logger that sends LiteLLM logs to Azure Sentinel via Azure Monitor Logs Ingestion API
    """

    def __init__(
        self,
        dcr_immutable_id: str | None = None,
        stream_name: str | None = None,
        endpoint: str | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        audit_stream_name: str | None = None,
        authority_host: str | None = None,
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
            audit_stream_name (str, optional): Stream name from DCR for audit logs.
                If not provided, will use AZURE_SENTINEL_AUDIT_STREAM_NAME env var or the standard stream name.
            authority_host (str, optional): Microsoft Entra authority host that issues the OAuth2 token,
                e.g. "https://login.microsoftonline.us" for Azure Government. If not provided, will use
                AZURE_SENTINEL_AUTHORITY_HOST or AZURE_AUTHORITY_HOST env vars, or default to the Azure
                Public Cloud authority. The Azure Monitor audience is derived from it.
        """
        self.async_httpx_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

        resolved_dcr_immutable_id: Final = dcr_immutable_id or os.getenv("AZURE_SENTINEL_DCR_IMMUTABLE_ID")
        resolved_stream_name: Final = stream_name or os.getenv("AZURE_SENTINEL_STREAM_NAME") or "Custom-LiteLLM"
        resolved_audit_stream_name: Final = (
            audit_stream_name or os.getenv("AZURE_SENTINEL_AUDIT_STREAM_NAME") or resolved_stream_name
        )
        resolved_endpoint: Final = endpoint or os.getenv("AZURE_SENTINEL_ENDPOINT")
        resolved_tenant_id: Final = tenant_id or os.getenv("AZURE_SENTINEL_TENANT_ID") or os.getenv("AZURE_TENANT_ID")
        resolved_client_id: Final = client_id or os.getenv("AZURE_SENTINEL_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
        resolved_client_secret: Final = (
            client_secret or os.getenv("AZURE_SENTINEL_CLIENT_SECRET") or os.getenv("AZURE_CLIENT_SECRET")
        )
        resolved_authority_host: Final = self._normalize_authority_host(
            authority_host
            or os.getenv("AZURE_SENTINEL_AUTHORITY_HOST")
            or os.getenv("AZURE_AUTHORITY_HOST")
            or DEFAULT_AZURE_AUTHORITY_HOST
        )

        if not resolved_dcr_immutable_id:
            raise ValueError(
                "AZURE_SENTINEL_DCR_IMMUTABLE_ID is required. Set it as an environment variable or pass dcr_immutable_id parameter."
            )
        if not resolved_endpoint:
            raise ValueError(
                "AZURE_SENTINEL_ENDPOINT is required. Set it as an environment variable or pass endpoint parameter."
            )
        if not resolved_tenant_id:
            raise ValueError(
                "AZURE_SENTINEL_TENANT_ID or AZURE_TENANT_ID is required. Set it as an environment variable or pass tenant_id parameter."
            )
        if not resolved_client_id:
            raise ValueError(
                "AZURE_SENTINEL_CLIENT_ID or AZURE_CLIENT_ID is required. Set it as an environment variable or pass client_id parameter."
            )
        if not resolved_client_secret:
            raise ValueError(
                "AZURE_SENTINEL_CLIENT_SECRET or AZURE_CLIENT_SECRET is required. Set it as an environment variable or pass client_secret parameter."
            )

        self.dcr_immutable_id = resolved_dcr_immutable_id
        self.stream_name = resolved_stream_name
        self.audit_stream_name = resolved_audit_stream_name
        self.endpoint = resolved_endpoint
        self.tenant_id = resolved_tenant_id
        self.client_id = resolved_client_id
        self.client_secret = resolved_client_secret

        # Build API endpoint: {Endpoint}/dataCollectionRules/{DCR Immutable ID}/streams/{Stream Name}?api-version=2023-01-01
        self.api_endpoint = self._build_api_endpoint(
            endpoint=resolved_endpoint,
            dcr_immutable_id=resolved_dcr_immutable_id,
            stream_name=resolved_stream_name,
        )
        self.audit_api_endpoint = self._build_api_endpoint(
            endpoint=resolved_endpoint,
            dcr_immutable_id=resolved_dcr_immutable_id,
            stream_name=resolved_audit_stream_name,
        )

        # OAuth2 scope for Azure Monitor
        self.authority_host = resolved_authority_host
        self.oauth_scope = self._resolve_oauth_scope(authority_host=resolved_authority_host)
        self.oauth_token: str | None = None
        self.oauth_token_expires_at: float | None = None

        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: list[StandardLoggingPayload] = []
        self.audit_log_queue: list[StandardAuditLogPayload] = []

    @staticmethod
    def _normalize_authority_host(authority_host: str) -> str:
        """
        Normalize an authority host into an absolute URL with no trailing slash.

        Accepts the scheme-qualified form litellm documents ("https://login.microsoftonline.us")
        and the bare-host form the azure-identity AzureAuthorityHosts constants use.
        """
        stripped: Final = authority_host.strip().rstrip("/")
        return stripped if "://" in stripped else f"https://{stripped}"

    @staticmethod
    def _resolve_oauth_scope(authority_host: str) -> str:
        """
        Map an authority host to the Azure Monitor Logs Ingestion audience for the same cloud,
        falling back to the Azure Public Cloud audience for an unrecognized host.
        """
        host: Final = urlparse(authority_host).hostname or ""
        return MONITOR_SCOPE_BY_AUTHORITY_HOST.get(host, DEFAULT_AZURE_MONITOR_SCOPE)

    @staticmethod
    def _build_api_endpoint(endpoint: str, dcr_immutable_id: str, stream_name: str) -> str:
        return f"{endpoint.rstrip('/')}/dataCollectionRules/{dcr_immutable_id}/streams/{stream_name}?api-version=2023-01-01"

    async def _get_oauth_token(self) -> str:
        """
        Get OAuth2 Bearer token for Azure Monitor Logs Ingestion API

        Returns:
            Bearer token string
        """
        if (
            self.oauth_token and self.oauth_token_expires_at and time.time() < self.oauth_token_expires_at - 60
        ):  # Refresh 60 seconds before expiry
            return self.oauth_token

        # Get new token using client credentials flow
        assert self.tenant_id is not None, "tenant_id is required"
        assert self.client_id is not None, "client_id is required"
        assert self.client_secret is not None, "client_secret is required"

        token_url: Final = f"{self.authority_host}/{self.tenant_id}/oauth2/v2.0/token"

        token_data: Final = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.oauth_scope,
            "grant_type": "client_credentials",
        }

        response: Final = await self.async_httpx_client.post(
            url=token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get OAuth2 token: {response.status_code} - {response.text}")

        token_response: Final = response.json()
        self.oauth_token = token_response.get("access_token")
        expires_in: Final = token_response.get("expires_in", 3600)

        if not self.oauth_token:
            raise Exception("OAuth2 token response did not contain access_token")

        self.oauth_token_expires_at = time.time() + expires_in

        return self.oauth_token

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log success events to Azure Sentinel

        - Gets StandardLoggingPayload from kwargs
        - Adds to batch queue
        - Flushes based on CustomBatchLogger settings

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            verbose_logger.debug("Azure Sentinel: Logging - Enters logging function for model %s", kwargs)
            standard_logging_payload: Final = kwargs.get("standard_logging_object", None)

            if standard_logging_payload is None:
                verbose_logger.warning("Azure Sentinel: standard_logging_object not found in kwargs")
                return

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception("Azure Sentinel Layer Error - %s\n%s", e, traceback.format_exc())

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
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
            standard_logging_payload: Final = kwargs.get("standard_logging_object", None)

            if standard_logging_payload is None:
                verbose_logger.warning("Azure Sentinel: standard_logging_object not found in kwargs")
                return

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception("Azure Sentinel Layer Error - %s\n%s", e, traceback.format_exc())

    async def async_log_audit_log_event(self, audit_log: StandardAuditLogPayload) -> None:
        """
        Async log LiteLLM audit log events to Azure Sentinel.

        Audit logs are queued separately from standard LLM logs so mixed callback
        usage never sends schema-mismatched records in the same ingestion batch.
        """
        try:
            verbose_logger.debug(
                "Azure Sentinel: Logging audit event id=%s action=%s table=%s",
                audit_log.get("id"),
                audit_log.get("action"),
                audit_log.get("table_name"),
            )

            self.audit_log_queue.append(audit_log)

            if len(self.audit_log_queue) >= self.batch_size:
                await self.async_send_audit_batch()

        except Exception as e:
            verbose_logger.exception("Azure Sentinel Audit Log Layer Error - %s\n%s", e, traceback.format_exc())

    async def async_send_batch(self):
        """
        Sends the batch of logs to Azure Monitor Logs Ingestion API

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        await self._async_send_batch_to_api(
            log_queue=self.log_queue,
            api_endpoint=self.api_endpoint,
            log_type="logs",
        )

    async def async_send_audit_batch(self):
        """
        Sends the batch of audit logs to Azure Monitor Logs Ingestion API
        """
        await self._async_send_batch_to_api(
            log_queue=self.audit_log_queue,
            api_endpoint=self.audit_api_endpoint,
            log_type="audit logs",
        )

    async def _async_send_batch_to_api(
        self,
        log_queue: list[StandardLoggingPayload | StandardAuditLogPayload],
        api_endpoint: str,
        log_type: str,
    ) -> None:
        try:
            if not log_queue:
                return

            verbose_logger.debug("Azure Sentinel - about to flush %s %s", len(log_queue), log_type)

            # Get OAuth2 token
            bearer_token: Final = await self._get_oauth_token()

            # Convert log queue to JSON array format expected by Logs Ingestion API
            # Each log entry should be a JSON object in the array
            body: Final = safe_dumps(log_queue)

            # Set headers for Logs Ingestion API
            headers: Final = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            }

            # Send the request
            response = await self.async_httpx_client.post(url=api_endpoint, data=body.encode("utf-8"), headers=headers)

            if response.status_code not in [200, 204]:
                verbose_logger.error(
                    "Azure Sentinel API error: status_code=%s, response=%s",
                    response.status_code,
                    response.text,
                )
                raise Exception(f"Failed to send logs to Azure Sentinel: {response.status_code} - {response.text}")

            verbose_logger.debug(
                "Azure Sentinel: Response from API status_code: %s",
                response.status_code,
            )

        except Exception as e:
            verbose_logger.exception("Azure Sentinel Error sending batch API - %s\n%s", e, traceback.format_exc())
        finally:
            log_queue.clear()

    async def flush_queue(self):
        if self.flush_lock is None:
            return

        async with self.flush_lock:
            if self.log_queue:
                await self.async_send_batch()
            if self.audit_log_queue:
                await self.async_send_audit_batch()
            self.last_flush_time = time.time()

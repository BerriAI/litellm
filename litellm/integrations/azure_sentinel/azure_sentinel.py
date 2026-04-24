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
import gzip
import os
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional

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

        self.dcr_immutable_id = dcr_immutable_id or os.getenv(
            "AZURE_SENTINEL_DCR_IMMUTABLE_ID"
        )
        self.stream_name = stream_name or os.getenv(
            "AZURE_SENTINEL_STREAM_NAME", "Custom-LiteLLM"
        )
        self.endpoint = endpoint or os.getenv("AZURE_SENTINEL_ENDPOINT")
        self.tenant_id = (
            tenant_id
            or os.getenv("AZURE_SENTINEL_TENANT_ID")
            or os.getenv("AZURE_TENANT_ID")
        )
        self.client_id = (
            client_id
            or os.getenv("AZURE_SENTINEL_CLIENT_ID")
            or os.getenv("AZURE_CLIENT_ID")
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
        self.api_endpoint = f"{self.endpoint.rstrip('/')}/dataCollectionRules/{self.dcr_immutable_id}/streams/{self.stream_name}?api-version=2023-01-01"

        # OAuth2 scope for Azure Monitor
        self.oauth_scope = "https://monitor.azure.com/.default"
        self.oauth_token: Optional[str] = None
        self.oauth_token_expires_at: Optional[float] = None

        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: List[StandardLoggingPayload] = []

        # When True, string fields (messages, response) are truncated to the
        # Azure Log Analytics column limit (256 KB / 262,144 chars). Azure
        # silently truncates at this limit anyway; doing it ourselves lets us
        # keep the tail (most recent content) and record metadata.
        # Controlled by AZURE_SENTINEL_TRUNCATE_CONTENT env var (default: false).
        truncate_env = os.getenv("AZURE_SENTINEL_TRUNCATE_CONTENT", "false")
        self.truncate_content = truncate_env.lower() in ("true", "1", "yes")

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

        token_url = (
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        )

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

    # Azure DCR Logs Ingestion API has a 1MB request size limit.
    # We target a conservative threshold to stay safely under the limit.
    MAX_BATCH_SIZE_BYTES = 950_000  # ~950KB uncompressed target per batch

    # Azure Log Analytics silently truncates string column values at 256 KB.
    # We enforce this limit ourselves so we can keep the tail (most recent
    # content) and record truncation metadata.
    MAX_COLUMN_CHARS = 262_144  # 256 KB Azure Log Analytics column limit

    def _enforce_column_limits(
        self, payload: StandardLoggingPayload
    ) -> StandardLoggingPayload:
        """
        Truncate messages/response string fields to the Azure Log Analytics
        column limit (262,144 chars). Keeps the *tail* of each field so that
        the most recent conversation turns and response text are preserved.

        Only called when ``self.truncate_content`` is True.

        Returns the original payload unchanged if neither field exceeds the
        limit, otherwise returns a deep copy with truncated fields and
        truncation metadata added.
        """
        limit = self.MAX_COLUMN_CHARS
        messages = payload.get("messages")
        response = payload.get("response")
        msg_str = str(messages) if messages is not None else ""
        resp_str = str(response) if response is not None else ""

        needs_truncation = len(msg_str) > limit or len(resp_str) > limit
        if not needs_truncation:
            return payload

        entry = deepcopy(payload)
        truncated_fields: List[str] = []
        prefix = "[truncated by litellm]..."
        tail_limit = limit - len(prefix)

        if len(msg_str) > limit:
            entry["messages"] = prefix + msg_str[-tail_limit:]
            truncated_fields.append("messages")

        if len(resp_str) > limit:
            entry["response"] = prefix + resp_str[-tail_limit:]
            truncated_fields.append("response")

        truncation_info: Dict[str, Any] = {
            "truncated": True,
            "truncate_reason": "azure_column_limit",
            "truncated_fields": truncated_fields,
            "original_messages_chars": len(msg_str),
            "original_response_chars": len(resp_str),
            "max_column_chars": limit,
        }

        if "metadata" in entry and isinstance(entry.get("metadata"), dict):
            entry["metadata"]["litellm_content_truncated"] = truncation_info  # type: ignore
        else:
            entry["litellm_content_truncated"] = truncation_info  # type: ignore

        verbose_logger.debug(
            "Azure Sentinel: column-level truncation (id=%s). "
            "messages: %d→%d chars, response: %d→%d chars",
            entry.get("id", "?"),
            len(msg_str),
            min(len(msg_str), limit),
            len(resp_str),
            min(len(resp_str), limit),
        )

        return entry

    def _split_into_batches(
        self, payloads: List[StandardLoggingPayload]
    ) -> List[bytes]:
        """
        Splits payloads into gzip-compressed batches that stay under
        MAX_BATCH_SIZE_BYTES (uncompressed) per batch.

        When truncate_content is enabled, enforces the Azure Log Analytics
        column limit (256 KB) on messages/response fields before batching.

        Returns a list of gzip-compressed byte strings, each representing
        a JSON array of log entries.
        """
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        batches: List[bytes] = []
        current_batch: list = []
        current_size = 2  # account for JSON array brackets '[]'

        for payload in payloads:
            # Enforce Azure column-level string limit when enabled
            if self.truncate_content:
                payload = self._enforce_column_limits(payload)

            entry_json = safe_dumps(payload)
            entry_size = len(entry_json.encode("utf-8"))

            # If a single entry exceeds the uncompressed batch limit,
            # send it alone in its own batch
            if entry_size + 2 > self.MAX_BATCH_SIZE_BYTES:
                # Flush any accumulated batch first
                if current_batch:
                    batch_body = safe_dumps(current_batch)
                    batches.append(gzip.compress(batch_body.encode("utf-8")))
                    current_batch = []
                    current_size = 2

                single_body = safe_dumps([payload])
                batches.append(gzip.compress(single_body.encode("utf-8")))
                continue

            # +1 for comma separator between entries
            separator = 1 if current_batch else 0
            if current_size + separator + entry_size > self.MAX_BATCH_SIZE_BYTES:
                # Current batch is full — flush it
                batch_body = safe_dumps(current_batch)
                batches.append(gzip.compress(batch_body.encode("utf-8")))
                current_batch = []
                current_size = 2

            current_batch.append(payload)
            current_size += entry_size + (1 if len(current_batch) > 1 else 0)

        # Flush remaining
        if current_batch:
            batch_body = safe_dumps(current_batch)
            batches.append(gzip.compress(batch_body.encode("utf-8")))

        return batches

    async def async_send_batch(self):
        """
        Sends the batch of logs to Azure Monitor Logs Ingestion API
        with gzip compression. Splits into multiple requests if the
        batch exceeds ~1MB uncompressed.

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                "Azure Sentinel - about to flush %s events", len(self.log_queue)
            )

            # Get OAuth2 token
            bearer_token = await self._get_oauth_token()

            # Split into size-limited, gzip-compressed batches
            compressed_batches = self._split_into_batches(self.log_queue)

            verbose_logger.debug(
                "Azure Sentinel - split into %s batch(es)", len(compressed_batches)
            )

            # Set headers for Logs Ingestion API with gzip encoding
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            }

            # Send each batch
            for i, compressed_body in enumerate(compressed_batches):
                response = await self.async_httpx_client.post(
                    url=self.api_endpoint,
                    content=compressed_body,
                    headers=headers,
                )

                if response.status_code not in [200, 204]:
                    verbose_logger.error(
                        "Azure Sentinel API error on batch %s/%s: status_code=%s, response=%s",
                        i + 1,
                        len(compressed_batches),
                        response.status_code,
                        response.text,
                    )
                    raise Exception(
                        f"Failed to send logs to Azure Sentinel: {response.status_code} - {response.text}"
                    )

                verbose_logger.debug(
                    "Azure Sentinel: batch %s/%s sent, status_code: %s",
                    i + 1,
                    len(compressed_batches),
                    response.status_code,
                )

        except Exception as e:
            verbose_logger.exception(
                f"Azure Sentinel Error sending batch API - {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            self.log_queue.clear()

"""
Faros AI (https://www.faros.ai/) integration.

Sends LiteLLM usage data to a Faros graph so LLM usage shows up alongside the
rest of an engineering org's productivity data.

Each successful LLM request is recorded as a Faros canonical
``vcs_UserToolUsage`` row (with its backing ``vcs_UserTool`` and ``vcs_User``
rows) via the Faros GraphQL API - the same write path Faros' own first-party
usage trackers use (see faros-ai/faros-vscode-extension).
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload

FAROS_DEFAULT_API_URL = "https://prod.api.faros.ai"
FAROS_DEFAULT_GRAPH = "default"
FAROS_DEFAULT_ORIGIN = "litellm"
FAROS_DEFAULT_USER_SOURCE = "LiteLLM"
FAROS_DEFAULT_TOOL_CATEGORY = "LiteLLM"

USAGE_MUTATION = (
    "mutation LiteLLMUserToolUsage($usages: [vcs_UserToolUsage_insert_input!]!) { "
    "insert_vcs_UserToolUsage(objects: $usages, on_conflict: {"
    "constraint: vcs_UserToolUsage_pkey, update_columns: [refreshedAt, origin]"
    "}) { affected_rows } }"
)


class FarosLogger(CustomBatchLogger):
    """
    Batches LiteLLM usage events and upserts them into a Faros graph.

    Environment Variables:
        FAROS_API_KEY: Faros API key (required)
        FAROS_API_URL: Faros API base url (default: https://prod.api.faros.ai)
        FAROS_GRAPH: Faros graph to write to (default: "default")
        FAROS_ORIGIN: Origin recorded on every row (default: "litellm")
        FAROS_USER_SOURCE: vcs_User.source for LiteLLM users (default: "LiteLLM")
        FAROS_TOOL_CATEGORY: vcs_UserTool.tool.category (default: "LiteLLM")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        graph: Optional[str] = None,
        origin: Optional[str] = None,
        user_source: Optional[str] = None,
        tool_category: Optional[str] = None,
        async_httpx_client: Optional[AsyncHTTPHandler] = None,
        **kwargs,
    ):
        self.api_key = api_key or os.getenv("FAROS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FAROS_API_KEY is not set. Set it in your environment or pass api_key to FarosLogger."
            )
        resolved_api_url = (
            api_url or os.getenv("FAROS_API_URL") or FAROS_DEFAULT_API_URL
        ).rstrip("/")
        self.graph = graph or os.getenv("FAROS_GRAPH") or FAROS_DEFAULT_GRAPH
        self.graphql_endpoint = f"{resolved_api_url}/graphs/{self.graph}/graphql"
        self.origin = origin or os.getenv("FAROS_ORIGIN") or FAROS_DEFAULT_ORIGIN
        self.user_source = (
            user_source or os.getenv("FAROS_USER_SOURCE") or FAROS_DEFAULT_USER_SOURCE
        )
        self.tool_category = (
            tool_category
            or os.getenv("FAROS_TOOL_CATEGORY")
            or FAROS_DEFAULT_TOOL_CATEGORY
        )
        self.async_httpx_client = async_httpx_client or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        super().__init__(**kwargs, flush_lock=asyncio.Lock())
        try:
            asyncio.create_task(self.periodic_flush())
        except RuntimeError:
            verbose_logger.debug(
                "FarosLogger: no running event loop; relying on batch_size flushes"
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if payload is None:
            verbose_logger.warning(
                "FarosLogger: standard_logging_object missing, skipping event"
            )
            return
        self.log_queue.append(self._usage_record(payload))
        if len(self.log_queue) >= self.batch_size:
            await self.flush_queue()

    def _usage_record(self, payload: StandardLoggingPayload) -> Dict[str, str]:
        metadata = payload.get("metadata") or {}
        key_user_id = metadata.get("user_api_key_user_id")
        if key_user_id == LITELLM_PROXY_ADMIN_NAME:
            # placeholder identity for master key / admin requests, not a real user
            key_user_id = None
        user_uid = (
            metadata.get("user_api_key_user_email")
            or key_user_id
            or payload.get("end_user")
            or "unknown"
        )
        used_at = datetime.fromtimestamp(
            payload["startTime"], tz=timezone.utc
        ).isoformat(timespec="milliseconds")
        return {"user_uid": user_uid, "used_at": used_at}

    def _usage_insert_input(self, record: Dict[str, str]) -> Dict[str, Any]:
        return {
            "usedAt": record["used_at"],
            "origin": self.origin,
            "userTool": {
                "data": {
                    "tool": {"category": self.tool_category},
                    "origin": self.origin,
                    "user": {
                        "data": {
                            "uid": record["user_uid"],
                            "source": self.user_source,
                            "origin": self.origin,
                        },
                        "on_conflict": {
                            "constraint": "vcs_User_pkey",
                            "update_columns": ["refreshedAt"],
                        },
                    },
                },
                "on_conflict": {
                    "constraint": "vcs_UserTool_pkey",
                    "update_columns": ["refreshedAt"],
                },
            },
        }

    async def async_send_batch(self):
        if not self.log_queue:
            return
        # (user, usedAt) is the row's primary key; a single upsert statement
        # cannot touch the same row twice
        unique_records = {
            (record["user_uid"], record["used_at"]): record for record in self.log_queue
        }
        usages: List[Dict[str, Any]] = [
            self._usage_insert_input(record) for record in unique_records.values()
        ]
        response = await self.async_httpx_client.post(
            url=self.graphql_endpoint,
            json={"query": USAGE_MUTATION, "variables": {"usages": usages}},
            headers={"authorization": self.api_key, "content-type": "application/json"},
        )
        response.raise_for_status()
        errors = response.json().get("errors")
        if errors:
            raise ValueError(f"Faros GraphQL request failed: {errors}")
        verbose_logger.debug(
            "FarosLogger: sent %s usage records to Faros graph %s",
            len(usages),
            self.graph,
        )

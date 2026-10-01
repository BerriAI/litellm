"""Databricks Zerobus logging integration."""

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Final
from urllib.parse import urlsplit

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.zerobus.client import ZerobusIngestClient, ZerobusIngestError
from litellm.integrations.zerobus.row import trace_row
from litellm.litellm_core_utils.redact_messages import (
    redacted_standard_logging_payload,
    should_redact_message_logging,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.zerobus import ZerobusConnection, ZerobusInitParams

_ENV_REFERENCE_PREFIX: Final = "os.environ/"


def _resolved_secret(value: str | None) -> str | None:
    """Resolve a config value that may name a secret; an unset ``os.environ/NAME`` stays unresolved."""
    if value is None:
        return None
    resolved: Final = get_secret_str(value)
    if resolved:
        return resolved
    return None if value.startswith(_ENV_REFERENCE_PREFIX) else value


def _configured_params() -> ZerobusInitParams:
    configured: Final = litellm.zerobus_params
    if isinstance(configured, ZerobusInitParams):
        return configured
    if isinstance(configured, Mapping):
        return ZerobusInitParams.model_validate(configured)
    return ZerobusInitParams()


def _setting(configured: str | None, env_var: str) -> str:
    """Prefer the configured value, falling back to the environment the proxy UI writes."""
    value: Final = _resolved_secret(configured) or get_secret_str(env_var)
    if not value:
        raise ValueError(
            f"zerobus logging requires {env_var}. Set it in the environment, or "
            f"litellm_settings.zerobus_params.{env_var.removeprefix('ZEROBUS_').lower()} in config.yaml"
        )
    return value


def _workspace_id(server_endpoint: str) -> str:
    """The Zerobus endpoint is ``https://<workspace_id>.zerobus.<region>.<cloud>``, so the id is its first label."""
    host: Final = urlsplit(server_endpoint).hostname or ""
    workspace_id: Final = host.split(".", 1)[0]
    if not workspace_id.isdigit():
        raise ValueError(
            f"ZEROBUS_SERVER_ENDPOINT {server_endpoint!r} does not look like "
            "https://<workspace_id>.zerobus.<region>.cloud.databricks.com"
        )
    return workspace_id


def _table_name(configured: str | None) -> str:
    table_name: Final = _setting(configured, "ZEROBUS_TABLE_NAME")
    if table_name.count(".") != 2:
        raise ValueError(f"ZEROBUS_TABLE_NAME {table_name!r} must be fully qualified as catalog.schema.table")
    return table_name


def connection_for(params: ZerobusInitParams) -> ZerobusConnection:
    """The connection configured right now, so a UI edit takes effect without a restart."""
    server_endpoint: Final = _setting(params.server_endpoint, "ZEROBUS_SERVER_ENDPOINT")
    return ZerobusConnection(
        workspace_url=_setting(params.workspace_url, "ZEROBUS_WORKSPACE_URL"),
        workspace_id=_workspace_id(server_endpoint),
        server_endpoint=server_endpoint,
        client_id=_setting(params.client_id, "ZEROBUS_CLIENT_ID"),
        client_secret=_setting(params.client_secret, "ZEROBUS_CLIENT_SECRET"),
        table_name=_table_name(params.table_name),
    )


class ZerobusLogger(CustomBatchLogger):
    preserve_events_added_during_flush = True

    def __init__(
        self,
        params: ZerobusInitParams | None = None,
        client: ZerobusIngestClient | None = None,
        start_periodic_flush: bool = True,
    ) -> None:
        resolved: Final = params if params is not None else _configured_params()
        self.params: Final = resolved
        self.given_client: Final = client
        self._cached_client: ZerobusIngestClient | None = None
        if client is None:
            connection_for(resolved)
        super().__init__(
            flush_lock=asyncio.Lock(),
            batch_size=resolved.batch_size,
            flush_interval=resolved.flush_interval,
            turn_off_message_logging=bool(resolved.turn_off_message_logging),
        )
        self._flushing: bool = False
        self._batch_flush_task: asyncio.Task[None] | None = None
        self._periodic_flush_task: asyncio.Task[None] | None = (
            self._start_periodic_flush_task() if start_periodic_flush else None
        )

    @property
    def client(self) -> ZerobusIngestClient:
        """A client for the current connection, kept while the connection is unchanged so its token is reused."""
        if self.given_client is not None:
            return self.given_client
        connection: Final = connection_for(self.params)
        cached: Final = self._cached_client
        if cached is not None and cached.connection == connection:
            return cached
        fresh: Final = ZerobusIngestClient(
            connection=connection,
            http_client=get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback),
        )
        self._cached_client = fresh
        return fresh

    def _start_periodic_flush_task(self) -> asyncio.Task[None] | None:
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(self.periodic_flush())

    def _start_batch_flush_task(self) -> None:
        if self._batch_flush_task is not None and not self._batch_flush_task.done():
            return
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._batch_flush_task = loop.create_task(self.flush_queue(skip_if_flushing=True))

    def _flush_task_is_alive(self) -> bool:
        task: Final = self._periodic_flush_task
        return task is not None and not task.done() and not task.get_loop().is_closed()

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        await self._enqueue(kwargs)

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        await self._enqueue(kwargs)

    async def _enqueue(self, kwargs: Mapping[str, object]) -> None:
        try:
            if not self._flush_task_is_alive():
                self._periodic_flush_task = self._start_periodic_flush_task()

            payload: Final = self._payload_for(kwargs)
            if payload is None:
                verbose_logger.debug("zerobus: event carried no standard_logging_object, skipping")
                return

            if self._flushing and len(self.log_queue) >= self.max_queue_size:
                verbose_logger.warning("zerobus: queue at %s rows during a flush, dropped a row", self.max_queue_size)
                return

            self.log_queue.append(trace_row(payload))
            self._drop_overflow()
            if len(self.log_queue) >= self.batch_size:
                self._start_batch_flush_task()
        except Exception:  # noqa: BLE001  # logging must never break the request path
            verbose_logger.exception("zerobus: failed to queue an event")

    def _payload_for(self, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        """The payload to buffer, redacted the way the framework redacts the success path."""
        details: Final = self.redact_standard_logging_payload_from_model_call_details(
            dict(kwargs)  # mutable-ok: both framework helpers take the call details as a dict
        )
        payload: Final = details.get("standard_logging_object")
        if not isinstance(payload, dict):
            return None
        if should_redact_message_logging(details):
            return redacted_standard_logging_payload(payload)
        return payload

    def _drop_overflow(self) -> None:
        """Trim the oldest rows, except mid flush when the in-flight batch is the head of the queue."""
        if self._flushing:
            return
        overflow: Final = len(self.log_queue) - self.max_queue_size
        if overflow <= 0:
            return
        del self.log_queue[:overflow]
        verbose_logger.warning("zerobus: queue over %s rows, dropped %s oldest", self.max_queue_size, overflow)

    async def flush_queue(self, skip_if_flushing: bool = False) -> None:
        if skip_if_flushing and self._flushing:
            return
        self._flushing = True
        try:
            await super().flush_queue()
        finally:
            self._flushing = False

    async def async_send_batch(self) -> None:
        """A retryable failure propagates so the rows are kept; a permanent one drops them so the queue moves on."""
        rows: Final = tuple(self.log_queue)
        if not rows:
            return
        failure: Final = await self.client.insert(rows)
        if failure is None:
            return
        if failure.retryable:
            raise ZerobusIngestError(failure.detail)
        verbose_logger.error("zerobus: dropping %s rows, %s", len(rows), failure.detail)

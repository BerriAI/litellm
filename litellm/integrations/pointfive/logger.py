"""
PointFive logging integration.

Buffers ``StandardLoggingPayload`` records and ships each flush as one gzipped
newline-delimited JSON object, rather than one object per request. Uploads go through a
presigned URL issued by the PointFive API, so the proxy needs no cloud credentials and
runs unchanged wherever it is hosted.
"""

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.pointfive.payload import chunk_lines, encode_lines, serialize_records
from litellm.integrations.pointfive.upload_client import PointFiveUploadClient, PointFiveUploadError
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.integrations.pointfive import DEFAULT_API_URL, PointFiveInitParams, PointFiveUploadFailure

_ENV_REFERENCE_PREFIX: Final = "os.environ/"


def _resolved_secret(value: str | None) -> str | None:
    """Resolve an ``os.environ/VAR`` config reference, the way config.yaml spells secrets."""
    if value is None or not value.startswith(_ENV_REFERENCE_PREFIX):
        return value
    return get_secret_str(value)


def _configured_params() -> PointFiveInitParams:
    """Read ``litellm.pointfive_params``, validating a raw config dict on the way through."""
    configured: Final = litellm.pointfive_params
    if isinstance(configured, PointFiveInitParams):
        return configured
    if isinstance(configured, Mapping):
        return PointFiveInitParams.model_validate(configured)
    return PointFiveInitParams()


def _resolved_api_key(params: PointFiveInitParams) -> str | None:
    """Prefer the configured key, falling back to the environment the proxy UI writes."""
    return _resolved_secret(params.api_key) or get_secret_str("POINTFIVE_API_KEY")


def _resolved_api_url(params: PointFiveInitParams) -> str:
    """Prefer the configured url, then the environment, then the public endpoint."""
    return _resolved_secret(params.api_url) or get_secret_str("POINTFIVE_API_URL") or DEFAULT_API_URL


def _upload_client_for(params: PointFiveInitParams) -> PointFiveUploadClient:
    """
    Build an upload client for the key and url configured right now.

    Resolved per call rather than kept: the proxy ui writes new values into the
    environment of a running proxy, and reading them once would need a restart to take
    effect. ``get_async_httpx_client`` is cached, so this reuses the same connections.
    """
    api_key: Final = _resolved_api_key(params)
    if not api_key:
        raise ValueError(
            "pointfive logging requires an api key. Set POINTFIVE_API_KEY, or "
            "litellm_settings.pointfive_params.api_key in config.yaml"
        )
    return PointFiveUploadClient(
        api_key=api_key,
        api_url=_resolved_api_url(params),
        http_client=get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback),
        max_retries=params.max_upload_retries,
    )


class PointFiveLogger(CustomBatchLogger):
    """Batching callback that ships LiteLLM request logs to PointFive."""

    preserve_events_added_during_flush = True

    def __init__(
        self,
        params: PointFiveInitParams | None = None,
        upload_client: PointFiveUploadClient | None = None,
        start_periodic_flush: bool = True,
    ) -> None:
        resolved: Final = params if params is not None else _configured_params()
        self.max_batch_bytes: Final = resolved.max_batch_bytes
        self.params: Final = resolved
        self.given_upload_client: Final = upload_client
        if upload_client is None:
            _upload_client_for(resolved)  # refuse to start without a key, rather than at the first flush
        super().__init__(
            flush_lock=asyncio.Lock(),
            batch_size=resolved.batch_size,
            flush_interval=resolved.flush_interval,
            turn_off_message_logging=bool(resolved.turn_off_message_logging),
        )
        # A caller that only wants one answer out of this logger, such as a health check,
        # passes start_periodic_flush=False so it leaves no flush task behind.
        self._flushing: bool = False
        self._periodic_flush_task: asyncio.Task[None] | None = (
            self._start_periodic_flush_task() if start_periodic_flush else None
        )

    @property
    def upload_client(self) -> PointFiveUploadClient:
        """The client for the currently configured key and url, so a ui edit needs no restart."""
        if self.given_upload_client is not None:
            return self.given_upload_client
        return _upload_client_for(self.params)

    def _start_periodic_flush_task(self) -> asyncio.Task[None] | None:
        """Start the periodic flush only once an event loop is actually running."""
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(self.periodic_flush())

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
        """Buffer one record, flushing early once the batch threshold is reached."""
        try:
            if self._periodic_flush_task is None or self._periodic_flush_task.done():
                self._periodic_flush_task = self._start_periodic_flush_task()

            payload: Final = kwargs.get("standard_logging_object")
            if not isinstance(payload, dict):
                verbose_logger.debug("pointfive: event carried no standard_logging_object, skipping")
                return

            self.log_queue.append(payload)
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue(skip_if_flushing=True)
        except Exception:  # noqa: BLE001  # logging must never break the request path
            verbose_logger.exception("pointfive: failed to queue an event")

    async def flush_queue(self, skip_if_flushing: bool = False) -> None:
        """
        Flush as usual, or report liveness when there is nothing to send.

        ``CustomBatchLogger`` skips an empty queue entirely, so without this an idle proxy
        would look identical to a dead one.

        ``skip_if_flushing`` is what a full batch passes. Uploading one takes seconds, and
        every event arriving meanwhile crosses the threshold too, so each would queue on the
        flush lock and then ship the handful of records left behind it. That turns one burst
        into a stream of tiny objects, which is what batching exists to avoid. The running
        flush already carries what is queued, and the interval catches whatever it missed.
        """
        if not self.log_queue:
            await self._ping()
            return
        if skip_if_flushing and self._flushing:
            return

        self._flushing = True
        try:
            await super().flush_queue()
        finally:
            self._flushing = False

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """Answer the proxy ui test button by asking the api whether it accepts this key."""
        try:
            failure: Final = await self.upload_client.ping()
        except ValueError as missing_key:
            return IntegrationHealthCheckStatus(status="unhealthy", error_message=str(missing_key))
        if failure is not None:
            return IntegrationHealthCheckStatus(status="unhealthy", error_message=failure.detail)
        return IntegrationHealthCheckStatus(status="healthy", error_message=None)

    async def _ping(self) -> None:
        """Report liveness, never failing the flush over it."""
        try:
            failure: Final = await self.upload_client.ping()
        except ValueError as missing_key:
            verbose_logger.warning("pointfive: liveness ping skipped, %s", missing_key)
            return
        if failure is not None:
            verbose_logger.warning("pointfive: liveness ping failed, %s", failure.detail)

    async def async_send_batch(self) -> None:
        """
        Upload everything queued, split into objects of at most ``max_batch_bytes``.

        A retryable failure propagates so ``CustomBatchLogger`` keeps the whole batch for
        the next flush, which can re-send objects that already landed, making delivery
        at-least-once. A rejection the server will refuse again drops the object instead,
        since holding it would block every record queued behind it.
        """
        pending: Final = tuple(self.log_queue)
        if not pending:
            return

        for chunk in chunk_lines(serialize_records(pending), self.max_batch_bytes):
            outcome = await self.upload_client.upload(await encode_lines(chunk))
            if not isinstance(outcome, PointFiveUploadFailure):
                continue
            if outcome.retryable:
                raise PointFiveUploadError(outcome.detail)
            verbose_logger.error("pointfive: dropping %s records, %s", len(chunk), outcome.detail)

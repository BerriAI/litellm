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
from litellm.litellm_core_utils.redact_messages import (
    redacted_standard_logging_payload,
    should_redact_message_logging,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.integrations.pointfive import DEFAULT_API_URL, PointFiveInitParams, PointFiveUploadFailure

_ENV_REFERENCE_PREFIX: Final = "os.environ/"


def _resolved_secret(value: str | None) -> str | None:
    """
    Resolve a config value that may name a secret, in any shape the secret manager accepts.

    A reference that resolves to nothing stays unresolved rather than falling back to its own
    text, so an unset ``os.environ/NAME`` reports a missing key instead of being sent as one.
    """
    if value is None:
        return None
    resolved: Final = get_secret_str(value)
    if resolved:
        return resolved
    return None if value.startswith(_ENV_REFERENCE_PREFIX) else value


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
        self._flushing: bool = False
        self._batch_flush_task: asyncio.Task[None] | None = None
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

    def _start_batch_flush_task(self) -> None:
        """
        Upload a full batch in the background, so no request waits on PointFive.

        Awaiting it here put the upload, its retries and their backoff on the caller's
        path, and a hung api held a response open for as long as the attempts took.
        """
        if self._batch_flush_task is not None and not self._batch_flush_task.done():
            return
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._batch_flush_task = loop.create_task(self.flush_queue(skip_if_flushing=True))

    def _flush_task_is_alive(self) -> bool:
        """A task whose loop has been closed never runs again, yet never reports itself done."""
        task: Final = self._periodic_flush_task
        return task is not None and not task.done() and not task.get_loop().is_closed()

    async def periodic_flush(self) -> None:
        """
        Report in straight away, then flush on the interval as usual.

        The inherited loop sleeps first, so a proxy that has just loaded the callback says
        nothing for a whole interval, five minutes by default. PointFive shows the integration
        as still waiting for its first call for all that time, which reads as a broken setup
        rather than an idle one. An empty queue makes this first cycle a ping, so a proxy with
        no traffic yet announces itself without uploading an object that holds no records.
        """
        await self.flush_queue(skip_if_flushing=True)
        await super().periodic_flush()

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
            if not self._flush_task_is_alive():
                self._periodic_flush_task = self._start_periodic_flush_task()

            record: Final = self._record_for(kwargs)
            if record is None:
                verbose_logger.debug("pointfive: event carried no standard_logging_object, skipping")
                return

            self.log_queue.append(record)
            self._drop_overflow()
            if len(self.log_queue) >= self.batch_size:
                self._start_batch_flush_task()
        except Exception:  # noqa: BLE001  # logging must never break the request path
            verbose_logger.exception("pointfive: failed to queue an event")

    def _record_for(self, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
        """
        The record to buffer, redacted the way the framework would have redacted it.

        A success reaches a callback already redacted, an async failure does not, so both
        the excluded-field list and this callback's own setting are applied here, then the
        global, per-request and header settings that only the framework's predicate knows.
        """
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
        """
        Hold the queue to its cap as records arrive, not only after a flush has failed.

        Never while a flush is running: it holds a snapshot taken by length, and trimming
        the front underneath it would make the post-flush drain remove records that arrived
        during the upload and were never sent. The next arrival after the flush trims.
        """
        if self._flushing:
            return
        overflow: Final = len(self.log_queue) - self.max_queue_size
        if overflow <= 0:
            return
        del self.log_queue[:overflow]
        verbose_logger.warning("pointfive: queue over %s records, dropped %s oldest", self.max_queue_size, overflow)

    async def flush_queue(self, skip_if_flushing: bool = False) -> None:
        """
        Flush as usual, or report liveness when there is nothing to send.

        ``CustomBatchLogger`` skips an empty queue entirely, so without this an idle proxy
        would look identical to a dead one.

        ``skip_if_flushing`` is what a full batch, and the loop's opening cycle, pass. Uploading one takes seconds, and
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

        A retryable failure propagates so ``CustomBatchLogger`` keeps the rest of the batch
        for the next flush; the records already shipped or already refused leave the queue
        first, so a retry re-sends at most the object that failed. A rejection the server
        will refuse again drops that object, since holding it would block every record
        queued behind it.
        """
        pending: Final = tuple(self.log_queue)
        if not pending:
            return

        client: Final = self.upload_client
        chunks: Final = chunk_lines(serialize_records(pending), self.max_batch_bytes)
        for index, chunk in enumerate(chunks):
            outcome = await client.upload(await encode_lines(chunk))
            if not isinstance(outcome, PointFiveUploadFailure):
                continue
            if outcome.retryable:
                del self.log_queue[: sum(len(shipped) for shipped in chunks[:index])]
                raise PointFiveUploadError(outcome.detail)
            verbose_logger.error("pointfive: dropping %s records, %s", len(chunk), outcome.detail)

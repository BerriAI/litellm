"""
New Relic Metric API Integration - sends per-team cost/usage metrics to /metric/v1

NR Reference API: https://docs.newrelic.com/docs/data-apis/ingest-apis/metric-api/introduction-metric-api/

`async_log_success_event` / `async_log_failure_event` queue one record per request;
at flush the queue is aggregated by (team, model group, model, provider, status)
into count/summary metrics. `interval.ms` is the real window between flushes,
computed at flush time.

Team-scoped by construction: the ingest key is injected explicitly and there is
deliberately no environment-variable fallback, so a team's metrics are never sent
with the proxy operator's credentials (mirrors ``allow_env_credentials=False`` on
the Datadog team logger).

Error policy on flush: 4xx drops the batch (a retry would fail identically; 403
is a permanent credential failure), 5xx/network re-queues capped at
``max_queue_size`` records with the oldest dropped.

For batching specific details see CustomBatchLogger class
"""

import asyncio
import gzip
import time
import traceback
from collections.abc import Mapping
from math import ceil
from types import MappingProxyType
from typing import Final

from httpx import HTTPStatusError, Response

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.newrelic import (
    NEWRELIC_DEFAULT_REGION,
    NEWRELIC_METRIC_ATTRIBUTE_MAX_LEN,
    NEWRELIC_METRIC_COMPLETION_TOKENS,
    NEWRELIC_METRIC_COST_USD,
    NEWRELIC_METRIC_ENDPOINT_BY_REGION,
    NEWRELIC_METRIC_PROMPT_TOKENS,
    NEWRELIC_METRIC_REQUEST_DURATION_MS,
    NEWRELIC_METRIC_REQUESTS,
    NEWRELIC_METRIC_TOTAL_TOKENS,
    NEWRELIC_METRICS_MAX_BATCH_SIZE,
    NEWRELIC_METRICS_MAX_DRAIN_PASSES,
    NEWRELIC_METRICS_MAX_RETRY_QUEUE_SIZE,
    NewRelicCountMetric,
    NewRelicMetric,
    NewRelicMetricCommon,
    NewRelicMetricEnvelope,
    NewRelicMetricRecord,
    NewRelicSummaryMetric,
    NewRelicSummaryValue,
)
from litellm.types.utils import StandardLoggingPayload

# 408 (request timeout) and 429 (rate limit) are transient client errors the
# Metric API expects a retry on, unlike 400/403 which a retry would only repeat.
_RETRYABLE_CLIENT_STATUSES: Final = frozenset({408, 429})


def resolve_newrelic_metric_endpoint(newrelic_region: str | None) -> str:
    if not newrelic_region:
        return NEWRELIC_METRIC_ENDPOINT_BY_REGION[NEWRELIC_DEFAULT_REGION]
    endpoint: Final = NEWRELIC_METRIC_ENDPOINT_BY_REGION.get(newrelic_region.lower())
    if endpoint is None:
        verbose_logger.warning(
            "New Relic: unknown newrelic_region %r; supported regions: %s. Using the default (US) endpoint.",
            newrelic_region,
            ", ".join(sorted(NEWRELIC_METRIC_ENDPOINT_BY_REGION)),
        )
        return NEWRELIC_METRIC_ENDPOINT_BY_REGION[NEWRELIC_DEFAULT_REGION]
    return endpoint


def _metric_record_from_payload(standard_logging_object: StandardLoggingPayload) -> NewRelicMetricRecord:
    metadata: Final = standard_logging_object.get("metadata")
    team_id: Final = ((metadata.get("user_api_key_team_id") or metadata.get("team_id")) if metadata else None) or ""
    team_alias: Final = (
        (metadata.get("user_api_key_team_alias") or metadata.get("team_alias")) if metadata else None
    ) or ""
    return NewRelicMetricRecord(
        team_id=team_id,
        team_alias=team_alias,
        model_group=standard_logging_object.get("model_group") or "",
        model=standard_logging_object.get("model") or "",
        custom_llm_provider=standard_logging_object.get("custom_llm_provider") or "",
        status=str(standard_logging_object.get("status") or "success"),
        response_cost=float(standard_logging_object.get("response_cost") or 0.0),
        prompt_tokens=int(standard_logging_object.get("prompt_tokens") or 0),
        completion_tokens=int(standard_logging_object.get("completion_tokens") or 0),
        total_tokens=int(standard_logging_object.get("total_tokens") or 0),
        duration_ms=float(standard_logging_object.get("response_time") or 0.0) * 1000.0,
    )


def _bucket_metrics(bucket_records: tuple[NewRelicMetricRecord, ...]) -> tuple[NewRelicMetric, ...]:
    first: Final = bucket_records[0]
    attributes: Final[Mapping[str, str]] = {  # mutable-ok: JSON leaf; safe_dumps stringifies MappingProxyType
        key: value[:NEWRELIC_METRIC_ATTRIBUTE_MAX_LEN]
        for key, value in (
            ("team_id", first.team_id),
            ("team_alias", first.team_alias),
            ("model_group", first.model_group),
            ("model", first.model),
            ("custom_llm_provider", first.custom_llm_provider),
            ("status", first.status),
        )
        if value
    }
    durations: Final = tuple(record.duration_ms for record in bucket_records)
    counts: Final[tuple[tuple[str, float], ...]] = (
        (NEWRELIC_METRIC_REQUESTS, float(len(bucket_records))),
        (NEWRELIC_METRIC_COST_USD, sum(record.response_cost for record in bucket_records)),
        (NEWRELIC_METRIC_PROMPT_TOKENS, float(sum(record.prompt_tokens for record in bucket_records))),
        (NEWRELIC_METRIC_COMPLETION_TOKENS, float(sum(record.completion_tokens for record in bucket_records))),
        (NEWRELIC_METRIC_TOTAL_TOKENS, float(sum(record.total_tokens for record in bucket_records))),
    )
    count_metrics: Final[tuple[NewRelicMetric, ...]] = tuple(
        NewRelicCountMetric(name=name, type="count", value=value, attributes=attributes) for name, value in counts
    )
    summary_metric: Final = NewRelicSummaryMetric(
        name=NEWRELIC_METRIC_REQUEST_DURATION_MS,
        type="summary",
        value=NewRelicSummaryValue(
            count=len(durations),
            sum=sum(durations),
            min=min(durations),
            max=max(durations),
        ),
        attributes=attributes,
    )
    return (*count_metrics, summary_metric)


def build_metric_payload(
    records: tuple[NewRelicMetricRecord, ...],
    *,
    window_start: float,
    now: float,
) -> tuple[NewRelicMetricEnvelope, ...]:
    """Aggregates records into one Metric API envelope for the flush window."""
    interval_ms: Final = max(1, int((now - window_start) * 1000))
    bucket_keys: Final = tuple(dict.fromkeys(record.bucket_key for record in records))
    metrics: Final = tuple(
        metric
        for key in bucket_keys
        for metric in _bucket_metrics(tuple(record for record in records if record.bucket_key == key))
    )
    common: Final[NewRelicMetricCommon] = {
        "timestamp": int(window_start * 1000),
        "interval.ms": interval_ms,
    }
    return (NewRelicMetricEnvelope(common=common, metrics=metrics),)


class NewRelicMetricsLogger(CustomBatchLogger):
    def __init__(
        self,
        newrelic_api_key: str,
        newrelic_region: str | None = None,
    ) -> None:
        if not newrelic_api_key:
            raise ValueError(
                "newrelic_api_key is required for NewRelicMetricsLogger; "
                "team-scoped metrics never fall back to environment credentials"
            )
        self.newrelic_api_key: Final = newrelic_api_key
        self.metric_api_url: Final = resolve_newrelic_metric_endpoint(newrelic_region)
        self.async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        self._stopped: bool = False
        self._drain_lock = asyncio.Lock()
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()
        super().__init__(
            flush_lock=self.flush_lock,
            batch_size=NEWRELIC_METRICS_MAX_BATCH_SIZE,
            max_queue_size=NEWRELIC_METRICS_MAX_RETRY_QUEUE_SIZE,
        )

    def stop(self) -> None:
        """Ends the periodic flush loop; called on DynamicLoggingCache eviction.

        Schedules one final drain of anything still queued, so eviction never
        silently discards records. Guarded so it can never raise into the
        cache's eviction path.
        """
        self._stopped = True
        try:
            asyncio.get_running_loop().create_task(self._final_drain())
        except Exception:  # noqa: BLE001  # no running loop / shutdown; the periodic loop's final drain still runs
            verbose_logger.debug("New Relic Metrics: could not schedule final drain on stop()", exc_info=True)

    async def _drain_with_retry(self) -> None:
        """Deliver everything queued on a stopped logger, or drop it with a log.

        A stopped logger has no periodic loop left, so every post-stop path
        funnels through here. ``_drain_lock`` serializes drains: a callback that
        appends and starts its own drain queues behind the running one instead
        of racing it. Each pass attempts the whole current queue in
        ``batch_size`` chunks, unlike the periodic path it does not stop at the
        first failing chunk, so a persistently failing head never starves the
        tail. Only after ``_MAX_DRAIN_PASSES`` against a permanently failing
        destination is the remainder dropped, and then only the records that were
        queued when this drain began, so every dropped record got the full retry
        budget: a record a callback appended mid-drain is not in that snapshot,
        so it is left for its own serialized drain rather than dropped after
        fewer attempts, and is never stranded.
        """
        async with self._drain_lock:
            attempted: Final = tuple(self.log_queue)
            for _pass in range(NEWRELIC_METRICS_MAX_DRAIN_PASSES):
                await self._drain_flush_once()
                if not self.log_queue:
                    return
                if _pass < NEWRELIC_METRICS_MAX_DRAIN_PASSES - 1:
                    await asyncio.sleep(2**_pass)
            async with self.flush_lock:
                tried_ids: Final = frozenset(id(record) for record in attempted)
                survivors: Final = tuple(record for record in self.log_queue if id(record) not in tried_ids)
                dropped: Final = len(self.log_queue) - len(survivors)
                if dropped:
                    verbose_logger.warning(
                        "New Relic Metrics: dropping %s records after %s drain passes",
                        dropped,
                        NEWRELIC_METRICS_MAX_DRAIN_PASSES,
                    )
                self.log_queue[:] = list(survivors)  # mutable-ok: leave late arrivals for the next serialized drain

    async def _drain_flush_once(self) -> None:
        """Attempt every queued record once, in ``batch_size`` chunks, without
        stopping at the first failing chunk so a persistently failing head does
        not starve the tail (the periodic ``flush_queue`` deliberately stops
        instead). Takes the queue under ``flush_lock`` and re-queues only the
        chunks a 5xx/network error left undelivered, so records a concurrent
        request appends during the sends survive for the next pass."""
        async with self.flush_lock:
            pending: Final = tuple(self.log_queue)
            window_start: Final = self.last_flush_time
            self.last_flush_time = time.time()
            del self.log_queue[:]
        if not pending:
            return
        chunks: Final = tuple(
            pending[start : start + self.batch_size] for start in range(0, len(pending), self.batch_size)
        )
        delivered: Final = tuple([await self._classify_and_send(chunk, window_start) for chunk in chunks])
        failed: Final = tuple(record for chunk, ok in zip(chunks, delivered) for record in (() if ok else chunk))
        if failed:
            self._requeue(failed)

    async def _final_drain(self) -> None:
        await self._drain_with_retry()

    async def periodic_flush(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.flush_interval)
            if self._stopped:
                break
            await self.flush_queue()
        await self._final_drain()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            await self._log_async_event(standard_logging_object=kwargs.get("standard_logging_object", None))
        except Exception as e:  # noqa: BLE001  # logging must never break the request path
            verbose_logger.exception("New Relic Metrics Layer Error - %s\n%s", e, traceback.format_exc())

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            await self._log_async_event(standard_logging_object=kwargs.get("standard_logging_object", None))
        except Exception as e:  # noqa: BLE001  # logging must never break the request path
            verbose_logger.exception("New Relic Metrics Layer Error - %s\n%s", e, traceback.format_exc())

    async def _log_async_event(self, standard_logging_object: StandardLoggingPayload | None) -> None:
        if standard_logging_object is None:
            raise ValueError("standard_logging_object not found in kwargs")
        self.log_queue.append(_metric_record_from_payload(standard_logging_object))
        if self._stopped:
            # A stopped logger has no periodic loop left; an in-flight callback
            # that appends after the eviction drain delivers its own record.
            await self._drain_with_retry()
            return
        if len(self.log_queue) >= self.batch_size:
            await self.flush_queue()

    async def flush_queue(self) -> None:
        async with self.flush_lock:
            window_start: Final = self.last_flush_time
            self.last_flush_time = time.time()
            queued: Final = len(self.log_queue)
            if not queued:
                return
            verbose_logger.debug("New Relic Metrics: Flushing %s queued records", queued)
            # Bounded by what is queued now: records appended mid-flush belong to
            # the next window, and looping until empty would never end under load.
            for _chunk in range(ceil(queued / self.batch_size)):
                if not await self.async_send_batch(window_start=window_start):
                    return

    async def async_send_batch(self, window_start: float | None = None) -> bool:
        """Sends the oldest ``batch_size`` records only, so a queue grown past that
        by re-queues cannot breach the Metric API data point cap in one request.
        Returns False once a chunk fails and is re-queued, so the caller stops."""
        if not self.log_queue:
            return False

        batch_to_send: Final[tuple[NewRelicMetricRecord, ...]] = tuple(self.log_queue[: self.batch_size])
        del self.log_queue[: len(batch_to_send)]

        delivered: Final = await self._classify_and_send(
            batch_to_send, window_start if window_start is not None else self.last_flush_time
        )
        if not delivered:
            self._requeue(batch_to_send)
        return delivered

    async def _classify_and_send(self, batch: tuple[NewRelicMetricRecord, ...], window_start: float) -> bool:
        """Send one chunk and classify the outcome, never touching the queue.
        Returns True when the batch is done with (delivered on any 2xx, or a 4xx
        a retry would only repeat, 403 being a permanent bad-key rejection), and
        False when a 5xx or network error means the caller should re-queue it.

        ``AsyncHTTPHandler.post`` raises ``HTTPStatusError`` on any non-2xx, so a
        4xx never returns a response here; the status is read off the raised
        error to keep the client-error path (drop) distinct from 5xx (retry)."""
        payload: Final = build_metric_payload(records=batch, window_start=window_start, now=time.time())
        try:
            status = (
                await self.async_send_compressed_data(payload)
            ).status_code  # rebind-ok: reassigned from the raised HTTPStatusError below
        except HTTPStatusError as e:
            status = e.response.status_code
        except Exception as e:  # noqa: BLE001  # transport/network failure re-queues the batch
            verbose_logger.warning(
                "New Relic Metrics: network error sending %s records, will retry - %s",
                len(batch),
                e,
            )
            return False

        if 200 <= status < 300:
            return True

        if 400 <= status < 500 and status not in _RETRYABLE_CLIENT_STATUSES:
            verbose_logger.warning(
                "New Relic Metrics: %s from Metric API%s, dropping %s records.",
                status,
                " (permanent credential failure: invalid or revoked team ingest key)" if status == 403 else "",
                len(batch),
            )
            return True

        verbose_logger.warning(
            "New Relic Metrics: %s from Metric API, will retry %s records",
            status,
            len(batch),
        )
        return False

    def _requeue(self, batch: tuple[NewRelicMetricRecord, ...]) -> None:
        """Prepends ``batch`` in place (never by assignment: records appended by
        concurrent requests during the flush await must survive), keeping
        chronological order so the cap drops the oldest records first."""
        self.log_queue[:0] = batch
        overflow: Final = len(self.log_queue) - self.max_queue_size
        if overflow > 0:
            del self.log_queue[:overflow]
            verbose_logger.warning(
                "New Relic Metrics: retry queue exceeded max_queue_size=%s; dropped %s oldest records.",
                self.max_queue_size,
                overflow,
            )

    async def async_send_compressed_data(self, payload: tuple[NewRelicMetricEnvelope, ...]) -> Response:
        compressed_data: Final = gzip.compress(safe_dumps(payload).encode("utf-8"))
        headers: Final[Mapping[str, str]] = MappingProxyType(
            {
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Api-Key": self.newrelic_api_key,
            }
        )
        return await self.async_client.post(
            url=self.metric_api_url,
            data=compressed_data,
            headers=headers,
        )

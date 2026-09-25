"""
Accumulates gateway request counts (SGR) recorded at the ASGI edge and commits
them to ``LiteLLM_DailyGatewayRequests``.

Unlike the spend queues this keeps no per-request item. A count is a pure
aggregate, so requests fold into an in-memory map as they finish. Every
dimension of the key is server-chosen and drawn from a fixed set: the date, the
category, and a route that the classifier maps to one of a closed list of
strings rather than passing the raw path through. Nothing a caller sends can
add a key, so the fold and the table it commits to are bounded by (days x
routes) however much traffic arrives, and the response path carries no
unbounded queue that would block once full.

A flush commits its whole snapshot as one multi-row ``INSERT ... ON CONFLICT DO
UPDATE`` rather than one upsert per key, so a worker costs the primary one
statement per interval however many routes it served. With
``use_redis_transaction_buffer`` on, workers instead push their snapshot to a
Redis list and one lock-holding pod folds every entry and writes the table, so
the deployment as a whole costs the primary one statement per interval.
"""

import json
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import MAX_REDIS_BUFFER_DEQUEUE_COUNT, REDIS_GATEWAY_REQUESTS_BUFFER_KEY
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory
from litellm.types.proxy.gateway_requests import (
    GatewayRequestCounts,
    GatewayRequestKey,
    GatewayRequestSnapshot,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_EMPTY: Final = GatewayRequestCounts(successful_requests=0, failed_requests=0)
_TABLE: Final = '"LiteLLM_DailyGatewayRequests"'
_COLUMNS_PER_ROW: Final = 5
_UTC_NOW: Final = "(NOW() AT TIME ZONE 'UTC')"
GATEWAY_REQUESTS_JOB_NAME: Final = "update_gateway_requests_job"

_BufferedRows: TypeAlias = tuple[tuple[str, str, str, int, int], ...]
_BUFFERED_ROWS: Final = TypeAdapter(_BufferedRows)
_BUFFERED_ENTRIES: Final = TypeAdapter(tuple[str | bytes, ...])
_NO_COUNTS: Final[GatewayRequestSnapshot] = MappingProxyType({})


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class GatewayRequestAccumulator:
    """Sink for the request-metrics middleware. ``record`` is sync and never awaits."""

    def __init__(self) -> None:
        self._counts: dict[GatewayRequestKey, GatewayRequestCounts] = {}  # mutable-ok: bounded fold, drained per flush

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None:
        key: Final = GatewayRequestKey(date=_utc_date(), category=category.value, route=route)
        self._counts[key] = self._counts.get(key, _EMPTY).plus(succeeded=200 <= status_code < 300)

    def drain(self) -> GatewayRequestSnapshot:
        drained: Final = self._counts
        self._counts = {}  # mutable-ok: the fold restarts empty; the drained map is handed off whole
        return drained

    def restore(self, snapshot: GatewayRequestSnapshot) -> None:
        """
        Merge un-committed counts back so the next flush retries them.

        A dropped flush would silently undercount the metric the dashboard now
        treats as the source of truth. Merging cannot grow without bound: keys
        collapse on collision, so the fold stays bounded by (date x category x
        route) however long the database is unreachable.

        This buys at-least-once, not exactly-once, and the cost is worth stating.
        The statement commits on the server before its acknowledgement is read, so
        a failure raised after the commit (a connection dropped while reading the
        acknowledgement) restores counts that are already persisted, and the next
        flush increments them a second time. Exactly-once would need a dedup key
        the upsert could ignore on replay. For a traffic-volume metric a rare
        overcount on a dropped acknowledgement beats losing a whole interval to
        every database blip, so the trade is deliberate.
        """
        self._counts = dict(fold_counts(chain(self._counts.items(), snapshot.items())))  # mutable-ok: fold replaced


def fold_counts(items: Iterable[tuple[GatewayRequestKey, GatewayRequestCounts]]) -> GatewayRequestSnapshot:
    """Sum counts key-wise; the result stays bounded by (date x category x route)."""
    folded: Final[dict[GatewayRequestKey, GatewayRequestCounts]] = {}  # mutable-ok: local fold returned once
    for key, counts in items:
        existing = folded.get(key, _EMPTY)
        folded[key] = GatewayRequestCounts(
            successful_requests=existing.successful_requests + counts.successful_requests,
            failed_requests=existing.failed_requests + counts.failed_requests,
        )
    return folded


def build_gateway_requests_upsert(snapshot: GatewayRequestSnapshot) -> tuple[str, tuple[str | int, ...]]:
    """
    One ``INSERT ... ON CONFLICT DO UPDATE`` that increments every (date, category,
    route) in the snapshot. Rows are ordered by the conflict key so concurrent
    writers lock rows in the same order and cannot deadlock.
    """
    ordered: Final = sorted(snapshot.items(), key=lambda item: (item[0].date, item[0].category, item[0].route))
    rows: Final = ", ".join(
        f"(${base + 1}::text, ${base + 2}::text, ${base + 3}::text, ${base + 4}::bigint, ${base + 5}::bigint, {_UTC_NOW})"
        for base in range(0, len(ordered) * _COLUMNS_PER_ROW, _COLUMNS_PER_ROW)
    )
    sql: Final = (
        f'INSERT INTO {_TABLE} ("date", "category", "route", "successful_requests", "failed_requests", "updated_at")\n'
        f"VALUES {rows}\n"
        'ON CONFLICT ("date", "category", "route") DO UPDATE SET\n'
        f'  "successful_requests" = {_TABLE}."successful_requests" + EXCLUDED."successful_requests",\n'
        f'  "failed_requests" = {_TABLE}."failed_requests" + EXCLUDED."failed_requests",\n'
        f'  "updated_at" = {_UTC_NOW}'
    )
    params: Final[tuple[str | int, ...]] = tuple(
        value
        for key, counts in ordered
        for value in (key.date, key.category, key.route, counts.successful_requests, counts.failed_requests)
    )
    return sql, params


async def commit_gateway_requests_to_db(
    *,
    prisma_client: "PrismaClient",
    snapshot: GatewayRequestSnapshot,
) -> None:
    """Increment every (date, category, route) in the snapshot with a single statement."""
    if not snapshot:
        return

    sql, params = build_gateway_requests_upsert(snapshot)
    await prisma_client.db.execute_raw(sql, *params)  # pyright: ignore[reportAny]  # untyped prisma client

    verbose_proxy_logger.debug(
        "Gateway request tracking - committed %d aggregated rows in one statement", len(snapshot)
    )


class GatewayRequestRedisBuffer:
    """
    Folds every worker's snapshot through one Redis list so a single pod per
    interval writes the table, mirroring the spend writer's transaction buffer.

    Each entry is one worker's snapshot as JSON rows; the lock holder pops them,
    sums them, and commits one statement. A commit failure pushes the summed
    rows back so the next holder retries, keeping the at-least-once guarantee.
    If that push fails too, the rows go back to the holder's own accumulator so
    they ride along with its next flush instead of vanishing with the pop.
    """

    def __init__(self, *, redis_cache: RedisCache, pod_lock_manager: PodLockManager) -> None:
        self._redis_cache: Final = redis_cache
        self._pod_lock_manager: Final = pod_lock_manager

    async def push(self, snapshot: GatewayRequestSnapshot) -> None:
        if not snapshot:
            return
        rows: Final[_BufferedRows] = tuple(
            (key.date, key.category, key.route, counts.successful_requests, counts.failed_requests)
            for key, counts in snapshot.items()
        )
        await self._redis_cache.async_rpush(key=REDIS_GATEWAY_REQUESTS_BUFFER_KEY, values=(json.dumps(rows),))

    async def _pop_batch(self) -> tuple[str | bytes, ...]:
        popped: Final[object] = await self._redis_cache.async_lpop(  # pyright: ignore[reportAny]  # redis returns Any
            key=REDIS_GATEWAY_REQUESTS_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT
        )
        if not popped:
            return ()
        return _BUFFERED_ENTRIES.validate_python(popped if isinstance(popped, list) else (popped,))

    async def _pop_all(self) -> AsyncIterator[str | bytes]:
        while True:
            batch = await self._pop_batch()
            for entry in batch:
                yield entry
            if len(batch) < MAX_REDIS_BUFFER_DEQUEUE_COUNT:
                return

    async def pop(self) -> GatewayRequestSnapshot:
        entries: Final = tuple([entry async for entry in self._pop_all()])
        return fold_counts(
            (
                GatewayRequestKey(date=date, category=category, route=route),
                GatewayRequestCounts(successful_requests=succeeded, failed_requests=failed),
            )
            for entry in entries
            for date, category, route, succeeded, failed in _BUFFERED_ROWS.validate_json(entry)
        )

    async def commit_if_leader(self, prisma_client: "PrismaClient") -> GatewayRequestSnapshot:
        """
        Drain the list and write it as one statement, but only on the pod holding the job lock.

        The lock is a lease, never released: the holder re-enters it on every flush and
        keeps committing alone until the TTL lapses, so the primary sees one statement
        per flush interval deployment-wide instead of one per worker.

        Returns the popped rows that could be neither committed nor re-queued, for the
        caller to keep in memory. Empty on success.
        """
        if not await self._pod_lock_manager.acquire_lock(cronjob_id=GATEWAY_REQUESTS_JOB_NAME):
            return _NO_COUNTS
        buffered: Final = await self.pop()
        try:
            await commit_gateway_requests_to_db(prisma_client=prisma_client, snapshot=buffered)
        except Exception:  # noqa: BLE001 -- a failed commit must not stop the scheduler
            verbose_proxy_logger.warning(
                "Gateway request tracking - failed to commit %d buffered rows, re-queuing to Redis for the next flush",
                len(buffered),
                exc_info=True,
            )
            return await self._requeue(buffered)
        return _NO_COUNTS

    async def _requeue(self, snapshot: GatewayRequestSnapshot) -> GatewayRequestSnapshot:
        try:
            await self.push(snapshot)
        except Exception:  # noqa: BLE001 -- the rows go back to the caller's accumulator instead
            verbose_proxy_logger.warning(
                "Gateway request tracking - Redis re-queue failed, keeping %d rows in memory for the next flush",
                len(snapshot),
                exc_info=True,
            )
            return snapshot
        return _NO_COUNTS


async def flush_gateway_requests(
    prisma_client: "PrismaClient",
    accumulator: GatewayRequestAccumulator,
    redis_buffer: GatewayRequestRedisBuffer | None = None,
) -> None:
    """
    Scheduler entrypoint. Never raises: a metering failure must not kill the job.

    With ``redis_buffer`` the snapshot goes to Redis and only the lease holder
    writes to Postgres. Shutdown passes no buffer so a departing worker writes its
    own counts directly instead of parking them behind a lease it may not hold.

    ``CancelledError`` is deliberately not caught, so a flush cancelled during
    shutdown drops its snapshot rather than restoring counts onto an accumulator
    the process is about to discard.
    """
    snapshot: Final = accumulator.drain()
    try:
        if redis_buffer is None:
            await commit_gateway_requests_to_db(prisma_client=prisma_client, snapshot=snapshot)
        else:
            await redis_buffer.push(snapshot)
    except Exception:  # noqa: BLE001 -- a failed flush must not stop the scheduler
        accumulator.restore(snapshot)
        verbose_proxy_logger.warning(
            "Gateway request tracking - failed to commit %d rows, retrying on the next flush",
            len(snapshot),
            exc_info=True,
        )
        return
    if redis_buffer is None:
        return
    try:
        accumulator.restore(await redis_buffer.commit_if_leader(prisma_client))
    except Exception:  # noqa: BLE001 -- entries still in Redis are drained by the next flush
        verbose_proxy_logger.warning(
            "Gateway request tracking - leader drain failed, buffered rows stay in Redis for the next flush",
            exc_info=True,
        )

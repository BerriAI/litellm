"""
Accumulates failed request counts per key, team, user and model group by HTTP
status and commits them to ``LiteLLM_DailyRequestErrors``.

The edge counters in ``LiteLLM_DailyGatewayFailedRequests`` say how many requests
failed with which status, but carry no caller dimension. This rollup is fed from
the spend writer, which sees the resolved key, team, user and model group of
every logged request together with the status code the logging callbacks
recorded, so the dashboard can answer who the failures land on. Only failures are
recorded, so the table grows with (days x failing callers x status codes), never
with successful traffic.

Flushing mirrors ``gateway_request_tracking``: one multi-row upsert per interval
per worker, or with ``use_redis_transaction_buffer`` one statement per interval
deployment-wide written by the lease holder.
"""

import asyncio
import json
from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    MAX_REDIS_BUFFER_DEQUEUE_COUNT,
    REDIS_REQUEST_ERRORS_BUFFER_KEY,
    REQUEST_ERRORS_MAX_ROWS_PER_UPSERT,
    REQUEST_ERRORS_UNKNOWN_STATUS_CODE,
)
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.types.proxy.request_errors import RequestErrorKey, RequestErrorSnapshot

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_REQUEST_ERROR_QUEUE_TARGET: Final = "request_error_queue"
_TABLE: Final = '"LiteLLM_DailyRequestErrors"'
_COLUMNS_PER_ROW: Final = 7
_UTC_NOW: Final = "(NOW() AT TIME ZONE 'UTC')"
REQUEST_ERRORS_JOB_NAME: Final = "update_request_errors_job"

_BufferedRow: TypeAlias = tuple[str, str, str, str, str, int, int]
_BUFFERED_ROWS: Final = TypeAdapter(tuple[_BufferedRow, ...])
_BUFFERED_ENTRIES: Final = TypeAdapter(tuple[str | bytes, ...])
_NO_COUNTS: Final[RequestErrorSnapshot] = MappingProxyType({})


_METADATA: Final = TypeAdapter(Mapping[str, object])


def status_code_from_metadata(metadata: Mapping[str, object]) -> int:
    """The HTTP status the logging callbacks recorded, or 0 when none was."""
    error_information: Final = metadata.get("error_information")
    if not isinstance(error_information, Mapping):
        return REQUEST_ERRORS_UNKNOWN_STATUS_CODE
    raw_code: Final = _METADATA.validate_python(error_information).get("error_code")
    try:
        code: Final = int(str(raw_code))
    except (TypeError, ValueError):
        return REQUEST_ERRORS_UNKNOWN_STATUS_CODE
    return code if 100 <= code <= 599 else REQUEST_ERRORS_UNKNOWN_STATUS_CODE


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _metadata(raw_metadata: object) -> Mapping[str, object]:
    """Spend logs carry metadata as a JSON string; accept an already-decoded mapping too."""
    if isinstance(raw_metadata, Mapping):
        return _METADATA.validate_python(raw_metadata)
    if not isinstance(raw_metadata, str):
        return {}
    try:
        return _METADATA.validate_json(raw_metadata)
    except ValueError:
        return {}


class RequestErrorAccumulator:
    """Sink for the spend writer. ``record`` is sync and never awaits."""

    def __init__(self) -> None:
        self._counts: dict[RequestErrorKey, int] = {}  # mutable-ok: bounded fold, drained per flush

    def record(
        self,
        *,
        payload: Mapping[str, object],
        request_status: Literal["success", "failure"],
        date: str,
        is_internal_call: bool,
    ) -> None:
        if request_status == "success" or is_internal_call:
            return
        key: Final = RequestErrorKey(
            date=date,
            api_key=_text(payload.get("api_key")),
            team_id=_text(payload.get("team_id")),
            user_id=_text(payload.get("user")),
            model_group=_text(payload.get("model_group")) or _text(payload.get("model")),
            status_code=status_code_from_metadata(_metadata(payload.get("metadata"))),
        )
        self._counts[key] = self._counts.get(key, 0) + 1

    def drain(self) -> RequestErrorSnapshot:
        drained: Final = self._counts
        self._counts = {}
        return drained

    def restore(self, snapshot: RequestErrorSnapshot) -> None:
        """Merge un-committed counts back so the next flush retries them (at-least-once)."""
        self._counts = dict(fold_counts(chain(self._counts.items(), snapshot.items())))


request_error_accumulator: Final = RequestErrorAccumulator()


def fold_counts(items: Iterable[tuple[RequestErrorKey, int]]) -> RequestErrorSnapshot:
    folded: Final[dict[RequestErrorKey, int]] = {}  # mutable-ok: local fold returned once
    for key, count in items:
        folded[key] = folded.get(key, 0) + count
    return folded


def _buffered_counts(entries: Iterable[str | bytes]) -> Iterator[tuple[RequestErrorKey, int]]:
    for entry in entries:
        for date, api_key, team_id, user_id, model_group, status_code, failed in _BUFFERED_ROWS.validate_json(entry):
            yield (
                RequestErrorKey(
                    date=date,
                    api_key=api_key,
                    team_id=team_id,
                    user_id=user_id,
                    model_group=model_group,
                    status_code=status_code,
                ),
                failed,
            )


def build_request_errors_upsert(snapshot: RequestErrorSnapshot) -> tuple[str, tuple[str | int, ...]]:
    """One statement; rows ordered by conflict key so concurrent writers cannot deadlock."""
    ordered: Final = sorted(
        ((key, count) for key, count in snapshot.items() if count > 0),
        key=lambda item: (
            item[0].date,
            item[0].api_key,
            item[0].team_id,
            item[0].user_id,
            item[0].model_group,
            item[0].status_code,
        ),
    )
    rows: Final = ", ".join(
        f"(${base + 1}::text, ${base + 2}::text, ${base + 3}::text, ${base + 4}::text, "
        f"${base + 5}::text, ${base + 6}::integer, ${base + 7}::bigint, {_UTC_NOW})"
        for base in range(0, len(ordered) * _COLUMNS_PER_ROW, _COLUMNS_PER_ROW)
    )
    sql: Final = (
        f"INSERT INTO {_TABLE} "
        '("date", "api_key", "team_id", "user_id", "model_group", "status_code", "failed_requests", "updated_at")\n'
        f"VALUES {rows}\n"
        'ON CONFLICT ("date", "api_key", "team_id", "user_id", "model_group", "status_code") DO UPDATE SET\n'
        f'  "failed_requests" = {_TABLE}."failed_requests" + EXCLUDED."failed_requests",\n'
        f'  "updated_at" = {_UTC_NOW}'
    )
    params: Final[tuple[str | int, ...]] = tuple(
        chain.from_iterable(
            (key.date, key.api_key, key.team_id, key.user_id, key.model_group, key.status_code, count)
            for key, count in ordered
        )
    )
    return sql, params


def _statement_chunks(snapshot: RequestErrorSnapshot) -> Iterator[RequestErrorSnapshot]:
    """Bounded statements: a backlog of many callers must not exceed the bind-parameter limit."""
    items: Final = tuple((key, count) for key, count in snapshot.items() if count > 0)
    for start in range(0, len(items), REQUEST_ERRORS_MAX_ROWS_PER_UPSERT):
        yield dict(items[start : start + REQUEST_ERRORS_MAX_ROWS_PER_UPSERT])


class RequestErrorCommitCancelled(asyncio.CancelledError):
    """Cancellation that interrupted a commit, carrying the rows that were not written yet."""

    def __init__(self, uncommitted: RequestErrorSnapshot) -> None:
        super().__init__()
        self.uncommitted: Final = uncommitted


def _rows_from(chunks: tuple[RequestErrorSnapshot, ...], index: int) -> RequestErrorSnapshot:
    return fold_counts(chain.from_iterable(remaining.items() for remaining in chunks[index:]))


async def commit_request_errors_to_db(
    *, prisma_client: "PrismaClient", snapshot: RequestErrorSnapshot
) -> RequestErrorSnapshot:
    """One bounded statement per chunk, stopping at the first failure.

    Returns the rows that were not written (the failed chunk and everything after it) so the
    caller retries only those; replaying the whole snapshot would count the landed chunks twice.
    """
    chunks: Final = tuple(_statement_chunks(snapshot))
    for index, chunk in enumerate(chunks):
        try:
            sql, params = build_request_errors_upsert(chunk)
            async with db_span("commit_request_errors", "LiteLLM_DailyRequestErrors"):
                await prisma_client.db.execute_raw(sql, *params)  # pyright: ignore[reportAny]  # untyped prisma client
        except asyncio.CancelledError:
            raise RequestErrorCommitCancelled(_rows_from(chunks, index)) from None
        except Exception:  # noqa: BLE001 -- keep the chunks already written, hand back the rest for retry
            verbose_proxy_logger.warning(
                "Request error tracking - statement %d of %d failed, %d rows go back for retry",
                index + 1,
                len(chunks),
                sum(len(remaining) for remaining in chunks[index:]),
                exc_info=True,
            )
            return _rows_from(chunks, index)
        verbose_proxy_logger.debug("Request error tracking - committed %d aggregated rows in one statement", len(chunk))
    return _NO_COUNTS


class RequestErrorRedisBuffer:
    """Folds every worker's snapshot through one Redis list so a single pod per interval writes the table."""

    def __init__(self, *, redis_cache: RedisCache, pod_lock_manager: PodLockManager) -> None:
        self._redis_cache: Final = redis_cache
        self._pod_lock_manager: Final = pod_lock_manager

    @with_service_target(_REQUEST_ERROR_QUEUE_TARGET)
    async def push(self, snapshot: RequestErrorSnapshot) -> None:
        if not snapshot:
            return
        rows: Final[tuple[_BufferedRow, ...]] = tuple(
            (key.date, key.api_key, key.team_id, key.user_id, key.model_group, key.status_code, count)
            for key, count in snapshot.items()
        )
        await self._rpush((json.dumps(rows),))

    @with_service_target(_REQUEST_ERROR_QUEUE_TARGET)
    async def _rpush(self, entries: tuple[str | bytes, ...]) -> None:
        if not entries:
            return
        await self._redis_cache.async_rpush(  # pyright: ignore[reportUnknownMemberType]  # redis cache span parameter is untyped
            key=REDIS_REQUEST_ERRORS_BUFFER_KEY, values=entries
        )

    @with_service_target(_REQUEST_ERROR_QUEUE_TARGET)
    async def _pop_batch(self) -> tuple[str | bytes, ...]:
        popped: Final[object] = await self._redis_cache.async_lpop(  # pyright: ignore[reportAny, reportUnknownMemberType]  # redis returns Any; span parameter is untyped
            key=REDIS_REQUEST_ERRORS_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT
        )
        if not popped:
            return ()
        return _BUFFERED_ENTRIES.validate_python(popped if isinstance(popped, list) else (popped,))

    async def pop(self) -> RequestErrorSnapshot:
        """Drains the Redis list; a cancellation mid-drain puts the popped entries back before propagating."""
        entries: Final[list[str | bytes]] = []  # mutable-ok: bounded drain, folded once per call
        try:
            while True:
                batch = await self._pop_batch()
                entries.extend(batch)
                if len(batch) < MAX_REDIS_BUFFER_DEQUEUE_COUNT:
                    break
        except asyncio.CancelledError:
            try:
                await asyncio.shield(self._rpush(tuple(entries)))
            except Exception:  # noqa: BLE001 -- Redis refused them, so the caller keeps the rows in memory
                raise RequestErrorCommitCancelled(fold_counts(_buffered_counts(entries))) from None
            raise
        except Exception:  # noqa: BLE001 -- fold what was already popped; unread entries stay queued in Redis
            verbose_proxy_logger.warning(
                "Request error tracking - Redis read failed, folding the entries already popped", exc_info=True
            )
        return fold_counts(_buffered_counts(entries))

    async def commit_if_leader(self, prisma_client: "PrismaClient") -> RequestErrorSnapshot:
        """Drain and write on the lease holder only; returns rows that could be neither committed nor re-queued."""
        if not await self._pod_lock_manager.acquire_lock(cronjob_id=REQUEST_ERRORS_JOB_NAME):
            return _NO_COUNTS
        try:
            buffered: Final = await self.pop()
            uncommitted: Final = await commit_request_errors_to_db(prisma_client=prisma_client, snapshot=buffered)
        except RequestErrorCommitCancelled as cancelled:
            raise RequestErrorCommitCancelled(await asyncio.shield(self._requeue(cancelled.uncommitted))) from None
        return await self._requeue(uncommitted)

    async def _requeue(self, uncommitted: RequestErrorSnapshot) -> RequestErrorSnapshot:
        """Pushes unwritten rows back to Redis; returns them when Redis refuses so they stay in memory."""
        if not uncommitted:
            return _NO_COUNTS
        try:
            await self.push(uncommitted)
        except Exception:  # noqa: BLE001 -- the rows go back to the caller's accumulator instead
            verbose_proxy_logger.warning(
                "Request error tracking - Redis re-queue failed, keeping %d rows in memory",
                len(uncommitted),
                exc_info=True,
            )
            return uncommitted
        verbose_proxy_logger.warning(
            "Request error tracking - re-queued %d uncommitted rows to Redis", len(uncommitted)
        )
        return _NO_COUNTS


async def flush_request_errors(
    prisma_client: "PrismaClient",
    accumulator: RequestErrorAccumulator,
    redis_buffer: RequestErrorRedisBuffer | None = None,
) -> None:
    """Scheduler entrypoint. Only cancellation propagates: a metering failure must not kill the job."""
    snapshot: Final = accumulator.drain()
    if redis_buffer is None:
        try:
            accumulator.restore(await commit_request_errors_to_db(prisma_client=prisma_client, snapshot=snapshot))
        except RequestErrorCommitCancelled as cancelled:
            accumulator.restore(cancelled.uncommitted)
            raise
        return
    try:
        await redis_buffer.push(snapshot)
    except Exception:  # noqa: BLE001 -- a failed push must not stop the scheduler
        accumulator.restore(snapshot)
        verbose_proxy_logger.warning(
            "Request error tracking - failed to buffer %d rows, retrying on the next flush",
            len(snapshot),
            exc_info=True,
        )
        return
    try:
        accumulator.restore(await redis_buffer.commit_if_leader(prisma_client))
    except RequestErrorCommitCancelled as cancelled:
        accumulator.restore(cancelled.uncommitted)
        raise
    except Exception:  # noqa: BLE001 -- entries still in Redis are drained by the next flush
        verbose_proxy_logger.warning(
            "Request error tracking - leader drain failed, buffered rows stay in Redis for the next flush",
            exc_info=True,
        )

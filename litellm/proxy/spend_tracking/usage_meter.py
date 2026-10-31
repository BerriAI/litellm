from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import accumulate, chain, count, takewhile
from threading import Lock
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from litellm._logging import verbose_proxy_logger
from litellm.accounting import (
    DeliveryTarget,
    Dimension,
    DimensionName,
    InboundUsageEvent,
    OutboundUsageEvent,
)
from litellm.proxy.db.accounting_ledger import AccountingConflictError, UsageWrite

_DAILY_TARGETS: Final = (
    (DimensionName.KEY, DeliveryTarget.DAILY_USER_SPEND),
    (DimensionName.USER, DeliveryTarget.DAILY_USER_SPEND),
    (DimensionName.TEAM, DeliveryTarget.DAILY_TEAM_SPEND),
    (DimensionName.ORGANIZATION, DeliveryTarget.DAILY_ORGANIZATION_SPEND),
    (DimensionName.END_USER, DeliveryTarget.DAILY_END_USER_SPEND),
    (DimensionName.AGENT, DeliveryTarget.DAILY_AGENT_SPEND),
    (DimensionName.TAG, DeliveryTarget.DAILY_TAG_SPEND),
    (DimensionName.MODEL, DeliveryTarget.DAILY_MODEL_USAGE),
    (DimensionName.TOOL, DeliveryTarget.DAILY_TOOL_SPEND),
)


def usage_delivery_targets(dimensions: tuple[Dimension, ...]) -> tuple[DeliveryTarget, ...]:
    names: Final = frozenset(dimension.name for dimension in dimensions)
    return tuple(dict.fromkeys(target for dimension, target in _DAILY_TARGETS if dimension in names))


class UsageCaptureSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=1000, ge=2, le=1000)
    flush_interval_seconds: float = Field(default=5.0, gt=0, le=10)
    max_pending_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_pending_records: int = Field(default=10000, ge=2)


class UsageCaptureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Pending:
    writes: tuple[UsageWrite, ...]
    size: int
    committed: Future[None]
    finished: asyncio.Event
    prerequisites: tuple[asyncio.Future[None], ...]


async def _wait_for_batch(wakeup: asyncio.Event, timeout: float) -> None:
    try:
        await asyncio.wait_for(wakeup.wait(), timeout)
    except asyncio.TimeoutError:
        return


def _report_failure(receipt: Future[None]) -> None:
    if not receipt.cancelled() and receipt.exception() is not None:
        verbose_proxy_logger.error("Usage capture did not acknowledge commit", exc_info=receipt.exception())


def _retryable_persistence_failure(error: Exception) -> bool:
    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler, exception_chain

    outage: Final = PrismaDBExceptionHandler.find_database_service_unavailable_error_in_chain(error)
    return (outage is not None and not PrismaDBExceptionHandler.is_permanent_database_fault(outage)) or any(
        isinstance(cause, Exception)
        and (
            PrismaDBExceptionHandler.is_deadlock_error(cause)
            or PrismaDBExceptionHandler.is_lock_timeout_error(cause)
            or PrismaDBExceptionHandler.postgres_sqlstate(cause) == "40001"
        )
        for cause in exception_chain(error)
    )


def _isolatable_persistence_failure(error: Exception) -> bool:
    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler, exception_chain

    return isinstance(error, AccountingConflictError) or any(
        isinstance(cause, Exception) and (PrismaDBExceptionHandler.postgres_sqlstate(cause) or "").startswith("22")
        for cause in exception_chain(error)
    )


class DurableUsageMeter:
    def __init__(
        self,
        write_batch: Callable[[tuple[UsageWrite, ...]], Awaitable[None]],
        settings: UsageCaptureSettings,
        *,
        wait_for_batch: Callable[[asyncio.Event, float], Awaitable[None]] = _wait_for_batch,
        retry_delay: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._write_batch: Final = write_batch
        self.settings: Final = settings
        self._wait_for_batch: Final = wait_for_batch
        self._retry_delay: Final = retry_delay
        self._loop: Final = asyncio.get_running_loop()
        self._admission: Final = Lock()
        self._queue: Final[deque[_Pending]] = deque()  # mutable-ok: one bounded producer/consumer queue
        self._wakeup: Final = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.pending_bytes = 0
        self.pending_records = 0

    def submit(
        self, writes: tuple[UsageWrite, ...], *, prerequisites: tuple[asyncio.Future[None], ...] = ()
    ) -> asyncio.Future[None]:
        size: Final = sum(len(write.envelope.encode("utf-8")) for write in writes)
        with self._admission:
            if (
                self._closed
                or not writes
                or len(writes) > self.settings.batch_size
                or self.pending_bytes + size > self.settings.max_pending_bytes
                or self.pending_records + len(writes) > self.settings.max_pending_records
            ):
                raise UsageCaptureError("Usage capture admission rejected: closed, empty or capacity exhausted")
            receipt: Final = Future[None]()
            receipt.set_running_or_notify_cancel()
            receipt.add_done_callback(_report_failure)
            self._queue.append(_Pending(writes, size, receipt, asyncio.Event(), prerequisites))
            self.pending_bytes += size
            self.pending_records += len(writes)
            self._loop.call_soon_threadsafe(self._ensure_worker)
        wrapped: Final = asyncio.wrap_future(receipt, loop=self._loop)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return wrapped

    def _ensure_worker(self) -> None:
        if self.pending_records >= self.settings.batch_size or self._queue and self._queue[0].prerequisites:
            self._wakeup.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="accounting usage capture")

    async def record(self, event: InboundUsageEvent | OutboundUsageEvent) -> None:
        envelope: Final = '{"event":' + event.model_dump_json() + "}"
        await asyncio.shield(self.submit((UsageWrite(event, envelope, usage_delivery_targets(event.dimensions)),)))

    async def flush(self) -> None:
        self._ensure_worker()
        self._wakeup.set()
        if self._task is not None:
            await asyncio.shield(self._task)

    async def close(self) -> None:
        with self._admission:
            self._closed = True
        await self.flush()

    async def stop(self, timeout: float) -> None:
        with self._admission:
            self._closed = True
        try:
            await asyncio.wait_for(self.close(), timeout)
        except asyncio.TimeoutError:
            if self._task is not None:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            while self._queue:
                self._finish(
                    self._take_batch(), UsageCaptureError("Usage capture stopped before commit acknowledgement")
                )
            verbose_proxy_logger.error("Usage capture shutdown deadline exhausted; commit acknowledgement incomplete")

    def _take_batch(self) -> tuple[_Pending, ...]:
        with self._admission:
            if self._queue[0].prerequisites:
                return (self._queue.popleft(),)
            independent: Final = takewhile(lambda item: not item.prerequisites, self._queue)
            sizes: Final = accumulate(len(item.writes) for item in independent)
            count: Final = sum(1 for _ in takewhile(lambda size: size <= self.settings.batch_size, sizes))
            return tuple(self._queue.popleft() for _ in range(count))

    async def _persist(self, writes: tuple[UsageWrite, ...]) -> None:
        for attempt in count():
            try:
                await self._write_batch(writes)
            except Exception as error:  # noqa: BLE001  # retry only classified database failures, preserving frozen writes
                if not _retryable_persistence_failure(error):
                    raise
                verbose_proxy_logger.warning("Usage persistence unavailable; retaining admitted batch for retry")
                await self._retry_delay(min(10.0, 0.2 * 2 ** min(attempt, 6)))
            else:
                return

    async def _commit(self, batch: tuple[_Pending, ...]) -> None:
        writes: Final = tuple(chain.from_iterable(item.writes for item in batch))
        try:
            if batch[0].prerequisites:
                outcomes: Final = await asyncio.gather(
                    *(asyncio.shield(receipt) for receipt in batch[0].prerequisites), return_exceptions=True
                )
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise UsageCaptureError("Usage dependency did not acknowledge commit") from outcome
            await self._persist(writes)
        except asyncio.CancelledError:
            self._finish(batch, UsageCaptureError("Usage commit acknowledgement interrupted"))
            raise
        except Exception as error:  # noqa: BLE001  # isolate data faults, surface systemic failures to all producers
            if len(batch) > 1 and _isolatable_persistence_failure(error):
                try:
                    for item in batch:
                        await self._commit((item,))
                except asyncio.CancelledError:
                    self._finish(batch, UsageCaptureError("Usage commit acknowledgement interrupted"))
                    raise
                return
            self._finish(
                batch,
                UsageCaptureError("Usage identity conflicts with accepted content")
                if isinstance(error, AccountingConflictError)
                else error,
            )
        else:
            self._finish(batch, None)

    def _finish(self, batch: tuple[_Pending, ...], error: Exception | None) -> None:
        for item in batch:
            if item.finished.is_set():
                continue
            item.finished.set()
            with self._admission:
                self.pending_bytes -= item.size
                self.pending_records -= len(item.writes)
            if item.committed.done():
                continue
            if error is None:
                item.committed.set_result(None)
            else:
                item.committed.set_exception(error)

    async def _run(self) -> None:
        try:
            while self._queue:
                await self._wait_for_batch(self._wakeup, self.settings.flush_interval_seconds)
                self._wakeup.clear()
                await self._commit(self._take_batch())
                if (
                    self._closed
                    or self.pending_records >= self.settings.batch_size
                    or self._queue
                    and self._queue[0].prerequisites
                ):
                    self._wakeup.set()
        except (Exception, asyncio.CancelledError):  # noqa: BLE001  # reject queued receipts if the writer exits abnormally
            while self._queue:
                self._finish(
                    self._take_batch(), UsageCaptureError("Usage capture stopped before commit acknowledgement")
                )
            raise

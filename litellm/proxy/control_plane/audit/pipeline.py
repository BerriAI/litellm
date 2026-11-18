"""Asynchronous delivery of audit records to configured sinks. One bounded queue and one worker per sink, so a sink
that is down never blocks another one or the request path. Full queue: drop the oldest and count it, or in strict
mode refuse the record so the caller can fail the request before it does anything."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy.control_plane.audit.ocsf import AuditRecord

Sleeper: TypeAlias = Callable[[float], Awaitable[None]]
Batch: TypeAlias = tuple[AuditRecord, ...]


@dataclass(frozen=True, slots=True)
class Delivered:
    count: int


@dataclass(frozen=True, slots=True)
class Undelivered:
    reason: str
    settled: int = 0
    rejected: int = 0


@dataclass(frozen=True, slots=True)
class Rejected:
    reason: str
    count: int | None = None


DeliveryResult: TypeAlias = Delivered | Undelivered | Rejected


class AuditSink(Protocol):
    @property
    def name(self) -> str: ...

    async def deliver(self, batch: Batch) -> DeliveryResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Accepted:
    pass


@dataclass(frozen=True, slots=True)
class Dropped:
    sink: str


@dataclass(frozen=True, slots=True)
class Refused:
    sink: str


Offer: TypeAlias = Accepted | Dropped | Refused
Mode: TypeAlias = Literal["drop_oldest", "strict"]
EXPONENT_CAP: Final = 30


@dataclass(frozen=True, slots=True)
class Backoff:
    base_seconds: float = 0.5
    max_seconds: float = 30.0

    def delay(self, failures: int) -> float:
        return min(self.max_seconds, self.base_seconds * (2.0 ** min(failures, EXPONENT_CAP)))


@dataclass(slots=True)
class SinkWorker:
    sink: AuditSink
    queue: asyncio.Queue[AuditRecord]
    batch_size: int
    mode: Mode
    backoff: Backoff = field(default_factory=Backoff)
    sleep: Sleeper = asyncio.sleep
    dropped: int = 0
    failures: int = 0
    rejected: int = 0
    task: asyncio.Task[None] | None = None
    inflight: Batch | None = None
    stopping: bool = False
    reservations: frozenset[object] = frozenset()

    def offer(self, record: AuditRecord, reservation: object | None = None) -> Offer:
        reserved: Final = reservation in self.reservations
        if reserved:
            self.release(reservation)
        if self.mode == "strict" and not reserved and self.full():
            return Refused(self.sink.name)
        if not self.queue.full():
            self.queue.put_nowait(record)
            return Accepted()
        self.queue.get_nowait()
        self.queue.put_nowait(record)
        self.dropped += 1
        verbose_proxy_logger.warning(
            "control_plane.audit sink %s queue full, dropped oldest (total %d)", self.sink.name, self.dropped
        )
        return Dropped(self.sink.name)

    def offer_unauthenticated(self, record: AuditRecord) -> Offer:
        if (
            self.queue.qsize() * 2 >= self.queue.maxsize
            or self.queue.qsize() + len(self.reservations) + 1 >= self.queue.maxsize
        ):
            self.dropped += 1
            verbose_proxy_logger.warning(
                "control_plane.audit sink %s dropped unauthenticated failure (total %d)",
                self.sink.name,
                self.dropped,
            )
            return Dropped(self.sink.name)
        self.queue.put_nowait(record)
        return Accepted()

    def full(self) -> bool:
        """Strict capacity is exhausted by occupied slots or a permanent audit loss."""
        return self.rejected > 0 or self.queue.qsize() + len(self.reservations) >= self.queue.maxsize

    def reserve(self, token: object) -> None:
        self.reservations = self.reservations | {token}

    def release(self, token: object) -> None:
        self.reservations = self.reservations - {token}

    def _take_batch(self, first: AuditRecord) -> Batch:
        rest: Final = tuple(self.queue.get_nowait() for _ in range(min(self.batch_size - 1, self.queue.qsize())))
        return (first, *rest)

    async def _deliver_until_done(self, batch: Batch) -> None:
        """Keeps the batch as ``inflight`` until the sink takes it, so a cancelled delivery can be redone."""
        self.inflight = batch
        current_batch: Batch = batch  # rebind-ok: retries advance through immutable unsettled suffixes
        while isinstance(result := await self.sink.deliver(current_batch), Undelivered):
            if result.rejected:
                self._record_rejection(result.rejected, result.reason)
            self.failures += 1
            verbose_proxy_logger.warning(
                "control_plane.audit sink %s failed (%s), retry %d", self.sink.name, result.reason, self.failures
            )
            current_batch = current_batch[result.settled :]
            self.inflight = current_batch
            if not current_batch:
                break
            await self.sleep(self.backoff.delay(self.failures - 1))
        if isinstance(result, Rejected):
            count: Final = len(current_batch) if result.count is None else result.count
            self._record_rejection(count, result.reason)
        self.failures = 0
        self.inflight = None

    def _record_rejection(self, count: int, reason: str) -> None:
        self.rejected += count
        verbose_proxy_logger.error(
            "control_plane.audit sink %s rejected %d records (%s), total %d",
            self.sink.name,
            count,
            reason,
            self.rejected,
        )

    async def flush_once(self) -> int:
        """Deliver what is queued right now, returns how many records went out. Tests drive this directly."""
        if self.queue.empty():
            return 0
        batch: Final = self._take_batch(self.queue.get_nowait())
        await self._deliver_until_done(batch)
        return len(batch)

    async def _next_batch(self) -> None:
        first: Final = await self.queue.get()
        await self._deliver_until_done(self._take_batch(first))

    async def run(self) -> None:
        while not self.stopping:
            await self._next_batch()

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run(), name=f"control_plane.audit.{self.sink.name}")

    async def close(self, drain_seconds: float, *, clock: Callable[[], float] | None = None) -> None:
        """Stops the worker, then drains its queue within the deadline."""
        now: Final = clock or asyncio.get_running_loop().time
        deadline: Final = now() + drain_seconds
        task: Final = self.task
        if task is not None:
            self.stopping = True
        delivery_finished: Final = (
            task is None
            or self.inflight is None
            or (task in (await asyncio.wait({task}, timeout=max(0.0, deadline - now())))[0] and not task.cancelled())
        )
        if task is not None:
            if self.inflight is None or not delivery_finished:
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001  # shutdown must continue after arbitrary sink worker failures
                verbose_proxy_logger.exception("control_plane.audit sink %s worker crashed", self.sink.name)
        if delivery_finished:
            try:
                await asyncio.wait_for(self._drain(), timeout=max(0.0, deadline - now()))
            except TimeoutError:
                self._warn_undelivered()
            except Exception:  # noqa: BLE001  # continue closing sinks after arbitrary drain failures
                verbose_proxy_logger.exception(
                    "control_plane.audit sink %s failed to drain at shutdown", self.sink.name
                )
        else:
            self._warn_undelivered()
        try:
            await self.sink.close()
        except Exception:  # noqa: BLE001  # close other sinks despite arbitrary sink close failures
            verbose_proxy_logger.exception("control_plane.audit sink %s failed to close", self.sink.name)

    def _warn_undelivered(self) -> None:
        verbose_proxy_logger.warning(
            "control_plane.audit sink %s left %d records undelivered at shutdown",
            self.sink.name,
            self.queue.qsize() + len(self.inflight or ()),
        )

    async def _drain(self) -> None:
        if self.inflight is not None:
            await self._deliver_until_done(self.inflight)
        while await self.flush_once():
            pass


@dataclass(frozen=True, slots=True)
class Reservation:
    """One held slot on every strict worker. Releasing twice is harmless, so the request can release in a
    ``finally`` after the audit record consumed it."""

    token: object
    workers: tuple[SinkWorker, ...]

    def release(self) -> None:
        for worker in self.workers:
            worker.release(self.token)


@dataclass(frozen=True, slots=True)
class AuditPipeline:
    workers: tuple[SinkWorker, ...]
    drain_seconds: float = 5.0

    def publish(self, record: AuditRecord, reservation: Reservation | None = None) -> tuple[Offer, ...]:
        return tuple(
            worker.offer(record, reservation.token if reservation is not None else None) for worker in self.workers
        )

    def publish_unauthenticated(self, record: AuditRecord) -> tuple[Offer, ...]:
        return tuple(worker.offer_unauthenticated(record) for worker in self.workers)

    def accepting(self) -> bool:
        """Refuse strict-mode writes after permanent audit loss or when the queue is full."""
        return not any(worker.mode == "strict" and worker.full() for worker in self.workers)

    def reserve(self) -> Reservation | None:
        """Atomically (no await) takes one slot on every strict worker so a write that passes the gate always has
        room for its record, however many writes are in flight. None when any strict queue is full."""
        if not self.accepting():
            return None
        token: Final = object()
        strict: Final = tuple(worker for worker in self.workers if worker.mode == "strict")
        for worker in strict:
            worker.reserve(token)
        return Reservation(token, strict)

    def start(self) -> None:
        for worker in self.workers:
            worker.start()

    async def close(self) -> None:
        results: Final = tuple(
            await asyncio.gather(
                *(worker.close(self.drain_seconds) for worker in self.workers),
                return_exceptions=True,
            )
        )
        for worker, result in zip(self.workers, results, strict=True):
            if isinstance(result, BaseException):
                verbose_proxy_logger.error(
                    "control_plane.audit sink %s failed during shutdown: %s", worker.sink.name, result
                )


def build_pipeline(
    sinks: tuple[AuditSink, ...],
    queue_size: int,
    batch_size: int,
    strict: bool,
    sleep: Sleeper = asyncio.sleep,
    drain_seconds: float = 5.0,
) -> AuditPipeline:
    return AuditPipeline(
        workers=tuple(
            SinkWorker(
                sink=sink,
                queue=asyncio.Queue(maxsize=queue_size),
                batch_size=batch_size,
                mode="strict" if strict else "drop_oldest",
                sleep=sleep,
            )
            for sink in sinks
        ),
        drain_seconds=drain_seconds,
    )


NO_PIPELINE: Final = AuditPipeline(workers=())

"""Asynchronous delivery of audit records to configured sinks. One bounded queue and one worker per sink, so a sink
that is down never blocks another one or the request path. Full queue: drop the oldest and count it, or in strict
mode refuse the record so the caller can fail the request before it does anything."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
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


DeliveryResult: TypeAlias = Delivered | Undelivered


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
    task: asyncio.Task[None] | None = None
    inflight: Batch | None = None
    reservations: frozenset[object] = frozenset()

    def offer(self, record: AuditRecord) -> Offer:
        if self.mode == "strict" and self.full():
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

    def full(self) -> bool:
        """Queued plus reserved slots. Reservations only matter in strict mode, where a refusal is the contract."""
        return self.queue.qsize() + len(self.reservations) >= self.queue.maxsize

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
        while isinstance(result := await self.sink.deliver(batch), Undelivered):
            self.failures += 1
            verbose_proxy_logger.warning(
                "control_plane.audit sink %s failed (%s), retry %d", self.sink.name, result.reason, self.failures
            )
            await self.sleep(self.backoff.delay(self.failures - 1))
        self.failures = 0
        self.inflight = None

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
        while True:
            await self._next_batch()

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self.run(), name=f"control_plane.audit.{self.sink.name}")

    async def close(self, drain_seconds: float) -> None:
        """Stops the worker, then delivers the batch it was holding and whatever is queued, within the deadline."""
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        try:
            await asyncio.wait_for(self._drain(), timeout=drain_seconds)
        except TimeoutError:
            verbose_proxy_logger.warning(
                "control_plane.audit sink %s left %d records undelivered at shutdown",
                self.sink.name,
                self.queue.qsize() + len(self.inflight or ()),
            )
        await self.sink.close()

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

    def publish(self, record: AuditRecord) -> tuple[Offer, ...]:
        return tuple(worker.offer(record) for worker in self.workers)

    def accepting(self) -> bool:
        """False only in strict mode with a full queue, the signal to answer 503 before a write goes out."""
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
        for worker in self.workers:
            await worker.close(self.drain_seconds)


def build_pipeline(
    sinks: tuple[AuditSink, ...], queue_size: int, batch_size: int, strict: bool, sleep: Sleeper = asyncio.sleep
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
        )
    )


NO_PIPELINE: Final = AuditPipeline(workers=())

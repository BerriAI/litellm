"""Background flushers.

Window resets need no worker (R10): rollover is implicit in the Lua window
period and the L3 flusher emits the audit breadcrumb at the boundary. Leadership
(pod_lock_manager) is injected as ``is_leader`` rather than imported, so a worker
is testable without Redis and the proxy owns lock acquisition.
"""

import asyncio
from typing import Awaitable, Callable, List, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.integrations.governor.model.audit import AuditEvent
from litellm.integrations.governor.model.errors import TierDegraded
from litellm.integrations.governor.plumbing.postgres import (
    PendingFlushQueue,
    PostgresCounterStore,
)

LeaderCheck = Callable[[], Awaitable[bool]]


class AuditSink(Protocol):
    async def emit(self, event: AuditEvent) -> None: ...


class _PeriodicWorker:
    def __init__(self, *, interval_s: float, is_leader: LeaderCheck | None) -> None:
        self._interval_s = interval_s
        self._is_leader = is_leader
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _tick(self) -> None:
        raise NotImplementedError

    async def _run(self) -> None:
        while not self._stop.is_set():
            if self._is_leader is None or await self._is_leader():
                await self._tick()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None


class L3Flusher(_PeriodicWorker):
    def __init__(
        self,
        queue: PendingFlushQueue,
        store: PostgresCounterStore,
        *,
        interval_s: float,
        is_leader: LeaderCheck | None = None,
    ) -> None:
        super().__init__(interval_s=interval_s, is_leader=is_leader)
        self._queue = queue
        self._store = store

    async def _tick(self) -> None:
        batch = await self._queue.drain()
        if not batch:
            return
        try:
            await self._store.flush(batch)
        except TierDegraded:
            await self._queue.requeue(batch)


class AuditFlusher(_PeriodicWorker):
    def __init__(
        self,
        sink: AuditSink,
        *,
        interval_s: float,
        is_leader: LeaderCheck | None = None,
    ) -> None:
        super().__init__(interval_s=interval_s, is_leader=is_leader)
        self._sink = sink
        self._buffer: List[AuditEvent] = []
        self._lock = asyncio.Lock()

    async def offer(self, event: AuditEvent) -> None:
        async with self._lock:
            self._buffer.append(event)

    async def _tick(self) -> None:
        async with self._lock:
            pending = self._buffer
            self._buffer = []
        for event in pending:
            try:
                await self._sink.emit(event)
            except Exception:
                verbose_proxy_logger.exception(
                    "governor audit sink failed for request_id=%s", event.request_id
                )

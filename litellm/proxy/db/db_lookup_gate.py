import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

from litellm.constants import (
    PROXY_DB_LOOKUP_DEADLINE_SECONDS,
    PROXY_DB_LOOKUP_MAX_CONCURRENCY,
)

LookupT = TypeVar("LookupT")


class LoopBoundSemaphore:
    __slots__ = ("_loop", "_semaphore", "_value")

    def __init__(self, value: int) -> None:
        self._value: Final = value
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def current(self) -> asyncio.Semaphore:
        loop: Final = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self._value)
            self._loop = loop
        return self._semaphore


class DBLookupDeadlineExceeded(asyncio.TimeoutError):
    def __init__(self, lookup: str, deadline_seconds: float) -> None:
        super().__init__(f"{lookup} lookup did not answer within {deadline_seconds:g}s")
        self.lookup: Final = lookup
        self.deadline_seconds: Final = deadline_seconds


class DBLookupStallTracker:
    __slots__ = ("_clock", "_last_hit")

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock: Final = clock
        self._last_hit: float | None = None

    def record_hit(self) -> None:
        self._last_hit = self._clock()

    def clear(self) -> None:
        self._last_hit = None

    def stalled_within(self, window_seconds: float) -> bool:
        if self._last_hit is None:
            return False
        return self._clock() - self._last_hit < window_seconds


db_lookup_gate: Final = LoopBoundSemaphore(PROXY_DB_LOOKUP_MAX_CONCURRENCY)
db_lookup_stall_tracker: Final = DBLookupStallTracker()


def _consume_abandoned_lookup(task: asyncio.Future[LookupT]) -> None:
    if not task.cancelled():
        task.exception()


async def bounded_db_lookup(
    lookup: Awaitable[LookupT],
    *,
    name: str,
    deadline_seconds: float | None = None,
    tracker: DBLookupStallTracker = db_lookup_stall_tracker,
) -> LookupT:
    timeout: Final = PROXY_DB_LOOKUP_DEADLINE_SECONDS if deadline_seconds is None else deadline_seconds
    task: Final = asyncio.ensure_future(lookup)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if task not in done:
        task.cancel()
        task.add_done_callback(_consume_abandoned_lookup)
        tracker.record_hit()
        raise DBLookupDeadlineExceeded(name, timeout)
    try:
        return task.result()
    except DBLookupDeadlineExceeded:
        raise
    except asyncio.TimeoutError as e:
        tracker.record_hit()
        raise DBLookupDeadlineExceeded(name, timeout) from e

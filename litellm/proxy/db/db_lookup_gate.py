import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final, TypeVar

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    PROXY_DB_LOOKUP_DEADLINE_SECONDS,
    PROXY_DB_LOOKUP_MAX_CONCURRENCY,
)

if TYPE_CHECKING:
    from prometheus_client import Gauge, Histogram

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


@dataclass(frozen=True, slots=True)
class DBLookupInFlightInstruments:
    in_flight: "Gauge"
    in_flight_at_admission: "Histogram"


_IN_FLIGHT_AT_ADMISSION_BUCKETS: Final = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 50, 100)


@cache
def prometheus_db_lookup_in_flight_instruments() -> DBLookupInFlightInstruments | None:
    try:
        from prometheus_client import Gauge, Histogram

        return DBLookupInFlightInstruments(
            in_flight=Gauge(
                "litellm_db_lookup_in_flight",
                "Bounded DB lookups currently running on a worker, the busiest worker under multiprocess",
                multiprocess_mode="livemax",
            ),
            in_flight_at_admission=Histogram(
                "litellm_db_lookup_in_flight_at_admission",
                "Bounded DB lookups running on the worker when a lookup was admitted, counting that lookup",
                buckets=_IN_FLIGHT_AT_ADMISSION_BUCKETS,
            ),
        )
    except Exception as e:  # noqa: BLE001  # a missing extra or a duplicate registration must never fail a lookup
        verbose_proxy_logger.warning("Could not register DB lookup in-flight metrics: %s", e)
        return None


class DBLookupInFlightTracker:
    __slots__ = ("_count", "_instruments")

    def __init__(
        self,
        instruments: Callable[[], DBLookupInFlightInstruments | None] = prometheus_db_lookup_in_flight_instruments,
    ) -> None:
        self._instruments: Final = instruments
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def admit(self) -> None:
        self._count += 1
        instruments: Final = self._instruments()
        if instruments is None:
            return
        instruments.in_flight.set(self._count)
        instruments.in_flight_at_admission.observe(self._count)

    def release(self) -> None:
        self._count -= 1
        instruments: Final = self._instruments()
        if instruments is None:
            return
        instruments.in_flight.set(self._count)


db_lookup_gate: Final = LoopBoundSemaphore(PROXY_DB_LOOKUP_MAX_CONCURRENCY)
db_lookup_stall_tracker: Final = DBLookupStallTracker()
db_lookup_in_flight: Final = DBLookupInFlightTracker()


def _consume_abandoned_lookup(task: asyncio.Future[LookupT]) -> None:
    if not task.cancelled():
        task.exception()


async def bounded_db_lookup(
    lookup: Awaitable[LookupT],
    *,
    name: str,
    deadline_seconds: float | None = None,
    tracker: DBLookupStallTracker = db_lookup_stall_tracker,
    in_flight: DBLookupInFlightTracker = db_lookup_in_flight,
) -> LookupT:
    timeout: Final = PROXY_DB_LOOKUP_DEADLINE_SECONDS if deadline_seconds is None else deadline_seconds
    task: Final = asyncio.ensure_future(lookup)
    in_flight.admit()
    task.add_done_callback(lambda _: in_flight.release())
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

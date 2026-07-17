from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Final, Literal, Protocol, cast


class _AsyncIteratorProtocol(Protocol):
    async def __anext__(self) -> object: ...


class _LeaseReleasingAsyncIterator:
    def __init__(self, iterator: object, lease: AdmissionLease) -> None:
        self._iterator = cast(  # cast-ok: async iterator capability is checked before this wrapper is created
            _AsyncIteratorProtocol, iterator
        )
        self._lease = lease

    def __aiter__(self) -> _LeaseReleasingAsyncIterator:
        return self

    async def __anext__(self) -> object:
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            await self._release()
            raise
        except BaseException:
            await self._release()
            raise

    async def aclose(self) -> None:
        try:
            close = getattr(self._iterator, "aclose", None)
            if close is not None:
                await cast(  # cast-ok: aclose is optional and obtained from the wrapped async iterator
                    Callable[[], Awaitable[None]], close
                )()
        finally:
            await self._release()

    async def _release(self) -> None:
        await self._lease.release()

    def __getattr__(self, name: str) -> object:
        return cast(  # cast-ok: dynamic iterator attributes are exposed as opaque objects
            object, getattr(self._iterator, name)
        )


class AdmissionClosedError(RuntimeError):
    pass


class AdmissionRejectedError(RuntimeError):
    pass


class AdmissionQueueTimeoutError(AdmissionRejectedError):
    pass


@dataclass(frozen=True, slots=True)
class AdmissionClass:
    name: str
    reservation: int
    priority: int


@dataclass(frozen=True, slots=True)
class _Waiter:
    admission_class: str
    priority: int
    sequence: int
    queued_at: float


class AdmissionMetrics:
    def __init__(self) -> None:
        self.admitted = 0
        self.queued = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0
        self.released = 0
        self.wait_time_total_s = 0.0
        self.max_wait_s = 0.0

    def record_admitted(self, wait_s: float) -> None:
        self.admitted += 1
        self.wait_time_total_s += wait_s
        self.max_wait_s = max(self.max_wait_s, wait_s)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "admitted": self.admitted,
            "queued": self.queued,
            "rejected": self.rejected,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "released": self.released,
            "wait_time_total_s": self.wait_time_total_s,
            "wait_time_mean_s": self.wait_time_total_s / self.admitted if self.admitted else 0.0,
            "wait_time_max_s": self.max_wait_s,
        }


class AdmissionLease:
    def __init__(self, owner: WeightedInFlightAdmission, admission_class: str) -> None:
        self._owner = owner
        self.admission_class = admission_class
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner.release(self.admission_class)

    def wrap_async_iterator(self, iterator: object) -> object:
        return _LeaseReleasingAsyncIterator(iterator, self)

    async def __aenter__(self) -> AdmissionLease:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.release()


class WeightedInFlightAdmission:
    def __init__(
        self,
        capacity: int,
        classes: tuple[AdmissionClass, ...],
        *,
        queue_timeout: float | None = None,
        overflow: Literal["wait", "reject"] = "wait",
        metrics: AdmissionMetrics | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not classes:
            raise ValueError("at least one admission class is required")
        if any(admission_class.reservation < 0 for admission_class in classes):
            raise ValueError("class reservations cannot be negative")
        if sum(admission_class.reservation for admission_class in classes) > capacity:
            raise ValueError("class reservations cannot exceed capacity")
        if queue_timeout is not None and queue_timeout < 0:
            raise ValueError("queue_timeout cannot be negative")
        if overflow not in ("wait", "reject"):
            raise ValueError("overflow must be 'wait' or 'reject'")
        names = tuple(admission_class.name for admission_class in classes)
        if len(set(names)) != len(names):
            raise ValueError("admission class names must be unique")

        self.capacity: Final = capacity
        self.queue_timeout: Final = queue_timeout
        self.overflow: Final = overflow
        self.metrics = metrics if metrics is not None else AdmissionMetrics()
        self._classes = {admission_class.name: admission_class for admission_class in classes}
        self._active_by_class = {name: 0 for name in names}
        self._condition = asyncio.Condition()
        self._waiters: list[_Waiter] = []
        self._next_sequence = 0
        self._closed = False

    @property
    def active(self) -> int:
        return sum(self._active_by_class.values())

    @property
    def active_by_class(self) -> dict[str, int]:
        return self._active_by_class.copy()

    @property
    def queued(self) -> int:
        return len(self._waiters)

    async def try_acquire(self, admission_class: str) -> AdmissionLease | None:
        async with self._condition:
            self._validate_class(admission_class)
            if self._closed:
                raise AdmissionClosedError("admission is closed")
            if self._waiters or not self._can_acquire(admission_class):
                return None
            self._active_by_class[admission_class] += 1
            self.metrics.record_admitted(0.0)
            return AdmissionLease(self, admission_class)

    async def acquire(self, admission_class: str) -> AdmissionLease:
        self._validate_class(admission_class)
        if self.queue_timeout is None:
            return await self._acquire_wait(admission_class, count_cancelled=True)
        try:
            return await asyncio.wait_for(
                self._acquire_wait(admission_class, count_cancelled=False), self.queue_timeout
            )
        except asyncio.TimeoutError as exc:
            self.metrics.timed_out += 1
            raise AdmissionQueueTimeoutError(
                f"admission queue timeout for class {admission_class} after {self.queue_timeout}s"
            ) from exc

    async def _acquire_wait(self, admission_class: str, *, count_cancelled: bool) -> AdmissionLease:
        async with self._condition:
            if self._closed:
                raise AdmissionClosedError("admission is closed")
            if not self._waiters and self._can_acquire(admission_class):
                self._active_by_class[admission_class] += 1
                self.metrics.record_admitted(0.0)
                return AdmissionLease(self, admission_class)
            if self.overflow == "reject":
                self.metrics.rejected += 1
                raise AdmissionRejectedError(f"admission capacity unavailable for class {admission_class}")

            waiter = _Waiter(
                admission_class=admission_class,
                priority=self._classes[admission_class].priority,
                sequence=self._next_sequence,
                queued_at=time.monotonic(),
            )
            self._next_sequence += 1
            self._waiters.append(waiter)
            self._waiters.sort(key=lambda item: (item.priority, item.sequence))
            self.metrics.queued += 1
            try:
                while True:
                    if self._closed:
                        self._waiters.remove(waiter)
                        self._condition.notify_all()
                        raise AdmissionClosedError("admission is closed")
                    if self._waiters[0] == waiter and self._can_acquire(admission_class):
                        self._waiters.pop(0)
                        self._active_by_class[admission_class] += 1
                        self.metrics.record_admitted(time.monotonic() - waiter.queued_at)
                        return AdmissionLease(self, admission_class)
                    await self._condition.wait()
            except asyncio.TimeoutError:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self.metrics.timed_out += 1
                self._condition.notify_all()
                raise
            except asyncio.CancelledError:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                if count_cancelled:
                    self.metrics.cancelled += 1
                raise

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def release(self, admission_class: str) -> None:
        async with self._condition:
            self._validate_class(admission_class)
            if self._active_by_class[admission_class] < 1:
                raise RuntimeError("admission lease released without an active permit")
            self._active_by_class[admission_class] -= 1
            self.metrics.released += 1
            self._condition.notify_all()

    def _validate_class(self, admission_class: str) -> None:
        if admission_class not in self._classes:
            raise ValueError(f"unknown admission class: {admission_class}")

    def _can_acquire(self, admission_class: str) -> bool:
        if self.active >= self.capacity:
            return False
        reserved_for_others = sum(
            max(candidate.reservation - self._active_by_class[candidate.name], 0)
            for candidate in self._classes.values()
            if candidate.priority < self._classes[admission_class].priority
        )
        return self.active < self.capacity - reserved_for_others

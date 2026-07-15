from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final


class _LeaseReleasingAsyncIterator:
    def __init__(self, iterator: object, lease: AdmissionLease) -> None:
        self._iterator = iterator
        self._lease = lease

    def __aiter__(self) -> _LeaseReleasingAsyncIterator:
        return self

    async def __anext__(self) -> object:
        try:
            next_item = getattr(self._iterator, "__anext__")
            return await next_item()
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
                await close()
        finally:
            await self._release()

    async def _release(self) -> None:
        await self._lease.release()

    def __getattr__(self, name: str) -> object:
        return getattr(self._iterator, name)


class AdmissionClosedError(RuntimeError):
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
    def __init__(self, capacity: int, classes: tuple[AdmissionClass, ...]) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not classes:
            raise ValueError("at least one admission class is required")
        if sum(admission_class.reservation for admission_class in classes) > capacity:
            raise ValueError("class reservations cannot exceed capacity")
        if any(admission_class.reservation < 0 for admission_class in classes):
            raise ValueError("class reservations cannot be negative")
        names = tuple(admission_class.name for admission_class in classes)
        if len(set(names)) != len(names):
            raise ValueError("admission class names must be unique")

        self.capacity: Final = capacity
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

    async def try_acquire(self, admission_class: str) -> AdmissionLease | None:
        async with self._condition:
            self._validate_class(admission_class)
            if self._closed:
                raise AdmissionClosedError("admission is closed")
            if self._waiters or not self._can_acquire(admission_class):
                return None
            self._active_by_class[admission_class] += 1
            return AdmissionLease(self, admission_class)

    async def acquire(self, admission_class: str) -> AdmissionLease:
        async with self._condition:
            self._validate_class(admission_class)
            if self._closed:
                raise AdmissionClosedError("admission is closed")
            if not self._waiters and self._can_acquire(admission_class):
                self._active_by_class[admission_class] += 1
                return AdmissionLease(self, admission_class)

            waiter = _Waiter(
                admission_class=admission_class,
                priority=self._classes[admission_class].priority,
                sequence=self._next_sequence,
            )
            self._next_sequence += 1
            self._waiters.append(waiter)
            self._waiters.sort(key=lambda item: (item.priority, item.sequence))
            try:
                while True:
                    if self._closed:
                        raise AdmissionClosedError("admission is closed")
                    if self._waiters[0] == waiter and self._can_acquire(admission_class):
                        self._waiters.pop(0)
                        self._active_by_class[admission_class] += 1
                        return AdmissionLease(self, admission_class)
                    await self._condition.wait()
            except asyncio.CancelledError:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def release(self, admission_class: str) -> None:
        async with self._condition:
            if self._active_by_class[admission_class] < 1:
                raise RuntimeError("admission lease released without an active permit")
            self._active_by_class[admission_class] -= 1
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

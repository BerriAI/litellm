"""
Adaptive in-flight concurrency limiter (AIMD, Vector ARC style).

Grows the limit additively after `limit` consecutive clean completions and
halves it only on an explicit throttle signal (429, 503, SlowDown, or a
transport error out of the PUT). With floor == ceiling it degenerates to a
fixed-width semaphore.
"""

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class PutSample:
    throttled: bool


class AdaptiveConcurrencyLimiter:
    """AIMD in-flight limiter used as `async with limiter:`."""

    def __init__(self, initial: int, floor: int, ceiling: int) -> None:
        if not 1 <= floor <= ceiling:
            raise ValueError(f"adaptive limiter bounds must satisfy 1 <= floor <= ceiling, got {floor}..{ceiling}")
        self._limit: int = min(max(initial, floor), ceiling)
        self._floor: Final[int] = floor
        self._ceiling: Final[int] = ceiling
        self._clean_streak: int = 0
        self._in_flight: int = 0
        self._waiters: deque[asyncio.Future[None]] = deque()  # mutable-ok: waiters queue up behind a full limit

    @property
    def limit(self) -> int:
        return self._limit

    async def __aenter__(self) -> "AdaptiveConcurrencyLimiter":
        if self._in_flight < self._limit:
            self._in_flight += 1
            return self
        waiter: Final = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self._in_flight -= 1
                self._grant()
            else:
                with suppress(ValueError):
                    self._waiters.remove(waiter)
            raise
        return self

    def _grant(self) -> None:
        while self._in_flight < self._limit and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            self._in_flight += 1
            waiter.set_result(None)

    async def __aexit__(self, *_: object) -> None:
        self._in_flight -= 1
        self._grant()

    def record(self, sample: PutSample) -> None:
        if sample.throttled:
            self._limit = max(self._floor, self._limit // 2)
            self._clean_streak = 0
            return
        self._clean_streak += 1
        if self._clean_streak >= self._limit and self._limit < self._ceiling:
            self._limit += 1
            self._clean_streak = 0
            self._grant()

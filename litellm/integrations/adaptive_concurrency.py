"""
Adaptive in-flight concurrency limiter (AIMD, Vector ARC style).

Grows the limit additively after `limit` consecutive clean completions and
halves it when the sink throttles or RTT inflates past a factor of the EWMA.
"""

import asyncio
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class PutSample:
    rtt_seconds: float
    throttled: bool  # 429, 503, any body <Code>SlowDown</Code>, or transport error/timeout


class AdaptiveConcurrencyLimiter:
    """AIMD in-flight limiter used as `async with limiter:`."""

    def __init__(
        self,
        initial: int,
        floor: int,
        ceiling: int,
        rtt_inflation_factor: float = 2.0,
        ewma_alpha: float = 0.4,
    ) -> None:
        if not 1 <= floor <= ceiling:
            raise ValueError(f"adaptive limiter bounds must satisfy 1 <= floor <= ceiling, got {floor}..{ceiling}")
        self._limit: int = min(max(initial, floor), ceiling)
        self._floor: Final[int] = floor
        self._ceiling: Final[int] = ceiling
        self._rtt_inflation_factor: Final[float] = rtt_inflation_factor
        self._ewma_alpha: Final[float] = ewma_alpha
        self._rtt_ewma: float | None = None
        self._clean_streak: int = 0
        self._in_flight: int = 0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def __aenter__(self) -> "AdaptiveConcurrencyLimiter":
        async with self._condition:
            await self._condition.wait_for(lambda: self._in_flight < self._limit)
            self._in_flight += 1
        return self

    async def __aexit__(self, *_: object) -> None:
        async with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    def record(self, sample: PutSample) -> None:
        ewma_before_update: Final[float | None] = self._rtt_ewma
        self._rtt_ewma = (
            sample.rtt_seconds
            if ewma_before_update is None
            else self._ewma_alpha * sample.rtt_seconds + (1 - self._ewma_alpha) * ewma_before_update
        )
        if sample.throttled or (
            ewma_before_update is not None and sample.rtt_seconds > self._rtt_inflation_factor * ewma_before_update
        ):
            self._limit = max(self._floor, self._limit // 2)
            self._clean_streak = 0
            return
        self._clean_streak += 1
        if self._clean_streak >= self._limit and self._limit < self._ceiling:
            self._limit += 1
            self._clean_streak = 0
            try:
                asyncio.get_running_loop().create_task(self._wake_waiters())
            except RuntimeError:
                pass

    async def _wake_waiters(self) -> None:
        async with self._condition:
            self._condition.notify_all()

import asyncio
from typing import Final

import pytest

from litellm.integrations.adaptive_concurrency import AdaptiveConcurrencyLimiter, PutSample


def _limiter(initial: int = 4, ceiling: int = 16) -> AdaptiveConcurrencyLimiter:
    return AdaptiveConcurrencyLimiter(initial=initial, floor=1, ceiling=ceiling)


@pytest.mark.asyncio
async def test_limit_grows_after_limit_clean_samples() -> None:
    limiter: Final = _limiter(initial=4)
    for _ in range(4):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    assert limiter.limit == 5


def test_limit_does_not_grow_before_the_streak_completes() -> None:
    limiter: Final = _limiter(initial=4)
    for _ in range(3):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    assert limiter.limit == 4


def test_throttled_sample_halves_the_limit() -> None:
    limiter: Final = _limiter(initial=16)
    limiter.record(PutSample(rtt_seconds=0.5, throttled=True))
    assert limiter.limit == 8


def test_rtt_above_twice_the_ewma_halves_the_limit() -> None:
    limiter: Final = _limiter(initial=16)
    for _ in range(20):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    limiter.record(PutSample(rtt_seconds=1.0, throttled=False))
    assert limiter.limit == limiter._ceiling // 2


def test_limit_clamps_at_the_floor() -> None:
    limiter: Final = _limiter(initial=1)
    limiter.record(PutSample(rtt_seconds=0.5, throttled=True))
    assert limiter.limit == 1


@pytest.mark.asyncio
async def test_limit_clamps_at_the_ceiling() -> None:
    limiter: Final = _limiter(initial=15, ceiling=16)
    for _ in range(1000):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    assert limiter.limit == 16


def test_throttled_sample_resets_the_clean_streak() -> None:
    limiter: Final = _limiter(initial=4, ceiling=32)
    for _ in range(3):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    limiter.record(PutSample(rtt_seconds=0.5, throttled=True))
    for _ in range(1):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    assert limiter.limit == 2


@pytest.mark.asyncio
async def test_growing_the_limit_wakes_a_waiting_acquirer() -> None:
    limiter: Final = AdaptiveConcurrencyLimiter(initial=1, floor=1, ceiling=4)
    acquired: Final[list[str]] = []  # mutable-ok: the waiter task appends to it across the await boundary
    released: Final = asyncio.Event()

    async def hold() -> None:
        async with limiter:
            await released.wait()

    holder: Final = asyncio.create_task(hold())

    async def waiter() -> None:
        async with limiter:
            acquired.append("waiter")

    pending: Final = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert not acquired

    # A clean sample grows limit 1 -> 2, waking the waiter even though the holder still waits.
    limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
    await asyncio.wait_for(asyncio.shield(pending), timeout=5)
    released.set()
    await asyncio.wait_for(holder, timeout=5)
    assert tuple(acquired) == ("waiter",)


@pytest.mark.asyncio
async def test_releasing_a_slot_wakes_exactly_one_waiter() -> None:
    limiter: Final = AdaptiveConcurrencyLimiter(initial=1, floor=1, ceiling=4)
    acquired: Final[list[str]] = []  # mutable-ok: the waiter tasks append to it across the await boundary
    release: Final = asyncio.Event()

    async def hold() -> None:
        async with limiter:
            await asyncio.sleep(0.05)

    async def waiter(name: str) -> None:
        async with limiter:
            acquired.append(name)
            await release.wait()

    holder: Final = asyncio.create_task(hold())
    waiters: Final = tuple(asyncio.create_task(waiter(f"w{i}")) for i in range(3))
    await asyncio.sleep(0.02)
    await asyncio.wait_for(holder, timeout=5)
    await asyncio.sleep(0.05)
    assert len(acquired) == 1

    # Growing the limit wakes the remaining waiters even though nothing else frees a slot.
    for _ in range(3):
        limiter.record(PutSample(rtt_seconds=0.1, throttled=False))
        await asyncio.sleep(0.05)
    assert len(acquired) == 3
    release.set()
    await asyncio.gather(*waiters)

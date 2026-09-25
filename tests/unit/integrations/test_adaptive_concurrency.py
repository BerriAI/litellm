import asyncio
from typing import Final

import pytest

from litellm.integrations.adaptive_concurrency import AdaptiveConcurrencyLimiter, PutSample


def _limiter(initial: int = 4, floor: int = 1, ceiling: int = 16) -> AdaptiveConcurrencyLimiter:
    return AdaptiveConcurrencyLimiter(initial=initial, floor=floor, ceiling=ceiling)


@pytest.mark.asyncio
async def test_limit_grows_after_limit_clean_samples() -> None:
    limiter: Final = _limiter(initial=4)
    for _ in range(4):
        limiter.record(PutSample(throttled=False))
    assert limiter.limit == 5


@pytest.mark.asyncio
async def test_limit_does_not_grow_before_the_streak_completes() -> None:
    limiter: Final = _limiter(initial=4)
    for _ in range(3):
        limiter.record(PutSample(throttled=False))
    assert limiter.limit == 4


@pytest.mark.asyncio
async def test_throttled_sample_halves_the_limit() -> None:
    limiter: Final = _limiter(initial=16)
    limiter.record(PutSample(throttled=True))
    assert limiter.limit == 8


@pytest.mark.asyncio
async def test_limit_clamps_at_the_floor() -> None:
    limiter: Final = _limiter(initial=4, floor=4)
    limiter.record(PutSample(throttled=True))
    assert limiter.limit == 4


@pytest.mark.asyncio
async def test_limit_clamps_at_the_ceiling() -> None:
    limiter: Final = _limiter(initial=15, ceiling=16)
    for _ in range(1000):
        limiter.record(PutSample(throttled=False))
    assert limiter.limit == 16


@pytest.mark.asyncio
async def test_throttled_sample_resets_the_clean_streak() -> None:
    limiter: Final = _limiter(initial=4, ceiling=32)
    for _ in range(3):
        limiter.record(PutSample(throttled=False))
    limiter.record(PutSample(throttled=True))
    limiter.record(PutSample(throttled=False))
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

    limiter.record(PutSample(throttled=False))
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

    for _ in range(3):
        limiter.record(PutSample(throttled=False))
        await asyncio.sleep(0.05)
    assert len(acquired) == 3
    release.set()
    await asyncio.gather(*waiters)


@pytest.mark.asyncio
async def test_double_cancel_during_release_leaves_in_flight_at_zero() -> None:
    limiter: Final = AdaptiveConcurrencyLimiter(initial=1, floor=1, ceiling=1)
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def hold() -> None:
        async with limiter:
            entered.set()
            await release.wait()

    holder: Final = asyncio.create_task(hold())
    await entered.wait()

    waiter: Final = asyncio.create_task(hold())
    await asyncio.sleep(0.02)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    await asyncio.wait_for(holder, timeout=5)
    holder.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass

    assert limiter._in_flight == 0


@pytest.mark.asyncio
async def test_a_cancelled_waiter_is_skipped_when_a_slot_frees() -> None:
    limiter: Final = AdaptiveConcurrencyLimiter(initial=1, floor=1, ceiling=1)
    acquired: Final[list[str]] = []  # mutable-ok: waiter tasks append across the await boundary
    first_entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def hold(name: str, entered: asyncio.Event | None = None) -> None:
        async with limiter:
            acquired.append(name)
            if entered is not None:
                entered.set()
            await release.wait()

    holder: Final = asyncio.create_task(hold("holder", first_entered))
    await first_entered.wait()
    doomed: Final = asyncio.create_task(hold("doomed"))
    next_waiter: Final = asyncio.create_task(hold("next"))
    await asyncio.sleep(0.02)
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed
    release.set()
    await asyncio.wait_for(holder, timeout=5)
    await asyncio.wait_for(next_waiter, timeout=5)

    assert "doomed" not in acquired
    assert "next" in acquired
    assert limiter._in_flight == 0

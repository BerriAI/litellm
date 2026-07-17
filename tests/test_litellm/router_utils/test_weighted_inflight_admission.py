import asyncio
from collections.abc import AsyncIterator, Callable
from typing import cast

import pytest

from litellm.router_utils.weighted_inflight_admission import (
    AdmissionClass,
    AdmissionClosedError,
    AdmissionQueueTimeoutError,
    AdmissionRejectedError,
    WeightedInFlightAdmission,
)


@pytest.fixture
def admission() -> WeightedInFlightAdmission:
    return WeightedInFlightAdmission(
        capacity=8,
        classes=(
            AdmissionClass(name="interactive", reservation=2, priority=0),
            AdmissionClass(name="background", reservation=6, priority=10),
        ),
    )


@pytest.mark.asyncio
async def test_background_cannot_consume_interactive_reservation(admission):
    leases = [await admission.acquire("background") for _ in range(6)]

    assert await admission.try_acquire("background") is None
    interactive_lease = await admission.try_acquire("interactive")
    assert interactive_lease is not None
    assert admission.active == 7

    await interactive_lease.release()
    await asyncio.gather(*(lease.release() for lease in leases))
    assert admission.active == 0


@pytest.mark.asyncio
async def test_idle_capacity_is_borrowable(admission):
    leases = [await admission.acquire("interactive") for _ in range(8)]

    assert len(leases) == 8
    assert admission.active == 8
    assert await admission.try_acquire("background") is None

    await asyncio.gather(*(lease.release() for lease in leases))


@pytest.mark.asyncio
async def test_release_is_idempotent(admission):
    lease = await admission.acquire("interactive")

    await lease.release()
    await lease.release()

    assert admission.active == 0
    assert admission.active_by_class["interactive"] == 0


@pytest.mark.asyncio
async def test_waiting_acquire_is_woken_by_release(admission):
    leases = [await admission.acquire("background") for _ in range(6)]
    leases.extend([await admission.acquire("interactive") for _ in range(2)])
    first_waiter = asyncio.create_task(admission.acquire("background"))
    second_waiter = asyncio.create_task(admission.acquire("interactive"))

    await asyncio.sleep(0)
    assert not first_waiter.done()
    assert not second_waiter.done()

    await leases[0].release()
    interactive_lease = await asyncio.wait_for(second_waiter, timeout=1)
    assert interactive_lease is not None
    assert not first_waiter.done()

    await leases[1].release()
    background_lease = await asyncio.wait_for(first_waiter, timeout=1)
    assert background_lease is not None

    await background_lease.release()
    await interactive_lease.release()
    await asyncio.gather(*(lease.release() for lease in leases[2:]))


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_consume_capacity(admission):
    leases = [await admission.acquire("background") for _ in range(6)]
    leases.extend([await admission.acquire("interactive") for _ in range(2)])
    waiter = asyncio.create_task(admission.acquire("interactive"))

    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    await leases[0].release()
    replacement = await asyncio.wait_for(admission.acquire("interactive"), timeout=1)
    assert replacement is not None

    await replacement.release()
    await asyncio.gather(*(lease.release() for lease in leases[1:]))


@pytest.mark.asyncio
async def test_close_rejects_new_requests_and_releases_waiters(admission):
    leases = [await admission.acquire("background") for _ in range(6)]
    leases.extend([await admission.acquire("interactive") for _ in range(2)])
    waiter = asyncio.create_task(admission.acquire("background"))

    await asyncio.sleep(0)
    await admission.close()

    with pytest.raises(AdmissionClosedError):
        await waiter
    assert admission.queued == 0
    with pytest.raises(AdmissionClosedError):
        await admission.acquire("interactive")
    with pytest.raises(AdmissionClosedError):
        await admission.try_acquire("interactive")
    await asyncio.gather(*(lease.release() for lease in leases))


@pytest.mark.asyncio
async def test_stream_release_happens_on_exhaustion(admission):
    lease = await admission.acquire("interactive")

    async def stream():
        yield "one"
        yield "two"

    wrapped = lease.wrap_async_iterator(stream())
    assert [item async for item in wrapped] == ["one", "two"]
    assert admission.active == 0


@pytest.mark.asyncio
async def test_stream_release_happens_on_close(admission):
    lease = await admission.acquire("interactive")

    async def stream():
        yield "one"
        await asyncio.sleep(10)

    wrapped = lease.wrap_async_iterator(stream())
    await wrapped.__anext__()
    await wrapped.aclose()

    assert admission.active == 0


@pytest.mark.asyncio
async def test_stream_release_happens_on_error(admission: WeightedInFlightAdmission):
    lease = await admission.acquire("interactive")

    async def stream():
        yield "one"
        raise RuntimeError("stream failed")

    wrapped = cast(AsyncIterator[str], lease.wrap_async_iterator(stream()))
    assert await anext(wrapped) == "one"
    with pytest.raises(RuntimeError, match="stream failed"):
        await anext(wrapped)

    assert admission.active == 0


@pytest.mark.asyncio
async def test_lease_context_manager_releases_permit(admission: WeightedInFlightAdmission):
    async with await admission.acquire("interactive") as lease:
        assert lease.admission_class == "interactive"
        assert admission.active == 1

    assert admission.active == 0


@pytest.mark.asyncio
async def test_unknown_class_and_invalid_release_are_rejected(admission: WeightedInFlightAdmission):
    with pytest.raises(ValueError, match="unknown admission class"):
        await admission.acquire("missing")
    with pytest.raises(ValueError, match="unknown admission class"):
        await admission.try_acquire("missing")
    with pytest.raises(RuntimeError, match="without an active permit"):
        await admission.release("interactive")


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: WeightedInFlightAdmission(0, (AdmissionClass("interactive", 0, 0),)),
            "capacity must be positive",
        ),
        (lambda: WeightedInFlightAdmission(1, ()), "at least one admission class"),
        (
            lambda: WeightedInFlightAdmission(1, (AdmissionClass("interactive", -1, 0),)),
            "cannot be negative",
        ),
        (
            lambda: WeightedInFlightAdmission(1, (AdmissionClass("interactive", 2, 0),)),
            "cannot exceed capacity",
        ),
        (
            lambda: WeightedInFlightAdmission(1, (AdmissionClass("interactive", 1, 0),), queue_timeout=-1),
            "queue_timeout cannot be negative",
        ),
        (
            lambda: WeightedInFlightAdmission(1, (AdmissionClass("interactive", 1, 0),), overflow="drop"),
            "overflow must be",
        ),
        (
            lambda: WeightedInFlightAdmission(
                1, (AdmissionClass("interactive", 0, 0), AdmissionClass("interactive", 0, 1))
            ),
            "names must be unique",
        ),
    ),
)
def test_invalid_configuration_is_rejected(factory: Callable[[], WeightedInFlightAdmission], message: str):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.asyncio
async def test_queue_timeout_cleans_waiter_and_records_metric():
    admission = WeightedInFlightAdmission(
        capacity=1,
        classes=(AdmissionClass(name="interactive", reservation=1, priority=0),),
        queue_timeout=0.01,
    )
    lease = await admission.acquire("interactive")
    with pytest.raises(AdmissionQueueTimeoutError):
        await admission.acquire("interactive")
    assert admission.queued == 0
    assert admission.metrics.snapshot()["timed_out"] == 1
    await lease.release()


@pytest.mark.asyncio
async def test_reject_overflow_records_metric():
    admission = WeightedInFlightAdmission(
        capacity=1,
        classes=(AdmissionClass(name="interactive", reservation=1, priority=0),),
        overflow="reject",
    )
    lease = await admission.acquire("interactive")
    with pytest.raises(AdmissionRejectedError):
        await admission.acquire("interactive")
    assert admission.queued == 0
    assert admission.metrics.snapshot()["rejected"] == 1
    await lease.release()

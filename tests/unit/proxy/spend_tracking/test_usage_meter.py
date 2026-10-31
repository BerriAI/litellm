import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from litellm.accounting import InboundUsageEvent, Metric, MetricName, Outcome
from litellm.proxy.db.accounting_ledger import AccountingConflictError, UsageWrite
from litellm.proxy.spend_tracking.usage_meter import DurableUsageMeter, UsageCaptureError, UsageCaptureSettings
from litellm.types.utils import CallTypes


def _write(identity: str) -> UsageWrite:
    event: Final = InboundUsageEvent(
        event_id=identity,
        request_id=identity,
        call_type=CallTypes.acompletion,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        outcome=Outcome.SUCCEEDED,
        dimensions=(),
        metrics=(Metric(name=MetricName.REQUESTS, value=Decimal(1)),),
    )
    return UsageWrite(event, json.dumps({"event": event.model_dump(mode="json")}))


class CommitBarrier:
    def __init__(self, *, conflict: str | None = None, failure: Exception | None = None) -> None:
        self.entered: Final = asyncio.Event()
        self.release: Final = asyncio.Event()
        self.committed: tuple[str, ...] = ()
        self.transactions = 0
        self.conflict: Final = conflict
        self.failure: Final = failure

    async def write(self, writes: tuple[UsageWrite, ...]) -> None:
        self.transactions += 1
        self.entered.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        identities: Final = tuple(write.event.event_id for write in writes)
        if self.conflict in identities:
            raise AccountingConflictError("Conflicting test event")
        self.committed += identities


async def _manual_flush(wakeup: asyncio.Event, timeout: float) -> None:
    await wakeup.wait()


@pytest.mark.asyncio
async def test_receipts_wait_for_one_batched_commit_and_keep_inflight_capacity_reserved() -> None:
    database: Final = CommitBarrier()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=2, max_pending_records=2))
    first: Final = meter.submit((_write("one"),))
    second: Final = meter.submit((_write("two"),))
    await database.entered.wait()
    assert not first.done() and not second.done()
    assert meter.pending_records == 2
    with pytest.raises(UsageCaptureError, match="capacity"):
        meter.submit((_write("three"),))
    database.release.set()
    await asyncio.gather(first, second)
    await meter.close()
    assert database.committed == ("one", "two")
    assert database.transactions == 1
    assert (meter.pending_bytes, meter.pending_records) == (0, 0)


@pytest.mark.asyncio
async def test_byte_limit_rejects_new_work_without_evicting_admitted_work() -> None:
    database: Final = CommitBarrier()
    write: Final = _write("large" * 300)
    size: Final = len(write.envelope.encode())
    meter: Final = DurableUsageMeter(
        database.write, UsageCaptureSettings(max_pending_bytes=size), wait_for_batch=_manual_flush
    )
    receipt: Final = meter.submit((write,))
    with pytest.raises(UsageCaptureError, match="capacity"):
        meter.submit((_write("extra"),))
    assert meter.pending_bytes == size
    database.release.set()
    await meter.close()
    await receipt
    assert database.committed == (write.event.event_id,)
    with pytest.raises(UsageCaptureError, match="closed"):
        meter.submit((_write("after-close"),))


@pytest.mark.asyncio
async def test_conflicting_group_rolls_back_and_valid_neighbor_still_commits() -> None:
    database: Final = CommitBarrier(conflict="conflict")
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=3))
    rejected: Final = meter.submit((_write("caller"), _write("conflict")))
    accepted: Final = meter.submit((_write("valid"),))
    await database.entered.wait()
    database.release.set()
    with pytest.raises(UsageCaptureError, match="identity"):
        await rejected
    await accepted
    await meter.close()
    assert database.committed == ("valid",)
    assert meter.pending_records == 0


@pytest.mark.asyncio
async def test_permanent_persistence_failure_reaches_each_receipt_and_releases_capacity() -> None:
    failure: Final = ValueError("invalid envelope")
    database: Final = CommitBarrier(failure=failure)
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=2))
    first: Final = meter.submit((_write("one"),))
    second: Final = meter.submit((_write("two"),))
    await database.entered.wait()
    database.release.set()
    results: Final = await asyncio.gather(first, second, return_exceptions=True)
    await meter.close()
    assert results == [failure, failure]
    assert meter.pending_bytes == 0


@pytest.mark.asyncio
async def test_cancelling_record_waiter_does_not_cancel_persistence() -> None:
    database: Final = CommitBarrier()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=2))
    waiting: Final = asyncio.create_task(meter.record(_write("caller").event))
    await database.entered.wait()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    database.release.set()
    await meter.close()
    assert database.committed == ("caller",)
    assert meter.pending_records == 0


@pytest.mark.asyncio
async def test_shutdown_deadline_rejects_unacknowledged_work_and_releases_capacity() -> None:
    database: Final = CommitBarrier()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=2))
    first: Final = meter.submit((_write("in-flight"), _write("in-flight-two")))
    await database.entered.wait()
    queued: Final = meter.submit((_write("queued"),))
    await meter.stop(0)
    results: Final = await asyncio.gather(first, queued, return_exceptions=True)
    assert all(isinstance(result, UsageCaptureError) for result in results)
    assert meter.pending_records == 0
    assert meter.pending_bytes == 0
    assert database.committed == ()


@pytest.mark.asyncio
async def test_shutdown_during_database_backoff_releases_reserved_capacity() -> None:
    retrying: Final = asyncio.Event()
    blocked: Final = asyncio.Event()

    async def unavailable(writes: tuple[UsageWrite, ...]) -> None:
        raise ConnectionError("database unavailable")

    async def backoff(delay: float) -> None:
        retrying.set()
        await blocked.wait()

    meter: Final = DurableUsageMeter(unavailable, UsageCaptureSettings(batch_size=2), retry_delay=backoff)
    receipt: Final = meter.submit((_write("one"), _write("two")))
    await retrying.wait()
    await meter.stop(0)
    with pytest.raises(UsageCaptureError, match="interrupted"):
        await receipt
    assert (meter.pending_records, meter.pending_bytes) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_closed_pooled_connection_retains_batch_until_automatic_recovery(wrapped: bool) -> None:
    import httpx
    from prisma.engine.errors import EngineRequestError
    from prisma.http import Response

    database: Final = CommitBarrier()
    attempts: Final = asyncio.Queue[tuple[UsageWrite, ...]]()
    retrying: Final = asyncio.Event()
    recovered: Final = asyncio.Event()
    closure: Final = EngineRequestError(
        Response(httpx.Response(500)), '{"error_code":"P1017","message":"Server has closed the connection."}'
    )
    failure: Final = RuntimeError("ledger transaction failed") if wrapped else closure
    if wrapped:
        failure.__cause__ = closure

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        attempts.put_nowait(writes)
        if attempts.qsize() == 1:
            raise failure
        await database.write(writes)

    async def backoff(delay: float) -> None:
        retrying.set()
        await recovered.wait()

    meter: Final = DurableUsageMeter(
        persist, UsageCaptureSettings(batch_size=2, max_pending_records=2), retry_delay=backoff
    )
    writes: Final = (_write("caller"), _write("attempt"))
    receipt: Final = meter.submit(writes)
    retry_signal: Final = asyncio.create_task(retrying.wait())
    reserved: Final = meter.pending_bytes
    try:
        await asyncio.wait((receipt, retry_signal), return_when=asyncio.FIRST_COMPLETED)
        assert not receipt.done(), "A dropped pooled connection must retain admitted usage for retry"
        assert (meter.pending_records, meter.pending_bytes) == (2, reserved)
        with pytest.raises(UsageCaptureError, match="capacity"):
            meter.submit((_write("overflow"),))
        database.release.set()
        recovered.set()
        await receipt
        assert attempts.get_nowait() == attempts.get_nowait() == writes
        assert database.committed == ("caller", "attempt")
        assert (meter.pending_records, meter.pending_bytes) == (0, 0)
    finally:
        retry_signal.cancel()
        await asyncio.gather(retry_signal, return_exceptions=True)
        await meter.stop(0)
        await asyncio.gather(receipt, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_commits", [False, True])
async def test_shutdown_during_conflict_isolation_completes_every_receipt(first_commits: bool) -> None:
    isolating: Final = asyncio.Event()
    blocked: Final = asyncio.Event()

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        if len(writes) > 1:
            raise AccountingConflictError("one producer conflicts")
        if first_commits and writes[0].event.event_id == "one":
            return
        isolating.set()
        await blocked.wait()

    meter: Final = DurableUsageMeter(persist, UsageCaptureSettings(batch_size=2))
    first: Final = meter.submit((_write("one"),))
    second: Final = meter.submit((_write("two"),))
    await isolating.wait()
    await meter.stop(0)
    results: Final = await asyncio.gather(first, second, return_exceptions=True)
    assert results[0] is None if first_commits else isinstance(results[0], UsageCaptureError)
    assert isinstance(results[1], UsageCaptureError)
    assert (meter.pending_records, meter.pending_bytes) == (0, 0)


@pytest.mark.asyncio
async def test_elapsed_flush_deadline_commits_a_partial_batch() -> None:
    from litellm.proxy.spend_tracking.usage_meter import _wait_for_batch

    database: Final = CommitBarrier()
    database.release.set()

    async def expired_flush(wakeup: asyncio.Event, timeout: float) -> None:
        assert timeout == 5.0
        await _wait_for_batch(wakeup, 0)

    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=10), wait_for_batch=expired_flush)
    receipt: Final = meter.submit((_write("partial-batch"),))
    try:
        await receipt
        assert database.committed == ("partial-batch",)
        assert database.transactions == 1
        assert (meter.pending_records, meter.pending_bytes) == (0, 0)
    finally:
        await meter.stop(0)


@pytest.mark.asyncio
async def test_burst_drains_full_batches_without_waiting_for_partial_batch_deadline() -> None:
    database: Final = CommitBarrier()
    database.release.set()
    waits: Final = asyncio.Queue[bool]()

    async def observe_wait(wakeup: asyncio.Event, timeout: float) -> None:
        waits.put_nowait(wakeup.is_set())
        await wakeup.wait()

    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=2), wait_for_batch=observe_wait)
    receipts: Final = tuple(meter.submit((_write(str(index)),)) for index in range(5))
    try:
        assert await waits.get() is True
        await asyncio.gather(*receipts[:2])
        assert await waits.get() is True, "A full queued batch must not wait for the flush deadline"
        await asyncio.gather(*receipts[2:4])
        assert await waits.get() is False
        assert database.committed == ("0", "1", "2", "3")
        assert meter.pending_records == 1
        assert not receipts[-1].done()
        await meter.close()
        await receipts[-1]
        assert database.committed == ("0", "1", "2", "3", "4")
        assert database.transactions == 3
        assert (meter.pending_records, meter.pending_bytes) == (0, 0)
    finally:
        await meter.stop(0)
        await asyncio.gather(*receipts, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["22P05", "22003", "42P01"])
async def test_invalid_submission_isolated_without_retrying_systemic_faults(sqlstate: str) -> None:
    from prisma.errors import DataError

    database: Final = CommitBarrier()
    database.release.set()
    attempts: Final = asyncio.Queue[tuple[str, ...]]()

    async def persist(writes: tuple[UsageWrite, ...]) -> None:
        identities: Final = tuple(write.event.event_id for write in writes)
        attempts.put_nowait(identities)
        if "invalid" in identities or sqlstate == "42P01":
            raise DataError(
                data={"user_facing_error": {"message": "database rejected input", "meta": {"code": sqlstate}}}
            )
        await database.write(writes)

    meter: Final = DurableUsageMeter(persist, UsageCaptureSettings(batch_size=3))
    rejected: Final = meter.submit((_write("caller"), _write("invalid")))
    neighbor: Final = meter.submit((_write("neighbor"),))
    try:
        with pytest.raises(DataError):
            await rejected
        if sqlstate == "42P01":
            with pytest.raises(DataError):
                await neighbor
            assert attempts.qsize() == 1
        else:
            await neighbor
            assert database.committed == ("neighbor",)
            assert attempts.qsize() == 3
    finally:
        await meter.close()
    assert (meter.pending_records, meter.pending_bytes) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency_outcome", ["commit", "failure", "cancel"])
async def test_commit_dependencies_preserve_capacity_and_isolate_failed_dependents(dependency_outcome: str) -> None:
    database: Final = CommitBarrier()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=3, max_pending_records=4))
    prerequisite: Final[asyncio.Future[None]] = asyncio.get_running_loop().create_future()
    dependent: Final = meter.submit((_write("parent"), _write("final")), prerequisites=(prerequisite,))
    neighbor: Final = meter.submit((_write("neighbor"),))
    assert meter.pending_records == 3
    assert not dependent.done()
    with pytest.raises(UsageCaptureError, match="capacity"):
        meter.submit((_write("excess-one"), _write("excess-two")))
    if dependency_outcome == "commit":
        prerequisite.set_result(None)
    elif dependency_outcome == "failure":
        prerequisite.set_exception(ValueError("dependency failed"))
    else:
        prerequisite.cancel()
    database.release.set()
    await meter.close()
    await neighbor
    if dependency_outcome == "commit":
        await dependent
        assert database.committed == ("parent", "final", "neighbor")
    else:
        with pytest.raises(UsageCaptureError, match="dependency"):
            await dependent
        assert database.committed == ("neighbor",)
    assert meter.pending_records == 0
    assert meter.pending_bytes == 0


@pytest.mark.asyncio
async def test_shutdown_does_not_cancel_dependency_receipt_or_leak_reserved_capacity() -> None:
    database: Final = CommitBarrier()
    meter: Final = DurableUsageMeter(database.write, UsageCaptureSettings(batch_size=2))
    prerequisite: Final[asyncio.Future[None]] = asyncio.get_running_loop().create_future()
    dependent: Final = meter.submit((_write("parent"), _write("final")), prerequisites=(prerequisite,))
    await meter.stop(0)
    with pytest.raises(UsageCaptureError, match="acknowledgement"):
        await dependent
    assert not prerequisite.done()
    assert database.committed == ()
    assert meter.pending_records == 0
    assert meter.pending_bytes == 0
    prerequisite.set_result(None)

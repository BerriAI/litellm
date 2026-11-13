import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import accumulate
from typing import Final

import pytest

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.log import build_audit_log
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import (
    Accepted,
    Batch,
    Delivered,
    DeliveryResult,
    Dropped,
    Refused,
    Rejected,
    Undelivered,
    build_pipeline,
)

T0: Final = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


@dataclass
class FakeSink:  # mutable-ok: records deliveries for assertions
    name: str = "fake"
    fail_first: int = 0
    fail_close: bool = False
    batches: list[Batch] = field(default_factory=list)
    closed: bool = False

    async def deliver(self, batch: Batch) -> DeliveryResult:
        if self.fail_first > 0:
            self.fail_first -= 1
            return Undelivered("sink down")
        self.batches.append(batch)
        return Delivered(len(batch))

    async def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("sink refused to close")


@dataclass
class PrefixSettlingSink:  # mutable-ok: records the batches received for assertions
    batches: list[Batch] = field(default_factory=list)
    name: str = "prefix"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        self.batches.append(batch)
        return Undelivered("partial delivery", settled=2) if len(self.batches) == 1 else Delivered(len(batch))

    async def close(self) -> None:
        return None


@dataclass
class Clock:  # mutable-ok: records sleeps for assertions
    slept: list[float] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _records(count: int) -> tuple[AuditRecord, ...]:
    def append_record(
        previous: tuple[ChainState, tuple[AuditRecord, ...]], event: AuditEvent
    ) -> tuple[ChainState, tuple[AuditRecord, ...]]:
        state, records = previous
        next_state, record = chain(event, Product("t"), state)
        return next_state, (*records, record)

    events: Final = tuple(
        AuditEvent(T0, "alice", None, "eu", "GET", f"/key/{index}", "allowed", 200, None, None)
        for index in range(count)
    )
    results: Final = tuple(accumulate(events, append_record, initial=(ChainState(), ())))
    return results[-1][1]


@pytest.mark.asyncio
async def test_records_are_delivered_in_order_in_batches_of_the_configured_size() -> None:
    sink, clock = FakeSink(), Clock()
    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=2, strict=False, sleep=clock.sleep)
    for record in _records(5):
        assert pipeline.publish(record) == (Accepted(),)
    (worker,) = pipeline.workers
    assert [await worker.flush_once() for _ in range(4)] == [2, 2, 1, 0]
    assert [[r.sequence for r in batch] for batch in sink.batches] == [[1, 2], [3, 4], [5]]


@pytest.mark.asyncio
async def test_a_full_queue_drops_the_oldest_and_counts_it_unless_strict() -> None:
    sink: Final = FakeSink()
    lenient: Final = build_pipeline((sink,), queue_size=2, batch_size=10, strict=False)
    one, two, three = _records(3)
    assert [lenient.publish(r) for r in (one, two, three)] == [(Accepted(),), (Accepted(),), (Dropped("fake"),)]
    (worker,) = lenient.workers
    assert worker.dropped == 1 and lenient.accepting() is True
    await worker.flush_once()
    assert [r.sequence for r in sink.batches[0]] == [2, 3]

    strict: Final = build_pipeline((FakeSink(),), queue_size=2, batch_size=10, strict=True)
    assert [strict.publish(r) for r in (one, two, three)] == [(Accepted(),), (Accepted(),), (Refused("fake"),)]
    assert strict.accepting() is False and strict.workers[0].queue.qsize() == 2


def test_a_strict_reservation_holds_the_slot_until_the_record_consumes_or_releases_it() -> None:
    strict: Final = build_pipeline((FakeSink(),), queue_size=1, batch_size=10, strict=True)
    one, two = _records(2)
    held: Final = strict.reserve()
    assert held is not None and strict.reserve() is None, "two in-flight writes cannot share one free slot"
    assert strict.publish(one) == (Refused("fake"),), "an unreserved event does not steal a reserved slot"
    held.release()
    assert strict.publish(one) == (Accepted(),)
    held.release()
    assert strict.publish(two) == (Refused("fake"),), "a second release does not free a slot that is in use"
    assert build_pipeline((FakeSink(),), queue_size=1, batch_size=10, strict=False).reserve() is not None


@dataclass(frozen=True, slots=True)
class RejectingSink:
    name: str = "rejecting"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        return Rejected("permanent rejection", len(batch))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_reserved_write_is_enqueued_after_a_previous_batch_is_rejected() -> None:
    accepted_sink: Final = FakeSink()
    pipeline: Final = build_pipeline((RejectingSink(), accepted_sink), queue_size=2, batch_size=1, strict=True)
    audit: Final = build_audit_log(pipeline=pipeline, now=lambda: T0)
    audit.record("alice", None, "eu", "POST", "/key/generate", "allowed", 200)
    held: Final = audit.reserve()
    assert held is not None
    assert [await worker.flush_once() for worker in pipeline.workers] == [1, 1]
    assert not audit.accepting() and audit.reserve() is None
    audit.record("alice", None, "eu", "POST", "/key/update", "allowed", 200, reservation=held)
    assert [worker.queue.qsize() for worker in pipeline.workers] == [1, 1]
    assert all(not worker.reservations for worker in pipeline.workers)
    held.release()
    assert pipeline.publish(_records(1)[0], held) == (Refused("rejecting"), Accepted())
    assert [await worker.flush_once() for worker in pipeline.workers] == [1, 1]
    assert accepted_sink.batches[-1][0].event.path == "/key/update"
    assert pipeline.workers[0].rejected == 2


@pytest.mark.asyncio
async def test_a_failing_sink_is_retried_with_backoff_and_the_batch_is_never_lost() -> None:
    sink, clock = FakeSink(fail_first=3), Clock()
    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False, sleep=clock.sleep)
    for record in _records(2):
        pipeline.publish(record)
    (worker,) = pipeline.workers
    assert await worker.flush_once() == 2
    assert clock.slept == [0.5, 1.0, 2.0] and [r.sequence for r in sink.batches[0]] == [1, 2]
    assert worker.failures == 0


@pytest.mark.asyncio
async def test_worker_retries_only_the_unsettled_suffix() -> None:
    sink: Final = PrefixSettlingSink()
    clock: Final = Clock()
    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=4, strict=False, sleep=clock.sleep)
    records: Final = _records(4)
    for record in records:
        pipeline.publish(record)

    (worker,) = pipeline.workers
    assert await worker.flush_once() == 4
    assert sink.batches == [records, records[2:]]
    assert clock.slept == [0.5]


@pytest.mark.asyncio
async def test_a_sink_down_for_thousands_of_attempts_still_gets_the_batch_when_it_recovers() -> None:
    sink, clock = FakeSink(fail_first=5_000), Clock()
    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False, sleep=clock.sleep)
    pipeline.publish(_records(1)[0])
    assert await pipeline.workers[0].flush_once() == 1
    assert len(clock.slept) == 5_000 and clock.slept[-1] == 30.0 and sink.batches


@pytest.mark.asyncio
async def test_each_sink_has_its_own_queue_so_one_down_sink_does_not_block_another() -> None:
    down, up, clock = FakeSink("down", fail_first=1), FakeSink("up"), Clock()
    pipeline: Final = build_pipeline((down, up), queue_size=10, batch_size=10, strict=False, sleep=clock.sleep)
    (record,) = _records(1)
    assert pipeline.publish(record) == (Accepted(), Accepted())
    await pipeline.workers[1].flush_once()
    assert up.batches and not down.batches
    await pipeline.workers[0].flush_once()
    assert down.batches


@pytest.mark.asyncio
async def test_start_restarts_a_worker_with_a_finished_task() -> None:
    sink: Final = FakeSink()
    pipeline: Final = build_pipeline((sink,), queue_size=2, batch_size=1, strict=False)
    (worker,) = pipeline.workers
    previous: Final = asyncio.create_task(asyncio.sleep(0))
    await previous
    worker.task = previous

    worker.start()

    assert worker.task is not None and worker.task is not previous and not worker.task.done()
    await worker.close(drain_seconds=0.1)


@pytest.mark.asyncio
async def test_start_runs_the_worker_and_close_drains_then_closes_the_sink() -> None:
    sink: Final = FakeSink()
    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False)
    pipeline.start()
    for record in _records(3):
        pipeline.publish(record)
    await asyncio.sleep(0)
    await pipeline.close()
    assert sum(len(batch) for batch in sink.batches) == 3 and sink.closed is True
    assert pipeline.workers[0].queue.empty()


@pytest.mark.asyncio
async def test_close_isolates_sink_failures_and_closes_every_sink(caplog: pytest.LogCaptureFixture) -> None:
    first, second = FakeSink("first", fail_close=True), FakeSink("second")
    pipeline: Final = build_pipeline((first, second), queue_size=2, batch_size=10, strict=False)
    (record,) = _records(1)
    pipeline.publish(record)

    await pipeline.close()

    assert first.closed and second.closed
    assert first.batches == [(record,)] and second.batches == [(record,)]
    assert "sink refused to close" in caplog.text


@dataclass
class StallingSink:  # mutable-ok: the test controls a blocked delivery
    stalled: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    batches: list[Batch] = field(default_factory=list)
    name: str = "stalling"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        self.calls += 1
        if self.calls == 1:
            self.stalled.set()
            await self.released.wait()
        self.batches.append(batch)
        return Delivered(len(batch))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_close_redelivers_the_batch_a_worker_was_holding_when_it_crashed() -> None:
    sink: Final = CrashingSink()
    pipeline: Final = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False)
    pipeline.start()
    for record in _records(2):
        pipeline.publish(record)
    await sink.crashed.wait()
    (worker,) = pipeline.workers
    task: Final = worker.task
    assert task is not None
    with pytest.raises(RuntimeError, match="delivery crashed"):
        await task
    await pipeline.close()
    assert [[r.sequence for r in batch] for batch in sink.batches] == [[1, 2]]


@dataclass
class CrashingSink:  # mutable-ok: the test observes the failed delivery and its retry
    crashed: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    batches: list[Batch] = field(default_factory=list)
    name: str = "crashing"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        self.calls += 1
        if self.calls == 1:
            self.crashed.set()
            raise RuntimeError("delivery crashed")
        self.batches.append(batch)
        return Delivered(len(batch))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_auth_failures_do_not_consume_strict_write_reservations() -> None:
    sink: Final = StallingSink()
    pipeline: Final = build_pipeline((sink,), queue_size=4, batch_size=1, strict=True)
    audit: Final = build_audit_log(pipeline=pipeline)
    pipeline.start()
    audit.record("anonymous", None, "control-plane", "GET", "/auth/me", "denied", 401, kind="authentication")
    await sink.stalled.wait()
    for _ in range(6):
        audit.record("anonymous", None, "control-plane", "GET", "/auth/me", "denied", 401, kind="authentication")
    (worker,) = pipeline.workers
    assert worker.queue.qsize() == 2
    assert worker.dropped == 4
    reservation: Final = audit.reserve()
    assert reservation is not None
    reservation.release()
    sink.released.set()
    await pipeline.close()


@pytest.mark.asyncio
async def test_unauthenticated_failures_keep_a_queue_slot_for_writes() -> None:
    sink: Final = FakeSink()
    pipeline: Final = build_pipeline((sink,), queue_size=2, batch_size=10, strict=False)
    audit: Final = build_audit_log(pipeline=pipeline, now=lambda: T0)

    audit.record_login("anonymous", "denied")
    audit.record_login("another-anonymous", "denied")

    (worker,) = pipeline.workers
    assert worker.queue.qsize() == 1
    assert worker.dropped == 1
    assert await worker.flush_once() == 1
    assert len(sink.batches) == 1 and len(sink.batches[0]) == 1
    assert sink.batches[0][0].event.actor == "anonymous"

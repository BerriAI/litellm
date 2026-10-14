import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

import pytest

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import (
    Accepted,
    Batch,
    Delivered,
    DeliveryResult,
    Dropped,
    Refused,
    Undelivered,
    build_pipeline,
)

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


@dataclass
class FakeSink:  # mutable-ok: records deliveries for assertions
    name: str = "fake"
    fail_first: int = 0
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


@dataclass
class Clock:  # mutable-ok: records sleeps for assertions
    slept: list[float] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _records(count: int) -> tuple[AuditRecord, ...]:
    state: Final[list[ChainState]] = [ChainState()]
    out: Final[list[AuditRecord]] = []
    for index in range(count):
        event = AuditEvent(T0, "alice", None, "eu", "GET", f"/key/{index}", "allowed", 200, None, None)
        state[0], record = chain(event, Product("t"), state[0])
        out.append(record)
    return tuple(out)


@pytest.mark.asyncio
async def test_records_are_delivered_in_order_in_batches_of_the_configured_size() -> None:
    sink, clock = FakeSink(), Clock()
    pipeline = build_pipeline((sink,), queue_size=10, batch_size=2, strict=False, sleep=clock.sleep)
    for record in _records(5):
        assert pipeline.publish(record) == (Accepted(),)
    (worker,) = pipeline.workers
    assert [await worker.flush_once() for _ in range(4)] == [2, 2, 1, 0]
    assert [[r.sequence for r in batch] for batch in sink.batches] == [[1, 2], [3, 4], [5]]


@pytest.mark.asyncio
async def test_a_full_queue_drops_the_oldest_and_counts_it_unless_strict() -> None:
    sink = FakeSink()
    lenient = build_pipeline((sink,), queue_size=2, batch_size=10, strict=False)
    one, two, three = _records(3)
    assert [lenient.publish(r) for r in (one, two, three)] == [(Accepted(),), (Accepted(),), (Dropped("fake"),)]
    (worker,) = lenient.workers
    assert worker.dropped == 1 and lenient.accepting() is True
    await worker.flush_once()
    assert [r.sequence for r in sink.batches[0]] == [2, 3]

    strict = build_pipeline((FakeSink(),), queue_size=2, batch_size=10, strict=True)
    assert [strict.publish(r) for r in (one, two, three)] == [(Accepted(),), (Accepted(),), (Refused("fake"),)]
    assert strict.accepting() is False and strict.workers[0].queue.qsize() == 2


def test_a_strict_reservation_holds_the_slot_until_the_record_consumes_or_releases_it() -> None:
    strict = build_pipeline((FakeSink(),), queue_size=1, batch_size=10, strict=True)
    one, two = _records(2)
    held = strict.reserve()
    assert held is not None and strict.reserve() is None, "two in-flight writes cannot share one free slot"
    assert strict.publish(one) == (Refused("fake"),), "an unreserved event does not steal a reserved slot"
    held.release()
    assert strict.publish(one) == (Accepted(),)
    held.release()
    assert strict.publish(two) == (Refused("fake"),), "a second release does not free a slot that is in use"
    assert build_pipeline((FakeSink(),), queue_size=1, batch_size=10, strict=False).reserve() is not None


@pytest.mark.asyncio
async def test_a_failing_sink_is_retried_with_backoff_and_the_batch_is_never_lost() -> None:
    sink, clock = FakeSink(fail_first=3), Clock()
    pipeline = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False, sleep=clock.sleep)
    for record in _records(2):
        pipeline.publish(record)
    (worker,) = pipeline.workers
    assert await worker.flush_once() == 2
    assert clock.slept == [0.5, 1.0, 2.0] and [r.sequence for r in sink.batches[0]] == [1, 2]
    assert worker.failures == 0


@pytest.mark.asyncio
async def test_a_sink_down_for_thousands_of_attempts_still_gets_the_batch_when_it_recovers() -> None:
    sink, clock = FakeSink(fail_first=5_000), Clock()
    pipeline = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False, sleep=clock.sleep)
    pipeline.publish(_records(1)[0])
    assert await pipeline.workers[0].flush_once() == 1
    assert len(clock.slept) == 5_000 and clock.slept[-1] == 30.0 and sink.batches


@pytest.mark.asyncio
async def test_each_sink_has_its_own_queue_so_one_down_sink_does_not_block_another() -> None:
    down, up, clock = FakeSink("down", fail_first=1), FakeSink("up"), Clock()
    pipeline = build_pipeline((down, up), queue_size=10, batch_size=10, strict=False, sleep=clock.sleep)
    (record,) = _records(1)
    assert pipeline.publish(record) == (Accepted(), Accepted())
    await pipeline.workers[1].flush_once()
    assert up.batches and not down.batches
    await pipeline.workers[0].flush_once()
    assert down.batches


@pytest.mark.asyncio
async def test_start_runs_the_worker_and_close_drains_then_closes_the_sink() -> None:
    sink = FakeSink()
    pipeline = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False)
    pipeline.start()
    for record in _records(3):
        pipeline.publish(record)
    await asyncio.sleep(0)
    await pipeline.close()
    assert sum(len(batch) for batch in sink.batches) == 3 and sink.closed is True
    assert pipeline.workers[0].queue.empty()


@dataclass
class StallingSink:  # mutable-ok: the first delivery hangs until shutdown interrupts it
    stalled: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    batches: list[Batch] = field(default_factory=list)
    name: str = "stalling"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        self.calls += 1
        if self.calls == 1:
            self.stalled.set()
            await asyncio.Event().wait()
        self.batches.append(batch)
        return Delivered(len(batch))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_close_redelivers_the_batch_a_worker_was_holding_when_it_was_stopped() -> None:
    sink = StallingSink()
    pipeline = build_pipeline((sink,), queue_size=10, batch_size=10, strict=False)
    pipeline.start()
    for record in _records(2):
        pipeline.publish(record)
    await sink.stalled.wait()
    await pipeline.close()
    assert [[r.sequence for r in batch] for batch in sink.batches] == [[1, 2]]

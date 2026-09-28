import asyncio
from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.hooks.fairness_queue import (
    ClassQueueState,
    FairQueue,
    InMemoryFairQueueStore,
    QueueAdmitted,
    QueueRejected,
    QueueReleased,
    QueueTicket,
)

MODEL: Final = "gpt-4o"
WEIGHTS: Final = MappingProxyType({"prod": 0.75, "batch": 0.25})


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _never_cancelled() -> bool:
    return False


async def _never_cancelled_after_probing_the_client() -> bool:
    await asyncio.sleep(0)
    return False


async def _always_cancelled() -> bool:
    return True


def _ticket(class_name: str, request_id: str) -> QueueTicket:
    return QueueTicket(model=MODEL, class_name=class_name, request_id=request_id)


class CapacityGate:
    """Admits `slots` requests total, recording the class of each admission in order."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.admitted: list[str] = []
        self.current: str | None = None

    def for_class(self, class_name: str):
        async def try_admit() -> bool:
            if self.slots <= 0:
                return False
            self.slots -= 1
            self.admitted.append(class_name)
            return True

        return try_admit


@pytest.mark.asyncio
async def test_rejects_immediately_when_class_queue_is_full():
    queue: Final = FairQueue()
    gate: Final = CapacityGate(slots=0)

    async def blocked_waiter(request_id: str):
        return await queue.wait_for_admission(
            ticket=_ticket("batch", request_id),
            weights=WEIGHTS,
            max_wait_seconds=0.5,
            max_depth=1,
            poll_interval_seconds=0.01,
            try_admit=gate.for_class("batch"),
            is_cancelled=_never_cancelled,
        )

    first: Final = asyncio.create_task(blocked_waiter("r1"))
    await asyncio.sleep(0.02)
    overflow: Final = await blocked_waiter("r2")
    assert overflow == QueueRejected(reason="queue_full", waited_seconds=0.0)
    assert queue.depths(MODEL)["batch"] == 1
    first_outcome: Final = await first
    assert isinstance(first_outcome, QueueRejected)
    assert first_outcome.reason == "queue_deadline_exceeded"
    assert queue.depths(MODEL)["batch"] == 0


@pytest.mark.asyncio
async def test_deadline_rejection_reports_wait_time_and_removes_ticket():
    clock: Final = FakeClock()
    queue: Final = FairQueue(clock=clock)
    attempts: list[float] = []

    async def try_admit() -> bool:
        attempts.append(clock.now)
        clock.now += 0.4
        return False

    outcome: Final = await queue.wait_for_admission(
        ticket=_ticket("prod", "r1"),
        weights=WEIGHTS,
        max_wait_seconds=1.0,
        max_depth=10,
        poll_interval_seconds=0.001,
        try_admit=try_admit,
        is_cancelled=_never_cancelled,
    )
    assert isinstance(outcome, QueueRejected)
    assert outcome.reason == "queue_deadline_exceeded"
    assert outcome.waited_seconds >= 1.0
    assert len(attempts) == 3
    assert queue.depths(MODEL)["prod"] == 0


@pytest.mark.asyncio
async def test_client_disconnect_cancels_queued_request_without_admitting():
    queue: Final = FairQueue()
    gate: Final = CapacityGate(slots=5)
    outcome: Final = await queue.wait_for_admission(
        ticket=_ticket("prod", "r1"),
        weights=WEIGHTS,
        max_wait_seconds=5.0,
        max_depth=10,
        poll_interval_seconds=0.001,
        try_admit=gate.for_class("prod"),
        is_cancelled=_always_cancelled,
    )
    assert isinstance(outcome, QueueRejected)
    assert outcome.reason == "client_disconnected"
    assert gate.admitted == []
    assert queue.depths(MODEL)["prod"] == 0


@pytest.mark.asyncio
async def test_waiter_is_released_without_admission_once_fairness_is_switched_off_mid_wait():
    clock: Final = FakeClock()
    queue: Final = FairQueue(clock=clock)
    gate: Final = CapacityGate(slots=0)
    fairness_enabled: Final = [True]

    async def switch_off_after(seconds: float) -> None:
        await asyncio.sleep(seconds)
        clock.now += 0.5
        fairness_enabled[0] = False

    switch: Final = asyncio.create_task(switch_off_after(0.02))
    outcome: Final = await queue.wait_for_admission(
        ticket=_ticket("batch", "r1"),
        weights=WEIGHTS,
        max_wait_seconds=30.0,
        max_depth=10,
        poll_interval_seconds=0.001,
        try_admit=gate.for_class("batch"),
        is_cancelled=_never_cancelled_after_probing_the_client,
        is_released=lambda: not fairness_enabled[0],
    )
    await switch
    assert outcome == QueueReleased(waited_seconds=0.5)
    assert gate.admitted == []
    assert queue.depths(MODEL)["batch"] == 0


@pytest.mark.asyncio
async def test_admitted_after_capacity_frees_reports_wait_and_stops_polling():
    clock: Final = FakeClock()
    queue: Final = FairQueue(clock=clock)
    calls: list[int] = []

    async def try_admit() -> bool:
        calls.append(1)
        clock.now += 0.25
        return len(calls) >= 3

    outcome: Final = await queue.wait_for_admission(
        ticket=_ticket("prod", "r1"),
        weights=WEIGHTS,
        max_wait_seconds=10.0,
        max_depth=10,
        poll_interval_seconds=0.001,
        try_admit=try_admit,
        is_cancelled=_never_cancelled,
    )
    assert outcome == QueueAdmitted(waited_seconds=0.75)
    assert len(calls) == 3
    assert queue.depths(MODEL)["prod"] == 0


@pytest.mark.asyncio
async def test_weighted_ordering_gives_reserved_share_of_turns_without_starving_low_class():
    store: Final = InMemoryFairQueueStore()
    queue: Final = FairQueue(store=store)
    gate: Final = CapacityGate(slots=0)

    async def waiter(class_name: str, request_id: str):
        return await queue.wait_for_admission(
            ticket=_ticket(class_name, request_id),
            weights=WEIGHTS,
            max_wait_seconds=30.0,
            max_depth=100,
            poll_interval_seconds=0.001,
            try_admit=gate.for_class(class_name),
            is_cancelled=_never_cancelled,
        )

    tasks: Final = [
        asyncio.create_task(waiter(class_name, f"{class_name}-{i}"))
        for class_name in ("prod", "batch")
        for i in range(4)
    ]
    await asyncio.sleep(0.05)
    assert gate.admitted == []
    gate.slots = 8
    outcomes: Final = await asyncio.gather(*tasks)
    assert all(isinstance(outcome, QueueAdmitted) for outcome in outcomes)
    assert sorted(gate.admitted) == ["batch"] * 4 + ["prod"] * 4
    first_four: Final = gate.admitted[:4]
    assert first_four.count("prod") == 3
    assert first_four.count("batch") == 1


class PerClassGate:
    """Admits per class while that class has slots, recording the class of each admission in order."""

    def __init__(self, **slots: int) -> None:
        self.slots = dict(slots)
        self.attempts: dict[str, int] = {}
        self.admitted: list[str] = []

    def for_class(self, class_name: str):
        async def try_admit() -> bool:
            self.attempts[class_name] = self.attempts.get(class_name, 0) + 1
            if self.slots.get(class_name, 0) <= 0:
                return False
            self.slots[class_name] -= 1
            self.admitted.append(class_name)
            return True

        return try_admit


def _waiter(queue: FairQueue, gate: PerClassGate, class_name: str, request_id: str, max_wait: float = 30.0):
    return asyncio.create_task(
        queue.wait_for_admission(
            ticket=_ticket(class_name, request_id),
            weights=WEIGHTS,
            max_wait_seconds=max_wait,
            max_depth=100,
            poll_interval_seconds=0.001,
            try_admit=gate.for_class(class_name),
            is_cancelled=_never_cancelled_after_probing_the_client,
        )
    )


@pytest.mark.asyncio
async def test_newer_low_share_request_does_not_jump_a_class_that_waited_through_failed_admits():
    queue: Final = FairQueue()
    gate: Final = PerClassGate(prod=0, batch=0)
    prod: Final = _waiter(queue, gate, "prod", "prod-1")
    while gate.attempts.get("prod", 0) < 30:
        await asyncio.sleep(0.001)
    batch: Final = _waiter(queue, gate, "batch", "batch-1")
    while queue.depths(MODEL).get("batch", 0) < 1:
        await asyncio.sleep(0)
    gate.slots["prod"] = 1
    gate.slots["batch"] = 1
    outcomes: Final = await asyncio.gather(prod, batch)
    assert all(isinstance(outcome, QueueAdmitted) for outcome in outcomes)
    assert gate.admitted == ["prod", "batch"]


@pytest.mark.asyncio
async def test_class_blocked_on_its_own_share_yields_the_slot_to_a_class_that_fits():
    queue: Final = FairQueue()
    gate: Final = PerClassGate(prod=0, batch=1)
    prod: Final = _waiter(queue, gate, "prod", "prod-1", max_wait=5.0)
    await asyncio.sleep(0.01)
    batch_outcome: Final = await _waiter(queue, gate, "batch", "batch-1", max_wait=1.0)
    assert isinstance(batch_outcome, QueueAdmitted)
    assert gate.admitted == ["batch"]
    prod.cancel()


def test_removing_the_head_that_yielded_hands_its_successor_a_fresh_turn():
    store: Final = InMemoryFairQueueStore()
    store.enqueue(_ticket("prod", "prod-1"), enqueued_at=1.0)
    store.enqueue(_ticket("prod", "prod-2"), enqueued_at=2.0)
    store.enqueue(_ticket("batch", "batch-1"), enqueued_at=3.0)
    store.yield_turn(MODEL, "prod")
    store.remove(_ticket("prod", "prod-1"))
    snapshot: Final = store.snapshot(MODEL, ("prod", "batch"))
    assert snapshot["prod"] == ClassQueueState(
        head_request_id="prod-2", head_enqueued_at=2.0, depth=1, pass_value=0.0, yielded=False
    ), snapshot
    assert snapshot["batch"].yielded is False, snapshot


@pytest.mark.asyncio
async def test_same_class_admits_in_arrival_order():
    queue: Final = FairQueue()
    gate: Final = CapacityGate(slots=0)
    order: list[str] = []

    async def waiter(request_id: str):
        async def try_admit() -> bool:
            if gate.slots <= 0:
                return False
            gate.slots -= 1
            order.append(request_id)
            return True

        return await queue.wait_for_admission(
            ticket=_ticket("prod", request_id),
            weights=WEIGHTS,
            max_wait_seconds=30.0,
            max_depth=100,
            poll_interval_seconds=0.001,
            try_admit=try_admit,
            is_cancelled=_never_cancelled,
        )

    tasks: list[asyncio.Task] = []
    for request_id in ("a", "b", "c"):
        tasks.append(asyncio.create_task(waiter(request_id)))
        await asyncio.sleep(0.005)
    gate.slots = 3
    await asyncio.gather(*tasks)
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_request_past_its_deadline_is_never_admitted_even_when_capacity_frees():
    clock: Final = FakeClock()
    queue: Final = FairQueue(clock=clock)
    gate: Final = CapacityGate(slots=5)

    async def slow_disconnect_probe() -> bool:
        clock.now += 2.0
        return False

    outcome: Final = await queue.wait_for_admission(
        ticket=_ticket("prod", "r1"),
        weights=WEIGHTS,
        max_wait_seconds=1.0,
        max_depth=10,
        poll_interval_seconds=0.001,
        try_admit=gate.for_class("prod"),
        is_cancelled=slow_disconnect_probe,
    )
    assert outcome == QueueRejected(reason="queue_deadline_exceeded", waited_seconds=2.0)
    assert gate.admitted == []
    assert queue.depths(MODEL)["prod"] == 0

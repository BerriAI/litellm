import asyncio
import logging
import selectors
from collections.abc import Callable, Coroutine, Sequence
from typing import Final, TypeVar

import pytest

from litellm._logging import verbose_router_logger
from litellm.router_utils.dynamic_queue import (
    OPERATION_TIMED_OUT,
    AdmitReply,
    Admitted,
    CoordinationUnavailable,
    DynamicQueue,
    Gone,
    Grant,
    LocalQueueBackend,
    MatchingFailure,
    OtherFailure,
    Outcome,
    PassThrough,
    PollReply,
    Probe,
    QueueFull,
    QueueSettings,
    QueueTimeout,
    Rejection,
    Released,
    Succeeded,
    Waiting,
    WaitRequest,
)

CAPACITY: Final = "deployment-a"
FAR: Final = 1_000_000
SAFEGUARD_S: Final = 10.0
MAX_STEPS: Final = 1_000
_T = TypeVar("_T")


class ManualTimer:
    """Time moves only when the test advances it. A test can wait until a given task, or any task, is parked on one of
    its sleeps; sleeps that other tasks run as operation guards never count as that task being parked."""

    def __init__(self) -> None:
        self.at = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None], asyncio.Task[object] | None]] = []
        self._changed = asyncio.Condition()

    def now(self) -> float:
        return self.at

    async def sleep(self, seconds: float) -> None:
        wake: Final = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.at + seconds, wake, asyncio.current_task()))
        async with self._changed:
            self._changed.notify_all()
        await wake

    def pending(self) -> int:
        return sum(1 for _, wake, _ in self._sleepers if not wake.done())

    def _is_parked(self, task: "asyncio.Task[object] | None", count: int) -> bool:
        if task is None:
            return self.pending() >= count
        return task.done() or any(owner is task and not wake.done() for _, wake, owner in self._sleepers)

    async def _until_parked(self, task: "asyncio.Task[object] | None", count: int) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self._is_parked(task, count))

    async def parked(self, task: "asyncio.Task[object] | None" = None, count: int = 1) -> None:
        watcher: Final = asyncio.create_task(self._until_parked(task, count))
        try:
            done, _ = await asyncio.wait(
                {watcher} if task is None else {watcher, task}, timeout=SAFEGUARD_S, return_when=asyncio.FIRST_COMPLETED
            )
            assert done, "deadlock safeguard: nothing parked on the timer"
        finally:
            await _reap(watcher)

    def advance(self, seconds: float) -> None:
        self.at += seconds
        for due, wake, _ in self._sleepers:
            if due <= self.at and not wake.done():
                wake.set_result(None)
        self._sleepers = [sleeper for sleeper in self._sleepers if not sleeper[1].done()]


async def _reap(*tasks: "asyncio.Task[object]") -> None:
    """Cancels and awaits every task a test spawned, failing instead of hanging if one ignores the cancellation."""
    for task in tasks:
        _ = task.cancel()
    if not tasks:
        return
    _, stuck = await asyncio.wait(tasks, timeout=SAFEGUARD_S)
    assert not stuck, "deadlock safeguard: a task ignored its cancellation"
    for task in tasks:
        _ = None if task.cancelled() else task.exception()


async def _finished(task: "asyncio.Task[_T]") -> _T:
    done, _ = await asyncio.wait({task}, timeout=SAFEGUARD_S)
    assert done, "deadlock safeguard: the task never finished"
    return task.result()


async def _signalled(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), SAFEGUARD_S)


def _backend(timer: ManualTimer, **overrides: int) -> LocalQueueBackend:
    return LocalQueueBackend(QueueSettings(**overrides), clock=timer.now)


async def _congest(backend: LocalQueueBackend, retry_after_ms: int = 0, capacity: str = CAPACITY) -> None:
    await backend.report(capacity, "in-flight", PassThrough(0), MatchingFailure(retry_after_ms))


def _request(waiter_id: str, priority: int = 0, capacity: str = CAPACITY) -> WaitRequest:
    return WaitRequest(capacity=capacity, waiter_id=waiter_id, priority=priority)


async def _enqueue(backend: LocalQueueBackend, *waiters: tuple[str, int]) -> None:
    replies: Final = [await backend.admit(_request(waiter_id, priority), FAR) for waiter_id, priority in waiters]
    assert all(isinstance(reply, Waiting) for reply in replies)


async def _polls(backend: LocalQueueBackend, waiter_ids: Sequence[str]) -> dict[str, PollReply]:
    return {waiter_id: await backend.poll(CAPACITY, waiter_id) for waiter_id in waiter_ids}


async def _probed(backend: LocalQueueBackend, waiter_ids: Sequence[str]) -> tuple[str, Probe]:
    replies: Final = await _polls(backend, waiter_ids)
    probes: Final = [(waiter_id, reply) for waiter_id, reply in replies.items() if isinstance(reply, Probe)]
    assert len(probes) == 1, replies
    assert all(isinstance(reply, (Probe, Waiting)) for reply in replies.values()), replies
    return probes[0]


async def _run(
    timer: ManualTimer, call: Coroutine[object, object, Admitted | Rejection], step: float
) -> Admitted | Rejection:
    """Advances time ``step`` at a time whenever the call is parked on a recheck, until it settles."""
    task: Final = asyncio.create_task(call)
    try:
        for _ in range(MAX_STEPS):
            await timer.parked(task)
            if task.done():
                return task.result()
            timer.advance(step)
        raise AssertionError("deadlock safeguard: the call never settled")
    finally:
        await _reap(task)


async def test_a_healthy_capacity_admits_every_attempt_without_holding_a_queue_place() -> None:
    backend: Final = _backend(ManualTimer(), max_waiting=1)

    replies: Final = [await backend.admit(_request(f"w{index}"), FAR) for index in range(3)]

    assert all(isinstance(reply, PassThrough) for reply in replies)


@pytest.mark.parametrize(
    ("outcome", "congested"),
    [(MatchingFailure(), True), (OtherFailure(), False), (Succeeded(), False)],
)
async def test_only_a_matching_failure_congests_a_healthy_capacity(outcome: Outcome, congested: bool) -> None:
    backend: Final = _backend(ManualTimer())

    await backend.report(CAPACITY, "in-flight", PassThrough(0), outcome)

    reply: Final = await backend.admit(_request("next"), FAR)
    assert isinstance(reply, Waiting) is congested
    assert isinstance(reply, PassThrough) is not congested


async def test_congestion_stays_on_the_capacity_that_failed() -> None:
    backend: Final = _backend(ManualTimer())
    await _congest(backend, capacity="shared-pool")

    assert isinstance(await backend.admit(_request("a", capacity="shared-pool"), FAR), Waiting)
    assert isinstance(await backend.admit(_request("b", capacity="other-pool"), FAR), PassThrough)


async def test_probes_go_in_priority_order_and_first_in_first_out_within_a_priority() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=100)
    await _congest(backend)
    arrivals: Final = (
        ("five-first", 5),
        ("minus-three-first", -3),
        ("zero", 0),
        ("five-second", 5),
        ("huge", 10**30),
        ("minus-three-second", -3),
        ("minus-huge", -(10**30)),
    )
    await _enqueue(backend, *arrivals)
    timer.advance(1.0)

    order: list[str] = []
    remaining: list[str] = [waiter_id for waiter_id, _ in arrivals]
    while remaining:
        waiter_id, probe = await _probed(backend, remaining)
        await backend.report(CAPACITY, waiter_id, probe, OtherFailure())
        order.append(waiter_id)
        remaining.remove(waiter_id)

    assert order == [
        "minus-huge",
        "minus-three-first",
        "minus-three-second",
        "zero",
        "five-first",
        "five-second",
        "huge",
    ]


async def test_probe_backoff_doubles_on_each_failed_probe_up_to_the_cap_and_honors_a_longer_retry_after() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, initial_backoff_ms=1_000, max_backoff_ms=3_000)
    await _congest(backend)
    await _enqueue(backend, ("a", 0), ("b", 0), ("c", 0), ("d", 0))
    probe_times_and_retry_after: Final = ((1.0, 0), (3.0, 0), (6.0, 10_000), (16.0, 0))

    for (due, retry_after_ms), waiter_id in zip(probe_times_and_retry_after, "abcd"):
        timer.at = due - 0.001
        assert isinstance(await backend.poll(CAPACITY, waiter_id), Waiting)
        timer.at = due
        probe = await backend.poll(CAPACITY, waiter_id)
        assert isinstance(probe, Probe)
        await backend.report(CAPACITY, waiter_id, probe, MatchingFailure(retry_after_ms))


async def test_recovery_needs_consecutive_probe_successes() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=2)
    await _congest(backend)
    await _enqueue(backend, ("a", 0), ("b", 1), ("c", 2), ("d", 3), ("behind", 4))
    timer.advance(1.0)
    waiters: Final = ["a", "b", "c", "d", "behind"]

    for waiter_id, outcome in (("a", Succeeded()), ("b", OtherFailure()), ("c", Succeeded())):
        probed, probe = await _probed(backend, waiters)
        assert probed == waiter_id
        await backend.report(CAPACITY, waiter_id, probe, outcome)
        waiters.remove(waiter_id)
    probed, probe = await _probed(backend, waiters)
    assert probed == "d"
    await backend.report(CAPACITY, "d", probe, Succeeded())

    assert isinstance(await backend.poll(CAPACITY, "behind"), Released)


async def _recovered(timer: ManualTimer, backend: LocalQueueBackend, *snapshot: tuple[str, int]) -> None:
    """Congests, queues a prober ahead of ``snapshot`` and recovers on its single success."""
    await _congest(backend)
    await _enqueue(backend, ("prober", -(10**9)), *snapshot)
    timer.advance(1.0)
    probe: Final = await backend.poll(CAPACITY, "prober")
    assert isinstance(probe, Probe)
    await backend.report(CAPACITY, "prober", probe, Succeeded())


async def test_a_recovery_snapshot_claims_in_priority_order_while_every_later_arrival_waits() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _recovered(timer, backend, ("low", 9), ("high", 1), ("high-second", 1))

    early_arrival: Final = await backend.admit(_request("arrival", priority=-100), FAR)
    low_first: Final = await backend.poll(CAPACITY, "low")
    second_first: Final = await backend.poll(CAPACITY, "high-second")
    high: Final = await backend.poll(CAPACITY, "high")
    arrival_mid: Final = await backend.poll(CAPACITY, "arrival")
    second: Final = await backend.poll(CAPACITY, "high-second")
    low: Final = await backend.poll(CAPACITY, "low")
    arrival_after: Final = await backend.poll(CAPACITY, "arrival")
    fresh: Final = await backend.admit(_request("fresh"), FAR)

    assert early_arrival == Waiting(ahead=3)
    assert low_first == Waiting(ahead=2)
    assert second_first == Waiting(ahead=1)
    assert isinstance(high, Released)
    assert arrival_mid == Waiting(ahead=2)
    assert isinstance(second, Released) and isinstance(low, Released)
    assert isinstance(arrival_after, PassThrough) and isinstance(fresh, PassThrough)


@pytest.mark.parametrize("leaves", ["abandon", "deadline"])
async def test_a_snapshot_member_that_leaves_stops_holding_back_the_rest(leaves: str) -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _congest(backend)
    _ = await backend.admit(_request("prober", priority=-1), FAR)
    _ = await backend.admit(_request("head"), 1_500)
    _ = await backend.admit(_request("next", priority=1), FAR)
    timer.advance(1.0)
    probe: Final = await backend.poll(CAPACITY, "prober")
    assert isinstance(probe, Probe)
    await backend.report(CAPACITY, "prober", probe, Succeeded())
    assert await backend.poll(CAPACITY, "next") == Waiting(ahead=1)

    if leaves == "abandon":
        await backend.abandon(CAPACITY, "head")
    else:
        timer.at = 1.5

    assert isinstance(await backend.poll(CAPACITY, "next"), Released)
    assert isinstance(await backend.poll(CAPACITY, "head"), Gone)


async def test_a_trigger_before_the_granted_probe_owner_polls_puts_it_back_in_its_place() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    await _enqueue(backend, ("owner", 0), ("later", 0))
    timer.advance(1.0)
    assert await backend.poll(CAPACITY, "later") == Waiting()

    await _congest(backend)

    assert await backend.poll(CAPACITY, "owner") == Waiting(probe_in_ms=1_000)
    timer.advance(1.0)
    regranted: Final = await backend.poll(CAPACITY, "owner")
    assert isinstance(regranted, Probe) and regranted.generation == 2
    assert await backend.poll(CAPACITY, "later") == Waiting()


async def test_a_trigger_after_recovery_folds_unclaimed_waiters_back_with_priority_and_deadline() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _congest(backend)
    _ = await backend.admit(_request("prober", priority=-1), FAR)
    _ = await backend.admit(_request("short-budget", priority=3), 3_000)
    _ = await backend.admit(_request("urgent", priority=1), FAR)
    timer.advance(1.0)
    probe: Final = await backend.poll(CAPACITY, "prober")
    assert isinstance(probe, Probe)
    await backend.report(CAPACITY, "prober", probe, Succeeded())

    await _congest(backend)

    assert await backend.poll(CAPACITY, "short-budget") == Waiting(probe_in_ms=1_000)
    timer.at = 2.0
    assert await backend.poll(CAPACITY, "short-budget") == Waiting()
    assert isinstance(await backend.poll(CAPACITY, "urgent"), Probe)
    timer.at = 3.0
    assert isinstance(await backend.poll(CAPACITY, "short-budget"), Gone)


async def test_a_congested_wait_with_no_budget_times_out_without_taking_a_due_probe() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    timer.advance(1.0)

    result: Final = await DynamicQueue(backend, timer).admit(_request("spent"), 0.0)

    assert isinstance(result, QueueTimeout)
    assert isinstance(await backend.admit(_request("next"), FAR), Probe)


async def test_a_waiter_reaching_its_deadline_as_the_probe_comes_due_is_not_granted() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    _ = await backend.admit(_request("due-now"), 1_000)
    _ = await backend.admit(_request("next"), FAR)
    timer.at = 1.0

    assert isinstance(await backend.poll(CAPACITY, "due-now"), Gone)
    assert isinstance(await backend.poll(CAPACITY, "next"), Probe)


async def test_a_probe_success_from_a_superseded_generation_does_not_recover() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _congest(backend)
    await _enqueue(backend, ("a", 0))
    timer.advance(1.0)
    stale: Final = await backend.poll(CAPACITY, "a")
    assert isinstance(stale, Probe)

    await _congest(backend)
    await backend.report(CAPACITY, "a", stale, Succeeded())

    assert isinstance(await backend.admit(_request("b"), FAR), Waiting)
    timer.advance(1.0)
    fresh: Final = await backend.poll(CAPACITY, "b")
    assert isinstance(fresh, Probe) and fresh.generation > stale.generation
    await backend.report(CAPACITY, "b", fresh, Succeeded())
    assert isinstance(await backend.admit(_request("c"), FAR), PassThrough)


async def test_an_unrenewed_probe_lease_expires_so_the_next_waiter_probes_and_the_late_result_is_ignored() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1, lease_ms=3_000)
    await _congest(backend)
    await _enqueue(backend, ("lost-worker", 0), ("b", 0))
    timer.advance(1.0)
    abandoned: Final = await backend.poll(CAPACITY, "lost-worker")
    assert isinstance(abandoned, Probe)

    timer.at = 3.999
    assert isinstance(await backend.poll(CAPACITY, "b"), Waiting)
    timer.at = 4.0
    takeover: Final = await backend.poll(CAPACITY, "b")
    await backend.report(CAPACITY, "lost-worker", abandoned, Succeeded())

    assert isinstance(takeover, Probe)
    assert isinstance(await backend.admit(_request("c"), FAR), Waiting)


async def test_a_probe_claimed_late_holds_a_full_lease_from_its_claim() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, lease_ms=3_000)
    await _congest(backend)
    await _enqueue(backend, ("owner", 0), ("later", 0))
    timer.at = 1.0
    assert await backend.poll(CAPACITY, "later") == Waiting()
    timer.at = 3.9
    claimed: Final = await backend.poll(CAPACITY, "owner")
    assert isinstance(claimed, Probe)

    timer.at = 6.5
    held: Final = await backend.poll(CAPACITY, "later")
    renewed: Final = await backend.renew(CAPACITY, "owner", claimed.token)

    assert held == Waiting()
    assert renewed is True


async def test_a_capacity_that_went_idle_and_congests_again_ignores_a_report_for_an_earlier_grant() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _congest(backend)
    timer.at = 1.0
    earlier: Final = await backend.admit(_request("p"), FAR)
    assert isinstance(earlier, Probe)
    await backend.report(CAPACITY, "p", earlier, Succeeded())
    recovered: Final = await backend.admit(_request("q"), FAR)
    assert isinstance(recovered, PassThrough) and recovered.generation > earlier.generation

    await _congest(backend)
    timer.at = 2.0
    current: Final = await backend.admit(_request("p"), FAR)
    await backend.report(CAPACITY, "p", earlier, Succeeded())

    assert isinstance(current, Probe) and current != earlier
    assert isinstance(await backend.admit(_request("c"), FAR), Waiting)


async def test_an_unclaimed_probe_holds_a_queue_place_until_its_owner_claims_it() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, max_waiting=2)
    await _congest(backend)
    await _enqueue(backend, ("owner", 0))
    timer.at = 1.0
    assert await backend.admit(_request("second"), FAR) == Waiting()

    refused: Final = await backend.admit(_request("third"), FAR)
    claimed: Final = await backend.poll(CAPACITY, "owner")
    after_claim: Final = await backend.admit(_request("third"), FAR)

    assert isinstance(refused, QueueFull)
    assert isinstance(claimed, Probe)
    assert after_claim == Waiting()


async def test_an_abandoned_congested_capacity_expires_after_the_state_ttl_and_ignores_its_old_probe() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1, lease_ms=10_000, state_ttl_ms=10_000)
    await _congest(backend)
    timer.at = 1.0
    abandoned: Final = await backend.admit(_request("p"), 1_000)
    assert isinstance(abandoned, Probe)

    timer.at = 21.0
    expired: Final = await backend.admit(_request("q"), FAR)
    await _congest(backend)
    timer.at = 22.0
    current: Final = await backend.admit(_request("p"), FAR)
    await backend.report(CAPACITY, "p", abandoned, Succeeded())

    assert isinstance(expired, PassThrough)
    assert isinstance(current, Probe) and current != abandoned
    assert isinstance(await backend.admit(_request("c"), FAR), Waiting)


async def test_a_congested_capacity_is_kept_through_a_retry_after_and_the_state_ttl_beyond_it() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, state_ttl_ms=10_000)
    await _congest(backend, retry_after_ms=60_000, capacity="kept")
    await _congest(backend, retry_after_ms=60_000, capacity="expiring")

    timer.at = 69.999
    kept: Final = await backend.admit(_request("a", capacity="kept"), FAR)
    timer.at = 70.0
    expired: Final = await backend.admit(_request("b", capacity="expiring"), FAR)

    assert isinstance(kept, Probe)
    assert isinstance(expired, PassThrough)


async def test_a_full_queue_refuses_new_waiters_until_one_leaves() -> None:
    backend: Final = _backend(ManualTimer(), max_waiting=2)
    await _congest(backend, retry_after_ms=FAR)
    await _enqueue(backend, ("a", 0), ("b", 0))

    assert isinstance(await backend.admit(_request("c"), FAR), QueueFull)
    await backend.abandon(CAPACITY, "a")
    assert isinstance(await backend.admit(_request("c"), FAR), Waiting)


async def test_a_waiter_times_out_on_its_budget_and_gives_its_place_back() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, max_waiting=1, recheck_ms=250)
    await _congest(backend, retry_after_ms=FAR)
    queue: Final = DynamicQueue(backend, timer)

    result: Final = await _run(timer, queue.admit(_request("a"), 1.0), 0.25)

    assert isinstance(result, QueueTimeout)
    assert timer.at == 1.0
    assert isinstance(await backend.admit(_request("b"), FAR), Waiting)


async def test_an_admitted_waiter_keeps_the_rest_of_its_budget_for_a_later_wait() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, recheck_ms=250, initial_backoff_ms=1_000)
    await _congest(backend)
    queue: Final = DynamicQueue(backend, timer)

    result: Final = await _run(timer, queue.admit(_request("a"), 2.5), 0.25)

    assert isinstance(result, Admitted) and isinstance(result.grant, Probe)
    assert result.remaining_s == 1.5


async def test_a_healthy_capacity_admits_even_with_no_budget_left() -> None:
    timer: Final = ManualTimer()
    queue: Final = DynamicQueue(_backend(timer), timer)

    result: Final = await queue.admit(_request("a"), 0.0)

    assert isinstance(result, Admitted) and isinstance(result.grant, PassThrough)


async def test_cancelling_a_waiter_removes_it_from_the_queue() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, max_waiting=1)
    await _congest(backend, retry_after_ms=FAR)
    waiter: Final = asyncio.create_task(DynamicQueue(backend, timer).admit(_request("a"), 30.0))
    try:
        await timer.parked(waiter)
        _ = waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert isinstance(await backend.admit(_request("b"), FAR), Waiting)


class StallsAfterEnqueue:
    """Enqueues for real, then never answers, like a Redis reply lost after the script ran."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        self.inner = inner
        self.enqueued = asyncio.Event()

    @property
    def settings(self) -> QueueSettings:
        return self.inner.settings

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        _ = await self.inner.admit(request, budget_ms, ready_in_ms)
        self.enqueued.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        return await self.inner.poll(capacity, waiter_id, ready_in_ms)

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        return await self.inner.renew(capacity, waiter_id, token)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return await self.inner.report(capacity, waiter_id, grant, outcome)

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        return await self.inner.abandon(capacity, waiter_id)


async def test_cancelling_while_the_enqueue_reply_is_in_flight_still_removes_the_waiter() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, max_waiting=1)
    await _congest(inner, retry_after_ms=FAR)
    stalling: Final = StallsAfterEnqueue(inner)
    waiter: Final = asyncio.create_task(DynamicQueue(stalling, timer).admit(_request("a"), 30.0))
    try:
        await _signalled(stalling.enqueued)
        _ = waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert isinstance(await inner.admit(_request("b"), FAR), Waiting)


async def _probe_attempt(timer: ManualTimer, backend: LocalQueueBackend) -> tuple[DynamicQueue, Admitted]:
    await _congest(backend)
    queue: Final = DynamicQueue(backend, timer)
    admitted: Final = await _run(timer, queue.admit(_request("prober"), 100.0), 0.25)
    assert isinstance(admitted, Admitted) and isinstance(admitted.grant, Probe)
    return queue, admitted


async def test_a_running_probe_renews_its_lease_and_leaving_unreported_hands_the_probe_on() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, lease_ms=3_000, recheck_ms=250)
    queue, admitted = await _probe_attempt(timer, backend)
    await _enqueue(backend, ("next", 0))

    async with queue.attempt(_request("prober"), admitted):
        for _ in range(5):
            await timer.parked()
            timer.advance(1.0)
        await timer.parked()
        held: Final = await backend.poll(CAPACITY, "next")

    handed_on: Final = await backend.poll(CAPACITY, "next")
    assert isinstance(held, Waiting)
    assert isinstance(handed_on, Probe)


async def test_a_reported_probe_success_recovers_the_capacity_for_the_waiters_behind_it() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    queue, admitted = await _probe_attempt(timer, backend)
    await _enqueue(backend, ("behind", 0))

    async with queue.attempt(_request("prober"), admitted) as attempt:
        assert await attempt.report(Succeeded()) is None

    assert isinstance(await backend.poll(CAPACITY, "behind"), Released)


class HoldsReleases(StallsAfterEnqueue):
    """Lets admission through, then holds every report and abandon until the test releases them."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        super().__init__(inner)
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        self.held.set()
        await self.release.wait()
        return await self.inner.report(capacity, waiter_id, grant, outcome)

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        self.held.set()
        await self.release.wait()
        return await self.inner.abandon(capacity, waiter_id)


async def _held_probe(timer: ManualTimer) -> tuple[LocalQueueBackend, HoldsReleases, Admitted]:
    inner: Final = _backend(timer, required_successes=1)
    await _congest(inner)
    holding: Final = HoldsReleases(inner)
    admitted: Final = await _run(timer, DynamicQueue(holding, timer).admit(_request("p"), 9.0), 1.0)
    assert isinstance(admitted, Admitted) and isinstance(admitted.grant, Probe)
    await _enqueue(inner, ("behind", 0))
    return inner, holding, admitted


@pytest.mark.parametrize("reports", [True, False])
async def test_cancelling_a_probe_while_its_release_is_in_flight_finishes_the_release_then_cancels(
    reports: bool,
) -> None:
    timer: Final = ManualTimer()
    inner, holding, admitted = await _held_probe(timer)
    attempt: Final = DynamicQueue(holding, timer).attempt(_request("p"), admitted)

    async def probe() -> None:
        async with attempt:
            if not reports:
                await asyncio.Event().wait()
            await timer.parked()
            _ = await attempt.report(Succeeded())

    running: Final = asyncio.create_task(probe())
    try:
        if not reports:
            await timer.parked()
            _ = running.cancel()
        await _signalled(holding.held)
        _ = running.cancel()
        _ = running.cancel()
        holding.release.set()
        with pytest.raises(asyncio.CancelledError):
            await _finished(running)
    finally:
        await _reap(running)

    after: Final = await inner.poll(CAPACITY, "behind")
    assert isinstance(after, Released if reports else Probe)
    assert timer.pending() == 0


class AnswersOnRelease(StallsAfterEnqueue):
    """Answers admission only when the test says so, like a slow Redis round trip."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        super().__init__(inner)
        self.release = asyncio.Event()

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        self.enqueued.set()
        await self.release.wait()
        return await self.inner.admit(request, budget_ms, ready_in_ms)


async def test_a_probe_granted_after_the_queue_budget_ran_out_is_given_back_as_a_timeout() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer)
    await _congest(inner)
    timer.at = 1.0
    slow: Final = AnswersOnRelease(inner)
    late: Final = asyncio.create_task(DynamicQueue(slow, timer).admit(_request("late"), 0.5))
    try:
        await _signalled(slow.enqueued)
        timer.advance(0.6)
        slow.release.set()
        result: Final = await _finished(late)
    finally:
        await _reap(late)

    assert isinstance(result, QueueTimeout)
    assert isinstance(await inner.admit(_request("next"), FAR), Probe)


async def test_an_unanswered_admission_is_given_up_after_the_operation_timeout() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, max_waiting=1, operation_timeout_ms=1_000)
    await _congest(inner, retry_after_ms=FAR)
    stalling: Final = StallsAfterEnqueue(inner)
    waiter: Final = asyncio.create_task(DynamicQueue(stalling, timer).admit(_request("a"), 30.0))
    try:
        await _signalled(stalling.enqueued)
        await timer.parked()
        timer.advance(1.0)
        result: Final = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert result == CoordinationUnavailable(reason=OPERATION_TIMED_OUT)
    assert isinstance(await inner.admit(_request("b"), FAR), Waiting)


class NeverReleases(StallsAfterEnqueue):
    """Admits normally but never answers a release, like Redis going silent mid-cleanup."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        super().__init__(inner)
        self.releasing = asyncio.Event()

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        self.releasing.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_cancelling_a_waiter_gives_up_an_unanswered_release_after_the_operation_timeout() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer)
    await _congest(inner, retry_after_ms=FAR)
    silent: Final = NeverReleases(inner)
    waiter: Final = asyncio.create_task(DynamicQueue(silent, timer).admit(_request("a"), 30.0))
    try:
        await timer.parked(waiter)
        _ = waiter.cancel()
        await _signalled(silent.releasing)
        await timer.parked()
        timer.advance(1.0)
        with pytest.raises(asyncio.CancelledError):
            await _finished(waiter)
    finally:
        await _reap(waiter)

    assert timer.pending() == 0


async def test_a_report_redis_never_answers_comes_back_as_unavailable_after_the_operation_timeout() -> None:
    timer: Final = ManualTimer()
    _, holding, admitted = await _held_probe(timer)
    attempt: Final = DynamicQueue(holding, timer).attempt(_request("p"), admitted)

    async def probe() -> None | CoordinationUnavailable:
        async with attempt:
            return await attempt.report(Succeeded())

    reporting: Final = asyncio.create_task(probe())
    try:
        await _signalled(holding.held)
        await timer.parked()
        timer.advance(1.0)
        result: Final = await _finished(reporting)
    finally:
        await _reap(reporting)

    assert result == CoordinationUnavailable(reason=OPERATION_TIMED_OUT)
    assert timer.pending() == 0


class ReportsQuiescence(selectors.DefaultSelector):
    """The selector of a test's own event loop. The loop asks it to wait only once nothing is left to run, so a hook set
    here fires exactly when every task is parked, and never while any of them keeps the loop busy."""

    def __init__(self) -> None:
        super().__init__()
        self.on_quiet: Callable[[], None] | None = None

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        on_quiet: Final = self.on_quiet
        if timeout == 0 or on_quiet is None:
            return super().select(timeout)
        self.on_quiet = None
        on_quiet()
        return super().select(0)


async def _quiet(selector: ReportsQuiescence) -> None:
    reached: Final = asyncio.get_running_loop().create_future()
    selector.on_quiet = lambda: None if reached.done() else reached.set_result(None)
    await asyncio.wait_for(reached, SAFEGUARD_S)


async def _a_held_report_parks_then_lands(selector: ReportsQuiescence) -> None:
    timer: Final = ManualTimer()
    inner, holding, admitted = await _held_probe(timer)
    attempt: Final = DynamicQueue(holding, timer).attempt(_request("p"), admitted)

    async def probe() -> None | CoordinationUnavailable:
        async with attempt:
            return await attempt.report(Succeeded())

    reporting: Final = asyncio.create_task(probe())
    try:
        await _signalled(holding.held)
        await _quiet(selector)
        parked: Final = not reporting.done()
        holding.release.set()
        result: Final = await _finished(reporting)
    finally:
        await _reap(reporting)

    assert parked
    assert result is None
    assert isinstance(await inner.poll(CAPACITY, "behind"), Released)
    assert timer.pending() == 0


def test_a_probe_report_held_by_its_backend_lets_the_loop_go_idle_and_lands_once_released() -> None:
    selector: Final = ReportsQuiescence()
    loop: Final = asyncio.SelectorEventLoop(selector)
    try:
        loop.run_until_complete(_a_held_report_parks_then_lands(selector))
    finally:
        loop.close()


class NeverAnswersPolls(StallsAfterEnqueue):
    """Admits normally, then every poll goes unanswered, like Redis going silent while a request waits."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        super().__init__(inner)
        self.polled = asyncio.Event()

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        self.polled.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("budget_s", "silence_s", "expected"),
    [
        (30.0, 1.0, CoordinationUnavailable(reason=OPERATION_TIMED_OUT)),
        (1.125, 0.125, QueueTimeout()),
    ],
    ids=["coordination_gave_out_first", "the_budget_ran_out_first"],
)
async def test_an_unanswered_poll_ends_the_wait_as_whichever_limit_ran_out_first(
    budget_s: float, silence_s: float, expected: object
) -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, max_waiting=1, operation_timeout_ms=1_000, recheck_ms=250)
    await _congest(inner, retry_after_ms=FAR)
    silent: Final = NeverAnswersPolls(inner)
    waiter: Final = asyncio.create_task(DynamicQueue(silent, timer).admit(_request("a"), budget_s))
    try:
        await timer.parked(waiter)
        timer.advance(1.0)
        await _signalled(silent.polled)
        await timer.parked()
        timer.advance(silence_s)
        result: Final = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert result == expected
    assert isinstance(await inner.admit(_request("b"), FAR), Waiting)


async def test_a_waiter_the_queue_dropped_meanwhile_times_out_instead_of_waiting_on() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, recheck_ms=250)
    await _congest(inner, retry_after_ms=FAR)
    waiter: Final = asyncio.create_task(DynamicQueue(inner, timer).admit(_request("dropped"), 30.0))
    try:
        await timer.parked(waiter)
        await inner.abandon(CAPACITY, "dropped")
        timer.advance(1.0)
        result: Final = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert isinstance(result, QueueTimeout)
    assert timer.at == 1.0


async def test_a_wait_whose_admission_reply_came_after_the_budget_times_out_and_leaves_the_queue() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, max_waiting=1)
    await _congest(inner, retry_after_ms=FAR)
    slow: Final = AnswersOnRelease(inner)
    late: Final = asyncio.create_task(DynamicQueue(slow, timer).admit(_request("late"), 0.5))
    try:
        await _signalled(slow.enqueued)
        timer.advance(0.6)
        slow.release.set()
        result: Final = await _finished(late)
    finally:
        await _reap(late)

    assert isinstance(result, QueueTimeout)
    assert isinstance(await inner.admit(_request("next"), FAR), Waiting)


async def test_reporting_a_probe_twice_counts_its_success_once() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=2)
    queue, admitted = await _probe_attempt(timer, backend)
    await _enqueue(backend, ("behind", 0))

    async with queue.attempt(_request("prober"), admitted) as attempt:
        first: Final = await attempt.report(Succeeded())
        second: Final = await attempt.report(Succeeded())

    assert (first, second) == (None, None)
    assert isinstance(await backend.poll(CAPACITY, "behind"), Probe)


class RecordsLostLeases(StallsAfterEnqueue):
    def __init__(self, inner: LocalQueueBackend) -> None:
        super().__init__(inner)
        self.lost = asyncio.Event()

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        renewed: Final = await self.inner.renew(capacity, waiter_id, token)
        if renewed is False:
            self.lost.set()
        return renewed


async def test_a_probe_that_lost_its_lease_stops_renewing_and_never_holds_the_next_probe() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, lease_ms=3_000, recheck_ms=250)
    await _congest(inner)
    recorder: Final = RecordsLostLeases(inner)
    queue: Final = DynamicQueue(recorder, timer)
    admitted: Final = await _run(timer, queue.admit(_request("prober"), 100.0), 0.25)
    assert isinstance(admitted, Admitted) and isinstance(admitted.grant, Probe)
    await _enqueue(inner, ("next", 0), ("third", 0))

    async with queue.attempt(_request("prober"), admitted):
        await timer.parked()
        await _congest(inner)
        timer.advance(1.0)
        await _signalled(recorder.lost)
        renewing_after_loss: Final = timer.pending()
        timer.at = 3.0
        next_probe: Final = await inner.poll(CAPACITY, "next")

    assert renewing_after_loss == 0
    assert isinstance(next_probe, Probe)
    assert isinstance(await inner.poll(CAPACITY, "third"), Waiting)


async def test_a_higher_priority_arrival_outranks_a_waiter_still_cooling_once_both_are_ready() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    timer.at = 1.0
    assert isinstance(await backend.admit(_request("p1", priority=1), FAR, ready_in_ms=4_000), Waiting)
    timer.at = 2.0
    assert isinstance(await backend.admit(_request("p0", priority=0), FAR, ready_in_ms=3_000), Waiting)

    timer.at = 4.5
    cooling: Final = {
        "p1": await backend.poll(CAPACITY, "p1", ready_in_ms=500),
        "p0": await backend.poll(CAPACITY, "p0", ready_in_ms=500),
    }
    timer.at = 5.0
    p1_turn: Final = await backend.poll(CAPACITY, "p1", ready_in_ms=0)
    p0_turn: Final = await backend.poll(CAPACITY, "p0", ready_in_ms=0)

    assert cooling == {"p1": Waiting(), "p0": Waiting()}
    assert p1_turn == Waiting()
    assert isinstance(p0_turn, Probe)


async def test_a_ready_waiter_is_granted_ahead_of_a_higher_priority_one_its_deployment_still_cools() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    timer.at = 1.0
    _ = await backend.admit(_request("cooling-p0", priority=0), FAR, ready_in_ms=10_000)

    ready_p5: Final = await backend.admit(_request("ready-p5", priority=5), FAR)

    assert isinstance(ready_p5, Probe)


async def test_a_refreshed_cooldown_puts_an_unclaimed_probe_back_in_its_place_without_a_new_generation() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    await _enqueue(backend, ("owner", 0), ("later", 1))
    timer.at = 1.0
    assert await backend.poll(CAPACITY, "later", ready_in_ms=5_000) == Waiting()

    refreshed: Final = await backend.poll(CAPACITY, "owner", ready_in_ms=2_000)
    timer.at = 3.0
    owner_turn: Final = await backend.poll(CAPACITY, "owner")

    assert refreshed == Waiting()
    assert isinstance(owner_turn, Probe) and owner_turn.generation == 1


async def test_a_matching_failure_between_the_readiness_read_and_the_claim_is_never_dispatched() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    await _enqueue(backend, ("owner", 0))
    timer.at = 1.0

    await _congest(backend)
    after_failure: Final = await backend.poll(CAPACITY, "owner", ready_in_ms=0)

    assert after_failure == Waiting(probe_in_ms=1_000)


async def test_a_recovery_snapshot_admits_its_first_ready_member_and_keeps_a_cooling_ones_place() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _recovered(timer, backend, ("first", 1), ("second", 2))

    first_cooling: Final = await backend.poll(CAPACITY, "first", ready_in_ms=2_000)
    second: Final = await backend.poll(CAPACITY, "second")
    timer.at = 3.0
    first: Final = await backend.poll(CAPACITY, "first")

    assert first_cooling == Waiting(ahead=0)
    assert isinstance(second, Released)
    assert isinstance(first, Released)


async def test_a_controller_wait_stays_queued_until_its_readiness_ends() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, recheck_ms=250)
    await _congest(backend)
    queue: Final = DynamicQueue(backend, timer)

    async def cooling_until_five() -> float:
        return max(5.0 - timer.now(), 0.0)

    admitted: Final = await _run(timer, queue.admit(_request("waiter"), 30.0, cooling_until_five), 0.25)

    assert isinstance(admitted, Admitted) and isinstance(admitted.grant, Probe)
    assert timer.at == 5.0


COOLING_AHEAD: Final = 70


async def test_a_ready_waiter_behind_more_cooling_waiters_than_one_sweep_still_gets_the_probe() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer)
    await _congest(backend)
    timer.at = 1.0
    cooling: Final = [
        await backend.admit(_request(f"cooling-{index}", priority=0), FAR, ready_in_ms=FAR)
        for index in range(COOLING_AHEAD)
    ]

    ready: Final = await backend.admit(_request("ready", priority=9), FAR)

    assert all(reply == Waiting() for reply in cooling)
    assert isinstance(ready, Probe)


async def test_a_ready_snapshot_member_behind_more_cooling_members_than_one_sweep_still_claims() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _congest(backend)
    _ = await backend.admit(_request("prober", priority=-1), FAR)
    for index in range(COOLING_AHEAD):
        _ = await backend.admit(_request(f"cooling-{index}", priority=0), FAR, ready_in_ms=FAR)
    _ = await backend.admit(_request("ready", priority=9), FAR)
    timer.at = 1.0
    probe: Final = await backend.poll(CAPACITY, "prober")
    assert isinstance(probe, Probe)
    await backend.report(CAPACITY, "prober", probe, Succeeded())

    claimed: Final = await backend.poll(CAPACITY, "ready")

    assert isinstance(claimed, Released)


def _readiness_lookup_taking(timer: ManualTimer, seconds: float) -> "Callable[[], Coroutine[object, object, float]]":
    async def lookup() -> float:
        timer.at += seconds
        return 0.0

    return lookup


async def _stalled_readiness() -> float:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


@pytest.mark.parametrize("stalls", [False, True], ids=["slow_lookup", "lookup_cut_off_at_the_budget"])
@pytest.mark.parametrize(("congested", "admitted"), [(False, True), (True, False)], ids=["healthy", "congested"])
async def test_a_readiness_lookup_that_used_up_the_budget_passes_a_healthy_capacity_and_never_queues_a_congested_one(
    stalls: bool, congested: bool, admitted: bool
) -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, max_waiting=1)
    if congested:
        await _congest(inner, retry_after_ms=FAR)
    readiness: Final = _stalled_readiness if stalls else _readiness_lookup_taking(timer, 0.125)
    admitting: Final = asyncio.create_task(DynamicQueue(inner, timer).admit(_request("late"), 0.125, readiness))
    try:
        if stalls:
            await timer.parked(count=1)
            timer.advance(0.125)
        result: Final = await _finished(admitting)
    finally:
        await _reap(admitting)
    next_arrival: Final = await inner.admit(_request("next"), FAR)

    if admitted:
        assert isinstance(result, Admitted) and isinstance(result.grant, PassThrough)
    else:
        assert result == QueueTimeout()
    assert timer.at == 0.125
    assert isinstance(next_arrival, PassThrough if admitted else Waiting)


async def test_a_poll_after_a_slow_readiness_lookup_only_gets_the_time_that_is_left() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, recheck_ms=250, operation_timeout_ms=1_000)
    await _congest(inner, retry_after_ms=FAR)
    silent: Final = NeverAnswersPolls(inner)
    waiter: Final = asyncio.create_task(
        DynamicQueue(silent, timer).admit(_request("a"), 2.0, _readiness_lookup_taking(timer, 0.125))
    )
    try:
        await timer.parked(waiter)
        assert timer.at == 0.125
        timer.advance(1.0)
        await _signalled(silent.polled)
        await timer.parked()
        timer.advance(0.75)
        result: Final = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert isinstance(result, QueueTimeout)
    assert timer.at == 2.0


class RecordsPolls(StallsAfterEnqueue):
    def __init__(self, inner: LocalQueueBackend, timer: ManualTimer) -> None:
        super().__init__(inner)
        self._timer = timer
        self.polled_at: list[float] = []  # mutable-ok: when each poll reached the backend

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        self.polled_at.append(self._timer.now())
        return await self.inner.poll(capacity, waiter_id, ready_in_ms)


@pytest.mark.parametrize(
    ("budget_s", "settled_at", "polls"),
    [(30.0, 20.0, 20), (2.5, 2.5, 2)],
    ids=["probe_comes_due", "budget_runs_out_first"],
)
async def test_a_wait_behind_a_long_probe_backoff_polls_once_a_second_and_never_past_the_probe_or_its_budget(
    budget_s: float, settled_at: float, polls: int
) -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, recheck_ms=250)
    await _congest(inner, retry_after_ms=20_000)
    recorder: Final = RecordsPolls(inner, timer)

    result: Final = await _run(timer, DynamicQueue(recorder, timer).admit(_request("a"), budget_s), 0.25)

    gaps: Final = [later - earlier for earlier, later in zip((0.0, *recorder.polled_at), recorder.polled_at)]
    assert isinstance(result, Admitted if budget_s > settled_at else QueueTimeout)
    assert timer.at == settled_at
    assert len(recorder.polled_at) == polls
    assert max(gaps) <= 1.0


async def test_a_wait_behind_a_probe_that_is_out_keeps_rechecking_so_its_recovery_releases_it_at_once() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, required_successes=1, recheck_ms=250)
    await _congest(inner, retry_after_ms=20_000)
    timer.at = 20.0
    probe: Final = await inner.admit(_request("prober"), FAR)
    assert isinstance(probe, Probe)
    waiter: Final = asyncio.create_task(DynamicQueue(inner, timer).admit(_request("behind"), 30.0))
    try:
        await timer.parked(waiter)
        await inner.report(CAPACITY, "prober", probe, Succeeded())
        timer.advance(0.25)
        released: Final = await _finished(waiter)
    finally:
        await _reap(waiter)

    assert isinstance(released, Admitted) and isinstance(released.grant, Released)
    assert timer.at == 20.25


async def test_a_wait_whose_poll_goes_unanswered_logs_why_and_after_how_long_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, operation_timeout_ms=1_000, recheck_ms=250)
    await _congest(inner, retry_after_ms=FAR)
    silent: Final = NeverAnswersPolls(inner)
    with caplog.at_level(logging.DEBUG, logger=verbose_router_logger.name):
        waiter: Final = asyncio.create_task(DynamicQueue(silent, timer).admit(_request("a"), 30.0))
        try:
            await timer.parked(waiter)
            timer.advance(1.0)
            await _signalled(silent.polled)
            await timer.parked()
            timer.advance(1.0)
            result: Final = await _finished(waiter)
        finally:
            await _reap(waiter)

    logged: Final = [record.getMessage() for record in caplog.records if "dynamic queue" in record.getMessage()]
    assert result == CoordinationUnavailable(reason=OPERATION_TIMED_OUT)
    assert logged == [
        "dynamic queue: coordination failed (operation_timed_out) after 2.000s waiting, going on without the queue"
    ]

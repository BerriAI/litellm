import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import AsyncExitStack
from types import MappingProxyType
from typing import Final

import pytest
import pytest_asyncio
from redis import Redis

from litellm.caching.redis_cache import RedisCache
from litellm.router_utils.dynamic_queue import (
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
    Rejection,
    Released,
    Succeeded,
    Waiting,
    WaitRequest,
)
from litellm.router_utils.dynamic_queue_redis import RedisQueueBackend, _keys
from tests.unit.router_utils.test_dynamic_queue import ManualTimer, _reap

FAR: Final = 1_000_000
CONFIGURED_SIZE: Final = 64


class SharedClock:
    """Stands in for Redis TIME so both workers see time move only when the test moves it."""

    def __init__(self) -> None:
        self.ms = 0

    def __call__(self) -> int:
        return self.ms


def _delete(*keys: str) -> None:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as client:
        _ = client.delete(*keys)


class WorkerFactory:
    def __init__(self, cleanup: AsyncExitStack) -> None:
        self._cleanup = cleanup
        self._capacity_keys: tuple[str, ...] = ()

    def owns(self, capacity: str) -> None:
        """Namespaced workers built after this delete ``capacity``'s keys under their namespace on teardown."""
        self._capacity_keys = _keys(capacity)

    def __call__(self, clock: SharedClock | None, namespace: str | None = None, **overrides: int) -> RedisQueueBackend:
        cache: Final = RedisCache(
            host=os.environ["REDIS_HOST"],
            port=int(os.environ["REDIS_PORT"]),
            client_name=f"dynamic-queue-test-{uuid.uuid4()}",
            namespace=namespace,
        )
        if namespace is not None:
            self._cleanup.callback(_delete, *(cache.check_and_fix_namespace(key=key) for key in self._capacity_keys))
        self._cleanup.push_async_callback(cache.disconnect)
        client: Final = cache.init_async_client()
        self._cleanup.push_async_callback(client.aclose)
        return RedisQueueBackend(cache, QueueSettings(**overrides), clock)

    def pair(self, clock: SharedClock, **overrides: int) -> tuple[RedisQueueBackend, RedisQueueBackend]:
        return self(clock, **overrides), self(clock, **overrides)


@pytest_asyncio.fixture(loop_scope="function")
async def workers() -> AsyncIterator[WorkerFactory]:
    async with AsyncExitStack() as cleanup:
        yield WorkerFactory(cleanup)


@pytest.fixture
def capacity() -> Iterator[str]:
    capacity_id: Final = f"integration-{uuid.uuid4()}"
    try:
        yield capacity_id
    finally:
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as client:
            _ = client.delete(*_keys(capacity_id))


async def _congest(worker: RedisQueueBackend, capacity: str, retry_after_ms: int = 0) -> None:
    assert await worker.report(capacity, "in-flight", PassThrough(0), MatchingFailure(retry_after_ms)) is None


async def _polls(workers: Mapping[str, RedisQueueBackend], capacity: str) -> dict[str, PollReply]:
    return {waiter_id: await worker.poll(capacity, waiter_id) for waiter_id, worker in workers.items()}


async def _probed(workers: Mapping[str, RedisQueueBackend], capacity: str) -> tuple[str, Probe]:
    replies: Final = await _polls(workers, capacity)
    probes: Final = [(waiter_id, reply) for waiter_id, reply in replies.items() if isinstance(reply, Probe)]
    assert len(probes) == 1, replies
    assert all(isinstance(reply, (Probe, Waiting)) for reply in replies.values()), replies
    return probes[0]


async def _enqueue(capacity: str, arrivals: Sequence[tuple[str, int, RedisQueueBackend]], budget_ms: int = FAR) -> None:
    for waiter_id, priority, worker in arrivals:
        reply: Final = await worker.admit(WaitRequest(capacity, waiter_id, priority), budget_ms)
        assert isinstance(reply, Waiting), (waiter_id, reply)


@pytest.mark.asyncio
async def test_waiters_on_two_workers_probe_in_one_priority_order_with_fifo_ties(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=100)
    await _congest(first, capacity)
    arrivals: Final = (
        ("five-first", 5, first),
        ("minus-three-first", -3, second),
        ("zero", 0, first),
        ("five-second", 5, second),
        ("huge", 2**80, first),
        ("minus-three-second", -3, first),
        ("minus-huge", -(2**80), second),
    )
    await _enqueue(capacity, arrivals)
    clock.ms = 1_000

    expected_order: Final = (
        "minus-huge",
        "minus-three-first",
        "minus-three-second",
        "zero",
        "five-first",
        "five-second",
        "huge",
    )
    participants: Final = MappingProxyType({waiter_id: worker for waiter_id, _, worker in arrivals})

    for index, expected in enumerate(expected_order):
        waiter_id, probe = await _probed(
            MappingProxyType({queued: participants[queued] for queued in expected_order[index:]}), capacity
        )
        assert waiter_id == expected
        assert await participants[waiter_id].report(capacity, waiter_id, probe, OtherFailure()) is None


@pytest.mark.asyncio
async def test_probe_backoff_doubles_to_its_cap_and_honors_a_longer_retry_after(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, initial_backoff_ms=1_000, max_backoff_ms=3_000, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [(waiter_id, 0, second) for waiter_id in "abcd"])

    for waiter_id, due, retry_after_ms in (("a", 1_000, 0), ("b", 3_000, 0), ("c", 6_000, 10_000), ("d", 16_000, 0)):
        clock.ms = due - 1
        assert isinstance(await second.poll(capacity, waiter_id), Waiting)
        clock.ms = due
        probe = await second.poll(capacity, waiter_id)
        assert isinstance(probe, Probe), (waiter_id, probe)
        assert await first.report(capacity, waiter_id, probe, MatchingFailure(retry_after_ms)) is None


@pytest.mark.asyncio
async def test_consecutive_successes_recover_into_a_snapshot_claimed_in_order_ahead_of_arrivals(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=2)
    await _congest(first, capacity)
    arrivals: Final = (
        ("a", 0, first),
        ("b", 1, second),
        ("c", 2, first),
        ("d", 3, second),
        ("e", 4, first),
        ("f", 4, second),
    )
    await _enqueue(capacity, arrivals)
    clock.ms = 1_000
    outcomes: Final = (("a", Succeeded()), ("b", OtherFailure()), ("c", Succeeded()), ("d", Succeeded()))

    for index, (expected, outcome) in enumerate(outcomes):
        waiter_id, probe = await _probed(
            MappingProxyType({queued: worker for queued, _, worker in arrivals[index:]}), capacity
        )
        assert waiter_id == expected
        assert await arrivals[index][2].report(capacity, waiter_id, probe, outcome) is None
    arrival: Final = await first.admit(WaitRequest(capacity, "after-recovery", -100), FAR)
    f_early: Final = await second.poll(capacity, "f")
    e: Final = await first.poll(capacity, "e")
    arrival_mid: Final = await first.poll(capacity, "after-recovery")
    f: Final = await second.poll(capacity, "f")
    arrival_after: Final = await first.poll(capacity, "after-recovery")

    assert arrival == Waiting(ahead=2)
    assert f_early == Waiting(ahead=1)
    assert isinstance(arrival_after, PassThrough)
    assert e == f == Released(arrival_after.generation)
    assert arrival_mid == Waiting(ahead=1)

    assert isinstance(e, Released)
    assert await first.report(capacity, "e", e, MatchingFailure(retry_after_ms=5_000)) is None
    held: Final = await second.admit(WaitRequest(capacity, "late", 0), FAR)
    clock.ms = 5_999
    still_held: Final = await second.poll(capacity, "late")
    clock.ms = 6_000
    reprobed: Final = await second.poll(capacity, "late")

    assert held == Waiting(probe_in_ms=5_000)
    assert still_held == Waiting(probe_in_ms=1)
    assert isinstance(reprobed, Probe) and reprobed.generation > e.generation


@pytest.mark.asyncio
async def test_a_success_from_a_superseded_generation_does_not_recover(capacity: str, workers: WorkerFactory) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=1)
    await _congest(first, capacity)
    await _enqueue(capacity, [("a", 0, second)])
    clock.ms = 1_000
    stale: Final = await second.poll(capacity, "a")
    assert isinstance(stale, Probe)

    await _congest(first, capacity)
    assert await second.report(capacity, "a", stale, Succeeded()) is None

    assert isinstance(await first.admit(WaitRequest(capacity, "b", 0), FAR), Waiting)
    clock.ms = 2_000
    fresh: Final = await first.poll(capacity, "b")
    assert isinstance(fresh, Probe) and fresh.generation > stale.generation
    assert await first.report(capacity, "b", fresh, Succeeded()) is None
    assert isinstance(await second.admit(WaitRequest(capacity, "c", 0), FAR), PassThrough)


@pytest.mark.asyncio
async def test_a_renewed_lease_stays_exclusive_and_an_unrenewed_one_expires_to_the_next_waiter(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=1, lease_ms=3_000, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("prober", 0, first), ("next", 0, second)])
    clock.ms = 1_000
    probe: Final = await first.poll(capacity, "prober")
    assert isinstance(probe, Probe)

    clock.ms = 3_500
    assert await first.renew(capacity, "prober", probe.token) is True
    clock.ms = 6_000
    held: Final = await second.poll(capacity, "next")
    clock.ms = 6_500
    takeover: Final = await second.poll(capacity, "next")

    assert isinstance(held, Waiting)
    assert isinstance(takeover, Probe)
    assert await first.renew(capacity, "prober", probe.token) is False
    assert await first.report(capacity, "prober", probe, Succeeded()) is None
    assert isinstance(await first.admit(WaitRequest(capacity, "later", 0), FAR), Waiting)


@pytest.mark.asyncio
async def test_a_probe_claimed_late_holds_a_full_lease_from_its_claim(capacity: str, workers: WorkerFactory) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, lease_ms=3_000, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("owner", 0, first), ("later", 0, second)])
    clock.ms = 1_000
    assert await second.poll(capacity, "later") == Waiting()
    clock.ms = 3_900
    claimed: Final = await first.poll(capacity, "owner")
    assert isinstance(claimed, Probe)

    clock.ms = 6_500
    held: Final = await second.poll(capacity, "later")
    renewed: Final = await first.renew(capacity, "owner", claimed.token)

    assert held == Waiting()
    assert renewed is True


@pytest.mark.asyncio
async def test_a_waiter_whose_worker_stops_polling_is_dropped_after_its_liveness_window(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    lost, alive = workers.pair(clock, liveness_ms=5_000)
    await _congest(lost, capacity, retry_after_ms=6_000)
    await _enqueue(capacity, [("from-lost-worker", 0, lost), ("from-live-worker", 0, alive)])

    clock.ms = 3_000
    assert isinstance(await alive.poll(capacity, "from-live-worker"), Waiting)
    clock.ms = 6_000

    assert isinstance(await alive.poll(capacity, "from-live-worker"), Probe)
    assert isinstance(await lost.poll(capacity, "from-lost-worker"), Gone)


@pytest.mark.asyncio
async def test_an_unclaimed_probe_holds_a_queue_place_until_its_owner_claims_it(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, max_waiting=2, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("owner", 0, first)])
    clock.ms = 1_000
    assert await second.admit(WaitRequest(capacity, "second", 0), FAR) == Waiting()

    refused: Final = await second.admit(WaitRequest(capacity, "third", 0), FAR)
    claimed: Final = await first.poll(capacity, "owner")
    after_claim: Final = await second.admit(WaitRequest(capacity, "third", 0), FAR)

    assert isinstance(refused, QueueFull)
    assert isinstance(claimed, Probe)
    assert after_claim == Waiting()


@pytest.mark.asyncio
async def test_a_full_queue_refuses_until_a_waiter_passes_its_deadline(capacity: str, workers: WorkerFactory) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, max_waiting=1, liveness_ms=FAR)
    await _congest(first, capacity, retry_after_ms=FAR)
    await _enqueue(capacity, [("a", 0, first)], budget_ms=2_000)

    assert isinstance(await second.admit(WaitRequest(capacity, "b", 0), FAR), QueueFull)
    clock.ms = 2_000
    assert isinstance(await second.admit(WaitRequest(capacity, "b", 0), FAR), Waiting)
    assert isinstance(await first.poll(capacity, "a"), Gone)


@pytest.mark.asyncio
async def test_an_abandoned_waiter_frees_its_place_on_every_worker(capacity: str, workers: WorkerFactory) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, max_waiting=1)
    await _congest(first, capacity, retry_after_ms=FAR)
    await _enqueue(capacity, [("a", 0, first)])

    assert await first.abandon(capacity, "a") is None

    assert isinstance(await second.admit(WaitRequest(capacity, "b", 0), FAR), Waiting)
    assert isinstance(await first.poll(capacity, "a"), Gone)


@pytest.mark.asyncio
async def test_redis_time_drives_the_queue_when_no_clock_is_injected(capacity: str, workers: WorkerFactory) -> None:
    first, second = workers(None), workers(None)

    assert isinstance(await first.admit(WaitRequest(capacity, "healthy", 0), FAR), PassThrough)
    await _congest(second, capacity, retry_after_ms=FAR)
    assert isinstance(await first.admit(WaitRequest(capacity, "queued", 0), FAR), Waiting)
    assert isinstance(await second.poll(capacity, "queued"), Waiting)
    assert await second.abandon(capacity, "queued") is None
    assert isinstance(await first.poll(capacity, "queued"), Gone)


@pytest.mark.asyncio
async def test_a_trigger_before_the_granted_probe_owner_polls_puts_it_back_in_its_place(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("owner", 0, first), ("later", 0, second)])
    clock.ms = 1_000
    assert await second.poll(capacity, "later") == Waiting()

    await _congest(second, capacity)

    assert await first.poll(capacity, "owner") == Waiting(probe_in_ms=1_000)
    clock.ms = 2_000
    assert await second.poll(capacity, "later") == Waiting()
    regranted: Final = await first.poll(capacity, "owner")
    assert isinstance(regranted, Probe) and regranted.generation == 2


@pytest.mark.asyncio
async def test_a_trigger_after_recovery_folds_unclaimed_waiters_back_with_priority_and_deadline(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=1, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("prober", -1, first), ("urgent", 1, second)])
    await _enqueue(capacity, [("short-budget", 3, first)], budget_ms=3_000)
    clock.ms = 1_000
    probe: Final = await first.poll(capacity, "prober")
    assert isinstance(probe, Probe)
    assert await first.report(capacity, "prober", probe, Succeeded()) is None

    await _congest(second, capacity)

    assert await first.poll(capacity, "short-budget") == Waiting(probe_in_ms=1_000)
    clock.ms = 2_000
    assert await first.poll(capacity, "short-budget") == Waiting()
    assert isinstance(await second.poll(capacity, "urgent"), Probe)
    clock.ms = 3_000
    assert isinstance(await first.poll(capacity, "short-budget"), Gone)


@pytest.mark.asyncio
async def test_a_congested_admit_with_no_budget_is_refused_even_when_a_probe_is_due(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock)
    await _congest(first, capacity)
    clock.ms = 1_000

    assert isinstance(await first.admit(WaitRequest(capacity, "spent", 0), 0), Gone)
    assert isinstance(await second.admit(WaitRequest(capacity, "next", 0), FAR), Probe)


@pytest.mark.asyncio
async def test_expired_waiters_beyond_the_sweep_limit_are_never_granted_or_revived(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, sweep_limit=2, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [(f"stale-{index}", -10, first) for index in range(6)], budget_ms=500)
    await _enqueue(capacity, [("live", 0, second)])
    clock.ms = 1_000

    unswept: Final = await first.poll(capacity, "stale-5")
    granted: Final = await second.poll(capacity, "live")
    stale_after: Final = [await first.poll(capacity, f"stale-{index}") for index in range(6)]

    assert isinstance(unswept, Gone)
    assert isinstance(granted, Probe)
    assert all(isinstance(reply, Gone) for reply in stale_after)


def _state_ttl_ms(capacity: str) -> int:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as redis:
        ttl: Final = redis.pttl(_keys(capacity)[0])
    assert isinstance(ttl, int)
    return ttl


@pytest.mark.asyncio
async def test_congestion_outlives_a_retry_after_longer_than_the_idle_ttl_and_then_expires(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    worker: Final = workers(clock, required_successes=1, state_ttl_ms=60_000)
    await _congest(worker, capacity, retry_after_ms=900_000)

    congested_ttl: Final = _state_ttl_ms(capacity)
    clock.ms = 900_000
    probe: Final = await worker.admit(WaitRequest(capacity, "prober", 0), FAR)
    assert isinstance(probe, Probe)
    assert await worker.report(capacity, "prober", probe, Succeeded()) is None
    recovered_ttl: Final = _state_ttl_ms(capacity)

    assert 900_000 < congested_ttl <= 960_000
    assert 0 < recovered_ttl <= 60_000


@pytest.mark.asyncio
async def test_a_shared_capacity_is_shared_within_one_redis_namespace_and_never_across_two(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    shared: Final = f"shared:{capacity}"
    workers.owns(shared)
    congesting: Final = workers(clock, namespace=f"router-a-{capacity}")
    same_router: Final = workers(clock, namespace=f"router-a-{capacity}")
    other_router: Final = workers(clock, namespace=f"router-b-{capacity}")
    await _congest(congesting, shared)

    assert isinstance(await same_router.admit(WaitRequest(shared, "same-namespace", 0), FAR), Waiting)
    assert isinstance(await other_router.admit(WaitRequest(shared, "other-namespace", 0), FAR), PassThrough)


@pytest.mark.asyncio
async def test_a_higher_priority_arrival_outranks_a_waiter_still_cooling_on_another_worker(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, liveness_ms=FAR)
    await _congest(first, capacity)
    clock.ms = 1_000
    assert isinstance(await first.admit(WaitRequest(capacity, "p1", 1), FAR, ready_in_ms=4_000), Waiting)
    clock.ms = 2_000
    assert isinstance(await second.admit(WaitRequest(capacity, "p0", 0), FAR, ready_in_ms=3_000), Waiting)

    clock.ms = 4_500
    cooling: Final = (
        await first.poll(capacity, "p1", ready_in_ms=500),
        await second.poll(capacity, "p0", ready_in_ms=500),
    )
    clock.ms = 5_000
    p1_turn: Final = await first.poll(capacity, "p1", ready_in_ms=0)
    p0_turn: Final = await second.poll(capacity, "p0", ready_in_ms=0)

    assert cooling == (Waiting(), Waiting())
    assert p1_turn == Waiting()
    assert isinstance(p0_turn, Probe)


@pytest.mark.asyncio
async def test_a_refreshed_cooldown_puts_an_unclaimed_probe_back_in_its_place(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("owner", 0, first), ("later", 1, second)])
    clock.ms = 1_000
    assert await second.poll(capacity, "later", ready_in_ms=5_000) == Waiting()

    refreshed: Final = await first.poll(capacity, "owner", ready_in_ms=2_000)
    clock.ms = 3_000
    owner_turn: Final = await first.poll(capacity, "owner")

    assert refreshed == Waiting()
    assert isinstance(owner_turn, Probe) and owner_turn.generation == 1


@pytest.mark.asyncio
async def test_a_recovery_snapshot_admits_its_first_ready_member_across_workers(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=1, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("prober", -5, first), ("cooling", 1, first), ("ready", 2, second)])
    clock.ms = 1_000
    probe: Final = await first.poll(capacity, "prober")
    assert isinstance(probe, Probe)
    assert await first.report(capacity, "prober", probe, Succeeded()) is None

    cooling_early: Final = await first.poll(capacity, "cooling", ready_in_ms=2_000)
    ready: Final = await second.poll(capacity, "ready")
    clock.ms = 3_000
    cooling_late: Final = await first.poll(capacity, "cooling")

    assert cooling_early == Waiting(ahead=0)
    assert isinstance(ready, Released)
    assert isinstance(cooling_late, Released)


COOLING_AHEAD: Final = 70


@pytest.mark.asyncio
async def test_a_ready_waiter_behind_more_cooling_waiters_than_one_sweep_still_gets_the_probe(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, liveness_ms=FAR)
    await _congest(first, capacity)
    clock.ms = 1_000
    for index in range(COOLING_AHEAD):
        reply = await first.admit(WaitRequest(capacity, f"cooling-{index}", 0), FAR, ready_in_ms=FAR)
        assert reply == Waiting(), (index, reply)

    ready: Final = await second.admit(WaitRequest(capacity, "ready", 9), FAR)

    assert isinstance(ready, Probe)


@pytest.mark.asyncio
async def test_a_queue_filled_to_its_configured_size_still_probes_its_last_ready_waiter_and_refuses_one_more(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, max_waiting=CONFIGURED_SIZE, liveness_ms=FAR)
    await _congest(first, capacity)
    for index in range(CONFIGURED_SIZE - 1):
        reply = await first.admit(WaitRequest(capacity, f"cooling-{index}", 0), FAR, ready_in_ms=FAR)
        assert reply == Waiting(probe_in_ms=1_000), (index, reply)
    assert await second.admit(WaitRequest(capacity, "ready", 9), FAR) == Waiting(probe_in_ms=1_000)

    one_more: Final = await first.admit(WaitRequest(capacity, "one-more", 0), FAR)
    clock.ms = 1_000
    ready: Final = await second.poll(capacity, "ready")

    assert isinstance(one_more, QueueFull)
    assert isinstance(ready, Probe)


@pytest.mark.asyncio
async def test_a_ready_snapshot_member_behind_more_cooling_members_than_one_sweep_still_claims(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    first, second = workers.pair(clock, required_successes=1, liveness_ms=FAR)
    await _congest(first, capacity)
    await _enqueue(capacity, [("prober", -1, first)])
    for index in range(COOLING_AHEAD):
        reply = await first.admit(WaitRequest(capacity, f"cooling-{index}", 0), FAR, ready_in_ms=FAR)
        assert reply == Waiting(probe_in_ms=1_000), (index, reply)
    await _enqueue(capacity, [("ready", 9, second)])
    clock.ms = 1_000
    probe: Final = await first.poll(capacity, "prober")
    assert isinstance(probe, Probe)
    assert await first.report(capacity, "prober", probe, Succeeded()) is None

    claimed: Final = await second.poll(capacity, "ready")

    assert isinstance(claimed, Released)


async def _probe_hints(
    backend: RedisQueueBackend | LocalQueueBackend, capacity: str, clock: SharedClock
) -> tuple[tuple[type[object], int | None], ...]:
    clock.ms = 0
    assert await backend.report(capacity, "in-flight", PassThrough(0), MatchingFailure(20_000)) is None
    first: Final = await backend.admit(WaitRequest(capacity, "a", 0), FAR)
    clock.ms = 5_000
    later: Final = await backend.poll(capacity, "a")
    clock.ms = 20_000
    probe: Final = await backend.poll(capacity, "a")
    behind_the_probe: Final = await backend.admit(WaitRequest(capacity, "b", 0), FAR)
    return tuple(
        (type(reply), reply.probe_in_ms if isinstance(reply, Waiting) else None)
        for reply in (first, later, probe, behind_the_probe)
    )


@pytest.mark.asyncio
async def test_a_wait_reply_hints_the_next_probe_only_while_no_probe_is_out_as_the_in_process_queue_does(
    capacity: str, workers: WorkerFactory
) -> None:
    clock: Final = SharedClock()
    in_redis: Final = await _probe_hints(workers(clock, liveness_ms=FAR), capacity, clock)
    in_process: Final = await _probe_hints(
        LocalQueueBackend(QueueSettings(liveness_ms=FAR), clock=lambda: clock.ms / 1000), capacity, clock
    )

    assert in_redis == in_process
    assert in_process == ((Waiting, 20_000), (Waiting, 15_000), (Probe, None), (Waiting, None))


SHORT_LIVENESS: Final = MappingProxyType({"liveness_ms": 900, "operation_timeout_ms": 200})


async def _driven(
    timer: ManualTimer,
    clock: SharedClock,
    waiting: "asyncio.Task[Admitted | Rejection]",
    until_s: float,
    tick_s: float = 0.25,
) -> None:
    """Moves the waiter's timer and Redis time together, ``tick_s`` at a time whenever the waiter is parked on a
    recheck, until it settles or ``until_s``."""
    for _ in range(10_000):
        await timer.parked(waiting)
        if waiting.done() or timer.at >= until_s:
            return
        timer.advance(tick_s)
        clock.ms = round(timer.at * 1000)
    raise AssertionError("deadlock safeguard: the wait never settled")


class Lagging:
    """Delays an admit's reply, a readiness read and a poll's request by ``lag_s``, on the waiter's timer and Redis
    time alike, each within the operation timeout."""

    def __init__(self, inner: RedisQueueBackend, timer: ManualTimer, clock: SharedClock, lag_s: float) -> None:
        self._inner = inner
        self._timer = timer
        self._clock = clock
        self._lag_s = lag_s

    @property
    def settings(self) -> QueueSettings:
        return self._inner.settings

    def _lag(self) -> None:
        self._timer.at += self._lag_s
        self._clock.ms = round(self._timer.at * 1000)

    async def readiness(self) -> float:
        self._lag()
        return 0.0

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        reply: Final = await self._inner.admit(request, budget_ms, ready_in_ms)
        self._lag()
        return reply

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        self._lag()
        return await self._inner.poll(capacity, waiter_id, ready_in_ms)

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        return await self._inner.renew(capacity, waiter_id, token)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return await self._inner.report(capacity, waiter_id, grant, outcome)

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        return await self._inner.abandon(capacity, waiter_id)


@pytest.mark.parametrize(
    ("liveness", "lag_s", "tick_s", "latest_s"),
    [
        (SHORT_LIVENESS, 0.0, 0.25, 5.0),
        (MappingProxyType({"liveness_ms": 1_200, "operation_timeout_ms": 200}), 0.19, 0.01, 5.7),
    ],
    ids=["prompt_replies", "replies_lagging_just_under_the_operation_timeout"],
)
@pytest.mark.asyncio
async def test_a_wait_behind_a_long_probe_backoff_outlives_a_short_liveness_until_its_probe(
    capacity: str,
    workers: WorkerFactory,
    liveness: Mapping[str, int],
    lag_s: float,
    tick_s: float,
    latest_s: float,
) -> None:
    timer: Final = ManualTimer()
    clock: Final = SharedClock()
    inner: Final = workers(clock, **liveness)
    await _congest(inner, capacity, retry_after_ms=5_000)
    worker: Final = Lagging(inner, timer, clock, lag_s)
    queue: Final = DynamicQueue(worker, timer)
    waiting: Final = asyncio.create_task(queue.admit(WaitRequest(capacity, "a", 0), 30.0, worker.readiness))
    try:
        await _driven(timer, clock, waiting, until_s=FAR, tick_s=tick_s)
        outcome: Final = waiting.result()
    finally:
        await _reap(waiting)

    assert isinstance(outcome, Admitted) and isinstance(outcome.grant, Probe)
    assert 5.0 <= timer.at <= latest_s


@pytest.mark.asyncio
async def test_cancelling_a_wait_during_a_long_probe_backoff_gives_its_place_back_on_every_worker(
    capacity: str, workers: WorkerFactory
) -> None:
    timer: Final = ManualTimer()
    clock: Final = SharedClock()
    waiter_worker, other_worker = workers.pair(clock, max_waiting=1, **SHORT_LIVENESS)
    await _congest(waiter_worker, capacity, retry_after_ms=5_000)
    waiting: Final = asyncio.create_task(DynamicQueue(waiter_worker, timer).admit(WaitRequest(capacity, "a", 0), 30.0))
    try:
        await _driven(timer, clock, waiting, until_s=2.0)
        held: Final = await other_worker.admit(WaitRequest(capacity, "b", 0), FAR)
        _ = waiting.cancel()
        cancelled: Final = (await asyncio.gather(waiting, return_exceptions=True))[0]
    finally:
        await _reap(waiting)
    freed: Final = await other_worker.admit(WaitRequest(capacity, "b", 0), FAR)

    assert isinstance(held, QueueFull)
    assert isinstance(cancelled, asyncio.CancelledError)
    assert freed == Waiting(probe_in_ms=3_000)

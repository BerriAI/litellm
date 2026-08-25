"""Leader election for auxiliary DB jobs, exercised against a live Redis.

Every test here stands up N PodLockManager instances with distinct pod ids
against one real Redis, which is the only place the compare-and-set, the TTL
expiry and the Lua compare-and-delete actually execute. fakeredis and mocks
cannot fail these.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Final

import pytest

from litellm._uuid import uuid
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.single_owner_job import (
    JobLease,
    WhenLockUnavailable,
    claim_once_per_window,
    run_as_single_owner,
)
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

pytestmark = pytest.mark.asyncio(loop_scope="module")

POD_COUNT: Final = 5


@pytest.fixture(scope="module")
def redis_cache() -> RedisCache:
    host: Final = os.environ.get("REDIS_HOST") or "127.0.0.1"
    port: Final = int(os.environ.get("REDIS_PORT") or "6379")
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        unreachable: Final = f"Redis at {host}:{port} is not reachable ({exc})."
        if os.environ.get("LITELLM_REQUIRE_LIVE_REDIS") == "1":
            pytest.fail(f"{unreachable} This suite is meaningless without it.")
        pytest.skip(f"{unreachable} Start one and set REDIS_HOST/REDIS_PORT to run this suite.")
    return RedisCache(host=host, port=port, password=os.environ.get("REDIS_PASSWORD"))


@pytest.fixture
def job_name(request: pytest.FixtureRequest) -> str:
    return f"lit5434-{request.node.name}-{uuid.uuid4()}"


@pytest.fixture
def pods(redis_cache: RedisCache) -> list[PodLockManager]:
    fleet: Final = [PodLockManager(redis_cache=redis_cache) for _ in range(POD_COUNT)]
    assert len({pod.pod_id for pod in fleet}) == POD_COUNT, "each pod must carry a distinct pod id"
    return fleet


def _recording_body(log: list[JobLease]) -> Callable[[JobLease], Awaitable[str]]:
    async def body(lease: JobLease) -> str:
        log.append(lease)
        return "ran"

    return body


async def _concurrent_tick(
    pods: list[PodLockManager],
    job_name: str,
    *,
    ttl_seconds: int,
    when_unavailable: WhenLockUnavailable,
) -> tuple[list[JobLease], list[str | None]]:
    """Fire one tick from every pod at once, holding whoever wins until all of them have tried to acquire.

    Without the gate a body that returns immediately lets the winner release before the next pod even
    attempts, so the pods would take the lease in sequence and the tick would look exclusive by luck.
    """
    leases: Final[list[JobLease]] = []
    gate: Final = asyncio.Event()

    async def body(lease: JobLease) -> str:
        leases.append(lease)
        await gate.wait()
        return "ran"

    tick: Final = asyncio.gather(
        *(
            run_as_single_owner(
                pod_lock_manager=pod,
                job_name=job_name,
                ttl_seconds=ttl_seconds,
                when_unavailable=when_unavailable,
                run=body,
            )
            for pod in pods
        )
    )
    await asyncio.sleep(0.5)
    gate.set()
    return leases, list(await tick)


async def _lease_ttl_ms(redis_cache: RedisCache, job_name: str) -> int:
    client: Final = redis_cache.init_async_client()
    return int(await client.pttl(PodLockManager.get_redis_lock_key(job_name)))


async def _lease_holder(redis_cache: RedisCache, job_name: str) -> str | None:
    raw: Final = await redis_cache.async_get_cache(PodLockManager.get_redis_lock_key(job_name))
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


async def test_only_one_pod_of_five_runs_the_tick(pods: list[PodLockManager], job_name: str) -> None:
    """Five pods firing the same tick concurrently: one body runs, four never start."""
    leases, results = await _concurrent_tick(
        pods, job_name, ttl_seconds=10, when_unavailable=WhenLockUnavailable.SKIP
    )

    assert leases == [JobLease.LEADER], f"exactly one body may run, and it must hold the lease: {leases}"
    assert results.count("ran") == 1, f"one call returns the body's value: {results}"
    assert results.count(None) == POD_COUNT - 1, f"the other four skip without running: {results}"


async def test_successor_runs_only_after_the_dead_owner_lease_expires(
    pods: list[PodLockManager], job_name: str
) -> None:
    """A pod that grabs the lease and dies blocks the job until the TTL runs out, then one successor takes over."""
    ttl_seconds: Final = 4
    dead_owner: Final = pods[0]
    survivors: Final = pods[1:]

    assert await dead_owner.acquire_lock(cronjob_id=job_name, ttl=ttl_seconds) is True
    took_lease_at: Final = time.monotonic()

    before_expiry, early = await _concurrent_tick(
        survivors, job_name, ttl_seconds=ttl_seconds, when_unavailable=WhenLockUnavailable.SKIP
    )
    probed_after: Final = time.monotonic() - took_lease_at
    assert probed_after < ttl_seconds, (
        f"the before-expiry probe took {probed_after:.1f}s on a {ttl_seconds}s lease, so it proves nothing; "
        "raise ttl_seconds if this machine is that slow"
    )
    assert before_expiry == [], "no survivor may run while the dead owner's lease is still live"
    assert early == [None] * len(survivors), f"every survivor skips before expiry: {early}"

    await asyncio.sleep(ttl_seconds - probed_after + 0.5)

    after_expiry, late = await _concurrent_tick(
        survivors, job_name, ttl_seconds=ttl_seconds, when_unavailable=WhenLockUnavailable.SKIP
    )
    assert after_expiry == [JobLease.LEADER], f"exactly one survivor takes over after expiry: {after_expiry}"
    assert late.count("ran") == 1, f"the successor's body value is returned once: {late}"


async def test_renewal_holds_the_lease_past_the_ttl_and_frees_it_on_completion(
    redis_cache: RedisCache, pods: list[PodLockManager], job_name: str
) -> None:
    """A body outliving its TTL keeps the lease the whole time, and gives it up the moment it finishes."""
    ttl_seconds: Final = 3
    owner: Final = pods[0]
    challenger: Final = pods[1]

    async def slow_body(lease: JobLease) -> JobLease:
        await asyncio.sleep(ttl_seconds * 2)
        return lease

    owner_task: Final = asyncio.create_task(
        run_as_single_owner(
            pod_lock_manager=owner,
            job_name=job_name,
            ttl_seconds=ttl_seconds,
            when_unavailable=WhenLockUnavailable.SKIP,
            run=slow_body,
        )
    )
    started: Final = time.monotonic()
    await asyncio.sleep(0.5)
    assert await _lease_holder(redis_cache, job_name) == owner.pod_id

    await asyncio.sleep(ttl_seconds + 0.5)
    elapsed: Final = time.monotonic() - started
    assert elapsed > ttl_seconds, "the challenger must fire strictly after the original lease would have lapsed"

    challenger_leases: Final[list[JobLease]] = []
    challenger_result: Final = await run_as_single_owner(
        pod_lock_manager=challenger,
        job_name=job_name,
        ttl_seconds=ttl_seconds,
        when_unavailable=WhenLockUnavailable.RUN,
        run=_recording_body(challenger_leases),
    )
    assert challenger_leases == [], (
        f"the lease must still belong to the running owner {elapsed:.1f}s in, well past its {ttl_seconds}s TTL, "
        f"so a challenger that is willing to run unguarded must still be turned away: {challenger_leases}"
    )
    assert challenger_result is None
    assert await _lease_holder(redis_cache, job_name) == owner.pod_id

    assert await owner_task == JobLease.LEADER
    assert await _lease_holder(redis_cache, job_name) is None, "the lease must be released once the body returns"

    handover_leases: Final[list[JobLease]] = []
    await run_as_single_owner(
        pod_lock_manager=challenger,
        job_name=job_name,
        ttl_seconds=ttl_seconds,
        when_unavailable=WhenLockUnavailable.SKIP,
        run=_recording_body(handover_leases),
    )
    assert handover_leases == [JobLease.LEADER], "the released lease is available to the next pod"


async def test_rolling_restart_never_skips_or_doubles_a_tick(redis_cache: RedisCache, job_name: str) -> None:
    """Across a rolling restart every tick is run by exactly one live pod, and the replacement picks the job up."""
    drained: Final = PodLockManager(redis_cache=redis_cache)
    kept: Final = PodLockManager(redis_cache=redis_cache)
    replacement: Final = PodLockManager(redis_cache=redis_cache)
    schedule: Final = (
        ("tick-0", (drained, kept)),
        ("tick-1", (drained, kept)),
        ("tick-2", (kept, replacement)),
        ("tick-3", (replacement,)),
    )

    async def run_tick(live_pods: tuple[PodLockManager, ...]) -> list[str]:
        leaders: list[str] = []
        gate: Final = asyncio.Event()

        async def claim(pod: PodLockManager) -> None:
            async def body(lease: JobLease) -> None:
                assert lease is JobLease.LEADER
                leaders.append(pod.pod_id)
                await gate.wait()

            await run_as_single_owner(
                pod_lock_manager=pod,
                job_name=job_name,
                ttl_seconds=10,
                when_unavailable=WhenLockUnavailable.SKIP,
                run=body,
            )

        tick: Final = asyncio.gather(*(claim(pod) for pod in live_pods))
        await asyncio.sleep(0.5)
        gate.set()
        await tick
        return leaders

    outcome: Final = [(label, live_pods, await run_tick(live_pods)) for label, live_pods in schedule]

    for label, live_pods, leaders in outcome:
        assert len(leaders) == 1, f"{label} must be run by exactly one pod, got {len(leaders)}"
        assert leaders[0] in {pod.pod_id for pod in live_pods}, f"{label} was run by a pod that was not live"

    after_restart: Final = {leaders[0] for _, _, leaders in outcome[2:]}
    assert drained.pod_id not in after_restart, "the drained pod must not run a tick after it is gone"
    assert outcome[-1][2] == [replacement.pod_id], "once only the replacement is left it must take the lease"
    assert await _lease_holder(redis_cache, job_name) is None, "no lease may outlive the last tick"


async def test_window_claim_is_taken_once_even_by_the_pod_that_won_it(
    pods: list[PodLockManager], job_name: str
) -> None:
    """One pod claims the window, and nobody, including the winner, claims it again until it rolls over."""
    window_seconds: Final = 2

    claims: Final = await asyncio.gather(
        *(
            claim_once_per_window(pod_lock_manager=pod, job_name=job_name, window_seconds=window_seconds)
            for pod in pods
        )
    )
    assert claims.count(True) == 1, f"exactly one pod may claim the window: {claims}"

    winner: Final = pods[claims.index(True)]
    repeat: Final = await claim_once_per_window(
        pod_lock_manager=winner, job_name=job_name, window_seconds=window_seconds
    )
    assert repeat is False, "the winner re-firing inside its own window must be refused"

    await asyncio.sleep(window_seconds + 0.5)

    next_window: Final = await asyncio.gather(
        *(
            claim_once_per_window(pod_lock_manager=pod, job_name=job_name, window_seconds=window_seconds)
            for pod in pods
        )
    )
    assert next_window.count(True) == 1, f"the next window is claimable exactly once: {next_window}"


async def test_renew_lock_extends_the_lease_only_for_its_owner(
    redis_cache: RedisCache, pods: list[PodLockManager], job_name: str
) -> None:
    """renew_lock is a compare-and-expire: a non-owner is refused and cannot push the owner's expiry out."""
    owner: Final = pods[0]
    intruder: Final = pods[1]
    initial_ttl: Final = 5
    extended_ttl: Final = 60

    assert await owner.acquire_lock(cronjob_id=job_name, ttl=initial_ttl) is True
    before: Final = await _lease_ttl_ms(redis_cache, job_name)
    assert 0 < before <= initial_ttl * 1000

    assert await intruder.renew_lock(cronjob_id=job_name, ttl=extended_ttl) is False
    after_intruder: Final = await _lease_ttl_ms(redis_cache, job_name)
    assert after_intruder <= before, f"a non-owner renewal must not move the expiry: {before}ms -> {after_intruder}ms"
    assert await _lease_holder(redis_cache, job_name) == owner.pod_id

    assert await owner.renew_lock(cronjob_id=job_name, ttl=extended_ttl) is True
    after_owner: Final = await _lease_ttl_ms(redis_cache, job_name)
    assert after_owner > initial_ttl * 1000, f"the owner's renewal must push the expiry out: {after_owner}ms"

    await owner.release_lock(cronjob_id=job_name)
    assert await owner.renew_lock(cronjob_id=job_name, ttl=extended_ttl) is False, "a released lease cannot be renewed"


async def test_release_lock_never_frees_a_lease_another_pod_has_taken_over(
    redis_cache: RedisCache, pods: list[PodLockManager], job_name: str
) -> None:
    """A stale owner finishing after its lease lapsed must not delete the lease its successor now holds."""
    stale_owner: Final = pods[0]
    successor: Final = pods[1]
    bystanders: Final = pods[2:]
    short_ttl: Final = 2

    assert await stale_owner.acquire_lock(cronjob_id=job_name, ttl=short_ttl) is True
    await asyncio.sleep(short_ttl + 0.5)
    assert await successor.acquire_lock(cronjob_id=job_name, ttl=30) is True

    await stale_owner.release_lock(cronjob_id=job_name)

    assert (
        await _lease_holder(redis_cache, job_name) == successor.pod_id
    ), "the stale owner's release must compare owners before deleting"

    leases, results = await _concurrent_tick(
        bystanders, job_name, ttl_seconds=30, when_unavailable=WhenLockUnavailable.SKIP
    )
    assert leases == [], "no pod may start while the successor still holds the lease"
    assert results == [None] * len(bystanders), f"every bystander skips: {results}"

    await successor.release_lock(cronjob_id=job_name)
    assert await _lease_holder(redis_cache, job_name) is None, "the owner's own release must free the lease"

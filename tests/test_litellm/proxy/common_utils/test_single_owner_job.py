import asyncio
import json
import time
from typing import Any

import pytest

from litellm.proxy.common_utils.single_owner_job import (
    JobLease,
    JobRole,
    WhenLockUnavailable,
    claim_once_per_window,
    run_as_single_owner,
)
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

JOB = "test_single_owner_job"


class FakeRedisCache:
    """In-memory cache with real SET NX / EX semantics and a real clock.

    ``supports_scripts`` selects between PodLockManager's Lua path and its
    GET-then-write fallback, so both are exercised by the same assertions.
    """

    def __init__(self, supports_scripts: bool = True, fail: bool = False):
        self.store: dict[str, tuple[Any, float | None]] = {}
        self.fail = fail
        if supports_scripts:
            self.async_register_script = self._register_script

    def _live(self, key: str) -> tuple[Any, float | None] | None:
        entry = self.store.get(key)
        if entry is None:
            return None
        _, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            del self.store[key]
            return None
        return entry

    async def async_set_cache(self, key: str, value: Any, nx: bool = False, ttl: int | None = None) -> bool:
        if self.fail:
            raise ConnectionError("redis down")
        if nx and self._live(key) is not None:
            return False
        self.store[key] = (value, None if ttl is None else time.monotonic() + ttl)
        return True

    async def async_get_cache(self, key: str) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        entry = self._live(key)
        return None if entry is None else entry[0]

    async def async_delete_cache(self, key: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        return 1 if self.store.pop(key, None) is not None else 0

    def ttl_remaining(self, key: str) -> float | None:
        entry = self._live(key)
        if entry is None or entry[1] is None:
            return None
        return entry[1] - time.monotonic()

    def _register_script(self, script: str):
        async def _run(keys: list[str], args: list[Any]) -> int:
            if self.fail:
                raise ConnectionError("redis down")
            key = keys[0]
            entry = self._live(key)
            # PodLockManager compares against the JSON encoding redis stores
            if entry is None or json.dumps(entry[0]) != args[0]:
                return 0
            if "del" in script:
                del self.store[key]
                return 1
            self.store[key] = (entry[0], time.monotonic() + int(args[1]))
            return 1

        return _run


def manager(**kwargs: Any) -> PodLockManager:
    return PodLockManager(redis_cache=FakeRedisCache(**kwargs))


def sharing(count: int, **kwargs: Any) -> list[PodLockManager]:
    """N pods with distinct ids against one cache, as separate replicas would be."""
    cache = FakeRedisCache(**kwargs)
    return [PodLockManager(redis_cache=cache) for _ in range(count)]


@pytest.mark.parametrize("supports_scripts", [True, False])
@pytest.mark.asyncio
async def test_exactly_one_pod_runs_a_tick(supports_scripts: bool):
    pods = sharing(5, supports_scripts=supports_scripts)
    ran: list[str] = []

    async def body(lease: JobLease) -> str:
        ran.append(lease.value)
        return "done"

    results = await asyncio.gather(
        *(
            run_as_single_owner(
                pod_lock_manager=pod,
                job_name=JOB,
                ttl_seconds=60,
                when_unavailable=WhenLockUnavailable.RUN,
                run=body,
            )
            for pod in pods
        )
    )

    assert ran == ["leader"]
    assert results.count("done") == 1
    assert results.count(None) == 4


@pytest.mark.asyncio
async def test_follower_does_not_run_and_leader_releases():
    pods = sharing(2)
    leader, follower = pods
    calls: list[JobLease] = []

    async def body(lease: JobLease) -> int:
        calls.append(lease)
        return 1

    assert await run_as_single_owner(
        pod_lock_manager=leader,
        job_name=JOB,
        ttl_seconds=60,
        when_unavailable=WhenLockUnavailable.RUN,
        run=body,
    ) == 1
    # the lease is released on the way out, so the next pod may take the next tick
    assert await run_as_single_owner(
        pod_lock_manager=follower,
        job_name=JOB,
        ttl_seconds=60,
        when_unavailable=WhenLockUnavailable.RUN,
        run=body,
    ) == 1
    assert calls == [JobLease.LEADER, JobLease.LEADER]


@pytest.mark.asyncio
async def test_a_held_lease_blocks_a_second_pod_for_the_whole_run():
    pods = sharing(2)
    holder, other = pods
    started = asyncio.Event()
    release = asyncio.Event()
    other_leases: list[JobLease] = []

    async def slow(_lease: JobLease) -> None:
        started.set()
        await release.wait()

    async def quick(lease: JobLease) -> None:
        other_leases.append(lease)

    holding = asyncio.create_task(
        run_as_single_owner(
            pod_lock_manager=holder,
            job_name=JOB,
            ttl_seconds=60,
            when_unavailable=WhenLockUnavailable.RUN,
            run=slow,
        )
    )
    await started.wait()
    assert (
        await run_as_single_owner(
            pod_lock_manager=other,
            job_name=JOB,
            ttl_seconds=60,
            when_unavailable=WhenLockUnavailable.RUN,
            run=quick,
        )
        is None
    )
    assert other_leases == []
    release.set()
    await holding


@pytest.mark.parametrize("supports_scripts", [True])
@pytest.mark.asyncio
async def test_renewal_keeps_a_run_that_outlives_its_ttl(supports_scripts: bool):
    """A body slower than the TTL keeps the lease, so no second pod joins mid-run."""
    pods = sharing(2, supports_scripts=supports_scripts)
    holder, other = pods
    ttl = 2
    started = asyncio.Event()
    joined: list[JobLease] = []

    async def slow(_lease: JobLease) -> None:
        started.set()
        await asyncio.sleep(ttl * 1.75)

    async def joiner(lease: JobLease) -> None:
        joined.append(lease)

    holding = asyncio.create_task(
        run_as_single_owner(
            pod_lock_manager=holder,
            job_name=JOB,
            ttl_seconds=ttl,
            when_unavailable=WhenLockUnavailable.RUN,
            run=slow,
        )
    )
    await started.wait()
    # strictly past the original TTL, so only a renewal can still be holding it
    await asyncio.sleep(ttl * 1.25)
    assert (
        await run_as_single_owner(
            pod_lock_manager=other,
            job_name=JOB,
            ttl_seconds=ttl,
            when_unavailable=WhenLockUnavailable.RUN,
            run=joiner,
        )
        is None
    ), "a renewed lease must still be held past its original TTL"
    assert joined == []

    await holding
    # released once the body finished, so the next tick elects freely
    await run_as_single_owner(
        pod_lock_manager=other,
        job_name=JOB,
        ttl_seconds=ttl,
        when_unavailable=WhenLockUnavailable.RUN,
        run=joiner,
    )
    assert joined == [JobLease.LEADER]


@pytest.mark.asyncio
async def test_an_expired_lease_fails_over_to_another_pod():
    pods = sharing(2)
    crashed, survivor = pods
    ttl = 1
    # a pod that took the lease and died: no renewal, no release
    assert await crashed.acquire_lock(cronjob_id=JOB, ttl=ttl) is True

    took: list[JobLease] = []

    async def body(lease: JobLease) -> None:
        took.append(lease)

    assert (
        await run_as_single_owner(
            pod_lock_manager=survivor,
            job_name=JOB,
            ttl_seconds=ttl,
            when_unavailable=WhenLockUnavailable.RUN,
            run=body,
        )
        is None
    ), "the lease must still be honoured before it expires"
    assert took == []

    await asyncio.sleep(ttl * 1.5)
    await run_as_single_owner(
        pod_lock_manager=survivor,
        job_name=JOB,
        ttl_seconds=ttl,
        when_unavailable=WhenLockUnavailable.RUN,
        run=body,
    )
    assert took == [JobLease.LEADER]


@pytest.mark.asyncio
async def test_unreachable_redis_runs_unguarded_or_skips_by_policy():
    seen: list[JobLease] = []

    async def body(lease: JobLease) -> str:
        seen.append(lease)
        return "ran"

    assert (
        await run_as_single_owner(
            pod_lock_manager=manager(fail=True),
            job_name=JOB,
            ttl_seconds=60,
            when_unavailable=WhenLockUnavailable.RUN,
            run=body,
        )
        == "ran"
    )
    assert seen == [JobLease.UNGUARDED]

    assert (
        await run_as_single_owner(
            pod_lock_manager=manager(fail=True),
            job_name=JOB,
            ttl_seconds=60,
            when_unavailable=WhenLockUnavailable.SKIP,
            run=body,
        )
        is None
    )
    assert seen == [JobLease.UNGUARDED], "SKIP must not run the body during an outage"


@pytest.mark.parametrize("lock_manager", [None, PodLockManager(redis_cache=None)])
@pytest.mark.asyncio
async def test_a_deployment_without_redis_runs_unguarded(lock_manager: PodLockManager | None):
    seen: list[JobLease] = []

    async def body(lease: JobLease) -> str:
        seen.append(lease)
        return "ran"

    assert (
        await run_as_single_owner(
            pod_lock_manager=lock_manager,
            job_name=JOB,
            ttl_seconds=60,
            when_unavailable=WhenLockUnavailable.SKIP,
            run=body,
        )
        == "ran"
    )
    assert seen == [JobLease.UNGUARDED]


class StaleReadCache(FakeRedisCache):
    """Reports a lease holder that the store has already moved past.

    This is what a GET-then-write renewal sees when the lease lapses and another
    pod takes it in the gap between the two calls: the read still names the old
    owner while the key already belongs to the successor.
    """

    def __init__(self, stale_holder: str):
        super().__init__(supports_scripts=False)
        self.stale_holder = stale_holder

    async def async_get_cache(self, key: str) -> Any:
        return self.stale_holder

    def committed(self, key: str) -> Any:
        entry = self._live(key)
        return None if entry is None else entry[0]


@pytest.mark.asyncio
async def test_renewal_never_takes_a_lease_back_from_a_successor():
    """Renewal has no safe non-atomic form, so where compare-and-expire cannot run it
    must report failure rather than write.

    A GET-then-SET writes unconditionally, so a lease taken over between the two calls
    is handed back to the pod that lost it and both then believe they own the job. That
    is the exclusivity this whole module exists to hold, so it must not be traded for a
    renewal a Redis without scripting could not have done atomically anyway.
    """
    successor_id = "successor-pod"
    owner = PodLockManager()
    owner.redis_cache = StaleReadCache(stale_holder=owner.pod_id)
    key = PodLockManager.get_redis_lock_key(JOB)
    # the successor already holds it; only the read is still reporting the old owner
    owner.redis_cache.store[key] = (successor_id, None)

    assert await owner.renew_lock(cronjob_id=JOB, ttl=30) is False
    assert owner.redis_cache.committed(key) == successor_id, (
        "renewal must not write the lease back to the pod that lost it"
    )


@pytest.mark.parametrize("supports_scripts", [True])
@pytest.mark.asyncio
async def test_renew_lock_only_extends_this_pods_lease(supports_scripts: bool):
    pods = sharing(2, supports_scripts=supports_scripts)
    owner, intruder = pods
    cache: FakeRedisCache = owner.redis_cache  # type: ignore[assignment]
    assert await owner.acquire_lock(cronjob_id=JOB, ttl=30) is True

    await asyncio.sleep(0.05)
    before = cache.ttl_remaining(PodLockManager.get_redis_lock_key(JOB))
    assert before is not None and before < 30

    assert await intruder.renew_lock(cronjob_id=JOB, ttl=30) is False
    assert cache.ttl_remaining(PodLockManager.get_redis_lock_key(JOB)) == pytest.approx(before, abs=0.05)

    assert await owner.renew_lock(cronjob_id=JOB, ttl=30) is True
    after = cache.ttl_remaining(PodLockManager.get_redis_lock_key(JOB))
    assert after is not None and after > before


@pytest.mark.asyncio
async def test_renew_lock_reports_a_lost_lease():
    pods = sharing(2)
    owner, thief = pods
    assert await owner.acquire_lock(cronjob_id=JOB, ttl=1) is True
    await asyncio.sleep(1.2)
    assert await thief.acquire_lock(cronjob_id=JOB, ttl=30) is True

    assert await owner.renew_lock(cronjob_id=JOB, ttl=30) is False
    assert await owner.redis_cache.async_get_cache(PodLockManager.get_redis_lock_key(JOB)) == thief.pod_id


@pytest.mark.asyncio
async def test_releasing_after_losing_the_lease_leaves_the_new_owner_alone():
    pods = sharing(2)
    loser, winner = pods
    assert await loser.acquire_lock(cronjob_id=JOB, ttl=1) is True
    await asyncio.sleep(1.2)
    assert await winner.acquire_lock(cronjob_id=JOB, ttl=30) is True

    await loser.release_lock(cronjob_id=JOB)

    assert await winner.redis_cache.async_get_cache(PodLockManager.get_redis_lock_key(JOB)) == winner.pod_id


@pytest.mark.asyncio
async def test_one_pod_claims_a_window_and_nobody_repeats_it():
    pods = sharing(4)
    claims = [
        await claim_once_per_window(pod_lock_manager=pod, job_name=JOB, window_seconds=1) for pod in pods
    ]
    assert claims.count(True) == 1

    winner = pods[claims.index(True)]
    # not reentrant: even the holder may not redo the window it already sent
    assert await claim_once_per_window(pod_lock_manager=winner, job_name=JOB, window_seconds=1) is False

    await asyncio.sleep(1.2)
    reclaims = [
        await claim_once_per_window(pod_lock_manager=pod, job_name=JOB, window_seconds=1) for pod in pods
    ]
    assert reclaims.count(True) == 1


@pytest.mark.asyncio
async def test_window_claim_runs_without_redis_and_skips_during_an_outage():
    assert await claim_once_per_window(pod_lock_manager=None, job_name=JOB, window_seconds=60) is True
    assert (
        await claim_once_per_window(
            pod_lock_manager=PodLockManager(redis_cache=None), job_name=JOB, window_seconds=60
        )
        is True
    )
    assert (
        await claim_once_per_window(pod_lock_manager=manager(fail=True), job_name=JOB, window_seconds=60) is False
    ), "a report that reaches a channel must not be sent when the claim cannot be proved"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, JobRole.ALL),
        ("", JobRole.ALL),
        ("   ", JobRole.ALL),
        ("all", JobRole.ALL),
        ("serving", JobRole.SERVING),
        ("worker", JobRole.WORKER),
        ("  WORKER  ", JobRole.WORKER),
        ("Serving", JobRole.SERVING),
        ("bogus", JobRole.ALL),
    ],
)
def test_job_role_parsing(raw: str | None, expected: JobRole):
    assert JobRole.from_env_value(raw) is expected


@pytest.mark.parametrize(
    "role, registers",
    [(JobRole.ALL, True), (JobRole.WORKER, True), (JobRole.SERVING, False)],
)
def test_only_a_serving_pod_skips_single_owner_jobs(role: JobRole, registers: bool):
    assert role.runs_single_owner_jobs is registers

"""Ownership rules for scheduled jobs that must run once per deployment.

Every proxy process registers the same scheduler, so a job with a shared
side effect runs once per pod unless something elects an owner.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from enum import Enum
from typing import Final, TypeVar

from litellm._logging import verbose_proxy_logger
from litellm.constants import SINGLE_OWNER_JOB_RENEWAL_DIVISOR
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

T = TypeVar("T")


class JobRole(Enum):
    """Which scheduled jobs a process registers."""

    ALL = "all"
    SERVING = "serving"
    WORKER = "worker"

    @classmethod
    def from_env_value(cls, raw: str | None) -> "JobRole":
        if raw is None or not raw.strip():
            return cls.ALL
        try:
            return cls(raw.strip().lower())
        except ValueError:
            verbose_proxy_logger.warning(
                "LITELLM_JOB_ROLE=%r is not one of %s; running every job as if it were %s",
                raw,
                tuple(role.value for role in cls),
                cls.ALL.value,
            )
            return cls.ALL

    @property
    def runs_single_owner_jobs(self) -> bool:
        return self is not JobRole.SERVING


class JobLease(Enum):
    """This pod's standing for one run of a single-owner job."""

    LEADER = "leader"
    FOLLOWER = "follower"
    UNGUARDED = "unguarded"


class WhenLockUnavailable(Enum):
    """What a pod does when the lock can be neither taken nor read.

    ``acquire_lock`` reports contention and an unreachable Redis identically, so
    each job has to say which way to resolve the ambiguity. Work whose duplicate
    is merely wasteful picks ``RUN``; work whose duplicate reaches a person or an
    external system picks ``SKIP``.
    """

    RUN = "run"
    SKIP = "skip"


async def run_as_single_owner(
    *,
    pod_lock_manager: PodLockManager | None,
    job_name: str,
    ttl_seconds: int,
    when_unavailable: WhenLockUnavailable,
    run: Callable[[JobLease], Awaitable[T]],
) -> T | None:
    """Run ``run`` on one pod, holding a lease that is renewed for its duration.

    Returns ``None`` without running when another pod owns this tick.

    Renewal is what makes the TTL a failover deadline rather than a run budget: a
    healthy owner keeps the lease however long its work takes, and a crashed one
    strands the job for at most ``ttl_seconds``.

    The lease is released when ``run`` returns, so it dedupes for the body's runtime
    and not for the TTL. Two pods whose ticks are further apart than that both run,
    which is what the scheduler's stagger makes ordinary rather than incidental, so
    ``run`` has to be idempotent. Work that must happen once per period regardless of
    when each pod fires wants ``claim_once_per_window`` instead, whose marker outlives
    the run.
    """
    manager: Final = pod_lock_manager
    if manager is None or manager.redis_cache is None:
        return await run(JobLease.UNGUARDED)

    lease: Final = await _acquire(
        manager=manager,
        job_name=job_name,
        ttl_seconds=ttl_seconds,
        when_unavailable=when_unavailable,
    )
    match lease:
        case JobLease.FOLLOWER:
            return None
        case JobLease.UNGUARDED:
            return await run(lease)
        case JobLease.LEADER:
            return await _run_holding_lease(
                manager=manager,
                job_name=job_name,
                ttl_seconds=ttl_seconds,
                run=run,
            )


async def claim_once_per_window(
    *,
    pod_lock_manager: PodLockManager | None,
    job_name: str,
    window_seconds: int,
) -> bool:
    """True when this pod may run the job for the current window.

    The lock is a done-marker rather than a lease: nobody releases it, so it
    expires with the window and the next window's first firer takes it. Each pod
    anchors its interval to its own boot time, so a lock that outlived only the
    run would let a later pod repeat the window.

    A deployment with no Redis at all runs, having no peer to duplicate against.
    A configured but unreachable Redis does not, because these windows end at
    someone's inbox.
    """
    if pod_lock_manager is None:
        return True
    claimed: Final = await pod_lock_manager.acquire_lock(
        cronjob_id=job_name,
        ttl=window_seconds,
        allow_reentrant=False,
    )
    return claimed is not False


async def _acquire(
    *,
    manager: PodLockManager,
    job_name: str,
    ttl_seconds: int,
    when_unavailable: WhenLockUnavailable,
) -> JobLease:
    if await manager.acquire_lock(cronjob_id=job_name, ttl=ttl_seconds):
        return JobLease.LEADER

    if await _lease_is_held(manager=manager, job_name=job_name):
        verbose_proxy_logger.debug("%s: another pod holds the lease, skipping this run", job_name)
        return JobLease.FOLLOWER

    match when_unavailable:
        case WhenLockUnavailable.RUN:
            verbose_proxy_logger.warning(
                "%s: could not take the lease and no other pod holds it, running unguarded rather than skipping",
                job_name,
            )
            return JobLease.UNGUARDED
        case WhenLockUnavailable.SKIP:
            verbose_proxy_logger.warning(
                "%s: could not take the lease and no other pod holds it, skipping rather than risking a duplicate",
                job_name,
            )
            return JobLease.FOLLOWER


async def _lease_is_held(*, manager: PodLockManager, job_name: str) -> bool:
    """An unreadable lease reports as unheld so the caller reaches its own
    ``when_unavailable`` policy instead of silently taking the follower branch.
    """
    if manager.redis_cache is None:
        return False
    try:
        lock_key: Final = manager.get_redis_lock_key(job_name)
        return bool(await manager.redis_cache.async_get_cache(lock_key))
    except Exception as exc:  # noqa: BLE001  # an unreadable lease must not decide the run
        verbose_proxy_logger.warning("%s: could not read the lease: %s", job_name, exc)
        return False


async def _run_holding_lease(
    *,
    manager: PodLockManager,
    job_name: str,
    ttl_seconds: int,
    run: Callable[[JobLease], Awaitable[T]],
) -> T:
    verbose_proxy_logger.info("%s: pod %s owns this run", job_name, manager.pod_id)
    renewer: Final = asyncio.create_task(
        _renew_until_cancelled(manager=manager, job_name=job_name, ttl_seconds=ttl_seconds)
    )
    try:
        return await run(JobLease.LEADER)
    finally:
        renewer.cancel()
        with suppress(asyncio.CancelledError):
            await renewer
        await manager.release_lock(cronjob_id=job_name)


async def _renew_until_cancelled(*, manager: PodLockManager, job_name: str, ttl_seconds: int) -> None:
    """Hold the lease until cancelled, or until it belongs to someone else.

    Renewal stops on the first failure so the new owner keeps what it took. The
    release on the way out compares owners, so it cannot steal the lease back.
    """
    interval: Final = max(1.0, ttl_seconds / SINGLE_OWNER_JOB_RENEWAL_DIVISOR)
    while True:
        await asyncio.sleep(interval)
        if not await manager.renew_lock(cronjob_id=job_name, ttl=ttl_seconds):
            verbose_proxy_logger.warning(
                "%s: pod %s lost the lease mid-run, a second pod may be running this job",
                job_name,
                manager.pod_id,
            )
            return

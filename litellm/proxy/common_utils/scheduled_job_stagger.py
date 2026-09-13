"""
Deterministic phase offsets for the proxy's scheduled background jobs.

APScheduler anchors an ``interval`` job at ``now + interval``, so every job registered in
the same startup shares one firing instant for the life of the process, and every replica
brought up by the same rollout shares it too. The result is a burst: each tick, every job
on every replica queries Postgres at the same moment, competing with the request path for
the connection pool. The product's own daily/monthly crons are worse still, since they name
a wall-clock instant that is identical on every replica by construction.

The fix is a phase offset derived from ``sha256(job_id, identity)``, where ``identity``
covers the pod and the worker process. Different jobs get different offsets, different
replicas get different offsets for the same job, and nothing collapses back onto a shared
instant after a restart. Hashing rather than randomising keeps a given process's schedule
stable for its whole life and lets the applied offsets be logged once and reasoned about
later.

The offset lives in the trigger rather than in a one-off ``next_run_time`` because a cron
trigger recomputes each fire from the wall clock and would otherwise snap straight back
onto the shared instant after its first shifted run.

Only schedules LiteLLM itself chose are shifted. Interval jobs are always eligible; cron
jobs only when their id is one of the product's own defaults, so an operator-supplied
crontab keeps the exact instant it asks for. A job whose call site passed an explicit
``next_run_time`` already anchors itself and is left alone.
"""

# apscheduler ships no type information, so its imports have no stubs. The Protocols below
# narrow everything it hands back, which is why this is the only diagnostic left to silence.
# pyright: reportMissingTypeStubs=false

import hashlib
import os
import socket
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Protocol

from apscheduler.events import EVENT_JOB_SUBMITTED
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    MONTHLY_SPEND_REPORT_JOB_ID,
    PROMETHEUS_FALLBACK_STATS_JOB_ID,
    PTU_ROLLUP_JOB_ID,
    PTU_ROLLUP_LOCK_TTL_SECONDS,
)
from litellm.proxy._types import ScheduledJobStaggerSettings

GENERAL_SETTINGS_KEY: Final = "scheduled_job_stagger"

#: Cron schedules LiteLLM picks on the operator's behalf, so shifting them changes nothing the
#: operator asked for. Every other cron trigger is an operator-supplied crontab, preserved exactly.
#:
#: The value is the span over which a second firing would redo work the first already did, which
#: is how long each job's leader-election lock stays held. Two replicas further apart than that
#: both find the key free and both run, which for the spend report means the customer gets it
#: twice. Offsets for these jobs are bounded by it, so widening the window cannot resurrect the
#: duplicate-work failure this feature exists to avoid.
DEFAULT_CRON_DEDUPE_SECONDS: Final = MappingProxyType(
    {
        MONTHLY_SPEND_REPORT_JOB_ID: 3600,
        PROMETHEUS_FALLBACK_STATS_JOB_ID: 3600,
        PTU_ROLLUP_JOB_ID: PTU_ROLLUP_LOCK_TTL_SECONDS,
    }
)


class Trigger(Protocol):
    """The one method APScheduler asks a trigger for"""

    def get_next_fire_time(self, previous_fire_time: datetime | None, now: datetime) -> datetime | None: ...


class ScheduledJob(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def trigger(self) -> Trigger: ...


class JobScheduler(Protocol):
    """The slice of ``AsyncIOScheduler`` this module uses, which ships no type information"""

    @property
    def running(self) -> bool: ...

    def get_jobs(self) -> Sequence[ScheduledJob]: ...

    def modify_job(self, job_id: str, *, trigger: Trigger) -> object: ...

    def add_listener(self, callback: Callable[["JobSubmission"], None], mask: int = ...) -> None: ...


class JobSubmission(Protocol):
    """An ``EVENT_JOB_SUBMITTED`` event"""

    @property
    def job_id(self) -> str: ...

    @property
    def scheduled_run_times(self) -> Sequence[datetime]: ...


class _OffsetTrigger:
    """
    Delegates to ``base`` on a clock rolled back by ``offset``, then rolls the answer
    forward again, so every fire lands exactly ``offset`` later than it otherwise would
    while the underlying schedule keeps its own semantics.

    Composed rather than derived from ``BaseTrigger``: APScheduler only ever asks a trigger
    for its next fire time, and it accepts this by virtual registration below.
    """

    __slots__ = ("base", "offset")

    def __init__(self, base: Trigger, offset: timedelta) -> None:
        self.base = base
        self.offset = offset

    def get_next_fire_time(self, previous_fire_time: datetime | None, now: datetime) -> datetime | None:
        shifted_previous: Final = None if previous_fire_time is None else previous_fire_time - self.offset
        next_fire_time: Final = self.base.get_next_fire_time(shifted_previous, now - self.offset)
        return None if next_fire_time is None else next_fire_time + self.offset

    def __str__(self) -> str:
        return f"{self.base}[+{int(self.offset.total_seconds())}s]"


# APScheduler type-checks assigned triggers with isinstance, so it has to accept this one
BaseTrigger.register(_OffsetTrigger)


def parse_stagger_settings(general_settings: Mapping[str, object]) -> ScheduledJobStaggerSettings:
    raw: Final = general_settings.get(GENERAL_SETTINGS_KEY)
    if raw is None:
        return ScheduledJobStaggerSettings()
    try:
        return ScheduledJobStaggerSettings.model_validate(raw)
    except ValidationError as exc:
        verbose_proxy_logger.warning(
            "Ignoring invalid general_settings.%s, falling back to defaults: %s",
            GENERAL_SETTINGS_KEY,
            exc,
        )
        return ScheduledJobStaggerSettings()


def resolve_stagger_identity(configured: str | None) -> str:
    """
    The value hashed alongside a job id to place this process in the stagger window.

    The process id is part of it because a pod runs one scheduler per uvicorn worker, and
    workers sharing a hostname would otherwise all land on the same offset. That makes the
    offsets change across restarts, which is what stops a simultaneous rollout from
    reconverging; the applied values are logged so a given run stays explainable.
    """
    host: Final = configured or os.getenv("POD_NAME") or os.getenv("HOSTNAME") or _hostname()
    return f"{host}:{os.getpid()}"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return str(uuid.uuid4())


def offset_seconds(*, job_id: str, identity: str, window_seconds: int) -> int:
    """A stable point in ``[0, window_seconds)`` for this job on this process"""
    if window_seconds <= 0:
        return 0
    digest: Final = hashlib.sha256(f"{job_id}\x00{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % window_seconds


def _interval_seconds(job: ScheduledJob) -> int | None:
    if not isinstance(job.trigger, IntervalTrigger):
        return None
    interval: Final = getattr(job.trigger, "interval", None)
    return int(interval.total_seconds()) if isinstance(interval, timedelta) else None


def _is_staggerable(job: ScheduledJob) -> bool:
    if hasattr(job, "next_run_time"):
        # the call site anchored the first fire itself
        return False
    if _interval_seconds(job) is not None:
        return True
    return job.id in DEFAULT_CRON_DEDUPE_SECONDS


def _window_for(*, job_id: str, period_seconds: int | None, settings: ScheduledJobStaggerSettings) -> int:
    """
    Exclusive upper bound on this job's offset. An interval job is never offset by more than
    one of its own periods, so it is not delayed past the wait it already had, and a
    leader-elected cron is never offset past the span in which a second replica would redo
    its work.
    """
    limits: Final = (settings.window_seconds, period_seconds, DEFAULT_CRON_DEDUPE_SECONDS.get(job_id))
    return min(limit for limit in limits if limit is not None)


def _clamped_override(*, job_id: str, requested: int) -> int:
    horizon: Final = DEFAULT_CRON_DEDUPE_SECONDS.get(job_id)
    if horizon is None or requested < horizon:
        return requested
    verbose_proxy_logger.warning(
        "general_settings.%s.offsets[%s]=%ss would place replicas more than %ss apart, "
        "which is long enough for a second replica to redo the run; using %ss instead",
        GENERAL_SETTINGS_KEY,
        job_id,
        requested,
        horizon,
        horizon - 1,
    )
    return horizon - 1


def _offset_for(
    *,
    job_id: str,
    period_seconds: int | None,
    staggerable: bool,
    settings: ScheduledJobStaggerSettings,
    identity: str,
) -> int:
    override: Final = settings.offsets.get(job_id)
    if override is not None:
        return _clamped_override(job_id=job_id, requested=max(0, override))
    if not staggerable:
        return 0
    return offset_seconds(
        job_id=job_id,
        identity=identity,
        window_seconds=_window_for(job_id=job_id, period_seconds=period_seconds, settings=settings),
    )


def stagger_trigger(
    *,
    job_id: str,
    trigger: Trigger,
    period_seconds: int | None,
    settings: ScheduledJobStaggerSettings,
    identity: str | None = None,
) -> Trigger:
    """
    The trigger a job should carry, shifted by its own share of the window.

    For a job registered against an already-running scheduler, which the startup sweep cannot
    reach: every job carries a ``next_run_time`` by then, so re-running the sweep would treat
    them all as self-anchored and change nothing.
    """
    offset: Final = _offset_for(
        job_id=job_id,
        period_seconds=period_seconds,
        staggerable=True,
        settings=settings,
        identity=identity or resolve_stagger_identity(settings.identity),
    )
    return trigger if offset == 0 else _OffsetTrigger(trigger, timedelta(seconds=offset))


def apply_scheduled_job_stagger(
    *,
    scheduler: JobScheduler,
    settings: ScheduledJobStaggerSettings,
    identity: str | None = None,
) -> Mapping[str, int]:
    """
    Shift each eligible job's schedule by its own offset. Call this once, after every job is
    registered and before the scheduler starts, so the offset is folded into the first fire
    rather than applied to a schedule already running.

    ``identity`` is resolved from the environment when the caller does not supply one.

    Returns the offset applied to every registered job, including the zeroes, so the caller
    and the logs describe the same thing.
    """
    resolved_identity: Final = identity or resolve_stagger_identity(settings.identity)
    if scheduler.running:
        # every job already carries a next_run_time by now, so the sweep would skip all of
        # them and report success while changing nothing
        verbose_proxy_logger.warning(
            "Scheduled job stagger skipped: the scheduler is already running, so offsets must be "
            "applied before it starts"
        )
        return MappingProxyType({job.id: 0 for job in scheduler.get_jobs()})
    if not settings.enabled:
        verbose_proxy_logger.info(
            "Scheduled job stagger disabled via general_settings.%s; all jobs keep their unshifted schedule",
            GENERAL_SETTINGS_KEY,
        )
        return MappingProxyType({job.id: 0 for job in scheduler.get_jobs()})

    offsets: Final = MappingProxyType(
        {
            job.id: _offset_for(
                job_id=job.id,
                period_seconds=_interval_seconds(job),
                staggerable=_is_staggerable(job),
                settings=settings,
                identity=resolved_identity,
            )
            for job in scheduler.get_jobs()
        }
    )
    for job in scheduler.get_jobs():
        if offsets[job.id] > 0:
            scheduler.modify_job(
                job.id,
                trigger=_OffsetTrigger(job.trigger, timedelta(seconds=offsets[job.id])),
            )

    verbose_proxy_logger.info(
        "Scheduled job stagger applied (identity=%s, window=%ss): %s",
        resolved_identity,
        settings.window_seconds,
        ", ".join(f"{job_id}=+{seconds}s" for job_id, seconds in sorted(offsets.items())),
    )
    return offsets


def attach_job_timing_logger(scheduler: JobScheduler) -> None:
    """Log each fire's scheduled instant against the instant it actually started"""
    scheduler.add_listener(_log_job_submitted, EVENT_JOB_SUBMITTED)


def _log_job_submitted(event: JobSubmission) -> None:
    if not event.scheduled_run_times:
        return
    scheduled: Final = event.scheduled_run_times[0]
    started: Final = datetime.now(scheduled.tzinfo)
    verbose_proxy_logger.debug(
        "Scheduled job %s started: scheduled_run_time=%s actual_start_time=%s delay=%.3fs",
        event.job_id,
        scheduled.isoformat(),
        started.isoformat(),
        (started - scheduled).total_seconds(),
    )

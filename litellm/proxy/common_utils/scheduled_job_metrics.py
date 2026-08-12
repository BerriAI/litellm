"""Turns APScheduler's own lifecycle events into scheduled-job telemetry.

Background jobs were previously invisible: nothing recorded which job ran on a
pod, when, for how long, whether it succeeded, or how much work it moved. During
an incident that left operators inferring job activity from database load.

A scheduler listener covers every registered job at once, including ones added
later, rather than each job growing its own instrumentation.

Job ids are the label, and every job litellm registers pins one. A caller that
omits ``id=`` gets a fresh uuid from APScheduler instead, which as a label would
grow without bound across pods and restarts, so those collapse into a single
bucket rather than being trusted.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MAX_INSTANCES,
    EVENT_JOB_MISSED,
    EVENT_JOB_SUBMITTED,
)

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from apscheduler.events import JobEvent
    from apscheduler.schedulers.base import BaseScheduler

# APScheduler assigns `uuid4().hex` when a caller omits `id=`.
_GENERATED_JOB_ID: Final = re.compile(r"\A[0-9a-f]{32}\Z")
_UNNAMED_JOB: Final = "unnamed_job"

_LISTENER_EVENT_MASK: Final = (
    EVENT_JOB_SUBMITTED | EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES
)


class JobResult(str, Enum):
    """Closed set of outcomes, so the ``result`` label stays bounded."""

    SUCCESS = "success"
    ERROR = "error"
    # The trigger fired but the run was skipped entirely.
    MISSED = "missed"
    # A previous run of the same job was still going. With max_instances=1 this
    # is how a job that overruns its interval reports falling behind.
    MAX_INSTANCES = "max_instances"


@dataclass(frozen=True, slots=True)
class JobRun:
    """One completed run of a scheduled job."""

    job_name: str
    result: JobResult
    duration_seconds: float | None
    items_processed: int | None


def _label_for(job_id: str) -> str:
    """The job id, unless APScheduler generated it."""
    return _UNNAMED_JOB if _GENERATED_JOB_ID.match(job_id) else job_id


def _items_processed(retval: object) -> int | None:
    """The item count a job reported, if it reported one.

    Jobs signal how much work they moved by returning a count. A job that
    returns anything else, which is most of them, simply has no count to
    publish. ``bool`` is excluded because it is an ``int`` subclass and a job
    returning ``True`` means success, not one item.
    """
    if isinstance(retval, bool) or not isinstance(retval, int):
        return None
    return retval


class ScheduledJobMetricsListener:
    """Pairs APScheduler's submit and completion events into job runs.

    Duration is measured across those two events because APScheduler does not
    report it. Runs are keyed by job id AND scheduled run time, because
    ``max_instances`` and ``coalesce`` are both env-overridable: raising the
    first puts several runs of one job in flight at once, and disabling the
    second makes one submission produce a completion per missed run time. Keying
    on the job id alone would let those runs consume each other's start times.
    """

    def __init__(self, *, monotonic: Final = time.monotonic) -> None:
        self._monotonic: Final = monotonic
        self._started_at: dict[str, float] = {}  # mutable-ok: start times arrive one scheduler event at a time

    def register(self, scheduler: BaseScheduler) -> None:
        scheduler.add_listener(self.handle, _LISTENER_EVENT_MASK)

    def handle(self, event: JobEvent) -> None:
        """Never raises. A listener that throws is swallowed by APScheduler and
        would leave the scheduler running with no telemetry and no explanation."""
        try:
            run: Final = self._to_run(event)
            if run is not None:
                self._publish(run)
        except Exception as e:  # noqa: BLE001  # telemetry must not disturb the scheduler
            verbose_proxy_logger.debug("scheduled job metrics listener failed: %s", e)

    @staticmethod
    def _key(job_id: str, scheduled_run_time: object) -> str:
        return f"{job_id}@{scheduled_run_time}"

    def _to_run(self, event: JobEvent) -> JobRun | None:
        job_name: Final = _label_for(event.job_id)
        if event.code == EVENT_JOB_SUBMITTED:
            now: Final = self._monotonic()
            for scheduled in getattr(event, "scheduled_run_times", ()) or (None,):
                self._started_at[self._key(event.job_id, scheduled)] = now
            return None

        # Neither of these follows a submission of its own. MAX_INSTANCES in
        # particular arrives while the previous run is still going, so popping
        # before this point would steal that run's start time and drop the
        # duration of exactly the overruns worth measuring.
        if event.code == EVENT_JOB_MISSED:
            return JobRun(job_name, JobResult.MISSED, None, None)
        if event.code == EVENT_JOB_MAX_INSTANCES:
            return JobRun(job_name, JobResult.MAX_INSTANCES, None, None)

        started_at: Final = self._started_at.pop(
            self._key(event.job_id, getattr(event, "scheduled_run_time", None)), None
        )
        duration: Final = None if started_at is None else self._monotonic() - started_at

        if event.code == EVENT_JOB_ERROR:
            return JobRun(job_name, JobResult.ERROR, duration, None)
        return JobRun(job_name, JobResult.SUCCESS, duration, _items_processed(getattr(event, "retval", None)))

    @staticmethod
    def _publish(run: JobRun) -> None:
        verbose_proxy_logger.info(
            "scheduled_job_completed job=%s result=%s duration_seconds=%s items_processed=%s",
            run.job_name,
            run.result.value,
            "unknown" if run.duration_seconds is None else f"{run.duration_seconds:.3f}",
            "unknown" if run.items_processed is None else run.items_processed,
        )

        from litellm.integrations.prometheus import PrometheusLogger

        logger: Final = PrometheusLogger.get_instance()
        if logger is not None:
            logger.record_scheduled_job_run(run)

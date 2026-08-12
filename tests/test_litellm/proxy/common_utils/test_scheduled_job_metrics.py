"""Scheduled-job telemetry, exercised against a real APScheduler.

The listener's whole job is to interpret APScheduler's event stream, so the
events come from a running scheduler rather than from hand-built objects. A
fabricated event proves only that the code reads the fields it was written to
read.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.common_utils.scheduled_job_metrics import (
    JobResult,
    ScheduledJobMetricsListener,
)


async def _drain(recorded, *, expected: int, timeout: float = 5.0):
    """Wait for the scheduler to deliver `expected` runs."""
    deadline = asyncio.get_running_loop().time() + timeout
    while len(recorded) < expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    return recorded


@pytest.fixture
def recorded():
    runs = []
    logger = MagicMock()
    logger.record_scheduled_job_run = runs.append
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.get_instance",
        return_value=logger,
    ):
        yield runs


@pytest.mark.asyncio
async def test_a_successful_job_reports_its_name_duration_and_result(recorded):
    scheduler = AsyncIOScheduler()
    ScheduledJobMetricsListener().register(scheduler)

    async def quick_job():
        await asyncio.sleep(0.05)

    scheduler.add_job(quick_job, "interval", seconds=60, id="quick_job", next_run_time=None)
    scheduler.start(paused=False)
    try:
        scheduler.get_job("quick_job").modify(next_run_time=__import__("datetime").datetime.now())
        await _drain(recorded, expected=1)
    finally:
        scheduler.shutdown(wait=False)

    assert len(recorded) == 1
    run = recorded[0]
    assert run.job_name == "quick_job"
    assert run.result is JobResult.SUCCESS
    assert run.duration_seconds is not None and run.duration_seconds >= 0.05, (
        f"duration must span the real execution, got {run.duration_seconds}"
    )
    assert run.items_processed is None


@pytest.mark.asyncio
async def test_a_failing_job_is_recorded_as_an_error_not_a_success(recorded):
    scheduler = AsyncIOScheduler()
    ScheduledJobMetricsListener().register(scheduler)

    async def broken_job():
        raise RuntimeError("job blew up")

    scheduler.add_job(broken_job, "interval", seconds=60, id="broken_job", next_run_time=None)
    scheduler.start(paused=False)
    try:
        scheduler.get_job("broken_job").modify(next_run_time=__import__("datetime").datetime.now())
        await _drain(recorded, expected=1)
    finally:
        scheduler.shutdown(wait=False)

    assert len(recorded) == 1
    assert recorded[0].result is JobResult.ERROR
    assert recorded[0].job_name == "broken_job"


@pytest.mark.asyncio
async def test_a_job_that_returns_a_count_publishes_it(recorded):
    """Item counts ride on the return value so a job opts in with one line
    instead of reaching for the metrics layer itself."""
    scheduler = AsyncIOScheduler()
    ScheduledJobMetricsListener().register(scheduler)

    async def counting_job():
        return 42

    scheduler.add_job(counting_job, "interval", seconds=60, id="counting_job", next_run_time=None)
    scheduler.start(paused=False)
    try:
        scheduler.get_job("counting_job").modify(next_run_time=__import__("datetime").datetime.now())
        await _drain(recorded, expected=1)
    finally:
        scheduler.shutdown(wait=False)

    assert recorded[0].items_processed == 42


@pytest.mark.asyncio
async def test_a_job_returning_true_is_not_read_as_one_item(recorded):
    """bool is an int subclass, so a job returning True would otherwise publish
    an item count of 1 that it never meant."""
    scheduler = AsyncIOScheduler()
    ScheduledJobMetricsListener().register(scheduler)

    async def boolean_job():
        return True

    scheduler.add_job(boolean_job, "interval", seconds=60, id="boolean_job", next_run_time=None)
    scheduler.start(paused=False)
    try:
        scheduler.get_job("boolean_job").modify(next_run_time=__import__("datetime").datetime.now())
        await _drain(recorded, expected=1)
    finally:
        scheduler.shutdown(wait=False)

    assert recorded[0].items_processed is None


@pytest.mark.asyncio
async def test_a_job_that_overruns_its_interval_reports_max_instances(recorded):
    """With max_instances=1 this is how a job falling behind its schedule
    surfaces, which is exactly the signal an operator wants during an incident."""
    import datetime

    scheduler = AsyncIOScheduler(job_defaults={"max_instances": 1, "coalesce": False})
    ScheduledJobMetricsListener().register(scheduler)

    async def slow_job():
        await asyncio.sleep(1.5)

    scheduler.add_job(slow_job, "interval", seconds=1, id="slow_job", next_run_time=datetime.datetime.now())
    scheduler.start(paused=False)
    try:
        await _drain(recorded, expected=1, timeout=6.0)
    finally:
        scheduler.shutdown(wait=False)

    results = {run.result for run in recorded}
    assert JobResult.MAX_INSTANCES in results, f"expected a max_instances skip, saw {results}"


@pytest.mark.asyncio
async def test_a_listener_failure_never_disturbs_the_scheduler(recorded):
    """APScheduler swallows listener exceptions, so a throwing listener would
    leave the scheduler running with no telemetry and nothing to explain it."""
    scheduler = AsyncIOScheduler()
    listener = ScheduledJobMetricsListener()
    listener.register(scheduler)

    ran = asyncio.Event()

    async def job():
        ran.set()

    with patch.object(ScheduledJobMetricsListener, "_publish", side_effect=RuntimeError("metrics down")):
        scheduler.add_job(job, "interval", seconds=60, id="job", next_run_time=__import__("datetime").datetime.now())
        scheduler.start(paused=False)
        try:
            await asyncio.wait_for(ran.wait(), timeout=5.0)
        finally:
            scheduler.shutdown(wait=False)


def test_duration_is_unknown_when_the_submit_event_was_missed():
    """A job already in flight when the listener registers has no start time.
    Reporting zero would put a false value on the duration histogram."""
    from apscheduler.events import EVENT_JOB_EXECUTED, JobExecutionEvent

    listener = ScheduledJobMetricsListener()
    event = JobExecutionEvent(EVENT_JOB_EXECUTED, "orphan_job", "default", None, retval=None)

    run = listener._to_run(event)

    assert run is not None
    assert run.duration_seconds is None
    assert run.result is JobResult.SUCCESS


def test_max_instances_does_not_steal_the_running_job_start_time():
    """APScheduler emits MAX_INSTANCES *instead of* SUBMITTED, while the previous
    run is still going, so popping the start time on that event would drop the
    duration of exactly the overrunning runs this metric exists to surface."""
    from apscheduler.events import (
        EVENT_JOB_EXECUTED,
        EVENT_JOB_MAX_INSTANCES,
        EVENT_JOB_SUBMITTED,
        JobExecutionEvent,
        JobSubmissionEvent,
    )

    # Exactly two reads: the submit, and the completion. A MAX_INSTANCES skip in
    # between must consume neither a tick nor the stored start time.
    ticks = iter([100.0, 103.5])
    listener = ScheduledJobMetricsListener(monotonic=lambda: next(ticks))

    listener._to_run(JobSubmissionEvent(EVENT_JOB_SUBMITTED, "slow_job", "default", [None]))
    skipped = listener._to_run(JobExecutionEvent(EVENT_JOB_MAX_INSTANCES, "slow_job", "default", None))
    finished = listener._to_run(JobExecutionEvent(EVENT_JOB_EXECUTED, "slow_job", "default", None, retval=None))

    assert skipped is not None and skipped.result is JobResult.MAX_INSTANCES
    assert finished is not None and finished.result is JobResult.SUCCESS
    assert finished.duration_seconds == pytest.approx(3.5), (
        "the overrunning run must keep its duration through a MAX_INSTANCES skip"
    )


def test_an_apscheduler_generated_job_id_does_not_become_an_unbounded_label():
    """A caller that omits id= gets uuid4().hex, which as a Prometheus label would
    grow without bound across pods and restarts."""
    from apscheduler.events import EVENT_JOB_EXECUTED, JobExecutionEvent

    listener = ScheduledJobMetricsListener()

    generated = listener._to_run(
        JobExecutionEvent(EVENT_JOB_EXECUTED, "e849e76a882d45ad9dc9965cd1c8a335", "default", None, retval=None)
    )
    pinned = listener._to_run(
        JobExecutionEvent(EVENT_JOB_EXECUTED, "update_spend_job", "default", None, retval=None)
    )

    assert generated is not None and generated.job_name == "unnamed_job"
    assert pinned is not None and pinned.job_name == "update_spend_job"


def test_every_job_litellm_registers_pins_an_explicit_id():
    """The label is only bounded because each add_job passes id=. This fails if a
    new registration forgets one."""
    import re
    from pathlib import Path

    import litellm

    root = Path(litellm.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"scheduler\.add_job\((.*?)\n\s*\)", source, re.DOTALL):
            if "id=" not in match.group(1):
                offenders.append(f"{path.relative_to(root)}: {match.group(1).strip().splitlines()[0]}")

    assert not offenders, "scheduler.add_job without an explicit id=: " + "; ".join(offenders)

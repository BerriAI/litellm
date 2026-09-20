import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import litellm.proxy.shutdown.scheduled_jobs as scheduled_jobs
from litellm.proxy.shutdown.scheduled_jobs import (
    AwaitableAsyncIOExecutor,
    stop_in_flight_scheduler_jobs,
    pause_scheduled_jobs,
)


class _Job:
    """A scheduled job that blocks until cancelled, or for ``work_seconds``, and records what it observed"""

    def __init__(self, swallow_cancellation: bool = False, work_seconds: float | None = None) -> None:
        self.started = asyncio.Event()
        self.events: list[str] = []
        self.swallow_cancellation = swallow_cancellation
        self.work_seconds = work_seconds

    async def run(self) -> None:
        self.started.set()
        try:
            if self.work_seconds is None:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(self.work_seconds)
                self.events.append("committed")
        except asyncio.CancelledError:
            self.events.append("cancelled")
            if self.swallow_cancellation:
                await asyncio.Event().wait()
            raise
        finally:
            self.events.append("finished")


@asynccontextmanager
async def _running_scheduler(*jobs: _Job) -> AsyncIterator[tuple[AsyncIOScheduler, AwaitableAsyncIOExecutor]]:
    """A started scheduler with every job in flight, stopped on the way out whatever the test did"""
    executor = AwaitableAsyncIOExecutor()
    scheduler = AsyncIOScheduler(executors={"default": executor})
    for index, job in enumerate(jobs):
        scheduler.add_job(job.run, id=f"job-{index}", next_run_time=datetime.now())
    scheduler.start()
    try:
        for job in jobs:
            await asyncio.wait_for(job.started.wait(), timeout=5)
        yield scheduler, executor
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)
        stragglers = executor.in_flight_jobs()
        for straggler in stragglers:
            straggler.cancel()
        await asyncio.gather(*stragglers, return_exceptions=True)


@pytest.mark.asyncio
async def test_in_flight_jobs_observe_cancellation_before_shutdown_returns():
    """The job's own CancelledError handler records how a run ended, so shutdown must wait for it"""
    job = _Job()
    async with _running_scheduler(job) as (scheduler, executor):
        await stop_in_flight_scheduler_jobs(scheduler, executor)

        assert job.events == ["cancelled", "finished"]
        assert scheduler.running is False
        assert executor.in_flight_jobs() == ()


@pytest.mark.asyncio
async def test_a_job_that_is_finishing_is_allowed_to_finish_rather_than_cancelled(monkeypatch):
    """A spend write cancelled mid-commit drops the rows it popped, so short jobs get to finish first"""
    monkeypatch.setattr(scheduled_jobs, "JOB_FINISH_TIMEOUT_SECONDS", 2.0)
    write = _Job(work_seconds=0.2)
    stuck = _Job()
    async with _running_scheduler(write, stuck) as (scheduler, executor):
        await stop_in_flight_scheduler_jobs(scheduler, executor)

        assert write.events == ["committed", "finished"]
        assert stuck.events == ["cancelled", "finished"]
        assert scheduler.running is False


@pytest.mark.asyncio
async def test_every_in_flight_job_is_cancelled_not_only_the_first():
    first, second = _Job(), _Job()
    async with _running_scheduler(first, second) as (scheduler, executor):
        await stop_in_flight_scheduler_jobs(scheduler, executor)

    assert first.events == ["cancelled", "finished"]
    assert second.events == ["cancelled", "finished"]


@pytest.mark.asyncio
async def test_a_job_that_ignores_cancellation_is_abandoned_after_the_timeout(monkeypatch, caplog):
    """A job that swallows CancelledError must not hold the pod past its termination grace period"""
    monkeypatch.setattr(scheduled_jobs, "JOB_CANCEL_TIMEOUT_SECONDS", 0.05)
    job = _Job(swallow_cancellation=True)
    async with _running_scheduler(job) as (scheduler, executor):
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            await stop_in_flight_scheduler_jobs(scheduler, executor)

        assert job.events == ["cancelled"]
        assert "1 scheduled job(s) did not finish within 0.05s of cancellation" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_with_nothing_in_flight_still_stops_the_scheduler():
    async with _running_scheduler() as (scheduler, executor):
        await stop_in_flight_scheduler_jobs(scheduler, executor)
        await asyncio.sleep(0)

        assert scheduler.running is False


@pytest.mark.asyncio
async def test_a_scheduler_that_never_started_is_left_alone():
    """The proxy runs without a scheduler when it has no database"""
    executor = AwaitableAsyncIOExecutor()
    scheduler = AsyncIOScheduler(executors={"default": executor})

    await stop_in_flight_scheduler_jobs(scheduler, executor)

    assert scheduler.running is False


@pytest.mark.asyncio
async def test_pausing_stops_new_jobs_from_starting_but_leaves_running_ones_alone():
    """A job due during the shutdown drain would only be cancelled, so it must not start at all"""
    running = _Job()
    async with _running_scheduler(running) as (scheduler, executor):
        late = _Job()
        scheduler.add_job(late.run, id="late", next_run_time=datetime.now() + timedelta(seconds=0.1))

        pause_scheduled_jobs(scheduler)
        await asyncio.sleep(0.3)

        assert late.started.is_set() is False
        assert running.events == []
        assert scheduler.running is True

        await stop_in_flight_scheduler_jobs(scheduler, executor)

    assert running.events == ["cancelled", "finished"]
    assert late.started.is_set() is False


@pytest.mark.asyncio
async def test_pausing_a_scheduler_that_never_started_is_a_no_op():
    scheduler = AsyncIOScheduler(executors={"default": AwaitableAsyncIOExecutor()})

    pause_scheduled_jobs(scheduler)

    assert scheduler.running is False

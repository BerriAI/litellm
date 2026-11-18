import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import psycopg
import pytest
from integration._support.database import scratch_database
from prisma import Prisma
from psycopg.rows import dict_row

from litellm.proxy.db.scheduled_job_runs import (
    JobResult,
    Schedule,
    ScheduledJob,
    ScheduledJobRunner,
    cron_schedule,
    interval_schedule,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_MIGRATION: Final = next(
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath("litellm/proxy-extras/litellm_proxy_extras/migrations")
    .glob("*_scheduled_job_runs/migration.sql")
)
_PODS: Final = 4


@dataclass(frozen=True)
class _Connection:
    db: Prisma


@dataclass(frozen=True)
class _Database:
    url: str
    pods: tuple["PrismaClient", ...]

    def rows(self, where: str = "TRUE") -> list[dict[str, object]]:
        with psycopg.connect(self.url, row_factory=dict_row) as connection:
            return connection.execute(
                f'SELECT * FROM "LiteLLM_ScheduledJobRun" WHERE {where} ORDER BY "created_at", "scheduled_for"'
            ).fetchall()

    def execute(self, statement: str, parameters: tuple[object, ...] = ()) -> None:
        with psycopg.connect(self.url, autocommit=True) as connection:
            connection.execute(statement, parameters)


class _Job:
    """A job whose runs are counted and can be held open until the test releases them."""

    def __init__(self, name: str = "job", lock_group: str = "group", schedule: str = "1h") -> None:
        self.name = name
        self.lock_group = lock_group
        self.schedule_value = schedule
        self.started = 0
        self.release = asyncio.Event()
        self.release.set()
        self.outcome: Callable[[], Awaitable[JobResult]] = self._succeed

    async def _succeed(self) -> JobResult:
        return "SUCCEEDED"

    async def _run(self, _prisma_client: object) -> JobResult:
        self.started += 1
        await self.release.wait()
        return await self.outcome()

    def _schedule(self) -> Schedule:
        parsed: Final = (
            cron_schedule("cron", self.schedule_value)
            if " " in self.schedule_value
            else interval_schedule("interval", self.schedule_value)
        )
        assert isinstance(parsed, Schedule)
        return parsed

    def definition(self) -> ScheduledJob:
        return ScheduledJob(
            name=self.name,
            lock_group=self.lock_group,
            schedule=self._schedule,
            lease=lambda: timedelta(minutes=5),
            run=self._run,
        )


@asynccontextmanager
async def _scratch() -> AsyncIterator[_Database]:
    with scratch_database("scheduled_jobs") as url:
        with psycopg.connect(url, autocommit=True) as setup:
            setup.execute(_MIGRATION.read_text())
        connections: Final = tuple(_Connection(Prisma(datasource={"url": url})) for _ in range(_PODS))
        for connection in connections:
            await connection.db.connect()
        try:
            yield _Database(
                url=url,
                pods=tuple(
                    cast("PrismaClient", connection)  # cast-ok: the runner only reaches the client through .db
                    for connection in connections
                ),
            )
        finally:
            for connection in connections:
                await connection.db.disconnect()


def _runners(job: _Job) -> tuple[ScheduledJobRunner, ...]:
    return tuple(ScheduledJobRunner(job.definition(), worker_id=f"pod-{index}") for index in range(_PODS))


async def _tick_all(runners: tuple[ScheduledJobRunner, ...], database: _Database) -> None:
    await asyncio.gather(*(runner.tick(pod) for runner, pod in zip(runners, database.pods)))


async def _until(condition: Callable[[], bool], seconds: float = 10) -> None:
    deadline: Final = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


async def test_every_pod_ticks_but_exactly_one_runs_and_the_next_run_is_planned_one_interval_after_it_finished() -> (
    None
):
    async with _scratch() as database:
        job: Final = _Job()
        job.release.clear()
        runners: Final = _runners(job)
        ticks: Final = asyncio.gather(*(runner.tick(pod) for runner, pod in zip(runners, database.pods)))
        await _until(lambda: job.started == 1)
        await asyncio.sleep(0.5)
        running: Final = database.rows("\"status\" = 'RUNNING'")
        assert job.started == 1
        assert len(running) == 1
        job.release.set()
        await ticks

        finished: Final = database.rows("\"status\" = 'SUCCEEDED'")
        pending: Final = database.rows("\"status\" = 'PENDING'")
        assert len(finished) == 1
        assert finished[0]["claimed_by"] == running[0]["claimed_by"]
        assert len(pending) == 1
        assert pending[0]["scheduled_for"] == finished[0]["finished_at"] + timedelta(hours=1)

        await _tick_all(runners, database)
        assert job.started == 1, "a run was started before the next occurrence was due"
        assert len(database.rows("\"status\" = 'PENDING'")) == 1


async def test_jobs_in_one_lock_group_never_run_at_the_same_time_on_different_pods() -> None:
    async with _scratch() as database:
        first: Final = _Job(name="first")
        second: Final = _Job(name="second")
        first.release.clear()
        first_runner: Final = ScheduledJobRunner(first.definition(), worker_id="pod-a")
        second_runner: Final = ScheduledJobRunner(second.definition(), worker_id="pod-b")
        first_tick: Final = asyncio.create_task(first_runner.tick(database.pods[0]))
        await _until(lambda: first.started == 1)

        await second_runner.tick(database.pods[1])
        assert second.started == 0
        assert [row["status"] for row in database.rows("\"job_name\" = 'second'")] == ["PENDING"]

        first.release.set()
        await first_tick
        await second_runner.tick(database.pods[1])
        assert second.started == 1


async def test_a_run_that_raises_is_recorded_as_failed_and_the_next_run_is_still_planned() -> None:
    async with _scratch() as database:
        job: Final = _Job()

        async def explode() -> JobResult:
            raise RuntimeError("disk on fire")

        job.outcome = explode
        await _tick_all(_runners(job), database)

        failed: Final = database.rows("\"status\" = 'FAILED'")
        assert len(failed) == 1
        assert len(database.rows("\"status\" = 'PENDING'")) == 1


async def test_a_run_held_by_a_dead_pod_is_failed_once_its_lease_expires_and_the_schedule_carries_on() -> None:
    async with _scratch() as database:
        database.execute(
            'INSERT INTO "LiteLLM_ScheduledJobRun" ("id", "job_name", "lock_group", "schedule", "scheduled_for", '
            '"status", "claimed_by", "claimed_at", "lease_expires_at") '
            "VALUES ('dead', 'job', 'group', 'interval:3600s', now() - interval '2 hours', 'RUNNING', 'dead-pod', "
            "now() - interval '2 hours', now() - interval '1 second')"
        )
        job: Final = _Job()
        await _tick_all(_runners(job), database)

        (reaped,) = database.rows("\"id\" = 'dead'")
        assert reaped["status"] == "FAILED"
        (pending,) = database.rows("\"status\" = 'PENDING'")
        assert pending["scheduled_for"] == reaped["finished_at"] + timedelta(hours=1)
        assert job.started == 0


async def test_a_run_held_by_a_live_pod_is_left_alone_and_nothing_else_starts() -> None:
    async with _scratch() as database:
        job: Final = _Job()
        job.release.clear()
        runners: Final = _runners(job)
        holder: Final = asyncio.create_task(runners[0].tick(database.pods[0]))
        await _until(lambda: job.started == 1)

        await asyncio.gather(*(runner.tick(pod) for runner, pod in zip(runners[1:], database.pods[1:])))
        assert [row["status"] for row in database.rows()] == ["RUNNING"]

        job.release.set()
        await holder
        assert job.started == 1


async def test_an_interval_already_overdue_when_the_pods_restart_runs_on_the_first_tick() -> None:
    async with _scratch() as database:
        database.execute(
            'INSERT INTO "LiteLLM_ScheduledJobRun" ("id", "job_name", "lock_group", "schedule", "scheduled_for", '
            '"status", "finished_at") '
            "VALUES ('yesterday', 'job', 'group', 'interval:86400s', now() - interval '2 days', 'SUCCEEDED', "
            "now() - interval '2 days')"
        )
        job: Final = _Job(schedule="1d")
        await _tick_all(_runners(job), database)

        assert job.started == 1
        assert [row["status"] for row in database.rows("\"id\" <> 'yesterday'")] == ["SUCCEEDED", "PENDING"]


async def test_a_schedule_change_replaces_the_pending_run_planned_from_the_old_schedule() -> None:
    async with _scratch() as database:
        job: Final = _Job(schedule="0 4 * * *")
        runners: Final = _runners(job)
        await _tick_all(runners, database)
        (old,) = database.rows("\"status\" = 'PENDING'")
        assert old["scheduled_for"].astimezone(timezone.utc).hour == 4

        job.schedule_value = "0 5 * * *"
        await _tick_all(runners, database)

        assert database.rows(f"\"id\" = '{old['id']}'")[0]["status"] == "CANCELLED"
        (replacement,) = database.rows("\"status\" = 'PENDING'")
        assert replacement["schedule"] == "cron:0 5 * * *"
        assert replacement["scheduled_for"].astimezone(timezone.utc).hour == 5
        assert replacement["scheduled_for"] > datetime.now(timezone.utc)
        assert job.started == 0


async def test_moving_a_cron_earlier_plans_the_earlier_occurrence_instead_of_skipping_a_day() -> None:
    async with _scratch() as database:
        now: Final = datetime.now(timezone.utc)
        job: Final = _Job(schedule=f"0 {(now + timedelta(hours=3)).hour} * * *")
        runners: Final = _runners(job)
        await _tick_all(runners, database)
        (old,) = database.rows("\"status\" = 'PENDING'")

        job.schedule_value = f"0 {(now + timedelta(hours=2)).hour} * * *"
        await _tick_all(runners, database)

        (replacement,) = database.rows("\"status\" = 'PENDING'")
        assert replacement["scheduled_for"] < old["scheduled_for"]
        assert replacement["scheduled_for"] - datetime.now(timezone.utc) < timedelta(hours=2)


async def test_a_run_cancelled_by_shutdown_is_recorded_as_failed_instead_of_holding_its_lease() -> None:
    async with _scratch() as database:
        job: Final = _Job()
        job.release.clear()
        runner: Final = ScheduledJobRunner(job.definition(), worker_id="pod-shutting-down")
        tick: Final = asyncio.create_task(runner.tick(database.pods[0]))
        await _until(lambda: job.started == 1)

        tick.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tick

        (cancelled,) = database.rows()
        assert cancelled["status"] == "FAILED"

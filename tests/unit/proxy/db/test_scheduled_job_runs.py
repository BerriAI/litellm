import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, cast

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from litellm.proxy.db.scheduled_job_runs import (
    InvalidSchedule,
    JobResult,
    JobScheduler,
    JobState,
    NewRun,
    PendingRun,
    Schedule,
    ScheduledJob,
    ScheduledJobRunner,
    TickPlan,
    cron_schedule,
    interval_schedule,
    next_run_at,
    plan_tick,
    register_poll,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

NOW: Final = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _schedule(value: str) -> Schedule:
    parsed: Final = cron_schedule("cron", value) if " " in value else interval_schedule("interval", value)
    assert not isinstance(parsed, InvalidSchedule)
    return parsed


def _state(
    *,
    group_running: bool = False,
    job_running: bool = False,
    last_finished_at: datetime | None = None,
    last_scheduled_for: datetime | None = None,
) -> JobState:
    return JobState(
        now=NOW,
        group_running=group_running,
        job_running=job_running,
        last_finished_at=last_finished_at,
        last_scheduled_for=last_scheduled_for,
    )


def _pending(run_id: str, schedule: str, scheduled_for: datetime) -> PendingRun:
    return PendingRun(id=run_id, schedule=schedule, scheduled_for=scheduled_for)


@pytest.mark.parametrize(
    ("schedule", "state", "expected"),
    [
        pytest.param("1h", _state(), NOW, id="interval-never-run-is-due-now"),
        pytest.param(
            "1h",
            _state(last_finished_at=NOW - timedelta(hours=3)),
            NOW - timedelta(hours=2),
            id="interval-counts-from-last-finish-so-a-restart-cannot-postpone-it",
        ),
        pytest.param(
            "1h",
            _state(last_finished_at=NOW - timedelta(minutes=10)),
            NOW + timedelta(minutes=50),
            id="interval-not-yet-elapsed",
        ),
        pytest.param("0 4 * * *", _state(), datetime(2026, 11, 5, 4, tzinfo=timezone.utc), id="cron-next-fire"),
        pytest.param(
            "0 12 * * *",
            _state(last_scheduled_for=NOW),
            datetime(2026, 11, 5, 12, tzinfo=timezone.utc),
            id="cron-never-replans-the-occurrence-it-just-had",
        ),
        pytest.param(
            "0 4 * * *",
            _state(last_scheduled_for=NOW - timedelta(days=3)),
            datetime(2026, 11, 5, 4, tzinfo=timezone.utc),
            id="cron-outage-collapses-to-one-next-fire",
        ),
    ],
)
def test_next_run_at(schedule: str, state: JobState, expected: datetime) -> None:
    assert next_run_at(_schedule(schedule), state) == expected


DUE: Final = NOW - timedelta(minutes=1)
LATER: Final = NOW + timedelta(hours=1)


@pytest.mark.parametrize(
    ("schedule", "state", "pending", "expected"),
    [
        pytest.param(
            "1h",
            _state(),
            (),
            TickPlan(cancel=(), insert=NewRun(id="new", schedule="interval:3600s", scheduled_for=NOW), claim="new"),
            id="first-ever-tick-plans-and-claims",
        ),
        pytest.param(
            "1h",
            _state(),
            (_pending("a", "interval:3600s", DUE),),
            TickPlan(cancel=(), insert=None, claim="a"),
            id="due-pending-is-claimed",
        ),
        pytest.param(
            "1h",
            _state(),
            (_pending("a", "interval:3600s", LATER),),
            TickPlan(cancel=(), insert=None, claim=None),
            id="future-pending-waits",
        ),
        pytest.param(
            "1h",
            _state(group_running=True),
            (_pending("a", "interval:3600s", DUE),),
            TickPlan(cancel=(), insert=None, claim=None),
            id="due-pending-waits-while-its-lock-group-runs",
        ),
        pytest.param(
            "1h",
            _state(group_running=True),
            (),
            TickPlan(cancel=(), insert=NewRun(id="new", schedule="interval:3600s", scheduled_for=NOW), claim=None),
            id="a-sibling-run-still-lets-the-next-run-be-planned",
        ),
        pytest.param(
            "1h",
            _state(group_running=True, job_running=True),
            (),
            TickPlan(cancel=(), insert=None, claim=None),
            id="nothing-is-planned-while-the-job-itself-runs",
        ),
        pytest.param(
            "2h",
            _state(last_finished_at=NOW - timedelta(minutes=30)),
            (_pending("old", "interval:3600s", NOW + timedelta(minutes=30)),),
            TickPlan(
                cancel=("old",),
                insert=NewRun(id="new", schedule="interval:7200s", scheduled_for=NOW + timedelta(minutes=90)),
                claim=None,
            ),
            id="schedule-change-replaces-the-pending-run",
        ),
        pytest.param(
            "1h",
            _state(),
            (_pending("a", "interval:3600s", DUE), _pending("b", "interval:3600s", LATER)),
            TickPlan(cancel=("b",), insert=None, claim="a"),
            id="a-duplicate-pending-run-is-cancelled",
        ),
    ],
)
def test_plan_tick(schedule: str, state: JobState, pending: tuple[PendingRun, ...], expected: TickPlan) -> None:
    assert plan_tick(_schedule(schedule), state, pending, "new") == expected


def test_an_invalid_schedule_replans_nothing_but_still_runs_what_the_last_valid_one_planned() -> None:
    pending: Final = (_pending("a", "interval:3600s", DUE),)
    plan: Final = plan_tick(InvalidSchedule("bad cron"), _state(), pending, "new")
    assert plan == TickPlan(cancel=(), insert=None, claim="a")


@pytest.mark.parametrize("value", ["not a cron", "0 0 * * * *", "0 0 31 2 *", "0 0 30 2 *", 5, None])
def test_cron_schedule_rejects_what_is_not_a_crontab(value: object) -> None:
    assert isinstance(cron_schedule("maximum_spend_logs_cleanup_cron", value), InvalidSchedule)


@pytest.mark.parametrize("value", ["soon", "99999999999d", "0s", 86400, None])
def test_interval_schedule_rejects_what_is_not_a_positive_duration(value: object) -> None:
    assert isinstance(interval_schedule("maximum_spend_logs_retention_interval", value), InvalidSchedule)


class _ScriptedDatabase:
    """Answers the runner's statements the way Postgres would for one due run, and records every write."""

    def __init__(self, *, locked: bool = True, finish_updates: int = 1, broken: bool = False) -> None:
        self.locked = locked
        self.finish_updates = finish_updates
        self.broken = broken
        self.writes: tuple[tuple[str, tuple[object, ...]], ...] = ()
        self.db = self

    @asynccontextmanager
    async def tx(self, timeout: timedelta) -> AsyncIterator["_ScriptedDatabase"]:
        if self.broken:
            raise ConnectionError("database unreachable")
        yield self

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        if "pg_try_advisory_xact_lock" in sql:
            return [{"locked": self.locked}]
        if "SELECT now()" in sql:
            return [
                {
                    "now": NOW,
                    "group_running": False,
                    "job_running": False,
                    "last_finished_at": None,
                    "last_scheduled_for": None,
                }
            ]
        if "'RUNNING', \"claimed_by\"" in sql:
            self.writes = (*self.writes, ("claim", args))
            return [{"id": args[0], "job_name": "job"}]
        return []

    async def execute_raw(self, sql: str, *args: object) -> int:
        if sql.lstrip().startswith("SET LOCAL"):
            return 0
        if 'SET "status" = $3' in sql:
            self.writes = (*self.writes, ("finish", args))
            self.locked = False
            return self.finish_updates
        if sql.lstrip().startswith("INSERT"):
            self.writes = (*self.writes, ("insert", args))
        return 1

    def client(self) -> "PrismaClient":
        return cast("PrismaClient", self)  # cast-ok: the runner only reaches the client through .db


def _finishes(database: _ScriptedDatabase) -> tuple[tuple[object, ...], ...]:
    return tuple(args for kind, args in database.writes if kind == "finish")


class _Job:
    def __init__(self, result: JobResult | BaseException = "SUCCEEDED") -> None:
        self.result = result
        self.runs = 0
        self.release = asyncio.Event()
        self.release.set()

    async def run(self, _prisma_client: object) -> JobResult:
        self.runs += 1
        await self.release.wait()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def runner(self) -> ScheduledJobRunner:
        definition: Final = ScheduledJob(
            name="job",
            lock_group="group",
            schedule=lambda: _schedule("1h"),
            lease=lambda: timedelta(minutes=5),
            run=self.run,
        )
        return ScheduledJobRunner(definition, worker_id="pod-a", new_id=lambda: "run-1")


async def test_a_due_run_is_claimed_by_this_pod_run_and_recorded_as_succeeded() -> None:
    database: Final = _ScriptedDatabase()
    job: Final = _Job("SUCCEEDED")
    await job.runner().tick(database.client())

    assert job.runs == 1
    assert ("claim", ("run-1", "pod-a", 300.0)) in database.writes
    assert _finishes(database) == (("run-1", "pod-a", "SUCCEEDED"),)


@pytest.mark.parametrize(
    "result",
    [pytest.param("FAILED", id="reported"), pytest.param(RuntimeError("disk on fire"), id="raised")],
)
async def test_a_failed_run_is_recorded_as_failed(result: JobResult | BaseException) -> None:
    database: Final = _ScriptedDatabase()
    await _Job(result).runner().tick(database.client())
    assert _finishes(database) == (("run-1", "pod-a", "FAILED"),)


async def test_a_pod_that_does_not_get_the_group_lock_neither_plans_nor_runs() -> None:
    database: Final = _ScriptedDatabase(locked=False)
    job: Final = _Job()
    await job.runner().tick(database.client())
    assert job.runs == 0
    assert database.writes == ()


async def test_an_unreachable_database_skips_the_tick_instead_of_raising_into_the_scheduler() -> None:
    job: Final = _Job()
    await job.runner().tick(_ScriptedDatabase(broken=True).client())
    assert job.runs == 0


async def test_a_poll_landing_while_this_pod_still_runs_the_job_does_not_touch_the_database() -> None:
    database: Final = _ScriptedDatabase()
    job: Final = _Job()
    job.release.clear()
    runner: Final = job.runner()
    first: Final = asyncio.create_task(runner.tick(database.client()))
    while job.runs == 0:
        await asyncio.sleep(0)
    writes_during_run: Final = database.writes

    await runner.tick(database.client())
    assert database.writes == writes_during_run

    job.release.set()
    await first
    assert job.runs == 1


async def test_a_run_cancelled_mid_flight_records_failed_and_still_cancels() -> None:
    database: Final = _ScriptedDatabase()
    job: Final = _Job()
    job.release.clear()
    tick: Final = asyncio.create_task(job.runner().tick(database.client()))
    while job.runs == 0:
        await asyncio.sleep(0)
    tick.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tick
    assert _finishes(database) == (("run-1", "pod-a", "FAILED"),)


async def test_a_result_that_arrives_after_the_lease_expired_is_reported_not_recorded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    database: Final = _ScriptedDatabase(finish_updates=0)
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _Job().runner().tick(database.client())
    assert any("after its lease expired" in record.getMessage() for record in caplog.records)


async def test_register_poll_adds_a_wanted_poll_once_and_removes_it_when_no_longer_wanted() -> None:
    scheduler: Final = AsyncIOScheduler()
    scheduler.start(paused=True)
    job_scheduler: Final = cast(JobScheduler, scheduler)  # cast-ok: apscheduler ships no type information
    definition: Final = _Job().runner().job

    def register(wanted: bool) -> None:
        register_poll(job_scheduler, None, "poll-id", definition, wanted=wanted, poll_seconds=30, misfire_grace_time=60)

    try:
        register(True)
        first: Final = scheduler.get_job("poll-id")
        assert first is not None
        register(True)
        assert scheduler.get_job("poll-id") is first, "re-registering replaced the live poll and its runner"
        register(False)
        assert scheduler.get_job("poll-id") is None
    finally:
        scheduler.shutdown(wait=False)

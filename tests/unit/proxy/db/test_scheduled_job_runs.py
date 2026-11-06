from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.db.scheduled_job_runs import (
    InvalidSchedule,
    JobState,
    NewRun,
    PendingRun,
    Schedule,
    TickPlan,
    cron_schedule,
    interval_schedule,
    next_run_at,
    plan_tick,
)

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


@pytest.mark.parametrize("value", ["not a cron", "0 0 * * * *", 5, None])
def test_cron_schedule_rejects_what_is_not_a_crontab(value: object) -> None:
    assert isinstance(cron_schedule("maximum_spend_logs_cleanup_cron", value), InvalidSchedule)


@pytest.mark.parametrize("value", ["soon", "99999999999d", "0s", 86400, None])
def test_interval_schedule_rejects_what_is_not_a_positive_duration(value: object) -> None:
    assert isinstance(interval_schedule("maximum_spend_logs_retention_interval", value), InvalidSchedule)

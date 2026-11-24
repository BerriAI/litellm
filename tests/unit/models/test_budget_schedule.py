from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.models.budget import BudgetLimitList, TeamBudgetLimitList
from litellm.models.budget_schedule import BudgetSchedule, resolve_schedules_at, scheduled_period, window_counter_name

BERLIN: Final = "Europe/Berlin"
WEEKDAYS: Final = ("mon", "tue", "wed", "thu", "fri")


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _team_limits(*windows: dict[str, object]) -> list[object]:
    return list(TypeAdapter(TeamBudgetLimitList).validate_python(list(windows)))


WORKDAY_MORNING: Final = BudgetSchedule(timezone=BERLIN, days=WEEKDAYS, start="08:00", end="13:00")
WORKDAY_AFTERNOON: Final = BudgetSchedule(timezone=BERLIN, days=WEEKDAYS, start="13:00", end="18:00")
WORKDAY_NIGHT: Final = BudgetSchedule(timezone=BERLIN, days=WEEKDAYS, start="18:00", end="08:00")
WEEKEND_DAY: Final = BudgetSchedule(timezone=BERLIN, days=("sat", "sun"), start="08:00", end="08:00")


@pytest.mark.parametrize(
    "schedule, at, period",
    [
        (WORKDAY_MORNING, "2026-11-25T07:00:00", ("2026-11-25T07:00:00", "2026-11-25T12:00:00")),
        (WORKDAY_MORNING, "2026-11-25T11:59:59", ("2026-11-25T07:00:00", "2026-11-25T12:00:00")),
        (WORKDAY_MORNING, "2026-11-25T12:00:00", None),
        (WORKDAY_AFTERNOON, "2026-11-25T12:00:00", ("2026-11-25T12:00:00", "2026-11-25T17:00:00")),
        (WORKDAY_NIGHT, "2026-11-25T23:30:00", ("2026-11-25T17:00:00", "2026-11-26T07:00:00")),
        (WORKDAY_NIGHT, "2026-11-26T06:59:00", ("2026-11-25T17:00:00", "2026-11-26T07:00:00")),
        (WORKDAY_NIGHT, "2026-11-28T06:00:00", ("2026-11-27T17:00:00", "2026-11-28T07:00:00")),
        (WORKDAY_NIGHT, "2026-11-28T08:00:00", None),
        (WORKDAY_MORNING, "2026-11-28T08:00:00", None),
        (WEEKEND_DAY, "2026-11-28T07:00:00", ("2026-11-28T07:00:00", "2026-11-29T07:00:00")),
        (WEEKEND_DAY, "2026-11-29T07:00:00", ("2026-11-29T07:00:00", "2026-11-30T07:00:00")),
        (WEEKEND_DAY, "2026-11-30T07:00:00", None),
    ],
    ids=[
        "morning_starts_inclusive",
        "morning_last_second",
        "morning_ends_exclusive",
        "afternoon_starts_at_morning_end",
        "night_before_midnight",
        "night_after_midnight_is_same_period",
        "friday_night_runs_into_saturday",
        "no_workday_night_on_saturday",
        "no_workday_morning_on_saturday",
        "weekend_saturday_full_day",
        "weekend_sunday_is_a_fresh_period",
        "weekend_ends_monday_morning",
    ],
)
def test_period_at_resolves_local_wall_clock_periods(
    schedule: BudgetSchedule, at: str, period: tuple[str, str] | None
) -> None:
    expected: Final = None if period is None else (_utc(period[0]), _utc(period[1]))
    assert schedule.period_at(_utc(at)) == expected


@pytest.mark.parametrize(
    "at, hours",
    [("2026-03-29T03:00:00", 13), ("2026-10-25T03:00:00", 15)],
    ids=["spring_forward_shortens_the_night", "fall_back_lengthens_the_night"],
)
def test_dst_shift_lands_on_the_overnight_period(at: str, hours: int) -> None:
    night: Final = BudgetSchedule(timezone=BERLIN, start="18:00", end="08:00")
    period: Final = night.period_at(_utc(at))
    assert period is not None
    start, end = period
    assert (end - start).total_seconds() == hours * 3600


def test_daytime_boundaries_keep_their_local_time_across_dst() -> None:
    morning: Final = BudgetSchedule(timezone=BERLIN, start="08:00", end="13:00")
    before: Final = morning.period_at(_utc("2026-03-28T07:30:00"))
    after: Final = morning.period_at(_utc("2026-03-30T06:30:00"))
    assert before == (_utc("2026-03-28T07:00:00"), _utc("2026-03-28T12:00:00"))
    assert after == (_utc("2026-03-30T06:00:00"), _utc("2026-03-30T11:00:00"))


def test_next_start_skips_days_outside_the_schedule() -> None:
    assert WORKDAY_MORNING.next_start(_utc("2026-11-27T12:00:00")) == _utc("2026-11-30T07:00:00")


def test_scheduled_window_derives_a_stable_budget_duration() -> None:
    first, second = _team_limits(
        {"max_budget": 10, "schedule": {"timezone": BERLIN, "days": ["fri", "mon"], "start": "08:00", "end": "13:00"}},
        {"max_budget": 20, "budget_duration": "7d"},
    )
    assert first.budget_duration == "sched:Europe/Berlin:mon,fri:08:00-13:00"
    assert BudgetSchedule.from_label(first.budget_duration) == first.schedule
    assert second.budget_duration == "7d"
    assert _team_limits(first.model_dump(mode="json"))[0] == first


def test_full_workday_and_weekend_schedule_is_accepted() -> None:
    limits: Final = _team_limits(
        *(
            {"max_budget": budget, "schedule": schedule.model_dump(mode="json")}
            for budget, schedule in (
                (10, WORKDAY_MORNING),
                (10, WORKDAY_AFTERNOON),
                (20, WORKDAY_NIGHT),
                (30, WEEKEND_DAY),
            )
        )
    )
    assert len({limit.budget_duration for limit in limits}) == 4


@pytest.mark.parametrize(
    "windows, message",
    [
        (
            [
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "start": "08:00", "end": "13:00"}},
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "days": ["sat"], "start": "12:00", "end": "14:00"}},
            ],
            "overlap",
        ),
        (
            [
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "days": ["fri"], "start": "18:00", "end": "08:00"}},
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "days": ["sat"], "start": "07:00", "end": "09:00"}},
            ],
            "overlap",
        ),
        (
            [
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "days": ["sun"], "start": "18:00", "end": "08:00"}},
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "days": ["mon"], "start": "07:00", "end": "09:00"}},
            ],
            "overlap",
        ),
        (
            [
                {"max_budget": 1, "schedule": {"timezone": BERLIN, "start": "08:00", "end": "13:00"}},
                {"max_budget": 1, "schedule": {"timezone": "UTC", "start": "13:00", "end": "18:00"}},
            ],
            "same timezone",
        ),
        (
            [{"max_budget": 1, "schedule": {"timezone": "Mars/Base", "start": "08:00", "end": "13:00"}}],
            "Unknown timezone",
        ),
        (
            [{"max_budget": 1, "schedule": {"timezone": BERLIN, "days": [], "start": "08:00", "end": "13:00"}}],
            "weekday",
        ),
        (
            [
                {
                    "max_budget": 1,
                    "schedule": {"timezone": BERLIN, "days": ["mon", "mon"], "start": "08:00", "end": "13:00"},
                }
            ],
            "weekday",
        ),
        ([{"max_budget": 1, "schedule": {"timezone": BERLIN, "start": "08:00:30", "end": "13:00"}}], "HH:MM"),
        (
            [
                {
                    "max_budget": 1,
                    "budget_duration": "1d",
                    "schedule": {"timezone": BERLIN, "start": "08:00", "end": "13:00"},
                }
            ],
            "derived from schedule",
        ),
    ],
    ids=[
        "same_hours_on_overlapping_days",
        "friday_night_overlaps_saturday_morning",
        "sunday_night_wraps_into_monday_morning",
        "mixed_timezones",
        "unknown_timezone",
        "no_days",
        "repeated_day",
        "seconds_in_time",
        "conflicting_budget_duration",
    ],
)
def test_ambiguous_or_invalid_schedules_are_rejected(windows: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _team_limits(*windows)


def test_schedules_are_rejected_outside_teams() -> None:
    with pytest.raises(ValidationError, match="only on teams"):
        TypeAdapter(BudgetLimitList).validate_python(
            [{"max_budget": 1, "schedule": {"timezone": BERLIN, "start": "08:00", "end": "13:00"}}]
        )


def test_each_period_gets_its_own_counter_name() -> None:
    label: Final = WORKDAY_MORNING.label()
    monday: Final = scheduled_period(label, _utc("2026-11-23T09:00:00"))
    tuesday: Final = scheduled_period(label, _utc("2026-11-24T09:00:00"))
    assert monday is not None and tuesday is not None
    assert window_counter_name(label, monday[0]) != window_counter_name(label, tuesday[0])
    assert window_counter_name(label, monday[0]) == window_counter_name(label, _utc("2026-11-23T07:00:00"))
    assert window_counter_name("7d", monday[0]) == "7d"


def test_charged_at_reads_naive_times_as_local_and_restores_the_clock() -> None:
    from litellm.models.budget_schedule import schedule_clock

    naive_local: Final = datetime(2026, 11, 25, 12, 0)
    with resolve_schedules_at(naive_local):
        assert schedule_clock() == naive_local.astimezone(timezone.utc)
    assert schedule_clock() != naive_local.astimezone(timezone.utc)

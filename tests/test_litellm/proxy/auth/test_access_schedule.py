"""Unit tests for recurring access-schedule evaluation on virtual keys."""

from datetime import datetime, timezone

import pytest

from litellm.proxy.auth.access_schedule import (
    AccessAllowed,
    AccessDenied,
    ScheduleAbsent,
    ScheduleInvalid,
    ScheduleValid,
    evaluate_access_schedule,
    is_within_schedule,
    parse_access_schedule,
)

WORKDAY_SCHEDULE = {
    "access_schedule": {
        "timezone": "Europe/Berlin",
        "windows": [
            {"days": ["mon", "tue", "wed", "thu", "fri"], "start": "09:00", "end": "18:00"}
        ],
    }
}


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_absent_when_no_permissions():
    assert isinstance(parse_access_schedule(None), ScheduleAbsent)
    assert isinstance(parse_access_schedule({}), ScheduleAbsent)
    assert isinstance(parse_access_schedule({"pii": False}), ScheduleAbsent)


def test_valid_schedule_parses():
    parsed = parse_access_schedule(WORKDAY_SCHEDULE)
    assert isinstance(parsed, ScheduleValid)
    assert parsed.schedule.timezone == "Europe/Berlin"
    assert len(parsed.schedule.windows) == 1


@pytest.mark.parametrize(
    "raw",
    [
        {"timezone": "Europe/Berlin"},
        {"timezone": "Europe/Berlin", "windows": []},
        {"timezone": "Not/AZone", "windows": [{"days": ["mon"], "start": "09:00", "end": "18:00"}]},
        {"timezone": "UTC", "windows": [{"days": [], "start": "09:00", "end": "18:00"}]},
        {"timezone": "UTC", "windows": [{"days": ["funday"], "start": "09:00", "end": "18:00"}]},
        {"timezone": "UTC", "windows": [{"days": ["mon"], "start": "9am", "end": "18:00"}]},
        {"timezone": "UTC", "windows": [{"days": ["mon"], "start": "09:00", "end": "18:00", "x": 1}]},
        {"timezone": "UTC", "windows": [{"days": ["mon"], "start": "09:00", "end": "09:00"}]},
        [{"days": ["mon"], "start": "09:00", "end": "18:00"}],
    ],
)
def test_invalid_schedules_are_rejected(raw):
    assert isinstance(parse_access_schedule({"access_schedule": raw}), ScheduleInvalid)


def test_within_window_allows():
    now = _utc(2026, 7, 24, 10)  # Fri 12:00 Berlin
    assert evaluate_access_schedule(WORKDAY_SCHEDULE, now) == AccessAllowed()


def test_start_is_inclusive():
    now = _utc(2026, 7, 24, 7)  # Fri 09:00 Berlin exactly
    assert isinstance(evaluate_access_schedule(WORKDAY_SCHEDULE, now), AccessAllowed)


def test_end_is_exclusive():
    now = _utc(2026, 7, 24, 16)  # Fri 18:00 Berlin exactly
    assert isinstance(evaluate_access_schedule(WORKDAY_SCHEDULE, now), AccessDenied)


def test_after_hours_denies():
    now = _utc(2026, 7, 24, 18)  # Fri 20:00 Berlin
    decision = evaluate_access_schedule(WORKDAY_SCHEDULE, now)
    assert isinstance(decision, AccessDenied)
    assert "access_schedule" in decision.reason


def test_before_hours_denies():
    now = _utc(2026, 7, 24, 6)  # Fri 08:00 Berlin
    assert isinstance(evaluate_access_schedule(WORKDAY_SCHEDULE, now), AccessDenied)


def test_weekend_denies():
    now = _utc(2026, 7, 26, 10)  # Sun 12:00 Berlin
    assert isinstance(evaluate_access_schedule(WORKDAY_SCHEDULE, now), AccessDenied)


def test_timezone_is_respected():
    schedule = {
        "access_schedule": {
            "timezone": "America/New_York",
            "windows": [{"days": ["fri"], "start": "09:00", "end": "18:00"}],
        }
    }
    # Fri 20:00 UTC == Fri 16:00 New York (inside)
    assert isinstance(evaluate_access_schedule(schedule, _utc(2026, 7, 24, 20)), AccessAllowed)
    # Fri 08:00 UTC == Fri 04:00 New York (before window)
    assert isinstance(evaluate_access_schedule(schedule, _utc(2026, 7, 24, 8)), AccessDenied)


OVERNIGHT_SCHEDULE = {
    "access_schedule": {
        "timezone": "UTC",
        "windows": [{"days": ["fri"], "start": "22:00", "end": "06:00"}],
    }
}


def test_overnight_open_on_start_day_evening():
    assert isinstance(evaluate_access_schedule(OVERNIGHT_SCHEDULE, _utc(2026, 7, 24, 23)), AccessAllowed)


def test_overnight_spills_into_next_morning():
    assert isinstance(evaluate_access_schedule(OVERNIGHT_SCHEDULE, _utc(2026, 7, 25, 5)), AccessAllowed)


def test_overnight_closed_after_end_next_morning():
    assert isinstance(evaluate_access_schedule(OVERNIGHT_SCHEDULE, _utc(2026, 7, 25, 7)), AccessDenied)


def test_overnight_closed_before_start_on_start_day():
    assert isinstance(evaluate_access_schedule(OVERNIGHT_SCHEDULE, _utc(2026, 7, 24, 21)), AccessDenied)


def test_overnight_does_not_open_on_non_start_evening():
    # Saturday 23:00 is not covered: only Friday evenings start the window
    assert isinstance(evaluate_access_schedule(OVERNIGHT_SCHEDULE, _utc(2026, 7, 25, 23)), AccessDenied)


def test_multiple_windows_any_match_allows():
    schedule = {
        "access_schedule": {
            "timezone": "UTC",
            "windows": [
                {"days": ["mon"], "start": "09:00", "end": "12:00"},
                {"days": ["mon"], "start": "13:00", "end": "17:00"},
            ],
        }
    }
    # Monday 2026-07-20
    assert isinstance(evaluate_access_schedule(schedule, _utc(2026, 7, 20, 10)), AccessAllowed)
    assert isinstance(evaluate_access_schedule(schedule, _utc(2026, 7, 20, 14)), AccessAllowed)
    # lunch gap 12:00-13:00 is denied
    assert isinstance(evaluate_access_schedule(schedule, _utc(2026, 7, 20, 12, 30)), AccessDenied)


def test_absent_schedule_allows():
    assert isinstance(evaluate_access_schedule({}, _utc(2026, 7, 26, 3)), AccessAllowed)


def test_invalid_persisted_schedule_fails_closed():
    decision = evaluate_access_schedule(
        {"access_schedule": {"timezone": "Not/AZone", "windows": []}},
        _utc(2026, 7, 24, 10),
    )
    assert isinstance(decision, AccessDenied)
    assert "fail closed" in decision.reason


def test_is_within_schedule_direct():
    parsed = parse_access_schedule(OVERNIGHT_SCHEDULE)
    assert isinstance(parsed, ScheduleValid)
    assert is_within_schedule(parsed.schedule, _utc(2026, 7, 24, 23)) is True
    assert is_within_schedule(parsed.schedule, _utc(2026, 7, 24, 21)) is False

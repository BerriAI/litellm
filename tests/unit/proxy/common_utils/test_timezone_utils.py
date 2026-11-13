from datetime import datetime, time, timezone
from typing import Final
from zoneinfo import ZoneInfo

import pytest


import litellm
from litellm.proxy.common_utils.timezone_utils import (
    BudgetResetSettings,
    compute_budget_reset_at,
    compute_budget_window_start,
    get_budget_reset_settings,
    get_budget_reset_time,
    get_budget_reset_timezone,
    parse_budget_reset_time,
)


@pytest.mark.parametrize(
    "duration,zone,reset_at,expected",
    (
        ("12mo", "UTC", "2029-01-01T00:00:00Z", "2028-01-01T00:00:00Z"),
        ("12mo", "UTC", "2101-01-01T00:00:00Z", "2100-01-01T00:00:00Z"),
        ("1mo", "America/New_York", "2026-12-01T00:00:00-05:00", "2026-11-01T04:00:00Z"),
        ("30d", "America/New_York", "2026-04-01T00:00:00-04:00", "2026-03-01T05:00:00Z"),
        ("1mo", "Australia/Sydney", "2026-11-01T00:00:00+11:00", "2026-09-30T14:00:00Z"),
        ("1mo", "Australia/Lord_Howe", "2026-05-01T00:00:00+10:30", "2026-03-31T13:00:00Z"),
        ("12mo", "Asia/Kathmandu", "2029-01-01T12:30:00+05:45", "2028-01-01T06:45:00Z"),
        ("1mo", "America/New_York", "2026-12-01T12:30:00-05:00", "2026-11-01T17:30:00Z"),
    ),
)
@pytest.mark.parametrize("storage", ("offset", "utc", "naive_utc"))
def test_calendar_window_survives_timestamp_storage(
    duration: str, zone: str, reset_at: str, expected: str, storage: str
) -> None:
    reset: Final = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
    stored: Final = {
        "offset": reset,
        "utc": reset.astimezone(timezone.utc),
        "naive_utc": reset.astimezone(timezone.utc).replace(tzinfo=None),
    }[storage]
    settings: Final = BudgetResetSettings(timezone=zone, reset_time_of_day=reset.time())
    start: Final = compute_budget_window_start(duration, stored, settings)

    assert start == datetime.fromisoformat(expected.replace("Z", "+00:00"))
    assert compute_budget_reset_at(duration, settings, now=start) == reset


def _restore_attr(obj, name, original):
    if original is None:
        if hasattr(obj, name):
            delattr(obj, name)
    else:
        setattr(obj, name, original)


def test_get_budget_reset_time():
    """
    Test that the budget reset time is set to the first of the next month
    """
    # Get the current date
    now = datetime.now(timezone.utc)

    # Calculate expected reset date (first of next month)
    if now.month == 12:
        expected_month = 1
        expected_year = now.year + 1
    else:
        expected_month = now.month + 1
        expected_year = now.year
    expected_reset_at = datetime(expected_year, expected_month, 1, tzinfo=timezone.utc)

    # Verify budget_reset_at is set to first of next month
    assert get_budget_reset_time(budget_duration="1mo") == expected_reset_at


def test_get_budget_reset_timezone_reads_litellm_attr():
    """
    Test that get_budget_reset_timezone reads from litellm.timezone attribute.
    """
    original = getattr(litellm, "timezone", None)
    try:
        litellm.timezone = "Asia/Tokyo"
        assert get_budget_reset_timezone() == "Asia/Tokyo"
    finally:
        if original is None:
            if hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")
        else:
            litellm.timezone = original


def test_get_budget_reset_timezone_fallback_utc():
    """
    Test that get_budget_reset_timezone falls back to UTC when litellm.timezone is not set.
    """
    original = getattr(litellm, "timezone", None)
    try:
        if hasattr(litellm, "timezone"):
            delattr(litellm, "timezone")
        assert get_budget_reset_timezone() == "UTC"
    finally:
        if original is not None:
            litellm.timezone = original


def test_get_budget_reset_timezone_fallback_on_none():
    """
    Test that get_budget_reset_timezone falls back to UTC when litellm.timezone is None.
    """
    original = getattr(litellm, "timezone", None)
    try:
        litellm.timezone = None
        assert get_budget_reset_timezone() == "UTC"
    finally:
        if original is None:
            if hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")
        else:
            litellm.timezone = original


def test_get_budget_reset_time_respects_timezone():
    """
    Test that get_budget_reset_time uses the configured timezone for reset calculation.
    A daily reset should align to midnight in the configured timezone.
    """
    original = getattr(litellm, "timezone", None)
    try:
        litellm.timezone = "Asia/Tokyo"
        reset_at = get_budget_reset_time(budget_duration="1d")
        # The reset time should be midnight in Asia/Tokyo
        tokyo_reset = reset_at.astimezone(ZoneInfo("Asia/Tokyo"))
        assert tokyo_reset.hour == 0
        assert tokyo_reset.minute == 0
        assert tokyo_reset.second == 0
    finally:
        if original is None:
            if hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")
        else:
            litellm.timezone = original


def test_parse_budget_reset_time_hh_mm():
    assert parse_budget_reset_time("12:00") == time(12, 0)


def test_parse_budget_reset_time_hh_mm_ss():
    assert parse_budget_reset_time("09:30:15") == time(9, 30, 15)


def test_parse_budget_reset_time_unset_defaults_to_midnight():
    assert parse_budget_reset_time(None) == time(0, 0)
    assert parse_budget_reset_time("") == time(0, 0)


def test_parse_budget_reset_time_invalid_string_raises():
    with pytest.raises(ValueError, match="hour 'HH:MM' or 'HH:MM:SS' string, e\\.g\\."):
        parse_budget_reset_time("25:00")
    with pytest.raises(ValueError, match="Invalid budget_reset_time 'noon'; expected a"):
        parse_budget_reset_time("noon")


def test_parse_budget_reset_time_non_string_raises():
    # Unquoted "12:00" in YAML parses to the int 720; it must fail loudly,
    # not silently fall back to midnight.
    with pytest.raises(ValueError, match="hour 'HH:MM' string, e\\.g\\."):
        parse_budget_reset_time(720)


def test_get_budget_reset_settings_reads_globals():
    orig_tz = getattr(litellm, "timezone", None)
    orig_rt = getattr(litellm, "budget_reset_time", None)
    try:
        litellm.timezone = "Asia/Jerusalem"
        litellm.budget_reset_time = "12:00"
        settings = get_budget_reset_settings()
        assert settings.timezone == "Asia/Jerusalem"
        assert settings.reset_time_of_day == time(12, 0)
    finally:
        _restore_attr(litellm, "timezone", orig_tz)
        _restore_attr(litellm, "budget_reset_time", orig_rt)


def test_compute_budget_reset_at_applies_offset():
    settings = BudgetResetSettings(
        timezone="Asia/Jerusalem", reset_time_of_day=time(12, 0)
    )
    reset_at = compute_budget_reset_at("1d", settings)
    jerusalem = reset_at.astimezone(ZoneInfo("Asia/Jerusalem"))
    assert jerusalem.hour == 12
    assert jerusalem.minute == 0
    assert reset_at > datetime.now(timezone.utc)


def test_get_budget_reset_time_honors_global_budget_reset_time():
    orig_tz = getattr(litellm, "timezone", None)
    orig_rt = getattr(litellm, "budget_reset_time", None)
    try:
        litellm.timezone = "UTC"
        litellm.budget_reset_time = "12:00"
        reset_at = get_budget_reset_time(budget_duration="1d")
        assert reset_at.astimezone(timezone.utc).hour == 12
        assert reset_at.astimezone(timezone.utc).minute == 0
    finally:
        _restore_attr(litellm, "timezone", orig_tz)
        _restore_attr(litellm, "budget_reset_time", orig_rt)

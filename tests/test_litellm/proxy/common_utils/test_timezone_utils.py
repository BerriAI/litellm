from datetime import datetime, time, timezone
from typing import Final
from zoneinfo import ZoneInfo

import pytest


import litellm
from litellm.proxy.common_utils.timezone_utils import (
    BudgetResetSettings,
    compute_budget_reset_at,
    get_budget_reset_settings,
    get_budget_reset_time,
    get_budget_reset_timezone,
    parse_budget_reset_time,
    get_daily_spend_bucket_date,
    get_daily_usage_timezone,
    parse_daily_usage_timezone,
)


@pytest.mark.parametrize(
    "timestamp, zone, expected",
    [
        ("2026-12-31T16:00:00Z", "Asia/Singapore", "2027-01-01"),
        ("2026-09-25T18:15:00Z", "Asia/Kathmandu", "2026-09-26"),
        ("2026-03-08T07:59:59Z", "America/Los_Angeles", "2026-03-07"),
        ("2026-03-08T08:00:00Z", "America/Los_Angeles", "2026-03-08"),
        ("2026-11-01T08:30:00Z", "America/Los_Angeles", "2026-11-01"),
        ("2026-11-01T09:30:00Z", "America/Los_Angeles", "2026-11-01"),
        ("2026-07-25T22:00:00-07:00", "UTC", "2026-07-26"),
        ("2026-07-25T22:00:00-07:00", None, "2026-07-25"),
    ],
)
def test_reporting_date_boundaries(timestamp: str, zone: str | None, expected: str) -> None:
    assert get_daily_spend_bucket_date(timestamp, zone) == expected


@pytest.mark.parametrize("raw", [False, 8, "", "Not/AZone"])
def test_reporting_timezone_rejects_invalid_config(raw: object) -> None:
    with pytest.raises((ValueError, KeyError)):
        parse_daily_usage_timezone(raw)


def test_reporting_timezone_is_independent_of_budget_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "timezone", "Asia/Singapore", raising=False)
    monkeypatch.setattr(litellm, "daily_usage_timezone", None)
    assert parse_daily_usage_timezone(None) is None
    assert get_daily_usage_timezone().key == "UTC"
    configured: Final = parse_daily_usage_timezone("America/Los_Angeles")
    monkeypatch.setattr(litellm, "daily_usage_timezone", configured)
    assert get_daily_usage_timezone().key == "America/Los_Angeles"
    assert get_budget_reset_timezone() == "Asia/Singapore"


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

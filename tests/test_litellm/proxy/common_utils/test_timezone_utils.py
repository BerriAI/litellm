import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.common_utils.timezone_utils import (
    get_budget_reset_time,
    get_budget_reset_timezone,
    get_weekly_budget_reset_day,
)


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


def test_get_weekly_budget_reset_day_reads_litellm_attr():
    """
    Test that get_weekly_budget_reset_day reads from litellm.weekly_budget_reset_day attribute.
    """
    original = getattr(litellm, "weekly_budget_reset_day", None)
    try:
        litellm.weekly_budget_reset_day = "sunday"
        assert get_weekly_budget_reset_day() == "sunday"
    finally:
        if original is None:
            if hasattr(litellm, "weekly_budget_reset_day"):
                delattr(litellm, "weekly_budget_reset_day")
        else:
            litellm.weekly_budget_reset_day = original


def test_get_weekly_budget_reset_day_fallback_monday():
    """
    Test that get_weekly_budget_reset_day falls back to "monday" when not set.
    """
    original = getattr(litellm, "weekly_budget_reset_day", None)
    try:
        if hasattr(litellm, "weekly_budget_reset_day"):
            delattr(litellm, "weekly_budget_reset_day")
        assert get_weekly_budget_reset_day() == "monday"
    finally:
        if original is not None:
            litellm.weekly_budget_reset_day = original


def test_get_budget_reset_time_respects_weekly_reset_day():
    """
    Test that get_budget_reset_time uses the configured weekly_budget_reset_day.
    A weekly reset should land on the configured day at midnight.
    """
    original_tz = getattr(litellm, "timezone", None)
    original_day = getattr(litellm, "weekly_budget_reset_day", None)
    try:
        litellm.timezone = "UTC"
        litellm.weekly_budget_reset_day = "sunday"
        reset_at = get_budget_reset_time(budget_duration="7d")
        # The reset should be on a Sunday at midnight UTC
        assert reset_at.weekday() == 6  # Sunday
        assert reset_at.hour == 0
        assert reset_at.minute == 0
        assert reset_at.second == 0
    finally:
        if original_tz is None:
            if hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")
        else:
            litellm.timezone = original_tz
        if original_day is None:
            if hasattr(litellm, "weekly_budget_reset_day"):
                delattr(litellm, "weekly_budget_reset_day")
        else:
            litellm.weekly_budget_reset_day = original_day


def test_get_budget_reset_time_weekly_defaults_to_monday():
    """
    Test that a weekly reset defaults to Monday when weekly_budget_reset_day is not set.
    """
    original_tz = getattr(litellm, "timezone", None)
    original_day = getattr(litellm, "weekly_budget_reset_day", None)
    try:
        litellm.timezone = "UTC"
        if hasattr(litellm, "weekly_budget_reset_day"):
            delattr(litellm, "weekly_budget_reset_day")
        reset_at = get_budget_reset_time(budget_duration="7d")
        # The reset should be on a Monday at midnight UTC
        assert reset_at.weekday() == 0  # Monday
        assert reset_at.hour == 0
        assert reset_at.minute == 0
        assert reset_at.second == 0
    finally:
        if original_tz is None:
            if hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")
        else:
            litellm.timezone = original_tz
        if original_day is not None:
            litellm.weekly_budget_reset_day = original_day

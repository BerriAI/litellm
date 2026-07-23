import os
import sys
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.common_utils.timezone_utils import (
    BudgetResetSettings,
    compute_budget_reset_at,
    get_budget_reset_settings,
    get_budget_reset_time,
    get_budget_reset_timezone,
    parse_budget_reset_time,
    validate_budget_duration,
)


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
    with pytest.raises(ValueError):
        parse_budget_reset_time("25:00")
    with pytest.raises(ValueError):
        parse_budget_reset_time("noon")


def test_parse_budget_reset_time_non_string_raises():
    # Unquoted "12:00" in YAML parses to the int 720; it must fail loudly,
    # not silently fall back to midnight.
    with pytest.raises(ValueError):
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
    settings = BudgetResetSettings(timezone="Asia/Jerusalem", reset_time_of_day=time(12, 0))
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


class TestValidateBudgetDuration:
    """`validate_budget_duration` is the fail-closed write-boundary guard shared by
    the key/team/customer/org/budget endpoints. It must reject any value the budget
    reset job can't honor, so a bad duration can never be persisted and then silently
    reset on the wrong cadence.
    """

    def test_none_is_a_noop(self):
        assert validate_budget_duration(None) is None

    @pytest.mark.parametrize("duration", ["1s", "30m", "1h", "24h", "7d", "30d", "1mo"])
    def test_canonical_durations_pass(self, duration):
        assert validate_budget_duration(duration) is None

    @pytest.mark.parametrize("duration", ["hourly", "daily", "weekly", "monthly", " MONTHLY ", "Weekly"])
    def test_word_form_durations_pass(self, duration):
        assert validate_budget_duration(duration) is None

    @pytest.mark.parametrize("duration", ["garbage", "5x", "abc", "", "d30", "1 day"])
    def test_unparseable_durations_raise_400(self, duration):
        with pytest.raises(HTTPException) as exc_info:
            validate_budget_duration(duration)
        assert exc_info.value.status_code == 400
        assert "budget_duration" in exc_info.value.detail["error"]

    @pytest.mark.parametrize("duration", ["0s", "0m", "0h", "0d", "0w"])
    def test_non_positive_durations_raise_400(self, duration):
        with pytest.raises(HTTPException) as exc_info:
            validate_budget_duration(duration)
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize("duration", ["1s", "30m", "1h", "24h", "7d", "30d", "1mo", "hourly", "weekly", "monthly"])
    def test_accepted_iff_reset_job_can_compute_it(self, duration):
        validate_budget_duration(duration)
        assert duration_in_seconds(duration) > 0

import unittest
from datetime import datetime, time, timedelta, timezone
from typing import Final
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import litellm.litellm_core_utils.duration_parser as duration_parser
from litellm.litellm_core_utils.duration_parser import (
    duration_in_seconds,
    duration_in_seconds_from,
    get_annual_budget_window_start,
    get_budget_window_end,
    get_budget_window_start,
    get_next_standardized_reset_time,
    is_annual_budget_duration,
)


class TestStandardizedResetTime(unittest.TestCase):
    def test_day_based_resets(self):
        """Test day-based reset durations (1d, 7d, 30d)"""
        # Base time: 2023-05-15 10:30:00 UTC
        base_time = datetime(2023, 5, 15, 10, 30, 0, tzinfo=timezone.utc)

        # Daily reset (1d) - should reset at next midnight
        daily_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=timezone.utc)
        daily_result = get_next_standardized_reset_time("1d", base_time, "UTC")
        self.assertEqual(daily_result, daily_expected)

        # Weekly reset (7d) - should reset on next Monday
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)  # A Wednesday
        weekly_expected = datetime(2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc)  # Next Monday
        weekly_result = get_next_standardized_reset_time("7d", wednesday, "UTC")
        self.assertEqual(weekly_result, weekly_expected)

        # Monthly reset (30d) - should reset on 1st of next month
        monthly_expected = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        monthly_result = get_next_standardized_reset_time("30d", base_time, "UTC")
        self.assertEqual(monthly_result, monthly_expected)

        # Custom day reset (3d) - should reset after 3 days
        custom_day_expected = datetime(2023, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
        custom_day_result = get_next_standardized_reset_time("3d", base_time, "UTC")
        self.assertEqual(custom_day_result, custom_day_expected)

    def test_week_based_resets(self):
        """Test week-based reset durations (1w, 2w).
        1w snaps to the next Monday at midnight (same as 7d).
        2w advances exactly 14 days from the current date at midnight.
        """
        # 1w from a Wednesday -> next Monday (5 days away, not 7)
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)
        weekly_expected = datetime(2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc)
        weekly_result = get_next_standardized_reset_time("1w", wednesday, "UTC")
        self.assertEqual(weekly_result, weekly_expected)

        # 2w from a Wednesday -> exactly 14 days out (lands on a Wednesday, not Monday)
        base_time = datetime(2023, 5, 17, 10, 30, 0, tzinfo=timezone.utc)
        two_week_expected = datetime(2023, 5, 31, 0, 0, 0, tzinfo=timezone.utc)
        two_week_result = get_next_standardized_reset_time("2w", base_time, "UTC")
        self.assertEqual(two_week_result, two_week_expected)

    def test_hour_minute_second_resets(self):
        """Test hour, minute, and second based reset durations"""
        # Base time: 2023-05-15 15:20:30 UTC (3:20:30 PM)
        base_time = datetime(2023, 5, 15, 15, 20, 30, tzinfo=timezone.utc)

        # 2-hour reset - should reset at next even hour (16:00)
        hour_expected = datetime(2023, 5, 15, 16, 0, 0, tzinfo=timezone.utc)
        hour_result = get_next_standardized_reset_time("2h", base_time, "UTC")
        self.assertEqual(hour_result, hour_expected)

        # 30-minute reset - should reset at next 30-minute mark (15:30)
        minute_expected = datetime(2023, 5, 15, 15, 30, 0, tzinfo=timezone.utc)
        minute_result = get_next_standardized_reset_time("30m", base_time, "UTC")
        self.assertEqual(minute_result, minute_expected)

        # 15-second reset - should reset at next 15-second mark (15:20:45)
        second_expected = datetime(2023, 5, 15, 15, 20, 45, tzinfo=timezone.utc)
        second_result = get_next_standardized_reset_time("15s", base_time, "UTC")
        self.assertEqual(second_result, second_expected)

    def test_timezone_handling(self):
        """Test timezone handling with different regions"""
        # Base time: 2023-05-15 22:30:00 UTC (late in UTC day)
        base_time = datetime(2023, 5, 15, 22, 30, 0, tzinfo=timezone.utc)

        # Test daily reset in different timezones
        # US/Eastern (UTC-4): 6:30 PM, so next reset is midnight same day
        eastern = ZoneInfo("US/Eastern")
        eastern_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=eastern)
        eastern_result = get_next_standardized_reset_time("1d", base_time, "US/Eastern")
        self.assertEqual(eastern_result, eastern_expected)

        # Asia/Kolkata (UTC+5:30): 4:00 AM next day, so next reset is midnight the day after
        ist = ZoneInfo("Asia/Kolkata")
        ist_expected = datetime(2023, 5, 17, 0, 0, 0, tzinfo=ist)
        ist_result = get_next_standardized_reset_time("1d", base_time, "Asia/Kolkata")
        self.assertEqual(ist_result, ist_expected)

        # Test hourly reset in different timezones
        pacific = ZoneInfo("US/Pacific")
        pacific_expected = datetime(2023, 5, 15, 16, 0, 0, tzinfo=pacific)
        pacific_result = get_next_standardized_reset_time("2h", base_time, "US/Pacific")
        self.assertEqual(pacific_result, pacific_expected)

        # Test minute reset in different timezones
        # Europe/London (UTC+1): 11:30 PM, so next 15m reset is 11:45 PM
        london = ZoneInfo("Europe/London")
        london_expected = datetime(2023, 5, 15, 23, 45, 0, tzinfo=london)
        london_result = get_next_standardized_reset_time("15m", base_time, "Europe/London")
        self.assertEqual(london_result, london_expected)

        # Test Bangkok timezone (UTC+7): 5:30 AM next day, so next reset is midnight the day after
        bangkok = ZoneInfo("Asia/Bangkok")
        bangkok_expected = datetime(2023, 5, 17, 0, 0, 0, tzinfo=bangkok)
        bangkok_result = get_next_standardized_reset_time("1d", base_time, "Asia/Bangkok")
        self.assertEqual(bangkok_result, bangkok_expected)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Exactly on hour boundary
        on_hour = datetime(2023, 5, 15, 14, 0, 0, tzinfo=timezone.utc)
        hour_expected = datetime(2023, 5, 15, 16, 0, 0, tzinfo=timezone.utc)
        hour_result = get_next_standardized_reset_time("2h", on_hour, "UTC")
        self.assertEqual(hour_result, hour_expected)

        # Exactly on minute boundary
        on_minute = datetime(2023, 5, 15, 14, 30, 0, tzinfo=timezone.utc)
        minute_expected = datetime(2023, 5, 15, 15, 0, 0, tzinfo=timezone.utc)
        minute_result = get_next_standardized_reset_time("30m", on_minute, "UTC")
        self.assertEqual(minute_result, minute_expected)

        # Near day boundary
        near_midnight = datetime(2023, 5, 15, 23, 50, 0, tzinfo=timezone.utc)

        # 30m near midnight - should roll over to next day
        midnight_minute_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=timezone.utc)
        midnight_minute_result = get_next_standardized_reset_time("30m", near_midnight, "UTC")
        self.assertEqual(midnight_minute_result, midnight_minute_expected)

        # Invalid timezone - should fall back to UTC
        invalid_tz_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=timezone.utc)
        invalid_tz_result = get_next_standardized_reset_time("1d", on_hour, "NonExistentTimeZone")
        self.assertEqual(invalid_tz_result, invalid_tz_expected)

    def test_iana_timezones_previously_unsupported(self):
        """Test IANA timezones that were previously unsupported by the hardcoded map."""
        # Base time: 2023-05-15 15:00:00 UTC
        base_time = datetime(2023, 5, 15, 15, 0, 0, tzinfo=timezone.utc)

        # Asia/Tokyo (UTC+9): 15:00 UTC = 00:00 JST May 16, exactly on midnight boundary → next day
        tokyo = ZoneInfo("Asia/Tokyo")
        tokyo_expected = datetime(2023, 5, 17, 0, 0, 0, tzinfo=tokyo)
        tokyo_result = get_next_standardized_reset_time("1d", base_time, "Asia/Tokyo")
        self.assertEqual(tokyo_result, tokyo_expected)

        # Australia/Sydney (UTC+10): 2023-05-16 01:00 AEST
        sydney = ZoneInfo("Australia/Sydney")
        # At 15:00 UTC it's 01:00 AEST May 16 → next midnight is May 17 00:00 AEST
        sydney_expected = datetime(2023, 5, 17, 0, 0, 0, tzinfo=sydney)
        sydney_result = get_next_standardized_reset_time("1d", base_time, "Australia/Sydney")
        self.assertEqual(sydney_result, sydney_expected)

        # America/Chicago (UTC-5): at 15:00 UTC it's 10:00 CDT → next midnight is May 16 00:00 CDT
        chicago = ZoneInfo("America/Chicago")
        chicago_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=chicago)
        chicago_result = get_next_standardized_reset_time("1d", base_time, "America/Chicago")
        self.assertEqual(chicago_result, chicago_expected)

    def test_dst_fall_back(self):
        """Test DST fall-back transition (clocks go back 1 hour)."""
        # US/Eastern DST ends first Sunday of November 2023 (Nov 5)
        # At 2023-11-05 05:30 UTC = 01:30 EDT (before fall-back)
        # After fall-back at 06:00 UTC = 01:00 EST
        pre_fallback = datetime(2023, 11, 5, 5, 30, 0, tzinfo=timezone.utc)
        eastern = ZoneInfo("US/Eastern")

        # Daily reset: next midnight should be Nov 6 00:00 EST
        expected = datetime(2023, 11, 6, 0, 0, 0, tzinfo=eastern)
        result = get_next_standardized_reset_time("1d", pre_fallback, "US/Eastern")
        self.assertEqual(result, expected)

    def test_dst_spring_forward(self):
        """Test DST spring-forward transition (clocks go forward 1 hour)."""
        # US/Eastern DST starts second Sunday of March 2023 (Mar 12)
        # At 2023-03-12 06:30 UTC = 01:30 EST (before spring-forward)
        # After spring-forward at 07:00 UTC = 03:00 EDT
        pre_spring = datetime(2023, 3, 12, 6, 30, 0, tzinfo=timezone.utc)
        eastern = ZoneInfo("US/Eastern")

        # Daily reset: next midnight should be Mar 13 00:00 EDT
        expected = datetime(2023, 3, 13, 0, 0, 0, tzinfo=eastern)
        result = get_next_standardized_reset_time("1d", pre_spring, "US/Eastern")
        self.assertEqual(result, expected)


@pytest.mark.parametrize(
    "now, zone, reset_time, expected",
    [
        (
            datetime(2026, 11, 12, tzinfo=timezone.utc),
            "UTC",
            time(0),
            datetime(2027, 1, 1, tzinfo=timezone.utc),
        ),
        (
            datetime(2028, 2, 29, tzinfo=timezone.utc),
            "UTC",
            time(0),
            datetime(2029, 1, 1, tzinfo=timezone.utc),
        ),
        (
            datetime(2027, 1, 1, tzinfo=timezone.utc),
            "UTC",
            time(0),
            datetime(2028, 1, 1, tzinfo=timezone.utc),
        ),
        (
            datetime(2027, 1, 1, 8, tzinfo=timezone.utc),
            "UTC",
            time(12),
            datetime(2027, 1, 1, 12, tzinfo=timezone.utc),
        ),
        (
            datetime(2027, 1, 1, 12, tzinfo=timezone.utc),
            "UTC",
            time(12),
            datetime(2028, 1, 1, 12, tzinfo=timezone.utc),
        ),
        (
            datetime(2027, 1, 1, 2, tzinfo=timezone.utc),
            "America/New_York",
            time(0),
            datetime(2027, 1, 1, tzinfo=ZoneInfo("America/New_York")),
        ),
    ],
)
def test_annual_reset_and_window_start(now: datetime, zone: str, reset_time: time, expected: datetime) -> None:
    reset: Final = get_next_standardized_reset_time("12mo", now, zone, reset_time)
    assert reset == expected
    assert get_budget_window_start("12mo", reset) == expected.replace(year=expected.year - 1)


@pytest.mark.parametrize(
    "long_spelling,short_form",
    [
        ("12months", "12mo"),
        ("12mos", "12mo"),
        ("1month", "1mo"),
        ("7days", "7d"),
        ("2weeks", "2w"),
        ("1hour", "1h"),
        ("5min", "5m"),
        ("30sec", "30s"),
    ],
)
def test_long_unit_spelling_resets_on_the_schedule_it_is_charged_for(long_spelling: str, short_form: str) -> None:
    now: Final = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
    assert duration_in_seconds_from(now, long_spelling) == duration_in_seconds_from(now, short_form)
    reset: Final = get_next_standardized_reset_time(long_spelling, now, "UTC")
    assert reset == get_next_standardized_reset_time(short_form, now, "UTC")
    assert get_budget_window_start(long_spelling, reset) == get_budget_window_start(short_form, reset)
    assert is_annual_budget_duration(long_spelling) == is_annual_budget_duration(short_form)


def test_long_month_spelling_without_a_calendar_schedule_keeps_its_nightly_reset() -> None:
    now: Final = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="1 month or 12 month"):
        get_next_standardized_reset_time("3mo", now, "UTC")
    reset: Final = get_next_standardized_reset_time("3months", now, "UTC")
    assert reset == datetime(2026, 11, 14, tzinfo=timezone.utc)
    assert get_budget_window_start("3months", reset) == datetime(2026, 11, 13, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "zone,reset_at,expected",
    (
        ("UTC", "2029-01-01T00:00:00+00:00", "2028-01-01T00:00:00+00:00"),
        ("UTC", "2101-01-01T00:00:00+00:00", "2100-01-01T00:00:00+00:00"),
        ("America/New_York", "2029-01-01T00:00:00-05:00", "2028-01-01T05:00:00+00:00"),
        ("Asia/Kathmandu", "2029-01-01T12:30:00+05:45", "2028-01-01T06:45:00+00:00"),
        ("Pacific/Apia", "2012-01-01T00:00:00+14:00", "2011-01-01T10:00:00+00:00"),
    ),
)
@pytest.mark.parametrize("storage", ("offset", "utc", "naive_utc"))
def test_annual_window_survives_timestamp_storage(zone: str, reset_at: str, expected: str, storage: str) -> None:
    reset: Final = datetime.fromisoformat(reset_at)
    stored: Final = {
        "offset": reset,
        "utc": reset.astimezone(timezone.utc),
        "naive_utc": reset.astimezone(timezone.utc).replace(tzinfo=None),
    }[storage]

    assert get_annual_budget_window_start(stored, zone) == datetime.fromisoformat(expected)


@pytest.mark.parametrize("microsecond", (0, 123000, 123456, 123999, 999999))
@pytest.mark.parametrize("duration", ("1s", "1h", "1d", "1w", "30d", "1mo", "12mo"))
def test_calendar_window_start_matches_database_millisecond_precision(duration: str, microsecond: int) -> None:
    reset: Final = datetime(2026, 11, 15, 5, 48, 2, microsecond, tzinfo=timezone.utc)
    expected: Final = get_budget_window_start(duration, reset.replace(microsecond=0)).replace(
        microsecond=microsecond // 1000 * 1000
    )
    assert get_budget_window_start(duration, reset) == expected


class TestResetTimeOfDay(unittest.TestCase):
    """A configurable reset_time_of_day shifts day/week/month resets off midnight."""

    def test_daily_reset_before_offset_is_today(self):
        now = datetime(2023, 5, 15, 8, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1d", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 15, 12, 0, 0, tzinfo=timezone.utc))

    def test_daily_reset_after_offset_is_tomorrow(self):
        now = datetime(2023, 5, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1d", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 16, 12, 0, 0, tzinfo=timezone.utc))

    def test_daily_reset_exactly_at_offset_rolls_forward(self):
        now = datetime(2023, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1d", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 16, 12, 0, 0, tzinfo=timezone.utc))

    def test_daily_reset_with_seconds_offset(self):
        now = datetime(2023, 5, 15, 8, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1d", now, "UTC", reset_time_of_day=time(9, 30, 15))
        self.assertEqual(result, datetime(2023, 5, 15, 9, 30, 15, tzinfo=timezone.utc))

    def test_offset_applies_in_configured_timezone(self):
        # 2023-05-15 22:30 UTC == 2023-05-16 01:30 in Jerusalem (IDT, UTC+3),
        # so the next noon-Jerusalem reset is 2023-05-16 12:00 IDT.
        now = datetime(2023, 5, 15, 22, 30, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1d", now, "Asia/Jerusalem", reset_time_of_day=time(12, 0))
        jerusalem = result.astimezone(ZoneInfo("Asia/Jerusalem"))
        self.assertEqual((jerusalem.year, jerusalem.month, jerusalem.day), (2023, 5, 16))
        self.assertEqual(jerusalem.hour, 12)
        self.assertEqual(jerusalem.minute, 0)

    def test_weekly_reset_lands_on_monday_at_offset(self):
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("7d", wednesday, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 22, 12, 0, 0, tzinfo=timezone.utc))

    def test_weekly_reset_today_is_monday_before_offset_is_today(self):
        monday_morning = datetime(2023, 5, 22, 9, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("7d", monday_morning, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 22, 12, 0, 0, tzinfo=timezone.utc))

    def test_weekly_reset_today_is_monday_after_offset_is_next_week(self):
        monday_afternoon = datetime(2023, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("7d", monday_afternoon, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 29, 12, 0, 0, tzinfo=timezone.utc))

    def test_monthly_30d_lands_on_first_at_offset(self):
        now = datetime(2023, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("30d", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_monthly_1mo_today_is_first_before_offset_is_today(self):
        now = datetime(2023, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1mo", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_monthly_year_rollover_at_offset(self):
        now = datetime(2023, 12, 15, 9, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("1mo", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_custom_day_reset_applies_offset(self):
        now = datetime(2023, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time("3d", now, "UTC", reset_time_of_day=time(12, 0))
        self.assertEqual(result, datetime(2023, 5, 18, 12, 0, 0, tzinfo=timezone.utc))

    def test_sub_day_durations_ignore_offset(self):
        base = datetime(2023, 5, 15, 15, 20, 30, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_standardized_reset_time("2h", base, "UTC", reset_time_of_day=time(12, 0)),
            datetime(2023, 5, 15, 16, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            get_next_standardized_reset_time("30m", base, "UTC", reset_time_of_day=time(12, 0)),
            datetime(2023, 5, 15, 15, 30, 0, tzinfo=timezone.utc),
        )

    def test_default_offset_is_midnight(self):
        now = datetime(2023, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_standardized_reset_time("1d", now, "UTC"),
            datetime(2023, 5, 16, 0, 0, 0, tzinfo=timezone.utc),
        )


class TestWordFormBudgetDurations(unittest.TestCase):
    """The Admin UI historically persisted word-form budget durations
    (hourly/daily/weekly/monthly). They must resolve to their real interval
    instead of silently collapsing to a next-midnight (daily) reset.
    """

    def test_word_forms_map_to_correct_reset_times(self):
        base_time = datetime(2023, 5, 17, 15, 20, 30, tzinfo=timezone.utc)

        self.assertEqual(
            get_next_standardized_reset_time("hourly", base_time, "UTC"),
            datetime(2023, 5, 17, 16, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            get_next_standardized_reset_time("daily", base_time, "UTC"),
            datetime(2023, 5, 18, 0, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            get_next_standardized_reset_time("weekly", base_time, "UTC"),
            datetime(2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            get_next_standardized_reset_time("monthly", base_time, "UTC"),
            datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

    def test_word_forms_are_not_all_collapsed_to_daily(self):
        base_time = datetime(2023, 5, 17, 15, 20, 30, tzinfo=timezone.utc)
        results = {
            word: get_next_standardized_reset_time(word, base_time, "UTC")
            for word in ("hourly", "daily", "weekly", "monthly")
        }
        self.assertEqual(len(set(results.values())), len(results))

    def test_word_forms_match_canonical_int_unit_forms(self):
        base_time = datetime(2023, 5, 17, 15, 20, 30, tzinfo=timezone.utc)
        for word, canonical in (("hourly", "1h"), ("daily", "24h"), ("weekly", "7d"), ("monthly", "30d")):
            self.assertEqual(
                get_next_standardized_reset_time(word, base_time, "UTC"),
                get_next_standardized_reset_time(canonical, base_time, "UTC"),
            )

    def test_word_forms_are_case_and_whitespace_insensitive(self):
        base_time = datetime(2023, 5, 17, 15, 20, 30, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_standardized_reset_time("  Monthly ", base_time, "UTC"),
            datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

    def test_duration_in_seconds_accepts_word_forms(self):
        self.assertEqual(duration_in_seconds("hourly"), 3600)
        self.assertEqual(duration_in_seconds("daily"), 86400)
        self.assertEqual(duration_in_seconds("weekly"), 604800)
        self.assertEqual(duration_in_seconds("monthly"), 2592000)

    def test_invalid_duration_logs_warning_and_falls_back(self):
        base_time = datetime(2023, 5, 15, 15, 0, 0, tzinfo=timezone.utc)
        with patch.object(duration_parser.verbose_logger, "warning") as mock_warning:
            result = get_next_standardized_reset_time("garbage", base_time, "UTC")
        self.assertEqual(result, datetime(2023, 5, 16, 0, 0, 0, tzinfo=timezone.utc))
        mock_warning.assert_called_once()
        self.assertIn("garbage", mock_warning.call_args.args)


class TestGetBudgetWindowStart(unittest.TestCase):
    def test_window_start_is_the_previous_reset_on_the_same_schedule(self):
        created_at: Final = datetime(2024, 10, 1, 0, 30, tzinfo=timezone.utc)
        durations: Final = "1d 24h daily 7d weekly 2w 10d 30d monthly 1mo 4h 5h 30m 45s 1hr fortnightly".split()
        for duration in durations:
            with self.subTest(duration=duration):
                reset_at = get_next_standardized_reset_time(duration, created_at, "UTC")
                window_start = get_budget_window_start(duration, reset_at)
                self.assertLessEqual(window_start, created_at)
                self.assertEqual(get_next_standardized_reset_time(duration, window_start, "UTC"), reset_at)

    def test_thirty_days_spans_the_calendar_month_it_resets_on(self):
        self.assertEqual(
            get_budget_window_start("30d", datetime(2024, 3, 1, tzinfo=timezone.utc)),
            datetime(2024, 2, 1, tzinfo=timezone.utc),
        )

    def test_months_clamp_to_the_shorter_month(self):
        self.assertEqual(
            get_budget_window_start("1mo", datetime(2024, 3, 31, tzinfo=timezone.utc)),
            datetime(2024, 2, 29, tzinfo=timezone.utc),
        )

    def test_unrecognized_duration_is_the_one_day_window_the_scheduler_falls_back_to(self):
        self.assertEqual(
            get_budget_window_start("1y", datetime(2024, 1, 16, tzinfo=timezone.utc)),
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("duration", "zone", "current_time", "expected"),
    (
        ("48h", "UTC", "2024-01-30T00:00:00+00:00", "2024-02-01T00:00:00+00:00"),
        ("2400h", "UTC", "2024-01-30T00:00:00+00:00", "2024-05-09T00:00:00+00:00"),
        ("49h", "UTC", "2024-01-30T14:15:30+00:00", "2024-02-01T01:00:00+00:00"),
        ("2885m", "UTC", "2024-01-30T14:15:30+00:00", "2024-02-01T14:05:00+00:00"),
        ("172805s", "Asia/Kolkata", "2024-01-30T08:45:30+00:00", "2024-02-01T08:45:05+00:00"),
    ),
)
def test_reset_rollover_keeps_all_elapsed_days(duration: str, zone: str, current_time: str, expected: str) -> None:
    reset_at: Final = get_next_standardized_reset_time(duration, datetime.fromisoformat(current_time), zone)

    assert reset_at.astimezone(timezone.utc) == datetime.fromisoformat(expected)


@pytest.mark.parametrize("duration", ("1d", "24h", "7d", "2w", "30d", "monthly", "1mo", "12mo", "5h", "45s"))
@pytest.mark.parametrize("zone", ("UTC", "America/Los_Angeles", "Asia/Kolkata"))
def test_committed_window_end_preserves_its_calendar_and_reset_time(duration: str, zone: str) -> None:
    for month in (3, 4, 11, 12):
        reset: Final = datetime(2026, month, 1, 2, 30, tzinfo=ZoneInfo(zone))
        start: Final = get_budget_window_start(duration, reset).astimezone(timezone.utc).replace(tzinfo=None)
        assert get_budget_window_end(duration, start, zone) == reset.astimezone(timezone.utc)


def test_month_end_uses_a_configured_reset_only_when_it_belongs_to_the_committed_window() -> None:
    reset: Final = datetime(2026, 3, 31, tzinfo=timezone.utc)
    assert get_budget_window_end("1mo", datetime(2026, 2, 28), reset_at=reset) == reset
    assert get_budget_window_end("1mo", datetime(2026, 1, 31), reset_at=reset) == datetime(
        2026, 2, 28, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    "start,duration,days",
    [
        (datetime(2026, 1, 15, 12, tzinfo=timezone.utc), "1mo", 31),
        (datetime(2026, 2, 10, 12, tzinfo=timezone.utc), "1mo", 28),
        (datetime(2026, 1, 31, 12, tzinfo=timezone.utc), "1mo", 28),
        (datetime(2028, 1, 31, 12, tzinfo=timezone.utc), "1mo", 29),
        (datetime(2026, 12, 15, 12, tzinfo=timezone.utc), "1mo", 31),
        (datetime(2026, 1, 15, 12, tzinfo=timezone.utc), "2mo", 59),
    ],
    ids=["january", "february", "clamped_to_february", "leap_february", "across_new_year", "two_months"],
)
def test_a_month_lasts_as_long_as_the_month_it_starts_in(start, duration, days):
    assert duration_in_seconds_from(start, duration) == days * 86400


@pytest.mark.parametrize("duration", ["45s", "30m", "1h", "1d", "7d", "2w", "30d", "daily", "monthly"])
def test_fixed_length_units_do_not_depend_on_the_start(duration):
    assert duration_in_seconds_from(datetime(2026, 2, 10, 12, tzinfo=timezone.utc), duration) == duration_in_seconds(
        duration
    )


def test_a_month_starting_now_matches_duration_in_seconds():
    now: Final = datetime(2026, 1, 31, 9, 30, 0, 250000, tzinfo=timezone.utc).astimezone()
    with patch.object(duration_parser.time_module, "time", return_value=now.timestamp()):
        assert duration_in_seconds("1mo") == duration_in_seconds_from(now, "1mo")


@pytest.mark.parametrize(
    "duration,next_date",
    (("1d", (2026, 11, 2)), ("30d", (2026, 12, 1)), ("1mo", (2026, 12, 1))),
)
@pytest.mark.parametrize("fold", (0, 1))
def test_calendar_reset_uses_only_the_first_occurrence(duration: str, next_date: tuple[int, int, int], fold: int) -> None:
    zone: Final = ZoneInfo("America/New_York")
    now: Final = datetime(2026, 11, 1, 1, 15, tzinfo=zone, fold=fold)
    expected: Final = datetime(*(next_date if fold else (2026, 11, 1)), 1, 30, tzinfo=zone)

    result: Final = get_next_standardized_reset_time(duration, now, zone.key, time(1, 30))

    assert result.astimezone(timezone.utc) == expected.astimezone(timezone.utc)


@pytest.mark.parametrize("hour,minute,next_day", ((1, 45, False), (3, 15, False), (3, 30, True), (3, 45, True)))
def test_daily_reset_resolves_the_spring_gap_before_comparing(hour: int, minute: int, next_day: bool) -> None:
    zone: Final = ZoneInfo("America/New_York")
    now: Final = datetime(2026, 3, 8, hour, minute, tzinfo=zone)
    expected: Final = datetime(2026, 3, 9, 2, 30, tzinfo=zone) if next_day else datetime(2026, 3, 8, 3, 30, tzinfo=zone)

    result: Final = get_next_standardized_reset_time("1d", now, zone.key, time(2, 30))

    assert result.astimezone(timezone.utc) == expected.astimezone(timezone.utc)
    assert result == result.astimezone(timezone.utc).astimezone(zone)


@pytest.mark.parametrize(
    "duration,reset_at,wall_time,expected",
    (
        ("1d", "2026-11-02T01:30:00-05:00", time(1, 30), "2026-11-01T01:30:00-04:00"),
        ("1mo", "2026-12-01T01:30:00-05:00", time(1, 30), "2026-11-01T01:30:00-04:00"),
        ("30d", "2026-03-01T02:30:00-05:00", time(2, 30), "2026-02-01T02:30:00-05:00"),
        ("1d", "2026-03-09T02:30:00-04:00", time(2, 30), "2026-03-08T03:30:00-04:00"),
        ("1d", "2026-03-08T03:30:00-04:00", time(2, 30), "2026-03-07T02:30:00-05:00"),
        ("1mo", "2026-03-15T09:15:00-04:00", time(2, 30), "2026-02-15T09:15:00-05:00"),
        ("1h", "2026-11-01T01:30:00-05:00", time(1, 30), "2026-11-01T01:30:00-04:00"),
    ),
)
def test_window_start_keeps_calendar_rules_and_elapsed_units(
    duration: str, reset_at: str, wall_time: time, expected: str
) -> None:
    result: Final = get_budget_window_start(
        duration, datetime.fromisoformat(reset_at), "America/New_York", wall_time
    )

    assert result.astimezone(timezone.utc) == datetime.fromisoformat(expected)


def test_window_end_after_a_spring_gap_returns_to_the_configured_wall_time() -> None:
    result: Final = get_budget_window_end(
        "1d",
        datetime.fromisoformat("2026-03-08T07:30:00+00:00"),
        "America/New_York",
        reset_time_of_day=time(2, 30),
    )

    assert result == datetime.fromisoformat("2026-03-09T06:30:00+00:00")


def test_daily_reset_resolves_a_half_hour_gap() -> None:
    zone: Final = ZoneInfo("Australia/Lord_Howe")
    now: Final = datetime(2026, 10, 4, 2, 35, tzinfo=zone)
    expected: Final = datetime(2026, 10, 4, 2, 45, tzinfo=zone)

    result: Final = get_next_standardized_reset_time("1d", now, zone.key, time(2, 15))

    assert result.astimezone(timezone.utc) == expected.astimezone(timezone.utc)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("duration", ("1h", "60m", "3600s"))
@pytest.mark.parametrize(
    "end",
    ("2026-11-01T06:30:00Z", "2026-11-01T07:30:00Z", "2026-03-08T07:30:00Z"),
)
def test_elapsed_window_bounds_do_not_depend_on_passing_the_zone_separately(duration: str, end: str) -> None:
    zone: Final = ZoneInfo("America/New_York")
    reset: Final = datetime.fromisoformat(end).astimezone(zone)
    start: Final = datetime.fromisoformat(end) - timedelta(hours=1)

    assert get_budget_window_start(duration, reset).astimezone(timezone.utc) == start
    assert get_budget_window_end(duration, start, zone.key, reset_at=reset) == reset.astimezone(timezone.utc)
    assert get_budget_window_end(duration, start, zone.key) == reset.astimezone(timezone.utc)


@pytest.mark.parametrize("duration", ("1h", "60m", "3600s"))
@pytest.mark.parametrize(
    "now",
    ("2026-11-01T05:15:00Z", "2026-11-01T06:15:00Z", "2026-03-08T06:45:00Z", "2026-03-08T07:15:00Z"),
)
def test_scheduled_elapsed_window_contains_now_through_both_dst_transitions(duration: str, now: str) -> None:
    instant: Final = datetime.fromisoformat(now)
    end: Final = get_next_standardized_reset_time(duration, instant, "America/New_York")
    start: Final = get_budget_window_start(duration, end)

    assert start.astimezone(timezone.utc) <= instant < end.astimezone(timezone.utc)
    assert end.astimezone(timezone.utc) - start.astimezone(timezone.utc) == timedelta(hours=1)


@pytest.mark.parametrize("duration", ("12h", "5h", "90m", "45s"))
@pytest.mark.parametrize(
    "now,zone",
    (
        ("2026-11-01T05:59:40Z", "America/New_York"),
        ("2026-03-08T06:59:40Z", "America/New_York"),
        ("2026-04-04T14:59:40Z", "Australia/Lord_Howe"),
        ("2026-10-03T15:29:40Z", "Australia/Lord_Howe"),
    ),
)
@pytest.mark.parametrize("saved_schedule", (False, True))
def test_elapsed_schedule_never_shortens_a_later_allowance(
    duration: str, now: str, zone: str, saved_schedule: bool
) -> None:
    current: Final = datetime.fromisoformat(now)
    anchor: Final = datetime(2026, 1, 1, 4, 17, tzinfo=timezone.utc) if saved_schedule else None
    interval: Final = timedelta(seconds=duration_in_seconds(duration))
    first: Final = get_next_standardized_reset_time(
        duration, current, zone, elapsed_anchor=anchor
    ).astimezone(timezone.utc)
    assert first - interval <= current < first
    for offset in range(6):
        boundary: Final = first + offset * interval
        following: Final = get_next_standardized_reset_time(
            duration, boundary, zone, elapsed_anchor=boundary
        ).astimezone(timezone.utc)
        assert following - boundary == interval


@pytest.mark.parametrize(
    "duration,zone,expected",
    (
        ("45m", "UTC", "2026-07-15T10:45:00Z"),
        ("5h", "UTC", "2026-07-15T15:00:00Z"),
        ("daily", "Asia/Tokyo", "2026-07-16T00:00:00+09:00"),
        ("1h", "Asia/Kolkata", "2026-07-15T16:00:00+05:30"),
        ("24h", "America/New_York", "2026-07-16T00:00:00-04:00"),
        ("1d", "Asia/Tokyo", "2026-07-16T00:00:00+09:00"),
        ("1mo", "UTC", "2026-08-01T00:00:00Z"),
    ),
)
def test_new_schedules_keep_their_local_alignment(duration: str, zone: str, expected: str) -> None:
    now: Final = datetime(2026, 7, 15, 10, 17, tzinfo=timezone.utc)

    assert get_next_standardized_reset_time(duration, now, zone) == datetime.fromisoformat(expected)


@pytest.mark.parametrize(
    "duration,boundary,expected",
    (
        ("45m", "2026-07-15T11:30:00Z", "2026-07-15T12:15:00Z"),
        ("5h", "2026-07-16T01:00:00Z", "2026-07-16T06:00:00Z"),
        ("45s", "2026-07-15T10:18:30Z", "2026-07-15T10:19:15Z"),
    ),
)
def test_saved_elapsed_schedules_do_not_shorten_at_local_unit_rollover(
    duration: str, boundary: str, expected: str
) -> None:
    saved_reset: Final = datetime.fromisoformat(boundary)

    assert get_next_standardized_reset_time(
        duration, saved_reset, "UTC", elapsed_anchor=saved_reset
    ) == datetime.fromisoformat(expected)


@pytest.mark.parametrize("duration", ("24h", "daily"))
@pytest.mark.parametrize(
    "zone,now,expected",
    (
        ("Asia/Tokyo", "2026-07-15T10:17:00Z", "2026-07-16T00:00:00+09:00"),
        ("Asia/Kolkata", "2026-07-15T10:17:00Z", "2026-07-16T00:00:00+05:30"),
        ("America/New_York", "2026-07-15T10:17:00Z", "2026-07-16T00:00:00-04:00"),
        ("UTC", "2026-07-15T10:17:00Z", "2026-07-16T00:00:00+00:00"),
        ("America/New_York", "2026-11-01T05:30:00Z", "2026-11-02T00:00:00-05:00"),
        ("America/New_York", "2026-11-01T06:30:00Z", "2026-11-02T00:00:00-05:00"),
        ("America/New_York", "2026-03-08T06:30:00Z", "2026-03-09T00:00:00-04:00"),
        ("America/Santiago", "2026-09-05T12:00:00Z", "2026-09-06T01:00:00-03:00"),
        ("America/Havana", "2026-03-08T03:00:00Z", "2026-03-08T01:00:00-04:00"),
    ),
)
def test_whole_day_hour_durations_reset_at_local_midnight(duration: str, zone: str, now: str, expected: str) -> None:
    current: Final = datetime.fromisoformat(now)
    reset: Final = get_next_standardized_reset_time(
        duration, current, zone, time(1, 30), elapsed_anchor=datetime(2020, 1, 1, 7, 13, tzinfo=timezone.utc)
    )
    start: Final = get_budget_window_start(duration, reset, zone)
    previous_midnight: Final = datetime(
        current.astimezone(ZoneInfo(zone)).year, current.astimezone(ZoneInfo(zone)).month,
        current.astimezone(ZoneInfo(zone)).day, tzinfo=ZoneInfo(zone),
    )

    assert reset == datetime.fromisoformat(expected)
    assert start.astimezone(timezone.utc) == previous_midnight.astimezone(timezone.utc)
    assert start.astimezone(timezone.utc) <= current < reset.astimezone(timezone.utc)
    assert get_budget_window_end(duration, start, zone) == reset.astimezone(timezone.utc)


@pytest.mark.parametrize(
    "duration,expected",
    (("48h", "2026-07-17T00:00:00+09:00"), ("168h", "2026-07-22T00:00:00+09:00"), ("720h", "2026-08-14T00:00:00+09:00")),
)
def test_multi_day_hour_durations_count_local_days_from_today(duration: str, expected: str) -> None:
    reset: Final = get_next_standardized_reset_time(duration, datetime(2026, 7, 15, 10, 17, tzinfo=timezone.utc), "Asia/Tokyo")

    assert reset == datetime.fromisoformat(expected)


def test_zero_hours_still_expires_immediately() -> None:
    now: Final = datetime(2026, 7, 15, 10, 17, tzinfo=timezone.utc)

    assert get_next_standardized_reset_time("0h", now, "Asia/Tokyo") == now

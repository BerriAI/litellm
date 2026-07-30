import unittest
from datetime import datetime, time, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import litellm.litellm_core_utils.duration_parser as duration_parser
from litellm.litellm_core_utils.duration_parser import (
    duration_in_seconds,
    get_next_rolling_reset_time,
    get_next_standardized_reset_time,
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
        weekly_expected = datetime(
            2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc
        )  # Next Monday
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
        # US/Pacific (UTC-7): 3:30 PM, so next 2h reset is 4:00 PM
        pacific = ZoneInfo("US/Pacific")
        pacific_expected = datetime(2023, 5, 15, 16, 0, 0, tzinfo=pacific)
        pacific_result = get_next_standardized_reset_time("2h", base_time, "US/Pacific")
        self.assertEqual(pacific_result, pacific_expected)

        # Test minute reset in different timezones
        # Europe/London (UTC+1): 11:30 PM, so next 15m reset is 11:45 PM
        london = ZoneInfo("Europe/London")
        london_expected = datetime(2023, 5, 15, 23, 45, 0, tzinfo=london)
        london_result = get_next_standardized_reset_time(
            "15m", base_time, "Europe/London"
        )
        self.assertEqual(london_result, london_expected)

        # Test Bangkok timezone (UTC+7): 5:30 AM next day, so next reset is midnight the day after
        bangkok = ZoneInfo("Asia/Bangkok")
        bangkok_expected = datetime(2023, 5, 17, 0, 0, 0, tzinfo=bangkok)
        bangkok_result = get_next_standardized_reset_time(
            "1d", base_time, "Asia/Bangkok"
        )
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
        midnight_minute_result = get_next_standardized_reset_time(
            "30m", near_midnight, "UTC"
        )
        self.assertEqual(midnight_minute_result, midnight_minute_expected)

        # Invalid timezone - should fall back to UTC
        invalid_tz_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=timezone.utc)
        invalid_tz_result = get_next_standardized_reset_time(
            "1d", on_hour, "NonExistentTimeZone"
        )
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
        sydney_result = get_next_standardized_reset_time(
            "1d", base_time, "Australia/Sydney"
        )
        self.assertEqual(sydney_result, sydney_expected)

        # America/Chicago (UTC-5): at 15:00 UTC it's 10:00 CDT → next midnight is May 16 00:00 CDT
        chicago = ZoneInfo("America/Chicago")
        chicago_expected = datetime(2023, 5, 16, 0, 0, 0, tzinfo=chicago)
        chicago_result = get_next_standardized_reset_time(
            "1d", base_time, "America/Chicago"
        )
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


class TestResetTimeOfDay(unittest.TestCase):
    """A configurable reset_time_of_day shifts day/week/month resets off midnight."""

    def test_daily_reset_before_offset_is_today(self):
        now = datetime(2023, 5, 15, 8, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1d", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 15, 12, 0, 0, tzinfo=timezone.utc))

    def test_daily_reset_after_offset_is_tomorrow(self):
        now = datetime(2023, 5, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1d", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 16, 12, 0, 0, tzinfo=timezone.utc))

    def test_daily_reset_exactly_at_offset_rolls_forward(self):
        now = datetime(2023, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1d", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 16, 12, 0, 0, tzinfo=timezone.utc))

    def test_daily_reset_with_seconds_offset(self):
        now = datetime(2023, 5, 15, 8, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1d", now, "UTC", reset_time_of_day=time(9, 30, 15)
        )
        self.assertEqual(result, datetime(2023, 5, 15, 9, 30, 15, tzinfo=timezone.utc))

    def test_offset_applies_in_configured_timezone(self):
        # 2023-05-15 22:30 UTC == 2023-05-16 01:30 in Jerusalem (IDT, UTC+3),
        # so the next noon-Jerusalem reset is 2023-05-16 12:00 IDT.
        now = datetime(2023, 5, 15, 22, 30, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1d", now, "Asia/Jerusalem", reset_time_of_day=time(12, 0)
        )
        jerusalem = result.astimezone(ZoneInfo("Asia/Jerusalem"))
        self.assertEqual(
            (jerusalem.year, jerusalem.month, jerusalem.day), (2023, 5, 16)
        )
        self.assertEqual(jerusalem.hour, 12)
        self.assertEqual(jerusalem.minute, 0)

    def test_weekly_reset_lands_on_monday_at_offset(self):
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "7d", wednesday, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 22, 12, 0, 0, tzinfo=timezone.utc))

    def test_weekly_reset_today_is_monday_before_offset_is_today(self):
        monday_morning = datetime(2023, 5, 22, 9, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "7d", monday_morning, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 22, 12, 0, 0, tzinfo=timezone.utc))

    def test_weekly_reset_today_is_monday_after_offset_is_next_week(self):
        monday_afternoon = datetime(2023, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "7d", monday_afternoon, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 29, 12, 0, 0, tzinfo=timezone.utc))

    def test_monthly_30d_lands_on_first_at_offset(self):
        now = datetime(2023, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "30d", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_monthly_1mo_today_is_first_before_offset_is_today(self):
        now = datetime(2023, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1mo", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_monthly_year_rollover_at_offset(self):
        now = datetime(2023, 12, 15, 9, 0, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "1mo", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_custom_day_reset_applies_offset(self):
        now = datetime(2023, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = get_next_standardized_reset_time(
            "3d", now, "UTC", reset_time_of_day=time(12, 0)
        )
        self.assertEqual(result, datetime(2023, 5, 18, 12, 0, 0, tzinfo=timezone.utc))

    def test_sub_day_durations_ignore_offset(self):
        base = datetime(2023, 5, 15, 15, 20, 30, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_standardized_reset_time(
                "2h", base, "UTC", reset_time_of_day=time(12, 0)
            ),
            datetime(2023, 5, 15, 16, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            get_next_standardized_reset_time(
                "30m", base, "UTC", reset_time_of_day=time(12, 0)
            ),
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


class TestRollingResetTime(unittest.TestCase):
    """Rolling resets measure the duration from the budget's own anchor and keep its
    time of day, instead of snapping to a shared calendar boundary.
    """

    def test_rolling_month_lands_on_same_day_and_time_next_month(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("1mo", base_time, "UTC"),
            datetime(2026, 8, 30, 14, 37, 0, tzinfo=timezone.utc),
        )

    def test_rolling_differs_from_calendar_for_month_durations(self):
        """The whole point of the feature: 1mo and 30d must stop collapsing to the 1st."""
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        for duration in ("1mo", "30d"):
            calendar = get_next_standardized_reset_time(duration, base_time, "UTC")
            rolling = get_next_rolling_reset_time(duration, base_time, "UTC")
            self.assertEqual(calendar, datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc))
            self.assertNotEqual(rolling, calendar)
            self.assertEqual(rolling.hour, 14)
            self.assertEqual(rolling.minute, 37)

    def test_rolling_30d_is_exactly_30_days(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("30d", base_time, "UTC"),
            datetime(2026, 8, 29, 14, 37, 0, tzinfo=timezone.utc),
        )

    def test_rolling_month_clamps_to_shorter_target_month(self):
        base_time = datetime(2026, 1, 31, 9, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("1mo", base_time, "UTC"),
            datetime(2026, 2, 28, 9, 0, 0, tzinfo=timezone.utc),
        )

    def test_rolling_supports_multi_month_durations_that_calendar_rejects(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            get_next_standardized_reset_time("2mo", base_time, "UTC")
        self.assertEqual(
            get_next_rolling_reset_time("2mo", base_time, "UTC"),
            datetime(2026, 9, 30, 14, 37, 0, tzinfo=timezone.utc),
        )

    def test_rolling_year_end_rollover(self):
        base_time = datetime(2026, 12, 15, 9, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("1mo", base_time, "UTC"),
            datetime(2027, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
        )

    def test_previous_anchor_absorbs_a_late_reset_job(self):
        """A reset job that fires late must not push the window out permanently."""
        previous_reset_at = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        ran_late = previous_reset_at + timedelta(minutes=3)
        self.assertEqual(
            get_next_rolling_reset_time("1mo", ran_late, "UTC", previous_reset_at=previous_reset_at),
            datetime(2026, 8, 30, 14, 37, 0, tzinfo=timezone.utc),
        )

    def test_late_reset_job_does_not_accumulate_drift_over_many_periods(self):
        """Twelve consecutive late runs must not move the reset's time of day at all.

        Without anchoring on `previous_reset_at` each late run would add its own
        lateness, so the window would creep later every single period.
        """
        anchor = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        previous_reset_at = anchor
        for _ in range(12):
            ran_late = previous_reset_at + timedelta(minutes=7)
            previous_reset_at = get_next_rolling_reset_time(
                "1mo", ran_late, "UTC", previous_reset_at=previous_reset_at
            )
        self.assertEqual(previous_reset_at.hour, anchor.hour)
        self.assertEqual(previous_reset_at.minute, anchor.minute)
        self.assertEqual(previous_reset_at.second, anchor.second)

    def test_month_end_clamp_walks_the_day_backwards_and_never_forwards(self):
        """A day-30 anchor is pulled back to 28 by February and stays there.

        Inherent to anchoring on the previous reset: once clamped, the original
        day-of-month is lost. Pinned here so the behavior is deliberate.
        """
        previous_reset_at = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        days = []
        for _ in range(9):
            previous_reset_at = get_next_rolling_reset_time(
                "1mo", previous_reset_at, "UTC", previous_reset_at=previous_reset_at
            )
            days.append(previous_reset_at.day)

        self.assertEqual(days, [30, 30, 30, 30, 30, 30, 28, 28, 28])
        self.assertTrue(all(later <= earlier for earlier, later in zip(days, days[1:])))

    def test_stale_anchor_jumps_past_now(self):
        """After a long outage the next reset must be in the future, not the past."""
        previous_reset_at = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        now = datetime(2026, 11, 2, 8, 0, 0, tzinfo=timezone.utc)
        result = get_next_rolling_reset_time("1mo", now, "UTC", previous_reset_at=previous_reset_at)
        self.assertGreater(result, now)
        self.assertEqual(result, datetime(2026, 12, 2, 8, 0, 0, tzinfo=timezone.utc))

    def test_result_is_always_strictly_after_now(self):
        """An anchor exactly one period old must roll forward, never return now."""
        previous_reset_at = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        now = datetime(2026, 8, 30, 14, 37, 0, tzinfo=timezone.utc)
        result = get_next_rolling_reset_time("1mo", now, "UTC", previous_reset_at=previous_reset_at)
        self.assertGreater(result, now)

    def test_rolling_respects_timezone(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        result = get_next_rolling_reset_time("1mo", base_time, "Asia/Kolkata")
        self.assertEqual(result, datetime(2026, 8, 30, 14, 37, 0, tzinfo=timezone.utc))
        self.assertEqual(result.astimezone(ZoneInfo("Asia/Kolkata")).hour, 20)

    def test_naive_previous_anchor_is_treated_as_utc(self):
        previous_reset_at = datetime(2026, 7, 30, 14, 37, 0)
        ran_late = datetime(2026, 7, 30, 14, 40, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("1mo", ran_late, "UTC", previous_reset_at=previous_reset_at),
            datetime(2026, 8, 30, 14, 37, 0, tzinfo=timezone.utc),
        )

    def test_sub_day_durations_roll_from_now(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("12h", base_time, "UTC"),
            datetime(2026, 7, 31, 2, 37, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            get_next_rolling_reset_time("30m", base_time, "UTC"),
            datetime(2026, 7, 30, 15, 7, 0, tzinfo=timezone.utc),
        )

    def test_word_form_durations_are_normalized(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("monthly", base_time, "UTC"),
            get_next_rolling_reset_time("30d", base_time, "UTC"),
        )

    def test_invalid_duration_falls_back_to_calendar_instead_of_raising(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(
            get_next_rolling_reset_time("garbage", base_time, "UTC"),
            get_next_standardized_reset_time("garbage", base_time, "UTC"),
        )

    def test_zero_duration_expires_immediately(self):
        base_time = datetime(2026, 7, 30, 14, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(get_next_rolling_reset_time("0d", base_time, "UTC"), base_time)


if __name__ == "__main__":
    unittest.main()

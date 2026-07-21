import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time


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

    def test_configurable_weekly_reset_day(self):
        """Test that weekly reset day can be configured via weekly_reset_day parameter."""
        # Wednesday May 17, 2023 at 15:45 UTC
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)

        # Default (Monday) - next Monday is May 22
        monday_result = get_next_standardized_reset_time("7d", wednesday, "UTC")
        self.assertEqual(monday_result, datetime(2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc))

        # Sunday reset - next Sunday is May 21
        sunday_result = get_next_standardized_reset_time(
            "7d", wednesday, "UTC", weekly_reset_day="sunday"
        )
        self.assertEqual(sunday_result, datetime(2023, 5, 21, 0, 0, 0, tzinfo=timezone.utc))

        # Wednesday reset - if today is Wednesday, next reset is 7 days later (May 24)
        wed_result = get_next_standardized_reset_time(
            "7d", wednesday, "UTC", weekly_reset_day="wednesday"
        )
        self.assertEqual(wed_result, datetime(2023, 5, 24, 0, 0, 0, tzinfo=timezone.utc))

        # Friday reset - next Friday is May 19
        fri_result = get_next_standardized_reset_time(
            "7d", wednesday, "UTC", weekly_reset_day="friday"
        )
        self.assertEqual(fri_result, datetime(2023, 5, 19, 0, 0, 0, tzinfo=timezone.utc))

    def test_configurable_weekly_reset_day_with_1w(self):
        """Test that 1w duration also respects weekly_reset_day."""
        # Saturday May 20, 2023 at 10:00 UTC
        saturday = datetime(2023, 5, 20, 10, 0, 0, tzinfo=timezone.utc)

        # Sunday reset - next Sunday is May 21 (1 day away)
        sunday_result = get_next_standardized_reset_time(
            "1w", saturday, "UTC", weekly_reset_day="sunday"
        )
        self.assertEqual(sunday_result, datetime(2023, 5, 21, 0, 0, 0, tzinfo=timezone.utc))

        # Tuesday reset - next Tuesday is May 23 (3 days away)
        tuesday_result = get_next_standardized_reset_time(
            "1w", saturday, "UTC", weekly_reset_day="tuesday"
        )
        self.assertEqual(tuesday_result, datetime(2023, 5, 23, 0, 0, 0, tzinfo=timezone.utc))

    def test_configurable_weekly_reset_day_on_reset_day(self):
        """Test that if today is the reset day, next reset is 7 days away."""
        # Monday May 15, 2023 at 08:00 UTC
        monday = datetime(2023, 5, 15, 8, 0, 0, tzinfo=timezone.utc)

        # Monday reset - since today is Monday, next reset is 7 days later (May 22)
        result = get_next_standardized_reset_time(
            "7d", monday, "UTC", weekly_reset_day="monday"
        )
        self.assertEqual(result, datetime(2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc))

        # Sunday reset - next Sunday is May 21 (6 days away)
        sunday_result = get_next_standardized_reset_time(
            "7d", monday, "UTC", weekly_reset_day="sunday"
        )
        self.assertEqual(sunday_result, datetime(2023, 5, 21, 0, 0, 0, tzinfo=timezone.utc))

    def test_invalid_weekly_reset_day_falls_back_to_monday(self):
        """Test that an invalid weekly_reset_day falls back to Monday and logs a warning."""
        # Wednesday May 17, 2023
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)

        # Invalid day - should fall back to Monday (next Monday = May 22)
        import logging

        with self.assertLogs("LiteLLM", level="WARNING") as log_ctx:
            result = get_next_standardized_reset_time(
                "7d", wednesday, "UTC", weekly_reset_day="funday"
            )
        self.assertEqual(result, datetime(2023, 5, 22, 0, 0, 0, tzinfo=timezone.utc))
        # Verify a warning was logged about the invalid day
        self.assertTrue(
            any("funday" in msg for msg in log_ctx.output),
            f"Expected warning about invalid day 'funday', got: {log_ctx.output}",
        )

    def test_weekly_reset_day_case_insensitive(self):
        """Test that weekly_reset_day is case-insensitive."""
        # Wednesday May 17, 2023
        wednesday = datetime(2023, 5, 17, 15, 45, 0, tzinfo=timezone.utc)

        # Uppercase - should work same as lowercase
        result = get_next_standardized_reset_time(
            "7d", wednesday, "UTC", weekly_reset_day="SUNDAY"
        )
        self.assertEqual(result, datetime(2023, 5, 21, 0, 0, 0, tzinfo=timezone.utc))

    def test_weekly_reset_day_with_timezone(self):
        """Test weekly_reset_day combined with a non-UTC timezone."""
        # Friday May 19, 2023 at 22:30 UTC = 15:30 US/Pacific (same day)
        friday_utc = datetime(2023, 5, 19, 22, 30, 0, tzinfo=timezone.utc)
        pacific = ZoneInfo("US/Pacific")

        # Sunday reset in Pacific - next Sunday midnight Pacific = May 21 00:00 PDT
        # = May 21 07:00 UTC
        result = get_next_standardized_reset_time(
            "7d", friday_utc, "US/Pacific", weekly_reset_day="sunday"
        )
        expected = datetime(2023, 5, 21, 0, 0, 0, tzinfo=pacific)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

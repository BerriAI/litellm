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
        tokyo_result = get_next_standardized_reset_time(
            "1d", base_time, "Asia/Tokyo"
        )
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


if __name__ == "__main__":
    unittest.main()

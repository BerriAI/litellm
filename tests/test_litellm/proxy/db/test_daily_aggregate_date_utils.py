"""
Tests for daily-aggregate `date` field conversion helpers.

These guard the boundary between the in-memory `YYYY-MM-DD` string representation
of a daily-aggregate transaction `date` and the `DateTime @db.Date` column
that Prisma now persists.
"""

from datetime import date, datetime, timezone

import pytest

from litellm.proxy.db.daily_aggregate_date_utils import to_date_str, to_db_date


class TestToDbDate:
    def test_should_convert_yyyy_mm_dd_string_to_utc_midnight_datetime(self):
        result = to_db_date("2026-04-03")
        assert result == datetime(2026, 4, 3, tzinfo=timezone.utc)
        assert result.tzinfo is not None

    def test_should_strip_time_component_from_iso_string(self):
        # In-memory transactions always carry a date-only string, but be lenient
        # so we can also ingest stray "YYYY-MM-DDTHH:MM:SS" inputs without
        # silently truncating to an unrelated date.
        result = to_db_date("2026-04-03T18:30:00")
        assert result == datetime(2026, 4, 3, tzinfo=timezone.utc)

    def test_should_convert_naive_datetime_to_utc_aware_datetime(self):
        naive = datetime(2026, 4, 3, 0, 0, 0)
        result = to_db_date(naive)
        assert result == datetime(2026, 4, 3, tzinfo=timezone.utc)

    def test_should_preserve_aware_datetime_unchanged(self):
        aware = datetime(2026, 4, 3, tzinfo=timezone.utc)
        assert to_db_date(aware) is aware

    def test_should_convert_date_object_to_utc_midnight_datetime(self):
        result = to_db_date(date(2026, 4, 3))
        assert result == datetime(2026, 4, 3, tzinfo=timezone.utc)

    def test_should_pass_through_none(self):
        assert to_db_date(None) is None

    def test_should_raise_on_non_date_like_value(self):
        with pytest.raises(TypeError):
            to_db_date(12345)  # type: ignore[arg-type]


class TestToDateStr:
    def test_should_format_datetime_as_yyyy_mm_dd(self):
        assert (
            to_date_str(datetime(2026, 4, 3, 18, 30, tzinfo=timezone.utc))
            == "2026-04-03"
        )

    def test_should_format_date_as_yyyy_mm_dd(self):
        assert to_date_str(date(2026, 4, 3)) == "2026-04-03"

    def test_should_pass_through_yyyy_mm_dd_string(self):
        assert to_date_str("2026-04-03") == "2026-04-03"

    def test_should_strip_time_component_from_iso_string(self):
        assert to_date_str("2026-04-03T00:00:00+00:00") == "2026-04-03"

    def test_should_return_none_for_none(self):
        assert to_date_str(None) is None

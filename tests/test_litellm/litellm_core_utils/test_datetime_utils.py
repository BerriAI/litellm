"""
parse_utc_datetime is the single allowed ISO-8601 parse entrypoint under litellm/proxy/
(enforced by semgrep). Naive input is defined as UTC; aware input keeps its offset. The
result must always be comparable against datetime.now(timezone.utc) without TypeError.
"""

from datetime import datetime, timedelta, timezone

import pytest

from litellm.litellm_core_utils.datetime_utils import parse_utc_datetime


def test_naive_string_is_assumed_utc():
    result = parse_utc_datetime("2026-01-20T00:00:00")
    assert result == datetime(2026, 1, 20, tzinfo=timezone.utc)


def test_z_suffix_string_parses_aware():
    result = parse_utc_datetime("2026-01-20T00:00:00Z")
    assert result == datetime(2026, 1, 20, tzinfo=timezone.utc)


def test_offset_string_keeps_offset():
    result = parse_utc_datetime("2026-01-20T05:30:00+05:30")
    assert result == datetime(2026, 1, 20, tzinfo=timezone.utc)
    assert result.utcoffset() == timedelta(hours=5, minutes=30)


def test_naive_datetime_passthrough_becomes_aware():
    result = parse_utc_datetime(datetime(2026, 1, 20))
    assert result == datetime(2026, 1, 20, tzinfo=timezone.utc)


def test_aware_datetime_passthrough_unchanged():
    aware = datetime(2026, 1, 20, tzinfo=timezone.utc)
    assert parse_utc_datetime(aware) is aware


def test_result_always_comparable_to_utc_now():
    for value in ("2026-01-20T00:00:00", "2026-01-20T00:00:00Z", datetime(2026, 1, 20)):
        assert parse_utc_datetime(value) < datetime.now(timezone.utc) or parse_utc_datetime(
            value
        ) >= datetime.now(timezone.utc)


def test_invalid_string_raises():
    with pytest.raises(ValueError):
        parse_utc_datetime("not-a-date")

"""
Pure-function tests for schedule parsing and next-run computation.
No DB, no FastAPI — just exercise the parsing branches.
"""

from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.scheduled_tasks.schedule import (
    _FAR_FUTURE,
    compute_next_run,
    validate_schedule,
)


class TestValidateSchedule:
    def test_interval_ok(self):
        validate_schedule(kind="interval", spec="30s", tz=None)
        validate_schedule(kind="interval", spec="5m", tz=None)
        validate_schedule(kind="interval", spec="2h", tz=None)
        validate_schedule(kind="interval", spec="1d", tz=None)

    def test_interval_bad_spec(self):
        with pytest.raises(ValueError, match="invalid interval"):
            validate_schedule(kind="interval", spec="banana", tz=None)
        with pytest.raises(ValueError, match="invalid interval"):
            validate_schedule(kind="interval", spec="5x", tz=None)

    def test_cron_ok(self):
        validate_schedule(kind="cron", spec="0 9 * * 1-5", tz="America/Los_Angeles")
        validate_schedule(kind="cron", spec="*/15 * * * *", tz=None)

    def test_cron_bad_spec(self):
        with pytest.raises(ValueError, match="invalid cron"):
            validate_schedule(kind="cron", spec="not a cron", tz=None)

    def test_cron_bad_tz(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_schedule(kind="cron", spec="0 9 * * *", tz="Europe/Atlantis")

    def test_once_iso_with_z(self):
        validate_schedule(kind="once", spec="2030-06-15T12:00:00Z", tz=None)

    def test_once_iso_with_offset(self):
        validate_schedule(
            kind="once",
            spec="2030-06-15T12:00:00+00:00",
            tz=None,
        )

    def test_once_iso_with_microseconds(self):
        validate_schedule(
            kind="once",
            spec="2030-06-15T12:00:00.123456Z",
            tz=None,
        )

    def test_once_naive_iso_treated_utc(self):
        validate_schedule(kind="once", spec="2030-06-15T12:00:00", tz=None)

    def test_once_rejects_relative_duration(self):
        with pytest.raises(ValueError, match="ISO-8601"):
            validate_schedule(kind="once", spec="1m", tz=None)

    def test_once_rejects_garbage(self):
        with pytest.raises(ValueError, match="ISO-8601"):
            validate_schedule(kind="once", spec="not a date", tz=None)

    def test_once_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_schedule(kind="once", spec="", tz=None)

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown schedule_kind"):
            validate_schedule(kind="weekly", spec="x", tz=None)


class TestComputeNextRun:
    def test_interval_advances(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(kind="interval", spec="5m", tz=None, from_time=now)
        assert nxt == now + timedelta(minutes=5)

    def test_interval_naive_input_treated_utc(self):
        naive = datetime(2026, 4, 29, 12, 0, 0)
        nxt = compute_next_run(kind="interval", spec="1h", tz=None, from_time=naive)
        assert nxt.tzinfo is not None
        assert nxt.utcoffset() == timedelta(0)

    def test_cron_returns_utc(self):
        now = datetime(2026, 4, 29, 0, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(
            kind="cron",
            spec="0 9 * * *",
            tz="America/Los_Angeles",
            from_time=now,
        )
        assert nxt.tzinfo is not None
        assert nxt.utcoffset() == timedelta(0)
        assert nxt > now

    def test_once_uses_schedule_spec_verbatim(self):
        """Regression: kind='once' must use the supplied ISO timestamp,
        not silently park at year 9999."""
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        fire_at = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(
            kind="once",
            spec=fire_at.isoformat(),
            tz=None,
            from_time=now,
        )
        assert nxt == fire_at
        assert nxt != _FAR_FUTURE

    def test_once_accepts_z_suffix(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(
            kind="once",
            spec="2030-06-15T12:00:00Z",
            tz=None,
            from_time=now,
        )
        assert nxt == datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_once_accepts_microseconds(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(
            kind="once",
            spec="2030-06-15T12:00:00.500000Z",
            tz=None,
            from_time=now,
        )
        assert nxt == datetime(2030, 6, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)

    def test_once_naive_input_treated_utc(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(
            kind="once",
            spec="2030-06-15T12:00:00",
            tz=None,
            from_time=now,
        )
        assert nxt == datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_once_rejects_unparseable_spec(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="ISO-8601"):
            compute_next_run(kind="once", spec="1m", tz=None, from_time=now)

    def test_unknown_kind_raises(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            compute_next_run(kind="weekly", spec="x", tz=None, from_time=now)

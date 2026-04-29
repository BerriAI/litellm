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

    def test_once_ok(self):
        validate_schedule(kind="once", spec="ignored", tz=None)

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

    def test_once_returns_far_future(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_run(kind="once", spec="x", tz=None, from_time=now)
        assert nxt == _FAR_FUTURE

    def test_unknown_kind_raises(self):
        now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            compute_next_run(kind="weekly", spec="x", tz=None, from_time=now)

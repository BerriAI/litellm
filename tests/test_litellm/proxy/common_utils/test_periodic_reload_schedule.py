import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.common_utils.periodic_reload_schedule import (
    ReloadSchedule,
    next_run_at,
    parse_reload_schedule,
    pod_reload_is_due,
    read_reload_schedule,
    record_reload_run,
    reload_schedule_status,
    serialize_reload_schedule,
)

LAST_RUN = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)


def test_parse_round_trips_through_serialization():
    schedule = ReloadSchedule(interval_hours=6, force_reload=True, last_run=LAST_RUN)

    assert parse_reload_schedule(json.loads(serialize_reload_schedule(schedule))) == schedule


def test_parse_treats_naive_last_run_as_utc():
    """Rows written before last_run was persisted as tz-aware must still parse"""
    schedule = parse_reload_schedule({"interval_hours": 6, "last_run": "2024-01-01T06:00:00"})

    assert schedule.last_run == LAST_RUN


@pytest.mark.parametrize(
    "param_value",
    [None, "not-a-dict", {}, {"interval_hours": "6"}, {"last_run": "garbage"}],
)
def test_parse_tolerates_unusable_values(param_value):
    schedule = parse_reload_schedule(param_value)

    assert schedule.interval_hours is None
    assert schedule.last_run is None


def test_status_reports_persisted_last_run_and_next_run():
    status = reload_schedule_status(ReloadSchedule(interval_hours=6, last_run=LAST_RUN))

    assert status == {
        "scheduled": True,
        "interval_hours": 6,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": "2024-01-01T12:00:00+00:00",
    }


def test_status_without_interval_is_not_scheduled():
    assert reload_schedule_status(None)["scheduled"] is False
    assert reload_schedule_status(ReloadSchedule(last_run=LAST_RUN)) == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": None,
    }


def test_next_run_needs_both_interval_and_last_run():
    assert next_run_at(ReloadSchedule(interval_hours=6)) is None
    assert next_run_at(ReloadSchedule(last_run=LAST_RUN)) is None


@pytest.mark.parametrize(
    "schedule, pod_last_reload, expected",
    [
        (ReloadSchedule(force_reload=True), LAST_RUN, True),
        (ReloadSchedule(), None, False),
        (ReloadSchedule(interval_hours=6), None, True),
        (ReloadSchedule(interval_hours=6), datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc), False),
        (ReloadSchedule(interval_hours=6), datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc), True),
    ],
)
def test_pod_reload_is_due(schedule, pod_last_reload, expected):
    assert (
        pod_reload_is_due(
            schedule=schedule,
            pod_last_reload=pod_last_reload,
            current_time=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            description="test",
        )
        is expected
    )


def test_pod_reload_decision_ignores_persisted_last_run():
    """
    A pod that has never reloaded must refresh its own copy even when another pod
    already stamped last_run within the interval
    """
    assert (
        pod_reload_is_due(
            schedule=ReloadSchedule(interval_hours=6, last_run=datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc)),
            pod_last_reload=None,
            current_time=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            description="test",
        )
        is True
    )


@pytest.mark.asyncio
async def test_record_reload_run_stamps_last_run_and_keeps_interval():
    existing = MagicMock()
    existing.param_value = {"interval_hours": 6, "force_reload": True}
    prisma_client = MagicMock()
    prisma_client.db.litellm_config.find_unique = AsyncMock(return_value=existing)
    prisma_client.db.litellm_config.upsert = AsyncMock(return_value=None)

    await record_reload_run(prisma_client, "model_cost_map_reload_config", LAST_RUN)

    written = json.loads(prisma_client.db.litellm_config.upsert.call_args[1]["data"]["update"]["param_value"])
    assert written == {
        "interval_hours": 6,
        "force_reload": False,
        "last_run": "2024-01-01T06:00:00+00:00",
    }


@pytest.mark.asyncio
async def test_read_reload_schedule_returns_none_for_missing_row():
    prisma_client = MagicMock()
    prisma_client.db.litellm_config.find_unique = AsyncMock(return_value=None)

    assert await read_reload_schedule(prisma_client, "model_cost_map_reload_config") is None

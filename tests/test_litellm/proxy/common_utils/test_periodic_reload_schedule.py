from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.common_utils.periodic_reload_schedule import (
    ReloadSchedule,
    next_run_at,
    parse_reload_schedule,
    pod_reload_is_due,
    read_reload_schedule,
    record_manual_reload,
    record_reload_run,
    reload_schedule_status,
    write_reload_interval,
)

LAST_RUN = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
NOW = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def _row(param_value=None, reload_requested_at=None, last_run_at=None):
    return SimpleNamespace(
        param_name="model_cost_map_reload_config",
        param_value=param_value,
        reload_requested_at=reload_requested_at,
        last_run_at=last_run_at,
    )


def _mock_prisma(row=None):
    prisma_client = MagicMock()
    prisma_client.db.litellm_config.find_unique = AsyncMock(return_value=row)
    prisma_client.db.litellm_config.upsert = AsyncMock(return_value=None)
    prisma_client.db.litellm_config.update_many = AsyncMock(return_value=1)
    return prisma_client


def test_parse_reads_interval_from_json_and_state_from_columns():
    schedule = parse_reload_schedule(
        _row(param_value={"interval_hours": 6}, reload_requested_at=NOW, last_run_at=LAST_RUN)
    )

    assert schedule == ReloadSchedule(interval_hours=6, reload_requested_at=NOW, last_run_at=LAST_RUN)


def test_parse_treats_naive_column_timestamps_as_utc():
    schedule = parse_reload_schedule(
        _row(reload_requested_at=NOW.replace(tzinfo=None), last_run_at=LAST_RUN.replace(tzinfo=None))
    )

    assert (schedule.reload_requested_at, schedule.last_run_at) == (NOW, LAST_RUN)


@pytest.mark.parametrize(
    "param_value",
    [None, "not-a-dict", {}, {"interval_hours": "6"}, {"interval_hours": None}],
)
def test_parse_tolerates_unusable_param_values(param_value):
    assert parse_reload_schedule(_row(param_value=param_value)).interval_hours is None


def test_parse_ignores_legacy_json_force_reload():
    """Rows written by pre-column versions carry force_reload in the JSON; honoring it
    would re-trigger a reload every poll because nothing clears the JSON copy"""
    schedule = parse_reload_schedule(_row(param_value={"interval_hours": 6, "force_reload": True}))

    assert schedule == ReloadSchedule(interval_hours=6, reload_requested_at=None, last_run_at=None)


def test_status_reports_persisted_last_run_and_next_run():
    status = reload_schedule_status(ReloadSchedule(interval_hours=6, last_run_at=LAST_RUN))

    assert status == {
        "scheduled": True,
        "interval_hours": 6,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": "2024-01-01T12:00:00+00:00",
    }


def test_status_without_interval_is_not_scheduled():
    assert reload_schedule_status(None)["scheduled"] is False
    assert reload_schedule_status(ReloadSchedule(last_run_at=LAST_RUN)) == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": None,
    }


def test_next_run_needs_both_interval_and_last_run():
    assert next_run_at(ReloadSchedule(interval_hours=6)) is None
    assert next_run_at(ReloadSchedule(last_run_at=LAST_RUN)) is None


@pytest.mark.parametrize(
    "schedule, pod_data_loaded_at, expected",
    [
        (ReloadSchedule(reload_requested_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)), LAST_RUN, True),
        (
            ReloadSchedule(
                interval_hours=6,
                reload_requested_at=datetime(2024, 1, 1, 11, 30, tzinfo=timezone.utc),
                last_run_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            ),
            datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            True,
        ),
        (ReloadSchedule(reload_requested_at=LAST_RUN), LAST_RUN, False),
        (ReloadSchedule(reload_requested_at=datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc)), LAST_RUN, False),
        (ReloadSchedule(reload_requested_at=LAST_RUN), datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc), False),
        (ReloadSchedule(interval_hours=6, reload_requested_at=LAST_RUN, last_run_at=LAST_RUN), LAST_RUN, True),
        (ReloadSchedule(), LAST_RUN, False),
        (ReloadSchedule(interval_hours=6), datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc), True),
        (
            ReloadSchedule(interval_hours=6, last_run_at=datetime(2024, 1, 1, 6, 30, tzinfo=timezone.utc)),
            datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            False,
        ),
        (ReloadSchedule(interval_hours=6, last_run_at=LAST_RUN), LAST_RUN, True),
    ],
)
def test_pod_reload_is_due(schedule, pod_data_loaded_at, expected):
    assert (
        pod_reload_is_due(
            schedule=schedule,
            pod_data_loaded_at=pod_data_loaded_at,
            current_time=NOW,
            description="test",
        )
        is expected
    )


def test_manual_request_reaches_every_pod_with_older_data():
    """Nothing clears reload_requested_at, so each pod reloads exactly once per request:
    due while its data predates the request, not due once refreshed or booted after it"""
    schedule = ReloadSchedule(reload_requested_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc))

    assert (
        pod_reload_is_due(schedule=schedule, pod_data_loaded_at=LAST_RUN, current_time=NOW, description="test") is True
    )
    assert (
        pod_reload_is_due(
            schedule=schedule,
            pod_data_loaded_at=datetime(2024, 1, 1, 11, 0, 1, tzinfo=timezone.utc),
            current_time=NOW,
            description="test",
        )
        is False
    )


def test_pod_reload_decision_ignores_persisted_last_run():
    """A pod with stale data must refresh even when another pod already stamped
    last_run_at within the interval"""
    assert (
        pod_reload_is_due(
            schedule=ReloadSchedule(interval_hours=6, last_run_at=datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc)),
            pod_data_loaded_at=LAST_RUN,
            current_time=NOW,
            description="test",
        )
        is True
    )


def test_schedule_that_never_ran_fires_immediately():
    """A fresh schedule must not wait a full interval for its first run, even on a pod
    whose own data is boot-fresh"""
    assert (
        pod_reload_is_due(
            schedule=ReloadSchedule(interval_hours=6, last_run_at=None),
            pod_data_loaded_at=datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc),
            current_time=NOW,
            description="test",
        )
        is True
    )


@pytest.mark.asyncio
async def test_read_reload_schedule_returns_none_for_missing_row():
    assert await read_reload_schedule(_mock_prisma(row=None), "model_cost_map_reload_config") is None


@pytest.mark.asyncio
async def test_read_reload_schedule_surfaces_request_on_interval_less_row():
    """A manual reload on a proxy with no schedule creates a row with only the columns
    set; the request must still reach other pods"""
    prisma_client = _mock_prisma(row=_row(param_value=None, reload_requested_at=NOW))

    schedule = await read_reload_schedule(prisma_client, "model_cost_map_reload_config")

    assert schedule == ReloadSchedule(interval_hours=None, reload_requested_at=NOW, last_run_at=None)


@pytest.mark.asyncio
async def test_write_reload_interval_touches_only_param_value():
    prisma_client = _mock_prisma()

    await write_reload_interval(prisma_client, "model_cost_map_reload_config", 12)

    data = prisma_client.db.litellm_config.upsert.await_args.kwargs["data"]
    assert set(data["update"]) == {"param_value"}
    assert set(data["create"]) == {"param_name", "param_value"}


@pytest.mark.asyncio
async def test_record_reload_run_updates_last_run_without_creating_or_clearing():
    """update_many so a schedule deleted mid-poll stays deleted, and the untouched
    reload_requested_at keeps fanning the request out to pods that have not seen it"""
    prisma_client = _mock_prisma()

    await record_reload_run(prisma_client, "model_cost_map_reload_config", LAST_RUN)

    kwargs = prisma_client.db.litellm_config.update_many.await_args.kwargs
    assert kwargs == {"data": {"last_run_at": LAST_RUN}, "where": {"param_name": "model_cost_map_reload_config"}}
    prisma_client.db.litellm_config.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_record_manual_reload_stamps_request_and_last_run():
    prisma_client = _mock_prisma()

    await record_manual_reload(prisma_client, "model_cost_map_reload_config", LAST_RUN)

    data = prisma_client.db.litellm_config.upsert.await_args.kwargs["data"]
    assert data["update"] == {"last_run_at": LAST_RUN, "reload_requested_at": LAST_RUN}
    assert data["create"] == {
        "param_name": "model_cost_map_reload_config",
        "last_run_at": LAST_RUN,
        "reload_requested_at": LAST_RUN,
    }

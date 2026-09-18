from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.common_utils.periodic_reload_schedule import (
    ReloadSchedule,
    clear_reload_interval,
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


def _row(param_value=None, reload_revision=0, last_run_at=None):
    return SimpleNamespace(
        param_name="model_cost_map_reload_config",
        param_value=param_value,
        reload_revision=reload_revision,
        last_run_at=last_run_at,
    )


def _mock_prisma(row=None, upserted_revision=1):
    prisma_client = MagicMock()
    prisma_client.db.litellm_config.find_unique = AsyncMock(return_value=row)
    prisma_client.db.litellm_config.upsert = AsyncMock(return_value=_row(reload_revision=upserted_revision))
    prisma_client.db.litellm_config.update_many = AsyncMock(return_value=1)
    return prisma_client


class _FakeConfigTable:
    """In-memory stand-in for the prisma LiteLLM_Config actions, faithful on the parts the
    revision depends on: upsert creates at the column default and applies ``{"increment": 1}``,
    and delete drops the row along with its counter"""

    def __init__(self):
        self._rows = {}

    async def find_unique(self, where):
        return self._rows.get(where["param_name"])

    async def upsert(self, where, data):
        row = self._rows.get(where["param_name"])
        if row is None:
            created = data["create"]
            row = _row(
                param_value=created.get("param_value"),
                reload_revision=created.get("reload_revision", 0),
                last_run_at=created.get("last_run_at"),
            )
            self._rows[where["param_name"]] = row
            return row
        self._apply(row, data["update"])
        return row

    async def update_many(self, data, where):
        row = self._rows.get(where["param_name"])
        if row is None:
            return 0
        self._apply(row, data)
        return 1

    async def delete(self, where):
        return self._rows.pop(where["param_name"], None)

    @staticmethod
    def _apply(row, data):
        if "param_value" in data and data["param_value"] is None:
            raise ValueError("`data.param_value`: A value is required but not set")
        for field, value in data.items():
            increment = value.get("increment") if isinstance(value, dict) else None
            setattr(row, field, getattr(row, field) + increment if increment is not None else value)


def _fake_prisma(table):
    prisma_client = MagicMock()
    prisma_client.db.litellm_config = table
    return prisma_client


def test_parse_reads_interval_from_json_and_state_from_columns():
    schedule = parse_reload_schedule(_row(param_value={"interval_hours": 6}, reload_revision=7, last_run_at=LAST_RUN))

    assert schedule == ReloadSchedule(interval_hours=6, reload_revision=7, last_run_at=LAST_RUN)


def test_parse_treats_naive_column_timestamps_as_utc():
    schedule = parse_reload_schedule(_row(last_run_at=LAST_RUN.replace(tzinfo=None)))

    assert schedule.last_run_at == LAST_RUN


def test_parse_defaults_revision_when_the_column_is_null():
    """Rows written before the column existed read back as NULL and must not crash the
    comparison; nobody has applied revision 0, so treating it as 0 is a no-op"""
    assert parse_reload_schedule(_row(reload_revision=None)).reload_revision == 0


@pytest.mark.parametrize(
    "param_value",
    [None, "not-a-dict", '{"interval_hours": "6"}', {}, {"interval_hours": "6"}, {"interval_hours": None}],
)
def test_parse_tolerates_unusable_param_values(param_value):
    assert parse_reload_schedule(_row(param_value=param_value)).interval_hours is None


def test_parse_reads_an_interval_still_encoded_as_json_text():
    """The interval is written with safe_dumps, so a raw row read can return it either
    decoded or as a string; reading a string as no schedule would silently stop the
    reloads an admin configured"""
    assert parse_reload_schedule(_row(param_value='{"interval_hours": 6}')).interval_hours == 6


def test_parse_ignores_legacy_json_force_reload():
    """Rows written by pre-column versions carry force_reload in the JSON; honoring it
    would re-trigger a reload every poll because nothing clears the JSON copy"""
    schedule = parse_reload_schedule(_row(param_value={"interval_hours": 6, "force_reload": True}))

    assert schedule == ReloadSchedule(interval_hours=6, reload_revision=0, last_run_at=None)


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
    "schedule, pod_applied_revision, pod_data_loaded_at, expected",
    [
        (ReloadSchedule(reload_revision=4), 3, NOW, True),
        (ReloadSchedule(interval_hours=6, reload_revision=4, last_run_at=LAST_RUN), 3, NOW, True),
        (ReloadSchedule(reload_revision=3), 3, LAST_RUN, False),
        (ReloadSchedule(reload_revision=4), 0, LAST_RUN, True),
        (ReloadSchedule(), 0, LAST_RUN, False),
        (ReloadSchedule(interval_hours=6), 0, datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc), True),
        (
            ReloadSchedule(interval_hours=6, last_run_at=datetime(2024, 1, 1, 6, 30, tzinfo=timezone.utc)),
            0,
            datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            False,
        ),
        (ReloadSchedule(interval_hours=6, last_run_at=LAST_RUN), 0, LAST_RUN, True),
    ],
)
def test_pod_reload_is_due(schedule, pod_applied_revision, pod_data_loaded_at, expected):
    assert (
        pod_reload_is_due(
            schedule=schedule,
            pod_applied_revision=pod_applied_revision,
            pod_data_loaded_at=pod_data_loaded_at,
            current_time=NOW,
            description="test",
        )
        is expected
    )


def test_manual_request_is_identified_not_ordered():
    """Comparing revisions for inequality rather than ordering timestamps: a pod applies a
    request once and is not due again, no matter how the clocks or precisions line up"""
    unapplied = pod_reload_is_due(
        schedule=ReloadSchedule(reload_revision=9),
        pod_applied_revision=8,
        pod_data_loaded_at=NOW,
        current_time=NOW,
        description="test",
    )
    applied = pod_reload_is_due(
        schedule=ReloadSchedule(reload_revision=9),
        pod_applied_revision=9,
        pod_data_loaded_at=NOW,
        current_time=NOW,
        description="test",
    )

    assert (unapplied, applied) == (True, False)


def test_booting_pod_serves_a_request_it_cannot_prove_it_already_has():
    """A pod that just booted cannot tell whether an outstanding request predates the prices
    it fetched at import, so it serves it. Adopting instead would strand it on stale prices
    with no interval configured to rescue it"""
    assert (
        pod_reload_is_due(
            schedule=ReloadSchedule(reload_revision=12),
            pod_applied_revision=0,
            pod_data_loaded_at=NOW,
            current_time=NOW,
            description="test",
        )
        is True
    )


def test_pod_reload_decision_ignores_persisted_last_run():
    """A pod with stale data must refresh even when another pod already stamped
    last_run_at within the interval"""
    assert (
        pod_reload_is_due(
            schedule=ReloadSchedule(interval_hours=6, last_run_at=datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc)),
            pod_applied_revision=0,
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
            pod_applied_revision=0,
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
async def test_read_reload_schedule_surfaces_revision_on_interval_less_row():
    """A manual reload on a proxy with no schedule creates a row with only the columns
    set; the revision must still reach other pods"""
    prisma_client = _mock_prisma(row=_row(param_value=None, reload_revision=3))

    schedule = await read_reload_schedule(prisma_client, "model_cost_map_reload_config")

    assert schedule == ReloadSchedule(interval_hours=None, reload_revision=3, last_run_at=None)


@pytest.mark.asyncio
async def test_write_reload_interval_touches_only_param_value():
    prisma_client = _mock_prisma()

    await write_reload_interval(prisma_client, "model_cost_map_reload_config", 12)

    data = prisma_client.db.litellm_config.upsert.await_args.kwargs["data"]
    assert set(data["update"]) == {"param_value"}
    assert set(data["create"]) == {"param_name", "param_value"}


@pytest.mark.asyncio
async def test_record_reload_run_updates_last_run_without_creating_or_bumping():
    """update_many so a schedule deleted mid-poll stays deleted, and the untouched revision
    keeps fanning the request out to pods that have not applied it"""
    prisma_client = _mock_prisma()

    await record_reload_run(prisma_client, "model_cost_map_reload_config", LAST_RUN)

    kwargs = prisma_client.db.litellm_config.update_many.await_args.kwargs
    assert kwargs == {"data": {"last_run_at": LAST_RUN}, "where": {"param_name": "model_cost_map_reload_config"}}
    prisma_client.db.litellm_config.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_cancelling_a_schedule_never_reissues_a_revision():
    """Cancelling must keep the row. The revision identifies a request rather than ordering
    one, so a counter restarted by a delete reissues a number pods already applied and their
    next manual reload is skipped everywhere but the pod that served it"""
    prisma_client = _fake_prisma(_FakeConfigTable())
    param_name = "model_cost_map_reload_config"
    await write_reload_interval(prisma_client, param_name, 6)
    pod_applied_revision = await record_manual_reload(prisma_client, param_name, LAST_RUN)

    await clear_reload_interval(prisma_client, param_name)
    republished = await record_manual_reload(prisma_client, param_name, NOW)

    assert (pod_applied_revision, republished) == (1, 2)
    schedule = await read_reload_schedule(prisma_client, param_name)
    assert schedule is not None
    assert (
        pod_reload_is_due(
            schedule=schedule,
            pod_applied_revision=pod_applied_revision,
            pod_data_loaded_at=NOW,
            current_time=NOW,
            description="test",
        )
        is True
    )


@pytest.mark.asyncio
async def test_cancelling_a_schedule_stops_it_while_keeping_the_recorded_run():
    """Dropping only the admin-owned param_value: the card must report no schedule without
    losing the last run it already showed"""
    prisma_client = _fake_prisma(_FakeConfigTable())
    param_name = "model_cost_map_reload_config"
    await write_reload_interval(prisma_client, param_name, 6)
    await record_reload_run(prisma_client, param_name, LAST_RUN)

    await clear_reload_interval(prisma_client, param_name)

    status = reload_schedule_status(await read_reload_schedule(prisma_client, param_name))
    assert status == {
        "scheduled": False,
        "interval_hours": None,
        "last_run": "2024-01-01T06:00:00+00:00",
        "next_run": None,
    }


@pytest.mark.asyncio
async def test_record_manual_reload_bumps_the_revision_atomically():
    """The increment must be delegated to the database: two concurrent requests that both
    read then wrote a computed value would publish the same revision and one would be lost"""
    prisma_client = _mock_prisma(upserted_revision=5)

    published = await record_manual_reload(prisma_client, "model_cost_map_reload_config", LAST_RUN)

    data = prisma_client.db.litellm_config.upsert.await_args.kwargs["data"]
    assert data["update"] == {"last_run_at": LAST_RUN, "reload_revision": {"increment": 1}}
    assert data["create"] == {
        "param_name": "model_cost_map_reload_config",
        "last_run_at": LAST_RUN,
        "reload_revision": 1,
    }
    assert published == 5

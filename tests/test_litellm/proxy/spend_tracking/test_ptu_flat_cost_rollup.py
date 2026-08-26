"""Tests for the per-model PTU flat-cost daily rollup."""

import json
import types
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.spend_tracking.ptu_flat_cost_rollup as ptu_rollup
from litellm.constants import PTU_ROLLUP_MAX_BACKFILL_DAYS, PTU_SENTINEL_API_KEY
from litellm.proxy.spend_tracking.ptu_feature_flag import PTU_COST_ATTRIBUTION_ENV_VAR
from litellm.types.router import ModelInfo
from litellm.proxy.spend_tracking.ptu_flat_cost_rollup import (
    PTUModel,
    _active_hours_on_day,
    _compute_daily_flat_cost,
    _parse_ptu_model,
    run_ptu_flat_cost_backfill,
    run_ptu_flat_cost_rollup,
    run_scheduled_ptu_rollup,
)

DAY = date(2026, 7, 30)
TODAY = date(2026, 7, 31)


# The endpoints require ptu_effective_from alongside the count and rate, so a fixture that
# omits it would exercise a shape the write path cannot produce. Tests about the start
# itself pass with_start=False.
_DEFAULT_PTU_START = "2020-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _ptu_enabled(monkeypatch):
    """PTU is gated off by default. These cover the rollup's mechanics, not the gate, so
    they run with it on; the gate itself is covered by its own test below."""
    monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")


_VALID_PTU = {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}


def _model_row(model_id="m1", model_name="gpt-4o-mini-ptu", model_info=None, with_start=True):
    row = MagicMock()
    row.model_id = model_id
    row.model_name = model_name
    if (
        with_start
        and isinstance(model_info, dict)
        and model_info.get("ptu_count") is not None
        and model_info.get("cost_per_ptu_per_hour") is not None
        and "ptu_effective_from" not in model_info
    ):
        model_info = {**model_info, "ptu_effective_from": _DEFAULT_PTU_START}
    row.model_info = model_info
    return row


def _model(**overrides):
    base = dict(model_id="m", model_name="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=2.0)
    base.update(overrides)
    return PTUModel(**base)


def test_full_day_when_no_window():
    # 5 PTU * $2.00/hr * 24h = $240
    assert _compute_daily_flat_cost(_model(), DAY) == pytest.approx(240.0)


def test_window_opening_at_2300_charges_one_hour():
    m = _model(effective_from=datetime(2026, 7, 30, 23, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == pytest.approx(1.0)
    # 5 * 2.0 * 1 = 10
    assert _compute_daily_flat_cost(m, DAY) == pytest.approx(10.0)


def test_window_closing_at_0600_charges_six_hours():
    m = _model(effective_to=datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == pytest.approx(6.0)
    assert _compute_daily_flat_cost(m, DAY) == pytest.approx(60.0)


def test_window_fully_covering_day_charges_24h():
    m = _model(
        effective_from=datetime(2026, 7, 1, tzinfo=timezone.utc),
        effective_to=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert _active_hours_on_day(m, DAY) == pytest.approx(24.0)


def test_window_before_day_charges_zero():
    m = _model(effective_to=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == 0.0
    assert _compute_daily_flat_cost(m, DAY) == 0.0


def test_window_after_day_charges_zero():
    m = _model(effective_from=datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == 0.0


def test_naive_effective_from_is_treated_as_utc():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-30T23:00:00",
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(1.0)


def test_effective_from_with_z_suffix_parses():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 1,
                "cost_per_ptu_per_hour": 1.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-30T18:00:00Z",
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(6.0)


@pytest.mark.parametrize(
    "model_info",
    [
        None,
        {},
        {"ptu_count": 5},
        {"cost_per_ptu_per_hour": 2.0},
        {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0},  # missing team_id
        {"ptu_count": 0, "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
        {"ptu_count": 5, "cost_per_ptu_per_hour": -1.0, "team_id": "t"},
        {"ptu_count": "not-int", "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
    ],
)
def test_parse_ptu_model_rejects_invalid(model_info):
    assert _parse_ptu_model(_model_row(model_info=model_info)) is None


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Keep the upsert retry backoff out of the test runtime."""
    monkeypatch.setattr(ptu_rollup, "_UPSERT_RETRY_BACKOFF_SECONDS", 0)


def _sentinel_row(row_id, team_id, model):
    row = MagicMock()
    row.id = row_id
    row.team_id = team_id
    row.model = model
    return row


def _prisma_with_models(rows, existing_sentinel_rows=()):
    prisma = MagicMock()
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=rows)
    daily = MagicMock()
    daily.find_many = AsyncMock(return_value=list(existing_sentinel_rows))
    daily.upsert = AsyncMock()
    daily.delete_many = AsyncMock()
    prisma.db = types.SimpleNamespace(litellm_proxymodeltable=model_table, litellm_dailyteamspend=daily)
    return prisma, daily


@pytest.mark.asyncio
async def test_rollup_writes_sentinel_row_with_hourly_cost():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "team_x"})]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 1
    created = table.upsert.await_args.kwargs["data"]["create"]
    assert created["api_key"] == PTU_SENTINEL_API_KEY
    assert created["ptu_flat_cost"] == pytest.approx(240.0)
    assert created["team_id"] == "team_x"
    # identity in the key, display beside it, so a rename cannot move the row
    assert created["model"] == "m1"
    assert created["model_group"] == "gpt-4o-mini-ptu"
    keyed = table.upsert.await_args.kwargs["where"][
        "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint"
    ]
    assert keyed["model"] == "m1"


@pytest.mark.asyncio
async def test_rollup_prunes_a_scanned_deployment_whose_ptu_config_is_gone():
    """A deployment the run can still see, and can therefore judge, is the one case where
    retracting the charge is justified."""
    prisma, table = _prisma_with_models(
        [_model_row(model_id="m1", model_info={"team_id": "team_x"})],
        existing_sentinel_rows=[_sentinel_row("stale-1", "team_x", "m1")],
    )

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 0
    table.upsert.assert_not_awaited()
    table.delete_many.assert_awaited_once()
    where = table.delete_many.await_args.kwargs["where"]
    assert where["date"] == DAY.isoformat()
    assert where["api_key"] == PTU_SENTINEL_API_KEY
    assert "lt" in where["updated_at"]
    assert where["model"]["in"] == ("m1",)


@pytest.mark.asyncio
async def test_rollup_writes_current_row_before_pruning_and_keeps_it():
    prisma, table = _prisma_with_models(
        [_model_row(model_id="ptu", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "team_x"})],
        existing_sentinel_rows=[
            _sentinel_row("live", "team_x", "gpt-4o-mini-ptu"),
            _sentinel_row("stale", "team_x", "removed-model"),
        ],
    )

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 1
    table.upsert.assert_awaited_once()
    # the upsert lands before the cutoff is applied, so the refreshed row is out of reach
    upsert_order = table.method_calls.index(("upsert", (), table.upsert.call_args.kwargs))
    assert upsert_order < [c[0] for c in table.method_calls].index("delete_many")


@pytest.mark.asyncio
async def test_two_deployments_sharing_a_name_get_a_row_each():
    """Keyed on the deployment id they no longer need collapsing, and each keeps its own
    amount. The read path merges them back under the shared display name."""
    rows = [
        _model_row(model_id="dep-b", model_info={"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "team_id": "team_x"}),
        _model_row(model_id="dep-a", model_info={"ptu_count": 3, "cost_per_ptu_per_hour": 1.0, "team_id": "team_x"}),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 2
    written = {c.kwargs["data"]["create"]["model"]: c.kwargs["data"]["create"] for c in table.upsert.await_args_list}
    assert set(written) == {"dep-a", "dep-b"}
    assert written["dep-a"]["ptu_flat_cost"] == pytest.approx(72.0)
    assert written["dep-b"]["ptu_flat_cost"] == pytest.approx(48.0)
    assert {row["model_group"] for row in written.values()} == {"gpt-4o-mini-ptu"}


@pytest.mark.asyncio
async def test_rollup_skips_zero_active_hours():
    rows = [
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "team_x",
                "ptu_effective_from": "2026-08-01T00:00:00Z",
            }
        )
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 0
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollup_skips_models_without_ptu_config():
    rows = [
        _model_row(model_id="plain", model_info={"team_id": "team_x"}),
        _model_row(model_id="ptu", model_info={"ptu_count": 3, "cost_per_ptu_per_hour": 1.0, "team_id": "team_y"}),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 1


def test_parse_ptu_model_skips_a_deployment_with_no_effective_start():
    """The endpoints require a start. A row without one predates that rule or was written
    around them, and inferring a start would bill days the deployment did not exist: before
    this, a windowless deployment accrued the whole cap window on its first run."""
    assert (
        _parse_ptu_model(
            _model_row(
                model_info={"ptu_count": 10, "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
                with_start=False,
            )
        )
        is None
    )


def test_parse_ptu_model_accepts_json_string_model_info():
    # Some query paths deliver model_info as a JSON string, not a dict.
    import json as _json

    raw = _json.dumps(
        {
            "ptu_count": 5,
            "cost_per_ptu_per_hour": 2.0,
            "team_id": "team_x",
            "ptu_effective_from": _DEFAULT_PTU_START,
        }
    )
    parsed = _parse_ptu_model(_model_row(model_info=raw))
    assert parsed is not None
    assert parsed.ptu_count == 5 and parsed.team_id == "team_x"


def test_parse_ptu_model_rejects_unparseable_string():
    assert _parse_ptu_model(_model_row(model_info="not-json")) is None


def test_parse_ptu_model_accepts_datetime_object_effective_from():
    # model_info can carry a real datetime object, not just an ISO string.
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "t",
                "ptu_effective_from": datetime(2026, 7, 30, 23, 0, tzinfo=timezone.utc),
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bounds",
    [
        {"ptu_effective_from": "not-a-date"},
        {"ptu_effective_to": 12345},
        {"ptu_effective_from": "not-a-date", "ptu_effective_to": 12345},
    ],
)
def test_parse_ptu_model_rejects_malformed_effective_dates(bounds):
    # Treating an unparseable bound as "no bound" would widen the window to the whole
    # day and overcharge, so the deployment is skipped until the config is fixed.
    parsed = _parse_ptu_model(
        _model_row(model_info={"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "team_id": "t", **bounds})
    )
    assert parsed is None


def test_parse_ptu_model_rejects_an_inverted_window():
    # An end at or before the start can only mean a broken config; charging it as an
    # open-ended window would bill a full day.
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 2,
                "cost_per_ptu_per_hour": 1.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-31T12:00:00Z",
                "ptu_effective_to": "2026-07-31T06:00:00Z",
            }
        )
    )
    assert parsed is None


def _router_entry(model_id="dep-1", model_name="gpt-4o-ptu", model_info=None, with_start=True):
    """A deployment as the router stores one: a plain dict whose id lives in model_info."""
    info = dict(model_info or {})
    if (
        with_start
        and info.get("ptu_count") is not None
        and info.get("cost_per_ptu_per_hour") is not None
        and "ptu_effective_from" not in info
    ):
        info["ptu_effective_from"] = _DEFAULT_PTU_START
    if model_id is not None:
        info["id"] = model_id
    return {
        "model_name": model_name,
        "litellm_params": {"model": "azure/gpt-4o"},
        "model_info": info,
    }


# A team-scoped deployment is stored under a synthetic routing key with the operator's name
# in team_public_model_name, while the same deployment in config.yaml carries the operator's
# name directly. Both must resolve to the same PTUModel or the two sources bill differently.
_PARITY_CASES = (
    (
        "an iso string start against the datetime pydantic coerces it to",
        {"ptu_effective_from": "2026-07-30T23:00:00Z"},
        {"ptu_effective_from": datetime(2026, 7, 30, 23, 0)},
        datetime(2026, 7, 30, 23, 0, tzinfo=timezone.utc),
        None,
    ),
    (
        "an open window, stored as null and dropped by exclude_none",
        {"ptu_effective_from": "2026-07-01T00:00:00Z", "ptu_effective_to": None},
        {"ptu_effective_from": datetime(2026, 7, 1, 0, 0)},
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        None,
    ),
    (
        "a closed window",
        {"ptu_effective_from": "2026-07-01T00:00:00Z", "ptu_effective_to": "2026-07-31T00:00:00Z"},
        {
            "ptu_effective_from": datetime(2026, 7, 1, 0, 0),
            "ptu_effective_to": datetime(2026, 7, 31, 0, 0),
        },
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 7, 31, tzinfo=timezone.utc),
    ),
)


@pytest.mark.parametrize(
    "db_window, router_window, expected_from, expected_to",
    [case[1:] for case in _PARITY_CASES],
    ids=[case[0] for case in _PARITY_CASES],
)
def test_a_deployment_parses_identically_from_the_db_and_from_config_yaml(
    db_window, router_window, expected_from, expected_to
):
    # The contract the router union is built on. If these ever diverge, a PTU deployment
    # declared in config.yaml is billed differently from the identical one in the database.
    from_db = _parse_ptu_model(
        _model_row(
            model_id="dep-1",
            model_name="model_name_t_9f3c2b",
            model_info={**_VALID_PTU, **db_window, "team_public_model_name": "gpt-4o-ptu"},
            with_start=False,
        )
    )
    from_router = _parse_ptu_model(
        ptu_rollup._router_deployment(
            _router_entry(model_id="dep-1", model_info={**_VALID_PTU, **router_window}, with_start=False)
        )
    )
    expected = PTUModel(
        model_id="dep-1",
        model_name="gpt-4o-ptu",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        effective_from=expected_from,
        effective_to=expected_to,
    )
    assert from_db == from_router == expected


@pytest.mark.parametrize("model_id", [None, "", 12345], ids=["absent", "blank", "not a string"])
def test_a_router_deployment_without_a_usable_id_is_dropped(model_id):
    # model_id keys the sentinel row, so an unusable one would file every such deployment
    # in a team onto one row and bill for a single reservation.
    entry = _router_entry(model_id=None, model_info=dict(_VALID_PTU))
    if model_id is not None:
        entry["model_info"]["id"] = model_id
    assert ptu_rollup._router_deployment(entry) is None


def test_a_router_deployment_keeps_the_name_the_operator_wrote():
    priced = _parse_ptu_model(
        ptu_rollup._router_deployment(_router_entry(model_name="gpt-4o-ptu", model_info=dict(_VALID_PTU)))
    )
    assert priced is not None
    assert priced.model_name == "gpt-4o-ptu"


def test_a_team_alias_on_a_router_deployment_still_wins_over_the_routing_name():
    # config.yaml can carry team_public_model_name to expose a team-facing alias, and the
    # charge has to file under the name that team calls rather than the routing one.
    priced = _parse_ptu_model(
        ptu_rollup._router_deployment(
            _router_entry(
                model_name="routing-name",
                model_info={**_VALID_PTU, "team_public_model_name": "public-alias"},
            )
        )
    )
    assert priced is not None
    assert priced.model_name == "public-alias"


def test_a_router_deployment_with_a_stringified_model_info_still_decodes():
    entry = _router_entry(model_info=dict(_VALID_PTU))
    entry["model_info"] = json.dumps(entry["model_info"])
    parsed = _parse_ptu_model(ptu_rollup._router_deployment(entry))
    assert parsed is not None
    assert parsed.model_id == "dep-1"


@pytest.mark.parametrize(
    "model_info",
    [None, "not json", 42, "[1, 2, 3]", '"a string"', "42", "true", "null"],
    ids=[
        "absent",
        "unparseable",
        "not a string or dict",
        "json array",
        "json string",
        "json number",
        "json bool",
        "json null",
    ],
)
def test_a_router_deployment_without_usable_model_info_is_dropped(model_info):
    # Valid JSON that is not an object decodes to a list or a scalar, and reading fields
    # off one raises rather than dropping the single bad deployment the rollup expects
    assert ptu_rollup._router_deployment({"model_name": "x", "model_info": model_info}) is None


@pytest.mark.parametrize(
    "model_info",
    ["[1, 2, 3]", '"a string"', "42", "true", "null"],
    ids=["json array", "json string", "json number", "json bool", "json null"],
)
def test_a_db_row_holding_non_object_json_is_dropped(model_info):
    assert _parse_ptu_model(_model_row(model_info=model_info, with_start=False)) is None


def test_a_router_deployment_does_not_alias_the_routers_own_model_info():
    # The rollup runs on a cron while requests are in flight, and the router rewrites
    # model_info in place, so a held record must neither observe nor cause those writes.
    live = {"id": "dep-1", **_VALID_PTU}
    record = ptu_rollup._router_deployment({"model_name": "x", "model_info": live})
    assert record is not None

    live["ptu_count"] = 999
    assert record.model_info["ptu_count"] == _VALID_PTU["ptu_count"]

    with pytest.raises(TypeError):
        record.model_info["ptu_count"] = 1


def test_a_raw_router_dict_is_not_a_deployment_record():
    # The parser reads attributes, so a router dict passed straight to it returns None
    # instead of raising. Skipping the factory would silently drop every config.yaml
    # deployment while leaving the suite green.
    assert _parse_ptu_model(_router_entry(model_info=dict(_VALID_PTU))) is None


@pytest.mark.asyncio
async def test_rollup_returns_empty_when_prisma_client_is_none():
    result = await run_ptu_flat_cost_rollup(None, target_date=DAY)
    assert result.models_processed == 0
    assert result.rows_written == 0
    assert result.day == DAY


@pytest.mark.asyncio
async def test_rollup_continues_after_a_failed_upsert():
    rows = [
        _model_row(model_id="a", model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"}),
        _model_row(model_id="b", model_info={"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "team_id": "team_b"}),
    ]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    # both models exhausted their retries, and the batch still ran to completion
    assert result.models_processed == 2
    assert result.rows_written == 0
    assert result.rows_failed == 2
    assert table.upsert.await_count == 2 * ptu_rollup._UPSERT_ATTEMPTS


@pytest.mark.asyncio
async def test_rollup_retries_a_transient_upsert_failure_and_succeeds():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=[RuntimeError("connection reset"), None])

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    # the retry writes the day's charge, so nothing is left for a manual rerun
    assert result.rows_written == 1
    assert result.rows_failed == 0
    assert table.upsert.await_count == 2


def _pod_lock(acquired):
    """A lock manager that acquires (or not) and, by default, still owns the lease."""
    lock = MagicMock()
    lock.pod_id = "this-pod"
    lock.redis_cache = MagicMock()
    lock.redis_cache.async_get_cache = AsyncMock(return_value="this-pod")
    lock.get_redis_lock_key = MagicMock(return_value="lock-key")
    lock.acquire_lock = AsyncMock(return_value=acquired)
    lock.release_lock = AsyncMock()
    return lock


@pytest.mark.asyncio
async def test_scheduled_rollup_skips_the_run_when_another_pod_holds_the_lock():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=False)

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    # the losing pod must not write or prune, or it could delete the winner's fresh rows
    assert result is None
    assert table.upsert.await_count == 0
    assert table.delete_many.await_count == 0
    lock.release_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_rollup_runs_and_releases_the_lock_when_it_wins():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=True)

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    assert result is not None and result.rows_written == 1
    lock.acquire_lock.assert_awaited_once()
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_rollup_releases_the_lock_even_when_the_run_raises():
    prisma, table = _prisma_with_models([])
    prisma.db.litellm_proxymodeltable.find_many = AsyncMock(side_effect=RuntimeError("db down"))
    lock = _pod_lock(acquired=True)

    with pytest.raises(RuntimeError):
        await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    # a stuck lock would block every later run until its TTL expires
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_rollup_runs_unguarded_without_a_redis_backed_lock():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=True)
    lock.redis_cache = None

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    # single-writer deployments have no lock to take, and must still reconcile the day
    assert result is not None and result.rows_written == 1
    lock.acquire_lock.assert_not_awaited()

    assert await run_scheduled_ptu_rollup(prisma, target_date=DAY) is not None


@pytest.mark.asyncio
async def test_rollup_skips_the_prune_when_a_replacement_write_failed():
    # The deployment was renamed, so the old sentinel row is stale only once its
    # replacement lands. Pruning against the intended charges after a failed write
    # would delete the old row and leave the team with no charge at all.
    prisma, table = _prisma_with_models(
        [
            _model_row(
                model_name="renamed-ptu", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
            )
        ],
        existing_sentinel_rows=[_sentinel_row("previous", "t", "old-name-ptu")],
    )
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_failed == 1
    table.delete_many.assert_not_awaited()


class _FakeSentinelTable:
    """In-memory LiteLLM_DailyTeamSpend that honours the sentinel key and prune predicate."""

    def __init__(self, upsert_gate=None):
        self.rows = {}
        self._upsert_gate = upsert_gate
        self.upsert_keys = []
        self.delete_many_calls = []
        self.find_many_calls = []

    async def upsert(self, where, data):
        if self._upsert_gate is not None:
            await self._upsert_gate.wait()
        key = where["team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint"]
        row_key = (key["team_id"], key["date"], key["api_key"], key["model"])
        self.upsert_keys.append(row_key)
        self.rows[row_key] = {
            "ptu_flat_cost": data["create"]["ptu_flat_cost"],
            "model_group": data["create"]["model_group"],
            "updated_at": datetime.now(timezone.utc),
        }

    async def delete_many(self, where):
        self.delete_many_calls.append(where)
        cutoff = where["updated_at"]["lt"]
        # honouring "model" matters: a fake that ignored an unknown clause would delete
        # the row the prune-scoping test exists to protect and still report a pass
        allowed = where.get("model", {}).get("in")
        doomed = [
            k
            for k, v in self.rows.items()
            if k[1] == where["date"]
            and k[2] == where["api_key"]
            and v["updated_at"] < cutoff
            and (allowed is None or k[3] in allowed)
        ]
        for k in doomed:
            del self.rows[k]
        return len(doomed)

    async def find_many(self, where=None):
        """Read back sentinel rows the way prisma would, honouring api_key and a date range."""
        self.find_many_calls.append(where)
        if not where or where.get("api_key") != PTU_SENTINEL_API_KEY:
            return []
        bounds = where.get("date") or {}
        return [
            _stored_sentinel_row(team_id, day, model_id, value.get("model_group"))
            for (team_id, day, api_key, model_id), value in self.rows.items()
            if api_key == PTU_SENTINEL_API_KEY and (not bounds or bounds["gte"] <= day <= bounds["lte"])
        ]

    def seed(self, team_id, day, model_id, flat_cost, updated_at=None, model_group=None):
        """Seed a row the way the rollup writes one: keyed on the deployment id."""
        self.rows[(team_id, day.isoformat(), PTU_SENTINEL_API_KEY, model_id)] = {
            "ptu_flat_cost": flat_cost,
            "model_group": model_group or model_id,
            "updated_at": updated_at or datetime.now(timezone.utc),
        }


def _stored_sentinel_row(team_id, day, model_id, model_group=None):
    row = MagicMock()
    row.team_id = team_id
    row.date = day
    row.model = model_id
    row.model_group = model_group
    return row


def _prisma_for(model_rows, daily_table):
    prisma = MagicMock()
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=model_rows)
    prisma.db = types.SimpleNamespace(litellm_proxymodeltable=model_table, litellm_dailyteamspend=daily_table)
    return prisma


@pytest.mark.asyncio
async def test_an_older_run_cannot_delete_a_newer_runs_row():
    """The race the absolute predicate exists for: an admin renames a PTU model while two
    pods are mid-rollup, so each pod prices a different model name. The pod that started
    first must not be able to delete the charge the second pod just wrote."""
    import asyncio

    gate = asyncio.Event()
    table = _FakeSentinelTable()
    ptu = {"ptu_count": 10, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}

    # pod A read the config before the second deployment appeared and is stalled mid-upsert
    slow_table = _FakeSentinelTable(upsert_gate=gate)
    slow_table.rows = table.rows
    pod_a = asyncio.create_task(
        run_ptu_flat_cost_rollup(
            _prisma_for([_model_row(model_id="dep-a", model_info=ptu)], slow_table), target_date=DAY
        )
    )
    await asyncio.sleep(0)  # let pod A capture run_started and reach the gate

    # pod B read a config that has since replaced it, and completes first
    await run_ptu_flat_cost_rollup(_prisma_for([_model_row(model_id="dep-b", model_info=ptu)], table), target_date=DAY)
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-b") in table.rows

    gate.set()
    await pod_a

    # pod A's cutoff predates every row written during the race, so its delete reaches none
    assert table.rows[("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-b")]["ptu_flat_cost"] == pytest.approx(480.0)


@pytest.mark.asyncio
async def test_a_later_clean_run_clears_the_row_the_race_left_behind():
    """The race can leave a charge for a no-longer-priced deployment in place for a day;
    the next run, seeing only the current config, must sweep it."""
    table = _FakeSentinelTable()
    ptu = {"ptu_count": 10, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
    stale_key = ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-retired")
    table.rows[stale_key] = {
        "ptu_flat_cost": 480.0,
        "model_group": "retired",
        "updated_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
    }

    await run_ptu_flat_cost_rollup(
        _prisma_for(
            [
                _model_row(model_id="dep-live", model_info=ptu),
                _model_row(model_id="dep-retired", model_info={"team_id": "t"}),
            ],
            table,
        ),
        target_date=DAY,
    )

    assert stale_key not in table.rows
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-live") in table.rows


@pytest.mark.asyncio
async def test_scheduled_rollup_alerts_when_a_team_charge_never_landed():
    """A failed charge is a silent underbill: the team shows no PTU cost for the date and
    the next cron run moves on to the next day. It has to reach an operator."""
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.rows_failed == 1
    alert.assert_awaited_once()
    message = alert.await_args.args[0]
    assert DAY.isoformat() in message
    assert "rerun" in message


@pytest.mark.asyncio
async def test_scheduled_rollup_stays_quiet_when_every_charge_landed():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.rows_failed == 0
    alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_rollup_alerts_once_a_ptu_window_has_closed():
    """Reserved capacity is billed until the deployment is deleted, so a closed window stops
    the attribution without stopping the charge. Nobody notices unless it is escalated."""
    ptu = {
        "ptu_count": 5,
        "cost_per_ptu_per_hour": 2.0,
        "team_id": "t",
        "ptu_effective_from": "2020-01-01T00:00:00Z",
        "ptu_effective_to": "2020-02-01T00:00:00Z",
    }
    prisma, _ = _prisma_with_models([_model_row(model_id="dep-lapsed", model_info=ptu)])
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.lapsed == ("gpt-4o-mini-ptu",)
    alert.assert_awaited_once()
    message = alert.await_args.args[0]
    assert "window has closed" in message
    assert "gpt-4o-mini-ptu" in message


@pytest.mark.asyncio
async def test_a_model_name_cannot_smuggle_slack_markup_into_the_alert():
    """The alert lands in an operator channel and a model name is operator-supplied, so an
    unescaped name could post a channel-wide mention."""
    ptu = {
        "ptu_count": 5,
        "cost_per_ptu_per_hour": 2.0,
        "team_id": "t",
        "ptu_effective_from": "2020-01-01T00:00:00Z",
        "ptu_effective_to": "2020-02-01T00:00:00Z",
    }
    row = _model_row(model_id="dep-x", model_name="<!channel> & <https://evil.example|click>", model_info=ptu)
    prisma, _ = _prisma_with_models([row])
    alert = AsyncMock()

    await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    message = alert.await_args.args[0]
    assert "<!channel>" not in message
    assert "&lt;!channel&gt;" in message


@pytest.mark.asyncio
async def test_an_open_ptu_window_raises_no_lapsed_alert():
    ptu = {
        "ptu_count": 5,
        "cost_per_ptu_per_hour": 2.0,
        "team_id": "t",
        "ptu_effective_from": "2020-01-01T00:00:00Z",
        "ptu_effective_to": "2999-01-01T00:00:00Z",
    }
    prisma, _ = _prisma_with_models([_model_row(model_id="dep-open", model_info=ptu)])
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.lapsed == ()
    alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_open_ended_ptu_window_raises_no_lapsed_alert():
    """No end bound means the operator never asked the attribution to stop."""
    ptu = {
        "ptu_count": 5,
        "cost_per_ptu_per_hour": 2.0,
        "team_id": "t",
        "ptu_effective_from": "2020-01-01T00:00:00Z",
    }
    prisma, _ = _prisma_with_models([_model_row(model_id="dep-forever", model_info=ptu)])
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.lapsed == ()
    alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_broken_alert_channel_does_not_fail_the_rollup():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_scheduled_ptu_rollup(
        prisma, target_date=DAY, alert=AsyncMock(side_effect=RuntimeError("slack down"))
    )

    # losing the alert must not also lose the run's result or leave the lock held
    assert result.rows_failed == 1


@pytest.mark.asyncio
async def test_scheduled_rollup_alerts_from_under_the_pod_lock_too():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    lock = _pod_lock(acquired=True)
    alert = AsyncMock()

    await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY, alert=alert)

    alert.assert_awaited_once()
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lock_read",
    [
        pytest.param(AsyncMock(side_effect=RuntimeError("redis down")), id="redis-unreachable"),
        pytest.param(AsyncMock(return_value=None), id="lock-key-missing"),
    ],
)
async def test_scheduled_rollup_runs_the_day_when_the_lock_is_unavailable_but_unheld(lock_read):
    """acquire_lock reports contention and a Redis outage identically. Treating both as
    "someone else has it" would skip the day on every pod at once, losing every team's
    charge for that date; the reconcile is safe to run twice, so the day wins."""
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=False)
    lock.redis_cache.async_get_cache = lock_read

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    assert result is not None and result.rows_written == 1
    table.upsert.assert_awaited_once()


def _team_scoped_row(public_name, model_id="m1", team_id="team_x", **ptu):
    """A deployment as POST /model/new actually stores it: synthetic routing name in
    model_name, the operator's chosen name in model_info.team_public_model_name."""
    info = {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": team_id, **ptu}
    info["team_public_model_name"] = public_name
    return _model_row(model_id=model_id, model_name=f"model_name_{team_id}_{model_id}-uuid", model_info=info)


def test_parse_ptu_model_keys_on_the_public_name_not_the_routing_key():
    # PTU requires a team_id, so every PTU deployment carries the synthetic model_name.
    # Keying the charge on it files the cost under a UUID no usage view can resolve.
    parsed = _parse_ptu_model(_team_scoped_row("gpt-4o"))
    assert parsed is not None
    assert parsed.model_name == "gpt-4o"


def test_parse_ptu_model_falls_back_to_model_name_without_a_public_name():
    parsed = _parse_ptu_model(
        _model_row(
            model_name="plain-deployment", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
        )
    )
    assert parsed is not None
    assert parsed.model_name == "plain-deployment"


@pytest.mark.parametrize("bad_public_name", ["", None, 123, {"nested": "value"}])
def test_parse_ptu_model_ignores_an_unusable_public_name(bad_public_name):
    row = _model_row(
        model_name="routing-key",
        model_info={
            "ptu_count": 5,
            "cost_per_ptu_per_hour": 2.0,
            "team_id": "t",
            "team_public_model_name": bad_public_name,
        },
    )
    parsed = _parse_ptu_model(row)
    assert parsed is not None
    assert parsed.model_name == "routing-key"


@pytest.mark.asyncio
async def test_team_scoped_deployments_key_on_their_id_and_display_the_public_name():
    """A team-scoped deployment's model_name is a synthetic routing key, so the row keys on
    the stable id and carries the operator-facing name alongside it for display."""
    rows = [
        _team_scoped_row("gpt-4o", model_id="dep-b", ptu_count=2, cost_per_ptu_per_hour=1.0),
        _team_scoped_row("gpt-4o", model_id="dep-a", ptu_count=3, cost_per_ptu_per_hour=1.0),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 2
    written = {c.kwargs["data"]["create"]["model"]: c.kwargs["data"]["create"] for c in table.upsert.await_args_list}
    assert set(written) == {"dep-a", "dep-b"}
    assert {row["model_group"] for row in written.values()} == {"gpt-4o"}
    assert sum(row["ptu_flat_cost"] for row in written.values()) == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Catch-up backfill: the days a once-daily "price yesterday" job never revisits
# ---------------------------------------------------------------------------


def _day(offset):
    """A UTC date relative to DAY, which is the last day the backfill may price."""
    return DAY + timedelta(days=offset)


def _windowed_row(effective_from=None, effective_to=None, **overrides):
    ptu = {
        "ptu_count": overrides.pop("ptu_count", 5),
        "cost_per_ptu_per_hour": overrides.pop("cost_per_ptu_per_hour", 2.0),
        "team_id": overrides.pop("team_id", "t"),
    }
    if effective_from is not None:
        ptu["ptu_effective_from"] = effective_from.isoformat()
    if effective_to is not None:
        ptu["ptu_effective_to"] = effective_to.isoformat()
    return _model_row(model_info=ptu, **overrides)


def _midnight(day):
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)


def _priced_dates(table):
    return sorted(key[1] for key in table.rows)


# --- R1: the gap is actually closed ----------------------------------------


@pytest.mark.asyncio
async def test_backfill_prices_every_elapsed_in_window_day():
    """The defect this exists for: an operator backdates a PTU window by 30 days, the
    config validates and persists, and the once-daily job prices only yesterday. Every
    elapsed day inside the declared window has to end up with a charge."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-29)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.days_scanned == 30
    assert result.rows_written == 30
    assert result.rows_failed == 0
    assert _priced_dates(table) == [_day(offset).isoformat() for offset in range(-29, 1)]
    assert all(row["ptu_flat_cost"] == pytest.approx(240.0) for row in table.rows.values())


@pytest.mark.asyncio
async def test_backfill_prices_a_day_the_daily_run_missed():
    """A pod restart across 00:15 loses exactly one day. Only that day may be written."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-2), "m1", 240.0)
    table.seed("t", _day(0), "m1", 240.0)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-2)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.rows_written == 1
    assert table.upsert_keys == [("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "m1")]


@pytest.mark.asyncio
async def test_backfill_prices_the_partial_first_day_by_active_hours():
    """A backfilled day is priced by the same hourly overlap as a live one, so a window
    opening at 08:01 charges the remaining 15h59m rather than a whole day."""
    table = _FakeSentinelTable()
    opens_at = datetime(_day(-1).year, _day(-1).month, _day(-1).day, 8, 1, tzinfo=timezone.utc)
    prisma = _prisma_for([_windowed_row(effective_from=opens_at)], table)

    await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    first_day = table.rows[("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "m1")]
    assert first_day["ptu_flat_cost"] == pytest.approx(10 * (15 + 59 / 60))
    assert table.rows[("t", _day(0).isoformat(), PTU_SENTINEL_API_KEY, "m1")]["ptu_flat_cost"] == pytest.approx(240.0)


@pytest.mark.asyncio
async def test_backfill_stops_at_yesterday():
    """A day that has not finished cannot be billed, however far into the future the
    declared window runs."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-2)), effective_to=_midnight(_day(30)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.end == DAY
    assert max(_priced_dates(table)) == DAY.isoformat()


# --- R2: history is never rewritten ----------------------------------------


@pytest.mark.asyncio
async def test_backfill_leaves_an_existing_row_untouched_when_config_changed():
    """A priced day keeps the amount it was billed at. Re-pricing it under today's rate
    would silently restate a closed day, which is worse than the gap being fixed."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-1), "m1", 240.0)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-1)), cost_per_ptu_per_hour=5.0)], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    already_priced = ("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "m1")
    assert already_priced not in table.upsert_keys
    assert table.rows[already_priced]["ptu_flat_cost"] == pytest.approx(240.0)
    assert result.rows_written == 1
    assert table.rows[("t", _day(0).isoformat(), PTU_SENTINEL_API_KEY, "m1")]["ptu_flat_cost"] == pytest.approx(600.0)


def test_parse_skips_a_count_too_large_to_price():
    """float(ptu_count) on an unbounded int raises OverflowError, which aborted the whole
    run rather than skipping the one deployment carrying it."""
    assert _parse_ptu_model(_model_row(model_info={**_VALID_PTU, "ptu_count": 10**400})) is None


@pytest.mark.parametrize("rate", ["NaN", "Infinity", "-Infinity"])
def test_parse_skips_a_non_finite_rate(rate):
    """NaN compares False against every bound, so a bare `< 0` check passed it through and
    the deployment accrued a flat cost of nan."""
    assert _parse_ptu_model(_model_row(model_info={**_VALID_PTU, "cost_per_ptu_per_hour": rate})) is None


def test_parse_still_accepts_config_at_the_bounds():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                **_VALID_PTU,
                "ptu_count": ModelInfo.MAX_PTU_COUNT,
                "cost_per_ptu_per_hour": ModelInfo.MAX_COST_PER_PTU_PER_HOUR,
            }
        )
    )
    assert parsed is not None and parsed.ptu_count == ModelInfo.MAX_PTU_COUNT


@pytest.mark.asyncio
async def test_a_bad_row_does_not_abort_pricing_for_other_teams():
    """One unusable deployment must not take the whole day's rollup down with it."""
    table = _FakeSentinelTable()
    prisma = _prisma_for(
        [
            _model_row(model_id="bad", model_info={**_VALID_PTU, "ptu_count": 10**400}),
            _model_row(model_id="good", model_info=_VALID_PTU),
        ],
        table,
    )

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 1
    assert result.models_processed == 1


@pytest.mark.asyncio
async def test_backfill_keeps_the_history_of_a_deployment_that_was_removed():
    """Deleting a deployment stops it accruing, it does not unbill the days it ran. The
    backfill deletes nothing, so a closed day survives its deployment."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-1), "dep-gone", 480.0, model_group="gone-model")
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-1)), model_id="dep-live")], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert table.rows[("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "dep-gone")]["ptu_flat_cost"] == 480.0
    assert table.delete_many_calls == []
    assert ("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "dep-live") in table.rows
    assert result.rows_written == 2


@pytest.mark.asyncio
async def test_backfill_keeps_history_after_every_ptu_deployment_is_gone():
    """With no PTU config left there is nothing to price, and nothing to delete either."""
    table = _FakeSentinelTable()
    for offset in (-2, -1):
        table.seed("t", _day(offset), "dep-gone", 480.0, model_group="gone-model")
    prisma = _prisma_for([_model_row(model_info={"base_model": "gpt-4o"})], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert len(table.rows) == 2
    assert table.delete_many_calls == []
    assert result.rows_written == 0


@pytest.mark.asyncio
async def test_backfill_keeps_history_when_a_window_is_narrowed():
    """Editing an effective window cannot rewrite a bill that was already correct."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-3), "dep-1", 480.0, model_group="ptu-a")
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-1)), model_id="dep-1")], table)

    await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert ("t", _day(-3).isoformat(), PTU_SENTINEL_API_KEY, "dep-1") in table.rows
    assert table.delete_many_calls == []


@pytest.mark.asyncio
async def test_backfill_never_prunes_by_timestamp():
    """Retirement is by identity. The catch-up must never take the single-day path's
    timestamp predicate, which needs the lock and agreeing clocks to be safe."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-3), "dep-1", 1.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-3)), model_id="dep-1")], table)

    await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert all("updated_at" not in (call or {}) for call in table.delete_many_calls)
    assert ("t", _day(-3).isoformat(), PTU_SENTINEL_API_KEY, "dep-1") in table.rows


# --- R3: a gap is per (team, model, date), not per date ---------------------


@pytest.mark.asyncio
async def test_backfill_fills_a_second_model_on_a_day_that_already_has_a_row():
    """A day is not covered just because something was priced on it."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-1), "a", 240.0)
    prisma = _prisma_for(
        [
            _windowed_row(effective_from=_midnight(_day(-1)), model_name="model-a", model_id="a"),
            _windowed_row(effective_from=_midnight(_day(-1)), model_name="model-b", model_id="b"),
        ],
        table,
    )

    await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert ("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "b") in table.upsert_keys
    assert ("t", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "a") not in table.upsert_keys


@pytest.mark.asyncio
async def test_backfill_fills_a_second_team_on_a_day_that_already_has_a_row():
    table = _FakeSentinelTable()
    table.seed("team-1", _day(-1), "a", 240.0)
    prisma = _prisma_for(
        [
            _windowed_row(effective_from=_midnight(_day(-1)), model_name="shared-name", model_id="a", team_id="team-1"),
            _windowed_row(effective_from=_midnight(_day(-1)), model_name="shared-name", model_id="b", team_id="team-2"),
        ],
        table,
    )

    await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert ("team-2", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "b") in table.upsert_keys
    assert ("team-1", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "a") not in table.upsert_keys


@pytest.mark.asyncio
async def test_backfill_keys_gaps_on_the_public_model_name():
    """Sentinel rows are written under the public name, so a gap check reading the
    synthetic routing key would never match one and would rewrite it on every run."""
    table = _FakeSentinelTable()
    table.seed("team_x", _day(-1), "m1", 240.0)
    row = _team_scoped_row(
        "gpt-4o",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=_midnight(_day(-1)).isoformat(),
    )
    prisma = _prisma_for([row], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert ("team_x", _day(-1).isoformat(), PTU_SENTINEL_API_KEY, "m1") not in table.upsert_keys
    assert result.rows_written == 1


# --- R4: bounds and convergence --------------------------------------------


@pytest.mark.asyncio
async def test_backfill_does_not_scan_before_effective_from():
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-2)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.start == _day(-2)
    assert result.days_scanned == 3


@pytest.mark.asyncio
async def test_backfill_caps_lookback_for_a_model_with_no_effective_from():
    """An open-ended window would otherwise scan back to the beginning of the table."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row()], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.start == DAY - timedelta(days=PTU_ROLLUP_MAX_BACKFILL_DAYS)
    assert result.days_scanned == PTU_ROLLUP_MAX_BACKFILL_DAYS + 1


@pytest.mark.asyncio
async def test_backfill_caps_lookback_for_a_window_older_than_the_cap():
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-400)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.start == DAY - timedelta(days=PTU_ROLLUP_MAX_BACKFILL_DAYS)


@pytest.mark.asyncio
async def test_backfill_writes_nothing_for_out_of_window_days():
    """A zero-cost day must write no row, or the gap check would read it as priced."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-2)), effective_to=_midnight(_day(-1)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.days_scanned == 3
    assert result.rows_written == 1
    assert _priced_dates(table) == [_day(-2).isoformat()]


@pytest.mark.asyncio
async def test_backfill_is_a_no_op_on_a_fully_priced_range():
    table = _FakeSentinelTable()
    for offset in (-2, -1, 0):
        table.seed("t", _day(offset), "m1", 240.0)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-2)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.rows_written == 0
    assert table.upsert_keys == []
    assert table.delete_many_calls == []


@pytest.mark.asyncio
async def test_backfill_run_twice_is_idempotent():
    """The second pass must be free, including leaving updated_at alone."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-3)))], table)

    await run_ptu_flat_cost_backfill(prisma, today=TODAY)
    snapshot = {key: dict(value) for key, value in table.rows.items()}
    table.upsert_keys.clear()

    second = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert second.rows_written == 0
    assert table.upsert_keys == []
    assert table.rows == snapshot


@pytest.mark.asyncio
async def test_backfill_does_nothing_without_ptu_config():
    table = _FakeSentinelTable()
    prisma = _prisma_for([_model_row(model_info={"base_model": "gpt-4o"})], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.days_scanned == 0
    assert result.rows_written == 0
    assert table.upsert_keys == []


@pytest.mark.asyncio
async def test_backfill_writes_nothing_for_a_window_that_opens_tomorrow():
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(5)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.days_scanned == 0
    assert table.upsert_keys == []


@pytest.mark.asyncio
async def test_backfill_returns_empty_when_prisma_client_is_none():
    result = await run_ptu_flat_cost_backfill(None, today=TODAY)

    assert result.rows_written == 0
    assert result.days_scanned == 0


@pytest.mark.asyncio
async def test_backfill_counts_a_charge_that_never_landed():
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-1)))], table)
    prisma.db.litellm_dailyteamspend.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.rows_written == 0
    assert result.rows_failed == 2


# --- R5: interaction with the daily path -----------------------------------


@pytest.mark.asyncio
async def test_scheduled_rollup_backfills_after_pricing_the_day():
    """The catch-up pass runs after the day's own rollup, so it sees yesterday already
    priced and does not write it a second time."""
    table = _FakeSentinelTable()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(yesterday - timedelta(days=2)))], table)

    await run_scheduled_ptu_rollup(prisma)

    yesterday_key = ("t", yesterday.isoformat(), PTU_SENTINEL_API_KEY, "m1")
    assert table.upsert_keys.count(yesterday_key) == 1
    assert len(table.rows) == 3


@pytest.mark.asyncio
async def test_scheduled_rollup_with_an_explicit_target_date_does_not_backfill():
    """An explicit date means reconcile exactly that day, so no catch-up pass runs."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-10)))], table)

    await run_scheduled_ptu_rollup(prisma, target_date=DAY)

    assert _priced_dates(table) == [DAY.isoformat()]
    assert table.find_many_calls == []


@pytest.mark.asyncio
async def test_scheduled_rollup_holds_one_lock_across_both_phases():
    """Backfill running outside the lock would let another pod's prune race its writes."""
    table = _FakeSentinelTable()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(yesterday - timedelta(days=3)))], table)
    rows_at_release = []
    lock = _pod_lock(acquired=True)
    lock.release_lock = AsyncMock(side_effect=lambda **kwargs: rows_at_release.append(len(table.rows)))

    await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock)

    lock.acquire_lock.assert_awaited_once()
    assert rows_at_release == [4]


@pytest.mark.asyncio
async def test_a_failing_backfill_does_not_lose_the_days_rollup_result():
    """The day's rollup has already run and committed; a broken catch-up pass must not
    swallow its result or raise into the scheduler."""
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row()], table)
    prisma.db.litellm_dailyteamspend.find_many = AsyncMock(side_effect=RuntimeError("read replica down"))

    result = await run_scheduled_ptu_rollup(prisma)

    assert result is not None
    assert result.rows_written == 1
    assert result.rows_failed == 0


@pytest.mark.asyncio
async def test_scheduled_rollup_alerts_when_a_backfill_charge_never_landed():
    """An unpriced day that stays unpriced is the silent underbill this work exists to
    remove, so it has to reach an operator too."""
    table = _FakeSentinelTable()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(yesterday - timedelta(days=1)))], table)
    prisma.db.litellm_dailyteamspend.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    alert = AsyncMock()

    await run_scheduled_ptu_rollup(prisma, alert=alert)

    messages = [call.args[0] for call in alert.await_args_list]
    assert any("backfill" in message for message in messages)
    assert any("unpriced" in message for message in messages)


@pytest.mark.asyncio
async def test_a_broken_alert_channel_does_not_fail_the_backfill():
    table = _FakeSentinelTable()
    prisma = _prisma_for([_windowed_row()], table)
    prisma.db.litellm_dailyteamspend.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_scheduled_ptu_rollup(prisma, alert=AsyncMock(side_effect=RuntimeError("slack down")))

    assert result.rows_failed == 1


# --- R6: the shape the cron actually calls ---------------------------------


@pytest.mark.asyncio
async def test_scheduled_rollup_with_no_target_date_closes_a_backdated_window():
    """The production call shape from proxy_server.py, on the real clock: no target_date,
    a window backdated 30 days, and every elapsed in-window day has to end up priced with
    no operator alert raised. Every other rollup test pins target_date, which is exactly
    why this regression shipped."""
    table = _FakeSentinelTable()
    today = datetime.now(timezone.utc).date()
    opened_on = today - timedelta(days=30)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(opened_on))], table)
    alert = AsyncMock()

    await run_scheduled_ptu_rollup(prisma, alert=alert)

    expected = [(opened_on + timedelta(days=offset)).isoformat() for offset in range(30)]
    assert _priced_dates(table) == expected
    alert.assert_not_awaited()


# --- R8: a rename must not re-price history under the new name ----------------


@pytest.mark.asyncio
async def test_backfill_does_not_double_price_a_day_after_a_rename():
    """A rename does not move the row, because the key is the deployment id. Every already
    priced day stays a single charge and only the unpriced day is written."""
    table = _FakeSentinelTable()
    for offset in (-2, -1):
        table.seed("t", _day(offset), "dep-1", 240.0, model_group="old-name")
    prisma = _prisma_for(
        [_windowed_row(effective_from=_midnight(_day(-2)), model_name="new-name", model_id="dep-1")], table
    )

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    priced_on = [key for key in table.rows if key[1] == _day(-1).isoformat()]
    assert len(priced_on) == 1, f"day {_day(-1)} carries two charges: {priced_on}"
    assert result.rows_written == 1
    assert table.upsert_keys == [("t", _day(0).isoformat(), PTU_SENTINEL_API_KEY, "dep-1")]


@pytest.mark.asyncio
async def test_backfill_still_prices_a_genuinely_missing_day_for_a_renamed_deployment():
    """Rename safety must not swallow real gaps: a day with no row for the deployment at
    all still gets one, and it carries the current display name."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-2), "dep-1", 240.0, model_group="old-name")
    prisma = _prisma_for(
        [_windowed_row(effective_from=_midnight(_day(-2)), model_name="new-name", model_id="dep-1")], table
    )

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.rows_written == 2
    assert sorted(key[1] for key in table.upsert_keys) == [_day(-1).isoformat(), _day(0).isoformat()]
    assert all(key[3] == "dep-1" for key in table.upsert_keys)
    assert table.rows[("t", _day(0).isoformat(), PTU_SENTINEL_API_KEY, "dep-1")]["model_group"] == "new-name"


@pytest.mark.asyncio
async def test_backfill_falls_back_to_the_name_when_a_row_carries_no_source_model_id():
    """A sentinel row whose display name is missing still counts as priced: identity is the
    model column, so the gap check never depends on the name being present."""
    table = _FakeSentinelTable()
    table.seed("t", _day(-1), "m1", 240.0)
    prisma = _prisma_for([_windowed_row(effective_from=_midnight(_day(-1)))], table)

    result = await run_ptu_flat_cost_backfill(prisma, today=TODAY)

    assert result.rows_written == 1
    assert table.upsert_keys == [("t", _day(0).isoformat(), PTU_SENTINEL_API_KEY, "m1")]


@pytest.mark.asyncio
async def test_two_runs_straddling_a_rename_collapse_onto_one_row():
    """The reported defect. Two runs holding different config views of the same deployment
    used to write two keys for one day; keyed on the id they write the same key, so the
    upsert collapses them instead of double charging."""
    table = _FakeSentinelTable()
    before = _prisma_for(
        [_windowed_row(effective_from=_midnight(_day(-1)), model_name="old-name", model_id="dep-1")], table
    )
    after = _prisma_for(
        [_windowed_row(effective_from=_midnight(_day(-1)), model_name="new-name", model_id="dep-1")], table
    )

    await run_ptu_flat_cost_backfill(before, today=TODAY)
    await run_ptu_flat_cost_backfill(after, today=TODAY)

    for offset in (-1, 0):
        charges = [key for key in table.rows if key[1] == _day(offset).isoformat()]
        assert len(charges) == 1, f"day {_day(offset)} carries {len(charges)} charges: {charges}"
    assert sum(row["ptu_flat_cost"] for row in table.rows.values()) == pytest.approx(480.0)


# --- R2: the interleaving that reproduced live on a four-pod rig -------------


@pytest.mark.asyncio
async def test_concurrent_runs_straddling_a_rename_write_one_row():
    """The exact shape reproduced on a live multi-pod rig, which double charged a day.

    Both pods read the day as unpriced before either writes, and a rename lands between
    their config reads. Keyed on the display name they produced two different composite
    keys and both rows survived, permanently. Keyed on the deployment id they produce the
    same key, so the upsert collapses them.
    """
    import asyncio

    table = _FakeSentinelTable()
    ptu = {"ptu_count": 10, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
    read_by_both = asyncio.Event()

    real_keys = ptu_rollup._existing_sentinel_keys
    arrivals = []

    async def gated_keys(*args, **kwargs):
        """Hold the first caller until the second has also read, so neither sees the other."""
        keys = await real_keys(*args, **kwargs)
        arrivals.append(1)
        if len(arrivals) >= 2:
            read_by_both.set()
        await read_by_both.wait()
        return keys

    ptu_rollup._existing_sentinel_keys = gated_keys
    try:
        pod_a = asyncio.create_task(
            run_ptu_flat_cost_backfill(
                _prisma_for([_model_row(model_id="dep-1", model_name="old-name", model_info=ptu)], table),
                today=TODAY,
            )
        )
        pod_b = asyncio.create_task(
            run_ptu_flat_cost_backfill(
                _prisma_for([_model_row(model_id="dep-1", model_name="new-name", model_info=ptu)], table),
                today=TODAY,
            )
        )
        await asyncio.wait_for(asyncio.gather(pod_a, pod_b), timeout=5)
    finally:
        ptu_rollup._existing_sentinel_keys = real_keys

    for day, count in sorted((key[1], 1) for key in table.rows):
        assert count == 1
    per_day = {}
    for team_id, day, api_key, model_id in table.rows:
        per_day[day] = per_day.get(day, 0) + 1
    assert set(per_day.values()) == {1}, f"a day carries more than one charge: {per_day}"
    assert {key[3] for key in table.rows} == {"dep-1"}


@pytest.mark.asyncio
async def test_a_rate_change_between_concurrent_runs_leaves_one_row():
    """Two pods disagreeing on the rate, not just the name, still land on one row. Last
    writer wins on the amount, which is self-consistent rather than a second charge."""
    table = _FakeSentinelTable()
    cheap = _prisma_for(
        [_windowed_row(effective_from=_midnight(_day(-1)), model_id="dep-1", cost_per_ptu_per_hour=2.0)], table
    )
    dear = _prisma_for(
        [_windowed_row(effective_from=_midnight(_day(-1)), model_id="dep-1", cost_per_ptu_per_hour=4.0)], table
    )

    await run_ptu_flat_cost_backfill(cheap, today=TODAY)
    await run_ptu_flat_cost_backfill(dear, today=TODAY)

    assert len(table.rows) == 2  # one per elapsed in-window day, not per config view
    assert all(row["ptu_flat_cost"] == pytest.approx(240.0) for row in table.rows.values())


# --- R6: the prune is the one operation that needs the lock -------------------


@pytest.mark.asyncio
async def test_an_unguarded_run_writes_but_does_not_prune():
    """Without the cross-pod lock the upserts still run, since they are idempotent, but the
    delete does not: its cutoff and the rows' updated_at come from different hosts, so a pod
    whose clock runs ahead would sweep a charge a concurrent pod just wrote."""
    table = _FakeSentinelTable()
    table.seed("t", DAY, "dep-gone", 480.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    prisma = _prisma_for(
        [_model_row(model_id="dep-live", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})],
        table,
    )

    await run_scheduled_ptu_rollup(prisma, target_date=DAY)

    assert table.delete_many_calls == []
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-gone") in table.rows
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-live") in table.rows


@pytest.mark.asyncio
async def test_a_run_holding_the_lock_still_prunes():
    """Losing the sweep entirely would leave stale charges forever, so the guarded path,
    which is the normal one, keeps it."""
    table = _FakeSentinelTable()
    table.seed("t", DAY, "dep-unpriced", 480.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    prisma = _prisma_for(
        [
            _model_row(model_id="dep-live", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}),
            _model_row(model_id="dep-unpriced", model_info={"team_id": "t"}),
        ],
        table,
    )

    await run_scheduled_ptu_rollup(prisma, pod_lock_manager=_pod_lock(acquired=True), target_date=DAY)

    assert table.delete_many_calls != []
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-unpriced") not in table.rows
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-live") in table.rows


@pytest.mark.asyncio
async def test_the_prune_cutoff_allows_for_clock_skew_between_hosts():
    """A row written seconds ago by a pod whose clock lags must survive; one written hours
    ago by a previous run must not. The grace separates the two populations without
    requiring the hosts' clocks to agree."""
    table = _FakeSentinelTable()
    just_written = datetime.now(timezone.utc) - timedelta(seconds=30)
    table.seed("t", DAY, "dep-concurrent", 480.0, updated_at=just_written)
    table.seed("t", DAY, "dep-stale", 480.0, updated_at=datetime.now(timezone.utc) - timedelta(hours=6))
    prisma = _prisma_for([_model_row(model_id="dep-concurrent"), _model_row(model_id="dep-stale")], table)

    await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-concurrent") in table.rows, (
        "a charge written 30s ago by a lagging pod was swept"
    )
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-stale") not in table.rows


@pytest.mark.asyncio
async def test_a_run_pricing_config_cannot_prune_a_row_it_did_not_scan(monkeypatch):
    """Staleness alone stops being evidence once two hosts hold different configuration: a
    row this run never considered belongs to a deployment another host is pricing from its
    own file, and sweeping it drops that charge."""
    table = _FakeSentinelTable()
    table.seed("t", DAY, "dep-elsewhere", 480.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    entry = _router_entry(model_id="cfg-here", model_info=dict(_VALID_PTU))
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(entry))

    await run_scheduled_ptu_rollup(_prisma_for([], table), pod_lock_manager=_pod_lock(acquired=True), target_date=DAY)

    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-elsewhere") in table.rows
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "cfg-here") in table.rows
    assert table.delete_many_calls[-1]["model"]["in"] == ("cfg-here",)


@pytest.mark.asyncio
async def test_a_deployment_deleted_from_the_table_keeps_the_day_it_was_charged(monkeypatch):
    """The accepted cost of bounding the prune, driven through the sequence that produces
    it: charge the day while the deployment exists, remove it, run the day again. Nothing
    scans it now, so nothing may judge its row, and the amount it was billed stands."""
    table = _FakeSentinelTable()
    ptu = {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
    live_row = _model_row(model_id="dep-live", model_info=ptu)
    doomed_row = _model_row(model_id="dep-doomed", model_info=ptu)
    charged_key = ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-doomed")
    monkeypatch.setattr(
        ptu_rollup, "_running_router", lambda: _router_holding(_router_entry(model_id="cfg", model_info=dict(ptu)))
    )

    await run_scheduled_ptu_rollup(
        _prisma_for([live_row, doomed_row], table), pod_lock_manager=_pod_lock(acquired=True), target_date=DAY
    )
    billed = table.rows[charged_key]["ptu_flat_cost"]
    table.rows[charged_key]["updated_at"] = datetime(2020, 1, 1, tzinfo=timezone.utc)

    await run_scheduled_ptu_rollup(
        _prisma_for([live_row], table), pod_lock_manager=_pod_lock(acquired=True), target_date=DAY
    )

    assert table.rows[charged_key]["ptu_flat_cost"] == billed
    assert "dep-doomed" not in table.delete_many_calls[-1]["model"]["in"]


@pytest.mark.asyncio
async def test_a_charge_the_run_cannot_reassess_is_left_alone():
    """A written charge records capacity that was reserved. A deployment absent from every
    source this run reads cannot be reassessed, and another host may be the one declaring
    it, so retracting the charge would drop money the provider still invoiced."""
    table = _FakeSentinelTable()
    table.seed("t", DAY, "dep-gone", 480.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    prisma = _prisma_for(
        [_model_row(model_id="dep-live", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})],
        table,
    )

    await run_scheduled_ptu_rollup(prisma, pod_lock_manager=_pod_lock(acquired=True), target_date=DAY)

    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-gone") in table.rows
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-live") in table.rows


@pytest.mark.asyncio
async def test_every_deployment_that_prices_is_inside_the_set_that_bounds_the_prune():
    """The bound has to be a superset of what the same run wrote, or a run's own charge
    could fall outside its own delete filter and never be reconciled."""
    table = _FakeSentinelTable()
    prisma = _prisma_for(
        [
            _model_row(model_id="dep-a", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}),
            _model_row(model_id="dep-b", model_info={"ptu_count": 9, "cost_per_ptu_per_hour": 1.0, "team_id": "u"}),
            _model_row(model_id="dep-unpriced", model_info={"team_id": "t"}),
        ],
        table,
    )

    loaded = await ptu_rollup._load_ptu_models(prisma)

    assert {model.model_id for model in loaded.models} <= loaded.scanned_ids
    assert loaded.scanned_ids == {"dep-a", "dep-b", "dep-unpriced"}


@pytest.mark.asyncio
async def test_a_priced_deployment_is_in_the_bound_even_with_an_id_the_scan_skips():
    """The bound is built by construction rather than by coincidence. The row scan drops a
    falsy id while the parser still prices one, and a charge outside its own run's delete
    filter could never be reconciled by any later run."""
    prisma = _prisma_for(
        [_model_row(model_id="", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})],
        _FakeSentinelTable(),
    )

    loaded = await ptu_rollup._load_ptu_models(prisma)

    assert {model.model_id for model in loaded.models} <= loaded.scanned_ids


@pytest.mark.asyncio
async def test_the_prune_splits_the_id_set_across_statements(monkeypatch):
    """Every id is one bind variable and the server refuses a statement carrying more than
    32767, so a proxy with that many deployments would fail the prune outright, and with it
    the rest of the scheduled run."""
    monkeypatch.setattr(ptu_rollup, "_PRUNE_ID_CHUNK_SIZE", 2)
    table = _FakeSentinelTable()
    ptu = {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
    deployments = [_model_row(model_id=f"dep-{n}", model_info=ptu) for n in range(4)]
    monkeypatch.setattr(
        ptu_rollup, "_running_router", lambda: _router_holding(_router_entry(model_id="dep-4", model_info=dict(ptu)))
    )
    table.seed("t", DAY, "dep-3", 480.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    await run_scheduled_ptu_rollup(
        _prisma_for(deployments, table), pod_lock_manager=_pod_lock(acquired=True), target_date=DAY
    )

    chunks = [call["model"]["in"] for call in table.delete_many_calls]
    assert len(chunks) == 3
    assert all(len(chunk) <= 2 for chunk in chunks)
    assert sorted(i for chunk in chunks for i in chunk) == [f"dep-{n}" for n in range(5)]


@pytest.mark.asyncio
async def test_scheduled_rollup_writes_nothing_when_ptu_attribution_is_disabled(monkeypatch):
    """Startup already skips scheduling the cron, so this guards the function itself: a
    deployment that never opted in accrues nothing whatever route reaches the rollup."""
    monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)
    table = _FakeSentinelTable()
    prisma = _prisma_for([_model_row(model_info=_VALID_PTU)], table)

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=None, alert=None)

    assert result is None
    assert table.rows == {}
    assert table.upsert_keys == []


# --- config.yaml deployments reach the rollup through the router ----------------


def _router_holding(*entries):
    """A stand-in for the proxy's router, carrying whatever model_list is passed."""
    return types.SimpleNamespace(model_list=list(entries))


@pytest.mark.asyncio
async def test_a_config_declared_deployment_is_priced(monkeypatch):
    """The whole point. A PTU deployment the proxy only knows from config.yaml is not in
    LiteLLM_ProxyModelTable, so a DB-only scan bills the provider's reservation to nobody."""
    entry = _router_entry(model_id="cfg-1", model_name="gpt-4o-ptu", model_info=dict(_VALID_PTU))
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(entry))

    loaded = await ptu_rollup._load_ptu_models(_prisma_for([], _FakeSentinelTable()))

    assert [(m.model_id, m.model_name, m.team_id) for m in loaded.models] == [("cfg-1", "gpt-4o-ptu", "t")]
    assert "cfg-1" in loaded.scanned_ids


@pytest.mark.asyncio
async def test_a_database_backed_router_entry_is_not_counted_twice(monkeypatch):
    """Every deployment loaded from the table is also in the router, flagged db_model. Pricing
    both copies would write two charges for one reservation."""
    row = _model_row(model_id="db-1", model_info=dict(_VALID_PTU))
    mirrored = _router_entry(model_id="db-1", model_info={**_VALID_PTU, "db_model": True})
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(mirrored))

    loaded = await ptu_rollup._load_ptu_models(_prisma_for([row], _FakeSentinelTable()))

    assert [m.model_id for m in loaded.models] == ["db-1"]


@pytest.mark.asyncio
async def test_a_router_entry_sharing_an_id_with_the_table_is_priced_once(monkeypatch):
    """db_model is data the router carries rather than something this module controls, so the
    id anti-join is what actually maps onto the failure: two charges under one id."""
    row = _model_row(model_id="both-1", model_info=dict(_VALID_PTU))
    unflagged = _router_entry(model_id="both-1", model_info=dict(_VALID_PTU))
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(unflagged))

    loaded = await ptu_rollup._load_ptu_models(_prisma_for([row], _FakeSentinelTable()))

    assert [m.model_id for m in loaded.models] == ["both-1"]


@pytest.mark.asyncio
async def test_a_client_credential_clone_is_not_priced(monkeypatch):
    """Supplying an api_key on a request mints a clone of the deployment under a fresh id,
    carrying the source's PTU config. Pricing it bills one reservation per distinct caller key."""
    source = _router_entry(model_id="cfg-1", model_info=dict(_VALID_PTU))
    clone = _router_entry(model_id="cfg-1-clone", model_info={**_VALID_PTU, "original_model_id": "cfg-1"})
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(source, clone))

    loaded = await ptu_rollup._load_ptu_models(_prisma_for([], _FakeSentinelTable()))

    assert [m.model_id for m in loaded.models] == ["cfg-1"]


@pytest.mark.asyncio
async def test_a_config_deployment_without_ptu_config_is_scanned_but_not_priced(monkeypatch):
    """It has to stay in the scanned set or its leftover sentinel rows become unprunable."""
    entry = _router_entry(model_id="cfg-plain", model_info={"team_id": "t"})
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(entry))

    loaded = await ptu_rollup._load_ptu_models(_prisma_for([], _FakeSentinelTable()))

    assert loaded.models == ()
    assert "cfg-plain" in loaded.scanned_ids


@pytest.mark.asyncio
async def test_no_router_in_the_process_prices_the_database_alone(monkeypatch):
    """The rollup is importable and callable outside a running proxy."""
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: None)

    loaded = await ptu_rollup._load_ptu_models(
        _prisma_for([_model_row(model_id="db-1", model_info=dict(_VALID_PTU))], _FakeSentinelTable())
    )

    assert [m.model_id for m in loaded.models] == ["db-1"]


@pytest.mark.asyncio
async def test_a_config_deployment_is_charged_end_to_end(monkeypatch):
    """Through the scheduled entry point, so the charge lands in a sentinel row rather than
    stopping at the loader."""
    table = _FakeSentinelTable()
    entry = _router_entry(model_id="cfg-1", model_name="gpt-4o-ptu", model_info=dict(_VALID_PTU))
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(entry))

    await run_scheduled_ptu_rollup(_prisma_for([], table), pod_lock_manager=_pod_lock(acquired=True), target_date=DAY)

    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "cfg-1") in table.rows


@pytest.mark.asyncio
async def test_a_stale_database_backed_router_entry_is_not_treated_as_config(monkeypatch):
    """The reconcile can leave a deployment on the router after its row is gone. The id
    anti-join cannot see that one, so the flag is what keeps it from being priced as though
    config.yaml had declared it."""
    stale = _router_entry(model_id="db-gone", model_info={**_VALID_PTU, "db_model": True})
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(stale))

    loaded = await ptu_rollup._load_ptu_models(_prisma_for([], _FakeSentinelTable()))

    assert loaded.models == ()


def test_the_router_lookup_reads_the_proxys_own_global():
    """Every other config test replaces this helper, so without one test driving the real
    body a typo in the module path or the attribute name leaves the whole feature dead in
    production with the suite still green."""
    import sys
    import types as _types

    assert ptu_rollup._running_router() is None or "litellm.proxy.proxy_server" in sys.modules

    sentinel = object()
    stub = _types.SimpleNamespace(llm_router=sentinel)
    real = sys.modules.get("litellm.proxy.proxy_server")
    sys.modules["litellm.proxy.proxy_server"] = stub
    try:
        assert ptu_rollup._running_router() is sentinel
        del stub.llm_router
        assert ptu_rollup._running_router() is None
    finally:
        if real is None:
            del sys.modules["litellm.proxy.proxy_server"]
        else:
            sys.modules["litellm.proxy.proxy_server"] = real


def test_the_router_lookup_returns_none_outside_a_proxy():
    import sys

    real = sys.modules.pop("litellm.proxy.proxy_server", None)
    try:
        assert ptu_rollup._running_router() is None
    finally:
        if real is not None:
            sys.modules["litellm.proxy.proxy_server"] = real


def test_the_prune_filter_is_a_plain_dict():
    """The query builder serialises the mapping it is handed and rejects a read-only view of
    one, which the in-memory table in these tests accepts happily. Only a live run caught it."""
    chunk = ("dep-a", "dep-b")
    predicate = ptu_rollup._prune_filter(date_str=DAY.isoformat(), cutoff=datetime.now(timezone.utc), chunk=chunk)

    assert type(predicate) is dict
    assert type(predicate["updated_at"]) is dict
    assert type(predicate["model"]) is dict
    assert predicate["model"]["in"] == chunk


@pytest.mark.asyncio
async def test_a_run_that_scanned_nothing_issues_no_delete_statements():
    """The window where a master-key rotation wipes and recreates the model table. A run that
    can see no deployment can reassess none of them, so it must not reach for the day's rows."""
    table = _FakeSentinelTable()
    table.seed("t", DAY, "dep-orphan", 240.0, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    await run_ptu_flat_cost_rollup(_prisma_for([], table), target_date=DAY)

    assert table.delete_many_calls == []
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "dep-orphan") in table.rows


@pytest.mark.asyncio
async def test_the_catch_up_pass_reaches_a_config_declared_deployment(monkeypatch):
    """The catch-up shares the loader, so config deployments join it without being wired in.
    That is what prices the elapsed days of a reservation declared before today."""
    table = _FakeSentinelTable()
    now = datetime.now(timezone.utc)
    started = (now - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    entry = _router_entry(
        model_id="cfg-back",
        model_info={"ptu_count": 100, "cost_per_ptu_per_hour": 0.02, "team_id": "t", "ptu_effective_from": started},
    )
    monkeypatch.setattr(ptu_rollup, "_running_router", lambda: _router_holding(entry))

    await run_scheduled_ptu_rollup(_prisma_for([], table), pod_lock_manager=_pod_lock(acquired=True))

    charged = sorted(day for (_, day, _, model) in table.rows if model == "cfg-back")
    yesterday = (now.date() - timedelta(days=1)).isoformat()
    assert len(charged) == 3, charged
    assert charged[-1] == yesterday
    assert all(row["ptu_flat_cost"] == pytest.approx(48.0) for row in table.rows.values())

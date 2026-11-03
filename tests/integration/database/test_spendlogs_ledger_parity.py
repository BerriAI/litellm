import asyncio
import json
import os
import uuid
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import psycopg
from psycopg import sql
import pytest
import litellm

from litellm.accounting.application import apply_deliveries
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.constants import BILLED_RETRY_ATTEMPT_METADATA_KEY, INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy.spend_tracking.spend_tracking_utils import spend_log_row_with_retained_metadata
from litellm.proxy import proxy_server
from litellm.proxy._types import SpendLogsMetadataFields
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.model_usage_rollup import flush_model_usage_transactions
from litellm.proxy.guardrails.usage_tracking import PendingRollups, process_spend_logs_guardrail_usage
from litellm.proxy.utils import PrismaClient
from litellm.proxy.hooks.model_budget_shared_windows import DatabaseSharedWindows
from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.repositories.accounting_delivery_repository import AccountingDeliveryRepository
from litellm.repositories.table_repositories import SpendLogsRepository
from litellm.responses.streaming_iterator import drain_logging_tasks
from litellm.types.utils import ModelResponse, Usage
from tests.integration._support.database import scratch_database

_ROOT: Final = Path(__file__).resolve().parents[3]
_NOW: Final = datetime(2026, 11, 2, tzinfo=timezone.utc)
_ENTITY_TABLES: Final = (
    "LiteLLM_VerificationToken",
    "LiteLLM_UserTable",
    "LiteLLM_TeamTable",
    "LiteLLM_TeamMembership",
    "LiteLLM_OrganizationTable",
    "LiteLLM_OrganizationMembership",
    "LiteLLM_EndUserTable",
    "LiteLLM_AgentsTable",
    "LiteLLM_TagTable",
    "LiteLLM_ProjectTable",
    "LiteLLM_ModelAccessGroupBudgetTable",
)
_DAILY_TABLES: Final = (
    "LiteLLM_DailyUserSpend",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_DailyOrganizationSpend",
    "LiteLLM_DailyEndUserSpend",
    "LiteLLM_DailyAgentSpend",
    "LiteLLM_DailyTagSpend",
    "LiteLLM_DailyModelUsage",
    "LiteLLM_DailyGuardrailUsageUnits",
    "LiteLLM_BudgetWindowSpend",
)


@pytest.fixture(scope="module")
def ledger_schema() -> str:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "prisma",
            "migrate",
            "diff",
            "--from-empty",
            "--to-schema-datamodel",
            str(_ROOT / "schema.prisma"),
            "--script",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    ).stdout


@pytest.fixture
def parity_database(ledger_schema: str) -> Iterator[str]:
    with scratch_database("spendlogs_parity") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(ledger_schema)
        connection.execute(
            "INSERT INTO \"LiteLLM_BudgetTable\" (budget_id, created_by, updated_by) VALUES ('budget', 'test', 'test')"
        )
        connection.execute("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('user')")
        connection.execute("INSERT INTO \"LiteLLM_TeamTable\" (team_id) VALUES ('team')")
        connection.execute(
            'INSERT INTO "LiteLLM_OrganizationTable" '
            "(organization_id, organization_alias, budget_id, created_by, updated_by) "
            "VALUES ('org', 'org', 'budget', 'test', 'test')"
        )
        connection.execute("INSERT INTO \"LiteLLM_TeamMembership\" (team_id, user_id) VALUES ('team', 'user')")
        connection.execute(
            "INSERT INTO \"LiteLLM_OrganizationMembership\" (organization_id, user_id) VALUES ('org', 'user')"
        )
        connection.execute("INSERT INTO \"LiteLLM_VerificationToken\" (token) VALUES ('key')")
        connection.execute("INSERT INTO \"LiteLLM_EndUserTable\" (user_id) VALUES ('end-user')")
        connection.execute(
            'INSERT INTO "LiteLLM_AgentsTable" '
            "(agent_id, agent_name, agent_card_params, created_by, updated_by) "
            "VALUES ('agent', 'agent', '{}', 'test', 'test')"
        )
        connection.execute(
            'INSERT INTO "LiteLLM_ProjectTable" (project_id, created_by, updated_by) '
            "VALUES ('project', 'test', 'test')"
        )
        connection.execute("INSERT INTO \"LiteLLM_TagTable\" (tag_name) VALUES ('tag')")
        connection.execute("INSERT INTO \"LiteLLM_ModelAccessGroupBudgetTable\" (access_group_name) VALUES ('group')")
        yield url


def _snapshot(url: str) -> dict[str, list[object]]:
    with psycopg.connect(url) as connection:
        return {
            table: [
                row[0]
                for row in connection.execute(
                    sql.SQL(
                        "SELECT to_jsonb(t) - 'id' - 'created_at' - 'updated_at' - 'request_id' - 'last_active' FROM {} t "
                        "ORDER BY (to_jsonb(t) - 'id' - 'created_at' - 'updated_at' - 'request_id' - 'last_active')::text"
                    ).format(sql.Identifier(table))
                ).fetchall()
            ]
            for table in (*_ENTITY_TABLES, *_DAILY_TABLES)
        }


def _reset(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as connection:
        for table in _ENTITY_TABLES:
            connection.execute(sql.SQL("UPDATE {} SET spend=0").format(sql.Identifier(table)))
        for table in (*_DAILY_TABLES, "LiteLLM_SpendLogs", "LiteLLM_AccountingDelivery"):
            connection.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(table)))
        connection.execute('UPDATE "LiteLLM_VerificationToken" SET total_spend=0')
        connection.execute('UPDATE "LiteLLM_TeamMembership" SET total_spend=0')


async def _assert_legacy_database_parity(
    parity_database: str,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    origin: str | None,
    cost: float,
    retention: str,
    reset_kind: str | None = None,
    correction_order: str | None = None,
    toggle_after_capture: bool = False,
    fallback_narrowing: bool = False,
    failing_table: str | None = None,
):
    monkeypatch.setitem(
        litellm.model_cost,
        "ledger-test",
        {
            "input_cost_per_token": 0.002,
            "output_cost_per_token": 0.004,
            "cache_read_input_token_cost": 0.0005,
            "cache_creation_input_token_cost": 0.003,
            "litellm_provider": "openai",
            "mode": "chat",
            "max_tokens": 4096,
        },
    )
    if fallback_narrowing:
        monkeypatch.setattr(proxy_server, "llm_router", litellm.Router(model_list=[{
            "model_name": "ledger-test",
            "litellm_params": {"model": "openai/ledger-test", "api_key": "test"},
            "model_info": {"id": "deployment", "access_groups": ["group"]},
        }]))
        with psycopg.connect(parity_database, autocommit=True) as connection:
            connection.execute(
                "INSERT INTO \"LiteLLM_ModelAccessGroupBudgetTable\" (access_group_name) VALUES ('unserved-group')"
            )
    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    client.spend_log_transactions = []
    client.model_usage_transactions = []
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    await client.db.connect()
    results = []
    try:
        for path in ("legacy", "deferred_legacy", "ledger"):
            _reset(parity_database)
            if reset_kind is not None:
                _prime_reset(parity_database, reset_kind)
            monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": path == "ledger"})
            monkeypatch.setattr(proxy_server, "disable_spend_logs", path == "ledger" and retention == "lean")
            writer = DBSpendUpdateWriter()
            monkeypatch.setattr(proxy_server.proxy_logging_obj, "db_spend_update_writer", writer)
            monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
            monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache())
            for cache_key in ("key", "team_id:team"):
                await proxy_server.user_api_key_cache.async_set_cache(
                    key=cache_key,
                    value={
                        "budget_limits": [
                            {"budget_duration": "1d", "max_budget": 100, "reset_at": "2026-11-04T00:00:00+00:00"}
                        ]
                    },
                )
            request_kwargs = {
                "model": "ledger-test",
                "custom_llm_provider": "openai",
                "call_type": "acompletion",
                "litellm_call_id": "request",
                "standard_logging_object": {
                    "model": "ledger-test",
                    "model_group": "ledger-test",
                    "metadata": {
                        "user_api_key_hash": "key",
                        "usage_object": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "cache_read_input_tokens": 4,
                            "cache_creation_input_tokens": 1,
                        },
                    },
                    "model_map_information": {},
                    "hidden_params": {},
                    "request_tags": ["tag"],
                    "model_id": "deployment",
                    "response_cost": cost,
                    "autorouter_savings": 0.125,
                    "guardrail_information": [
                        {
                            "guardrail_id": "guard",
                            "guardrail_name": "guard",
                            "guardrail_status": "success",
                            "guardrail_usage": {"text": 2},
                            "guardrail_cost_by_unit": {"text": 0.02},
                            "guardrail_cost_in_spend": True,
                        }
                    ],
                },
                "agent_id": "agent",
                "user": "end-user",
                "litellm_params": {
                    "metadata": {
                        "user_api_key": "key",
                        "user_api_key_model_max_budget": {"ledger-test": {"max_budget": 100, "budget_duration": "1d"}},
                        "user_api_key_user_id": "user",
                        "user_api_key_team_id": "team",
                        "user_api_key_org_id": "org",
                        "user_api_key_project_id": "project",
                        "user_api_key_end_user_id": "end-user",
                        "tags": ["tag"],
                        "user_api_key_matched_model_access_groups": ["group", "unserved-group"] if fallback_narrowing else ["group"],
                        "model_info": {"id": "deployment"},
                        "litellm_gateway_injected_cache": "deployment",
                        "compression_savings": {"tokens_before": 20, "tokens_after": 10, "tokens_saved": 10},
                        "status": status,
                        **({BILLED_RETRY_ATTEMPT_METADATA_KEY: True} if origin == "billed_retry" else {}),
                        **({INTERNAL_CALL_ORIGIN_METADATA_KEY: origin} if origin == "auto_router" else {}),
                    },
                    "user": "end-user",
                },
            }
            limiter = PROXY_VirtualKeyModelMaxBudgetLimiter(
                dual_cache=proxy_server.spend_counter_cache,
                clock=lambda: _NOW.timestamp(),
                shared_windows=DatabaseSharedWindows(prisma_client_source=lambda: client),
                defer_window_spend=path != "legacy",
            )
            if status == "success":
                await limiter.async_log_success_event(request_kwargs, None, _NOW, _NOW)
            charged = await writer.update_database(
                token="key",
                user_id="user",
                end_user_id="end-user",
                team_id="team",
                org_id="org",
                project_id="project",
                kwargs=request_kwargs,
                completion_response=ModelResponse(
                    id="request", model="ledger-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
                ),
                start_time=_NOW,
                end_time=_NOW,
                response_cost=cost,
            )
            assert charged
            await proxy_server.increment_spend_counters(
                token="key",
                team_id="team",
                user_id="user",
                response_cost=cost,
                request_started_at=_NOW,
                accounting_path=path,
            )
            await drain_logging_tasks()
            rows = tuple(client.spend_log_transactions)
            client.spend_log_transactions = []
            assert len(rows) == 1
            if path == "ledger" and retention == "lean":
                assert rows[0]["messages"] is None
                assert rows[0]["response"] is None
                assert rows[0]["proxy_server_request"] is None
            stored_rows = (
                tuple(spend_log_row_with_retained_metadata(row, SpendLogsMetadataFields(include=())) for row in rows)
                if retention == "stripped"
                else rows
            )
            if toggle_after_capture:
                for enabled in (True, False, True):
                    monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": enabled})
                    await SpendLogsRepository(client).insert_with_deliveries(stored_rows)
            else:
                await SpendLogsRepository(client).insert_with_deliveries(stored_rows)
            if reset_kind is not None:
                await _apply_reset(reset_kind, client)
            if correction_order == "before":
                await _publish_savings_correction(client, stored_rows[0])
            if path != "ledger":
                await writer._commit_spend_updates_to_db_without_redis_buffer(client, 0, proxy_server.proxy_logging_obj)
                await writer.commit_daily_tag_spend_to_db(client, 0, proxy_server.proxy_logging_obj)
                await flush_model_usage_transactions(prisma_client=client, transactions=client.model_usage_transactions)
                client.model_usage_transactions = []
                await process_spend_logs_guardrail_usage(client, rows, pending=PendingRollups())
            else:

                async def apply(transaction, deliveries, writer: DBSpendUpdateWriter = writer):
                    await apply_deliveries(
                        transaction,
                        deliveries,
                        prisma_client=client,
                        writer=writer,
                        proxy_logging_obj=proxy_server.proxy_logging_obj,
                    )

                if failing_table is not None:
                    before_failure = _snapshot(parity_database)
                    with psycopg.connect(parity_database, autocommit=True) as connection:
                        connection.execute(sql.SQL('ALTER TABLE {} ADD CONSTRAINT reject_spend CHECK (spend = 0)').format(sql.Identifier(failing_table)))
                    failed = await AccountingDeliveryRepository(client.writer_db).apply_pending(apply)
                    assert (failed.applied, failed.failed) == (0, 1)
                    assert _snapshot(parity_database) == before_failure
                    with psycopg.connect(parity_database, autocommit=True) as connection:
                        assert connection.execute('SELECT state::text, attempts FROM "LiteLLM_AccountingDelivery"').fetchall() == [("pending", 1)]
                        connection.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT reject_spend').format(sql.Identifier(failing_table)))
                        connection.execute('UPDATE "LiteLLM_AccountingDelivery" SET next_attempt_at=CURRENT_TIMESTAMP')
                result = await AccountingDeliveryRepository(client.writer_db).apply_pending(apply)
                assert result.applied == 1, result
            if correction_order == "after":
                await _publish_savings_correction(client, stored_rows[0])
            results.append(_snapshot(parity_database))
        differences = {
            (index, table): (results[0][table], result[table])
            for index, result in enumerate(results[1:], 1)
            for table in results[0]
            if results[0][table] != result[table]
        }
        assert not differences, differences
        prior_spend = 5 if reset_kind is not None and reset_kind != "scheduled" else 0
        assert results[-1]["LiteLLM_UserTable"][0]["spend"] == cost + prior_spend
        daily = results[-1]["LiteLLM_DailyUserSpend"][0]
        assert daily["compression_savings_spend"] > 0, results[-1]["LiteLLM_DailyUserSpend"]
        assert daily["prompt_caching_savings_spend"] != 0
        assert daily["gateway_injected_caching_savings_spend"] != 0
        assert daily["autorouter_savings_spend"] == pytest.approx(
            (0 if origin == "auto_router" else 0.125) + (0.2 if correction_order else 0)
        )
        group_spend = {row["access_group_name"]: row["spend"] for row in results[-1]["LiteLLM_ModelAccessGroupBudgetTable"]}
        assert group_spend == {"group": cost + prior_spend, **({"unserved-group": prior_spend} if fallback_narrowing else {})}
        assert results[-1]["LiteLLM_DailyGuardrailUsageUnits"]
        windows = results[-1]["LiteLLM_BudgetWindowSpend"]
        assert len(windows) == (2 if cost else 0) + (1 if status == "success" else 0)
        assert all(row["spend"] == cost for row in windows)
        if reset_kind == "window_rollover":
            model_window = next(row for row in windows if row["entity_type"] == "model_max_budget")
            assert model_window["window_start"].startswith("2026-11-04")
        if reset_kind in ("key", "global", "scheduled"):
            assert results[-1]["LiteLLM_VerificationToken"][0]["spend"] == cost
        if reset_kind in ("global", "scheduled"):
            assert results[-1]["LiteLLM_TeamTable"][0]["spend"] == cost
        if reset_kind == "team_member":
            assert results[-1]["LiteLLM_TeamMembership"][0]["spend"] == cost
    finally:
        await client.db.disconnect()


@pytest.mark.parametrize(
    "status,origin,cost",
    [
        ("success", None, 0.25),
        ("failure", None, 0.25),
        ("success", None, 0.0),
        ("success", "auto_router", 0.25),
        ("failure", "billed_retry", 0.25),
    ],
)
@pytest.mark.parametrize("retention", ["full", "stripped", "lean"])
@pytest.mark.asyncio
async def test_stored_ledger_matches_real_legacy_database_updates(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, status: str, origin: str | None, cost: float, retention: str
):
    await _assert_legacy_database_parity(parity_database, monkeypatch, status, origin, cost, retention)


@pytest.mark.parametrize("reset_kind", ["scheduled", "key", "team_member", "global", "window_rollover"])
@pytest.mark.asyncio
async def test_pending_delivery_after_reset_matches_queued_legacy_spend(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, reset_kind: str
):
    await _assert_legacy_database_parity(parity_database, monkeypatch, "success", None, 0.25, "full", reset_kind)


def _prime_reset(url: str, reset_kind: str) -> None:
    with psycopg.connect(url, autocommit=True) as connection:
        for table in _ENTITY_TABLES:
            connection.execute(sql.SQL("UPDATE {} SET spend=5").format(sql.Identifier(table)))
        if reset_kind == "scheduled":
            for table in ("LiteLLM_VerificationToken", "LiteLLM_UserTable", "LiteLLM_TeamTable", "LiteLLM_BudgetTable"):
                connection.execute(
                    sql.SQL("UPDATE {} SET budget_duration='1d', budget_reset_at='2026-11-01'").format(
                        sql.Identifier(table)
                    )
                )
            connection.execute("UPDATE \"LiteLLM_ModelAccessGroupBudgetTable\" SET budget_id='budget'")


async def _apply_reset(reset_kind: str, client: PrismaClient) -> None:
    from litellm.proxy._types import ResetSpendRequest, UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.management_endpoints.key_management_endpoints import reset_key_spend_fn
    from litellm.proxy.management_endpoints.team_endpoints import reset_team_member_spend_fn
    from litellm.proxy.spend_tracking.spend_management_endpoints import global_spend_reset

    admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    if reset_kind == "window_rollover":
        from litellm.proxy.db.budget_window_spend_writer import roll_window_spend_row

        windows = await client.writer_db.query_raw('SELECT * FROM "LiteLLM_BudgetWindowSpend"')
        assert windows
        for window in windows:
            await roll_window_spend_row(
                client, window["entity_type"], window["entity_id"], window["window_duration"], _NOW + timedelta(days=2)
            )
    elif reset_kind == "global":
        await global_spend_reset()
    elif reset_kind == "key":
        await reset_key_spend_fn(
            key="key", data=ResetSpendRequest(reset_to=0), user_api_key_dict=admin, litellm_changed_by=None
        )
    elif reset_kind == "team_member":
        await reset_team_member_spend_fn(
            team_id="team", user_id="user", data=ResetSpendRequest(reset_to=0), user_api_key_dict=admin
        )
    else:
        await ResetBudgetJob(proxy_server.proxy_logging_obj, client).reset_budget()


@pytest.mark.parametrize("correction_order", ["before", "after"])
@pytest.mark.asyncio
async def test_baseline_correction_order_preserves_frozen_request_savings(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, correction_order: str
):
    await _assert_legacy_database_parity(
        parity_database, monkeypatch, "success", None, 0.25, "full", correction_order=correction_order
    )


async def _publish_savings_correction(client: PrismaClient, row) -> None:
    from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
    from litellm.proxy.db.baseline_accounting import (
        BaselinePublication,
        DailyBaselineAttribution,
        DailyBaselineTarget,
        _Change,
        _publish,
    )

    metadata_before = (
        await client.writer_db.query_raw(
            'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=$1', row["request_id"]
        )
    )[0]["metadata"]
    if isinstance(metadata_before, str):
        metadata_before = json.loads(metadata_before)
    change = _Change(
        request_id=row["request_id"],
        publication=BaselinePublication(
            comparison_id="autorouter-baseline:v3:" + "a" * 64,
            comparison_started_at=_NOW.timestamp(),
            status="estimated",
            reason="modeled",
            provenance="modeled",
            actual_spend=0.25,
            baseline_spend=0.575,
        ),
        api_key=row["api_key"],
        session_id="session",
        router_name="router",
        baseline_model="ledger-test",
        covered_delta=0,
        actual_delta=0.0,
        savings_delta=0.2,
        daily=DailyBaselineAttribution(
            date=_NOW.date().isoformat(),
            api_key=row["api_key"],
            model=row["model"],
            custom_llm_provider=row["custom_llm_provider"],
            model_group=row["model_group"],
            endpoint=ROUTE_ENDPOINT_MAPPING.get(row["call_type"]),
            targets=tuple(
                DailyBaselineTarget(entity=entity, entity_id=identity)
                for entity, identity in (
                    ("user", "user"),
                    ("team", "team"),
                    ("org", "org"),
                    ("end_user", "end-user"),
                    ("agent", "agent"),
                    ("tag", "tag"),
                )
            ),
        ),
    )
    async with client.writer_db.tx() as transaction:
        await _publish(transaction, (change,))
    metadata_after = (
        await client.writer_db.query_raw(
            'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=$1', row["request_id"]
        )
    )[0]["metadata"]
    if isinstance(metadata_after, str):
        metadata_after = json.loads(metadata_after)
    assert metadata_after["autorouter_savings"] == pytest.approx(0.325)
    if row["accounting_path"] == "ledger":
        assert metadata_after["accounting_savings"] == metadata_before["accounting_savings"]


@pytest.mark.asyncio
async def test_flag_changes_do_not_reclassify_or_reapply_buffered_rows(
    parity_database: str,
    monkeypatch: pytest.MonkeyPatch,
):
    await _assert_legacy_database_parity(
        parity_database,
        monkeypatch,
        "success",
        None,
        0.25,
        "stripped",
        toggle_after_capture=True,
    )


@pytest.mark.parametrize("entity_type", ["key", "team"])
@pytest.mark.asyncio
async def test_window_seed_excludes_pending_and_parked_but_preserves_legacy_cutoff(
    parity_database: str,
    monkeypatch: pytest.MonkeyPatch,
    entity_type: str,
):
    from datetime import timedelta
    from litellm.proxy.db.budget_window_spend_writer import spend_logs_seed_totals, commit_window_spend_updates

    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    await client.db.connect()
    try:
        for request_id, path, cost, offset in (
            ("old-legacy", "legacy", 2.0, -1),
            ("current-legacy", "legacy", 5.0, 0),
            ("applied", "ledger", 3.0, 1),
            ("pending", "ledger", 7.0, 0),
            ("parked", "ledger", 11.0, 0),
        ):
            await client.writer_db.execute_raw(
                'INSERT INTO "LiteLLM_SpendLogs" '
                '(request_id,call_type,api_key,team_id,spend,"startTime","endTime",accounting_path) '
                "VALUES ($1,'acompletion','key','team',$2::float8,$3::timestamp,$3::timestamp,$4)",
                request_id,
                cost,
                _NOW + timedelta(seconds=offset),
                path,
            )
        for request_id, state in (("applied", "applied"), ("pending", "pending"), ("parked", "parked")):
            await client.writer_db.execute_raw(
                'INSERT INTO "LiteLLM_AccountingDelivery" (request_id,state) VALUES ($1,$2::"AccountingDeliveryState")',
                request_id,
                state,
            )
        start = _NOW - timedelta(hours=1)
        totals = await spend_logs_seed_totals(client, entity_type, entity_type, start, _NOW)
        assert totals is not None and totals.total == 10.0 and totals.before_batch == 5.0
        unbounded = await spend_logs_seed_totals(client, entity_type, entity_type, start, None)
        assert unbounded is not None and unbounded.total == unbounded.before_batch == 10.0
        await commit_window_spend_updates(
            client,
            (
                {
                    "entity_type": entity_type,
                    "entity_id": entity_type,
                    "window_duration": "1d",
                    "window_start": start.isoformat(),
                    "started_at": _NOW.isoformat(),
                    "spend": 5.0,
                },
            ),
        )
        rows = await client.writer_db.query_raw('SELECT spend FROM "LiteLLM_BudgetWindowSpend"')
        assert rows[0]["spend"] == 10.0
    finally:
        await client.db.disconnect()


@pytest.mark.parametrize("disable_logs", [False, True])
@pytest.mark.asyncio
async def test_batch_claim_failure_queues_an_atomic_retry_and_repeated_retrieve_does_not_charge(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, disable_logs: bool
):
    from litellm.accounting.worker import flush_accounting_deliveries

    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    client.spend_log_transactions = []
    writer = DBSpendUpdateWriter()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", disable_logs)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    await client.db.connect()


    try:
        with psycopg.connect(parity_database, autocommit=True) as connection:
            connection.execute(
                'ALTER TABLE "LiteLLM_AccountingDelivery" ADD CONSTRAINT reject_batch '
                "CHECK (request_id <> 'batch_batch_cost')"
            )
        assert await _retrieve_batch(writer, 0.25)
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (0,)
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_AccountingDelivery"').fetchone() == (0,)
        assert len(client.spend_log_transactions) == 1
        queued = tuple(client.spend_log_transactions)
        assert queued[0]["accounting_path"] == "ledger"
        with psycopg.connect(parity_database, autocommit=True) as connection:
            connection.execute('ALTER TABLE "LiteLLM_AccountingDelivery" DROP CONSTRAINT reject_batch')
        assert await SpendLogsRepository(client).insert_with_deliveries(queued) == 1
        assert await _retrieve_batch(writer, 0.25) is False
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=\'key\'').fetchone() == (0.25,)
            assert connection.execute('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=\'user\'').fetchone() == (0.25,)
            assert connection.execute('SELECT state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [("applied",)]
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (1,)
    finally:
        await client.db.disconnect()


@pytest.mark.asyncio
async def test_worker_exposes_missing_spend_rows_as_delayed_then_parked_work(
    parity_database: str, monkeypatch: pytest.MonkeyPatch
):
    from prometheus_client import REGISTRY
    from litellm.accounting.worker import flush_accounting_deliveries

    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    writer = DBSpendUpdateWriter()
    await client.db.connect()
    try:
        with psycopg.connect(parity_database, autocommit=True) as connection:
            connection.execute(
                'INSERT INTO "LiteLLM_AccountingDelivery" (request_id, created_at) '
                "VALUES ('missing-spend-row', CURRENT_TIMESTAMP - INTERVAL '1 hour')"
            )
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        assert REGISTRY.get_sample_value("litellm_accounting_pending") == 1
        assert REGISTRY.get_sample_value("litellm_accounting_parked") == 0
        with psycopg.connect(parity_database, autocommit=True) as connection:
            assert connection.execute(
                'SELECT state::text, attempts, last_error FROM "LiteLLM_AccountingDelivery"'
            ).fetchall() == [("pending", 1, "ValueError")]
            connection.execute(
                'UPDATE "LiteLLM_AccountingDelivery" SET attempts=4, next_attempt_at=CURRENT_TIMESTAMP'
            )
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        assert REGISTRY.get_sample_value("litellm_accounting_pending") == 0
        assert REGISTRY.get_sample_value("litellm_accounting_parked") == 1
        with psycopg.connect(parity_database) as connection:
            assert connection.execute(
                'SELECT state::text, attempts, last_error FROM "LiteLLM_AccountingDelivery"'
            ).fetchall() == [("parked", 5, "ValueError")]
            assert connection.execute('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=\'user\'').fetchone() == (0,)
    finally:
        await client.db.disconnect()


@pytest.mark.asyncio
async def test_fallback_charges_only_groups_declared_by_the_served_deployment(
    parity_database: str, monkeypatch: pytest.MonkeyPatch
):
    await _assert_legacy_database_parity(
        parity_database, monkeypatch, "success", None, 0.25, "stripped", fallback_narrowing=True
    )


@pytest.mark.asyncio
async def test_worker_skips_another_leader_and_releases_its_redis_lock_after_retry(
    parity_database: str, monkeypatch: pytest.MonkeyPatch
):
    from litellm.accounting.worker import flush_accounting_deliveries
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

    monkeypatch.setenv("DATABASE_URL", parity_database)
    cache = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), namespace=uuid.uuid4().hex)
    other_leader = PodLockManager(cache)
    writer = DBSpendUpdateWriter(redis_cache=cache)
    writer.pod_lock_manager.redis_cache = cache
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    await client.db.connect()
    try:
        with psycopg.connect(parity_database, autocommit=True) as connection:
            connection.execute('INSERT INTO "LiteLLM_AccountingDelivery" (request_id) VALUES (\'missing-row\')')
        assert await other_leader.acquire_lock(cronjob_id="accounting_delivery", allow_reentrant=False)
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT attempts FROM "LiteLLM_AccountingDelivery"').fetchall() == [(0,)]
        await other_leader.release_lock(cronjob_id="accounting_delivery")
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT attempts FROM "LiteLLM_AccountingDelivery"').fetchall() == [(1,)]
        assert await other_leader.acquire_lock(cronjob_id="accounting_delivery", allow_reentrant=False)
    finally:
        await other_leader.release_lock(cronjob_id="accounting_delivery")
        await client.db.disconnect()
        await cache.disconnect()


async def _retrieve_batch(writer: DBSpendUpdateWriter, cost: float) -> bool:
    return await writer.update_database(
        token="key", user_id="user", end_user_id=None, team_id="team", org_id=None,
        kwargs={
            "model": "ledger-test", "call_type": "aretrieve_batch", "litellm_call_id": "batch",
            "litellm_params": {"metadata": {"status": "success", "user_api_key": "key"}},
        },
        completion_response={"id": "batch", "status": "completed"},
        start_time=_NOW, end_time=_NOW, response_cost=cost,
    )


@pytest.mark.parametrize("enable_before_takeover,delivery_unavailable", [(False, False), (True, False), (True, True)])
@pytest.mark.asyncio
async def test_legacy_batch_takeover_keeps_its_path_while_original_increments_are_queued(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, enable_before_takeover: bool, delivery_unavailable: bool
):
    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    client.spend_log_transactions = []
    client.model_usage_transactions = []
    writer = DBSpendUpdateWriter()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": False})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    await client.db.connect()
    try:
        assert await _retrieve_batch(writer, 0)
        await drain_logging_tasks()
        monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": enable_before_takeover})
        assert await _retrieve_batch(writer, 0.25)
        await drain_logging_tasks()
        assert await _retrieve_batch(writer, 0.25) is False
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=\'user\'').fetchone() == (0,)
            assert connection.execute('SELECT accounting_path, spend FROM "LiteLLM_SpendLogs"').fetchall() == [("legacy", 0.25)]
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_AccountingDelivery"').fetchone() == (0,)
        if delivery_unavailable:
            with psycopg.connect(parity_database, autocommit=True) as connection:
                connection.execute('ALTER TABLE "LiteLLM_AccountingDelivery" RENAME TO temporarily_unavailable_delivery')
        try:
            await writer.db_update_spend_transaction_handler(client, 0, proxy_server.proxy_logging_obj)
        finally:
            if delivery_unavailable:
                with psycopg.connect(parity_database, autocommit=True) as connection:
                    connection.execute('ALTER TABLE temporarily_unavailable_delivery RENAME TO "LiteLLM_AccountingDelivery"')
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=\'user\'').fetchone() == (0.25,)
            assert connection.execute('SELECT spend, api_requests FROM "LiteLLM_DailyUserSpend"').fetchall() == [(0.25, 2)]
    finally:
        await client.db.disconnect()


@pytest.mark.parametrize("failing_table", [
    "LiteLLM_UserTable", "LiteLLM_VerificationToken", "LiteLLM_TeamTable", "LiteLLM_TeamMembership",
    "LiteLLM_OrganizationMembership", "LiteLLM_OrganizationTable", "LiteLLM_ProjectTable", "LiteLLM_AgentsTable", "LiteLLM_EndUserTable",
])
@pytest.mark.asyncio
async def test_destination_failure_rolls_back_all_earlier_updates_before_retry(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, failing_table: str
):
    await _assert_legacy_database_parity(
        parity_database, monkeypatch, "success", None, 0.25, "full", failing_table=failing_table
    )


@pytest.mark.asyncio
async def test_cleanup_jobs_report_protected_rows_until_delivery_commits(
    parity_database: str, monkeypatch: pytest.MonkeyPatch
):
    from prometheus_client import REGISTRY
    from litellm.accounting.worker import flush_accounting_deliveries
    from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup
    from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import SpendLogsPartitionManager

    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    writer = DBSpendUpdateWriter()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    cleanup = SpendLogCleanup(clock=lambda: 0)
    await client.db.connect()
    try:
        assert await _retrieve_batch(writer, 0.25)
        with psycopg.connect(parity_database, autocommit=True) as connection:
            connection.execute('CREATE TABLE "LiteLLM_SpendLogs_p20261102" (request_id text)')
            connection.execute('INSERT INTO "LiteLLM_SpendLogs_p20261102" VALUES (\'batch_batch_cost\')')
        previous_rows = REGISTRY.get_sample_value("litellm_accounting_cleanup_postponed_total", {"kind": "rows"}) or 0
        previous_partitions = REGISTRY.get_sample_value("litellm_accounting_cleanup_postponed_total", {"kind": "partition"}) or 0
        assert await cleanup._execute_delete_batch(client, "", _NOW + timedelta(days=1), "LiteLLM_SpendLogs", 60) == 0
        assert await SpendLogsPartitionManager._drop_settled_partition(client, "LiteLLM_SpendLogs_p20261102", 1000) is False
        assert REGISTRY.get_sample_value("litellm_accounting_cleanup_postponed_total", {"kind": "rows"}) == previous_rows + 1
        assert REGISTRY.get_sample_value("litellm_accounting_cleanup_postponed_total", {"kind": "partition"}) == previous_partitions + 1
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (1,)
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs_p20261102"').fetchone() == (1,)
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        assert await cleanup._execute_delete_batch(client, "", _NOW + timedelta(days=1), "LiteLLM_SpendLogs", 60) == 1
        assert await SpendLogsPartitionManager._drop_settled_partition(client, "LiteLLM_SpendLogs_p20261102", 1000) is True
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (0,)
            assert connection.execute("SELECT to_regclass('\"LiteLLM_SpendLogs_p20261102\"')").fetchone() == (None,)
    finally:
        await client.db.disconnect()


@pytest.mark.parametrize("receipt_state", ["pending", "parked", "applied", "missing"])
@pytest.mark.asyncio
async def test_batch_takeover_uses_the_stored_ledger_stamp_after_flag_disable(
    parity_database: str, monkeypatch: pytest.MonkeyPatch, receipt_state: str
):
    from litellm.accounting.worker import flush_accounting_deliveries

    monkeypatch.setenv("DATABASE_URL", parity_database)
    client = PrismaClient(database_url=parity_database, proxy_logging_obj=proxy_server.proxy_logging_obj)
    writer = DBSpendUpdateWriter()
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    await client.db.connect()
    try:
        assert await _retrieve_batch(writer, 0)
        if receipt_state in ("applied", "missing"):
            await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        with psycopg.connect(parity_database, autocommit=True) as connection:
            if receipt_state == "missing":
                connection.execute('DELETE FROM "LiteLLM_AccountingDelivery"')
            if receipt_state == "parked":
                connection.execute('UPDATE "LiteLLM_AccountingDelivery" SET state=\'parked\'')
        monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": False})
        assert await _retrieve_batch(writer, 0.25)
        assert await _retrieve_batch(writer, 0.25) is False
        await flush_accounting_deliveries(writer, client, proxy_server.proxy_logging_obj)
        with psycopg.connect(parity_database) as connection:
            assert connection.execute('SELECT accounting_path, spend FROM "LiteLLM_SpendLogs"').fetchall() == [("ledger", 0.25)]
            expected_spend = 0 if receipt_state == "parked" else 0.25
            expected_state = "parked" if receipt_state == "parked" else "applied"
            assert connection.execute('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=\'user\'').fetchone() == (expected_spend,)
            assert connection.execute('SELECT state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [(expected_state,)]
    finally:
        await client.db.disconnect()

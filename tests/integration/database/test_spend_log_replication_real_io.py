import asyncio
import hmac
import json
import os
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Final,
    cast,  # noqa: TID251  # psycopg requires LiteralString for trusted checked-in SQL files
)
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from psycopg import Connection, sql
from psycopg.rows import TupleRow
from pydantic import JsonValue, TypeAdapter
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from typing_extensions import LiteralString, TypedDict

from litellm import Router
from litellm.exceptions import BudgetExceededError
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import check_end_user_budget, virtual_key_max_budget_check
from litellm.proxy.auth.fallback_budget import RouterFallbackBudgetCheck
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db import spend_log_replication
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import SpendLogsPartitionManager
from litellm.proxy.db.fleet_budget_spend import (
    read_replica_budget_spend,
    record_fleet_key_budget_reset,
    replicated_budget_entity_id,
)
from litellm.proxy.db.spend_log_replication import (
    STREAM_NAME,
    SpendLogEvent,
    claim_batch,
    connect_regional_db,
    consume_batch,
    ensure_group,
    import_batch,
    publish_batch,
    read_batch,
    replication_redis,
    trim_stream,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.spend_tracking.carried_budget_state import carry_team_and_user_budget_state
from litellm.proxy.spend_tracking.spend_management_endpoints import _ui_session_grouped_spend_logs, router
from litellm.proxy.utils import PrismaClient, ProxyLogging
from tests.integration._support.redis_process import OwnedRedis, owned_redis

_ROOT: Final = Path(__file__).resolve().parents[3]
_MIGRATION: Final = (
    _ROOT / "litellm/proxy-extras/litellm_proxy_extras/migrations/20261124040000_spend_log_replication/migration.sql"
)
_PARTITION_SCRIPT: Final = _ROOT / "litellm/scripts/db/partition_spend_logs.sql"
_UNPARTITION_SCRIPT: Final = _ROOT / "litellm/scripts/db/unpartition_spend_logs.sql"
_ENTRIES: Final = TypeAdapter(list[tuple[str, dict[str, str]]])
_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_REQUEST_ID_KEY: Final = "a3" * 32


def enable_capture(db: Connection[TupleRow], region: str) -> None:
    spend_log_replication.enable_capture(db, region, _REQUEST_ID_KEY)


def test_capture_rejects_a_database_without_the_insert_trigger(regional_db: Connection[TupleRow]) -> None:
    regional_db.execute('DROP TRIGGER litellm_capture_spend_log_insert ON "LiteLLM_SpendLogs"')

    with pytest.raises(ValueError, match="Apply the LiteLLM database migrations"):
        enable_capture(regional_db, "eu")

    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogReplication"').fetchall() == []


def execute_sql_file(db: Connection[TupleRow], path: Path) -> None:
    db.execute(cast("LiteralString", path.read_text()))  # cast-ok: checked-in SQL file


@pytest.fixture
def regional_db() -> Iterator[Connection[TupleRow]]:
    schema: Final = sql.Identifier(f"Stream_Replication_{uuid4().hex}")
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        try:
            db.execute(sql.SQL("SET search_path TO {}").format(schema))
            db.execute(
                'CREATE TABLE "LiteLLM_SpendLogs" (request_id text PRIMARY KEY, spend double precision NOT NULL, '
                """metadata jsonb NOT NULL DEFAULT '{}', created_at timestamp(3) NOT NULL,
                updated_at timestamp(3) NOT NULL, call_type text NOT NULL DEFAULT 'acompletion',
                api_key text NOT NULL DEFAULT '', total_tokens int NOT NULL DEFAULT 5,
                prompt_tokens int NOT NULL DEFAULT 2, completion_tokens int NOT NULL DEFAULT 3,
                "startTime" timestamp(3) NOT NULL DEFAULT '2026-11-23 12:00:00',
                "endTime" timestamp(3) NOT NULL DEFAULT '2026-11-23 12:00:01',
                "completionStartTime" timestamp(3), request_duration_ms int,
                model text NOT NULL DEFAULT '', model_id text, model_group text, custom_llm_provider text,
                "user" text, team_id text, organization_id text, status text, litellm_call_id text,
                end_user text, session_id text, api_base text, cache_hit text, cache_key text,
                requester_ip_address text, mcp_namespaced_tool_name text, billing_agent_id text, agent_id text,
                messages jsonb NOT NULL DEFAULT '{}', response jsonb NOT NULL DEFAULT '{}',
                proxy_server_request jsonb NOT NULL DEFAULT '{}', request_tags jsonb NOT NULL DEFAULT '[]')"""
            )
            execute_sql_file(db, _MIGRATION)
            yield db
        finally:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))


def record_spend(db: Connection[TupleRow], request_id: str, spend: float) -> None:
    db.execute(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id, spend, created_at, updated_at) '
        "VALUES (%s, %s, '2026-11-23 12:00:00', '2026-11-23 12:00:00')",
        (request_id, spend),
    )


@pytest.fixture
def stream_server(tmp_path: Path) -> Iterator[OwnedRedis]:
    with owned_redis(tmp_path) as server:
        yield server


@pytest_asyncio.fixture(loop_scope="function")
async def stream_redis(stream_server: OwnedRedis) -> AsyncIterator[Redis]:
    async with replication_redis(f"redis://{stream_server.host}:{stream_server.port}") as redis:
        yield redis


async def stream_payloads(redis: Redis) -> tuple[str, ...]:
    entries: Final = _ENTRIES.validate_python(await redis.xrange(STREAM_NAME))
    return tuple(fields["event"] for _, fields in entries)


def event(
    spend: float,
    updated_at: str,
    *,
    region: str = "us",
    start_time: str = "2026-11-23T12:00:00",
    request_id: str = "remote-request",
) -> str:
    return json.dumps(
        {
            "region": region,
            "row": {
                "request_id": request_id,
                "call_type": "acompletion",
                "spend": spend,
                "model": "fleet-model",
                "api_key": "a" * 64,
                "user": "fleet-user",
                "team_id": "fleet-team",
                "end_user": "fleet-end-user",
                "startTime": start_time,
                "endTime": "2026-11-23T12:00:01",
                "created_at": "2026-11-23T12:00:00",
                "updated_at": updated_at,
            },
        }
    )


async def test_replay_after_a_crash_before_ack_keeps_one_latest_row(
    regional_db: Connection[TupleRow], stream_redis: Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    original: Final = event(0.13, "2026-11-23T12:00:00")
    updated: Final = event(0.27, "2026-11-23T12:00:01")
    await ensure_group(stream_redis, "eu")
    for payload in (original, updated):
        await stream_redis.xadd(STREAM_NAME, {"event": payload})
    entries: Final = await read_batch(stream_redis, "eu", "crashed-worker")

    async def interrupted_ack(*_args: object) -> None:
        assert regional_db.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
        raise RedisConnectionError("worker lost its connection before acknowledging")

    with monkeypatch.context() as patch:
        patch.setattr(stream_redis, "xack", interrupted_ack)
        with pytest.raises(RedisConnectionError, match="before acknowledging"):
            await consume_batch(regional_db, stream_redis, entries, "eu")
    assert regional_db.execute('SELECT spend FROM "LiteLLM_SpendLogReporting"').fetchall() == [(0.27,)]

    monkeypatch.setattr(spend_log_replication, "_CLAIM_IDLE_MS", 0)
    _, reclaimed = await claim_batch(stream_redis, "eu", "replacement-worker", "0-0")
    assert reclaimed == entries
    await consume_batch(regional_db, stream_redis, reclaimed, "eu")
    for payload in (updated, updated, original):
        await stream_redis.xadd(STREAM_NAME, {"event": payload})
    await consume_batch(regional_db, stream_redis, await read_batch(stream_redis, "eu", "replacement-worker"), "eu")
    assert await stream_redis.xpending_range(STREAM_NAME, "eu", "-", "+", 10) == []
    assert regional_db.execute(
        'SELECT request_id, spend, model, metadata, source_region FROM "LiteLLM_SpendLogReporting"'
    ).fetchall() == [("remote-request", 0.27, "fleet-model", None, "us")]
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []


@pytest_asyncio.fixture(loop_scope="function")
async def fleet_prisma(
    regional_db: Connection[TupleRow], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[PrismaClient]:
    enable_capture(regional_db, "eu")
    current: Final = regional_db.execute("SELECT current_schema()").fetchone()
    assert current is not None and isinstance(current[0], str)
    parsed: Final = urlsplit(os.environ["DATABASE_URL"])
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": current[0]}))
    )
    monkeypatch.setenv("DATABASE_URL", scoped_url)
    prisma: Final = PrismaClient(scoped_url, ProxyLogging(user_api_key_cache=UserApiKeyCache()))
    await prisma.connect()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    try:
        yield prisma
    finally:
        await prisma.disconnect()


@pytest.mark.parametrize(
    "counter_key",
    ("spend:key:" + "a" * 64, "spend:team:fleet-team", "spend:user:fleet-user", "spend:end_user:fleet-end-user"),
)
async def test_fleet_budget_combines_local_and_remote_spend_once(
    regional_db: Connection[TupleRow], fleet_prisma: PrismaClient, counter_key: str
) -> None:
    remote: Final = event(4.0, "2026-11-23T12:00:00")
    duplicate: Final = event(4.0, "2026-11-23T12:00:00", region="ap")
    assert await proxy_server.get_current_spend(counter_key, 7.0, max_budget=10.0) == 7.0
    import_batch(regional_db, (remote, remote, duplicate), "eu")
    assert await proxy_server.get_current_spend(counter_key, 7.0, max_budget=10.0) == 11.0
    assert await proxy_server.get_current_spend(counter_key, 7.0) == 7.0
    assert (
        await read_replica_budget_spend(
            fleet_prisma, counter_key, window_start=datetime(2026, 11, 24, tzinfo=timezone.utc)
        )
        == 0.0
    )

    regional_db.execute('DELETE FROM "LiteLLM_ReplicatedSpendLogs"')
    import_batch(regional_db, (remote,), "eu")
    assert await proxy_server.get_current_spend(counter_key, 7.0, max_budget=10.0) == 11.0
    assert regional_db.execute('SELECT spend FROM "LiteLLM_SpendLogs"').fetchall() == []


async def test_replayed_stream_spend_blocks_an_otherwise_under_budget_key_once(
    regional_db: Connection[TupleRow], fleet_prisma: PrismaClient, stream_redis: Redis
) -> None:
    key: Final = UserAPIKeyAuth(token="a" * 64, spend=7.0, max_budget=10.0)
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    await virtual_key_max_budget_check(key, logging)
    await ensure_group(stream_redis, "eu")
    payload: Final = event(4.0, "2026-11-23T12:00:00")
    for _ in range(2):
        await stream_redis.xadd(STREAM_NAME, {"event": payload})
        await consume_batch(regional_db, stream_redis, await read_batch(stream_redis, "eu", "worker"), "eu")
        with pytest.raises(BudgetExceededError):
            await virtual_key_max_budget_check(key, logging)
        assert await read_replica_budget_spend(fleet_prisma, f"spend:key:{key.token}") == 4.0


@pytest.mark.usefixtures("fleet_prisma")
async def test_end_user_budget_excludes_replica_spend_from_previous_window(
    regional_db: Connection[TupleRow],
) -> None:
    end_user: Final = LiteLLM_EndUserTable.model_validate(
        {
            "user_id": "fleet-end-user",
            "blocked": False,
            "spend": 7.0,
            "litellm_budget_table": {
                "max_budget": 10.0,
                "budget_duration": "1d",
                "budget_reset_at": "2026-11-25T00:00:00Z",
            },
        }
    )
    import_batch(regional_db, (event(9.0, "2026-11-23T12:00:01"),), "eu")
    await check_end_user_budget(end_user, "/v1/chat/completions")

    current: Final = event(4.0, "2026-11-24T12:00:01", start_time="2026-11-24T12:00:00")
    import_batch(regional_db, (current,), "eu")
    with pytest.raises(BudgetExceededError) as exc:
        await check_end_user_budget(end_user, "/v1/chat/completions")
    assert exc.value.current_cost == 11.0


@pytest.mark.usefixtures("fleet_prisma")
@pytest.mark.parametrize("scope", ("key", "user"))
@pytest.mark.parametrize("duration", ("1d", None))
async def test_paid_fallback_only_counts_replica_spend_in_the_active_budget_window(
    regional_db: Connection[TupleRow], scope: str, duration: str | None
) -> None:
    reset_at: Final = datetime(2026, 11, 25, tzinfo=timezone.utc) if duration is not None else None
    token: Final = UserAPIKeyAuth(
        token="a" * 64,
        user_id="fleet-user",
        spend=7.0,
        user_spend=7.0,
        max_budget=10.0 if scope == "key" else None,
        user_max_budget=10.0 if scope == "user" else None,
        budget_duration=duration if scope == "key" else None,
        budget_reset_at=reset_at if scope == "key" else None,
    )
    carry_team_and_user_budget_state(
        token, None, LiteLLM_UserTable(user_id="fleet-user", budget_duration=duration, budget_reset_at=reset_at)
    )
    router: Final = Router(
        model_list=[{"model_name": "paid", "litellm_params": {"model": "gpt-4o", "api_key": "unused"}}]
    )
    check: Final = RouterFallbackBudgetCheck(is_enforced=lambda: True)
    metadata: Final = {"user_api_key_auth": LiteLLMProxyRequestSetup.auth_for_request_metadata(token)}
    import_batch(regional_db, (event(9.0, "2026-11-23T12:00:01"),), "eu")
    assert await check(model="paid", request_kwargs={"metadata": metadata}, ctx=router._core_ctx) is (
        duration is not None
    )

    current: Final = event(4.0, "2026-11-24T12:00:01", start_time="2026-11-24T12:00:00", request_id="current-request")
    import_batch(regional_db, (current,), "eu")
    assert not await check(model="paid", request_kwargs={"metadata": metadata}, ctx=router._core_ctx)


async def test_key_reset_excludes_replayed_spend_without_resetting_the_team(
    regional_db: Connection[TupleRow], fleet_prisma: PrismaClient
) -> None:
    previous: Final = event(9.0, "2026-11-23T12:00:01")
    current: Final = event(4.0, "2026-11-24T12:00:01", start_time="2026-11-24T12:00:00")
    key: Final = "a" * 64
    import_batch(regional_db, (previous,), "eu")
    assert await read_replica_budget_spend(fleet_prisma, f"spend:key:{key}") == 9.0
    async with fleet_prisma.writer_db.tx() as tx:
        await record_fleet_key_budget_reset(tx, key, datetime(2026, 11, 24, tzinfo=timezone.utc))
    import_batch(regional_db, (previous, current), "eu")
    assert await read_replica_budget_spend(fleet_prisma, f"spend:key:{key}") == 4.0
    assert await read_replica_budget_spend(fleet_prisma, "spend:team:fleet-team") == 13.0


def test_remote_logs_do_not_reserve_local_request_ids_or_count_as_local_spend(
    regional_db: Connection[TupleRow],
) -> None:
    enable_capture(regional_db, "eu")
    import_batch(regional_db, (event(9.0, "2026-11-23T12:00:00"),), "eu")
    assert regional_db.execute('SELECT request_id, spend FROM "LiteLLM_SpendLogs"').fetchall() == []
    assert regional_db.execute('SELECT request_id, spend FROM "LiteLLM_SpendLogReporting"').fetchall() == [
        ("remote-request", 9.0)
    ]

    record_spend(regional_db, "remote-request", 0.04)
    import_batch(regional_db, (event(19.0, "2026-11-23T12:00:01"),), "eu")

    assert regional_db.execute('SELECT request_id, spend FROM "LiteLLM_SpendLogs"').fetchall() == [
        ("remote-request", 0.04)
    ]
    assert regional_db.execute(
        'SELECT request_id, spend, source_region FROM "LiteLLM_SpendLogReporting"'
    ).fetchall() == [("remote-request", 0.04, None)]
    assert regional_db.execute('SELECT "row"->>\'request_id\' FROM "LiteLLM_SpendLogOutbox"').fetchall() == [
        ("remote-request",)
    ]
    regional_db.execute('DELETE FROM "LiteLLM_SpendLogs"')
    import_batch(regional_db, (event(29.0, "2026-11-23T12:00:02"),), "eu")
    assert regional_db.execute('SELECT source_region, spend FROM "LiteLLM_FleetBudgetSpend"').fetchall() == [
        (None, 0.0)
    ]


async def test_worker_reclaims_pending_rows_and_publishes_the_local_outbox(
    regional_db: Connection[TupleRow], stream_redis: Redis, stream_server: OwnedRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    await ensure_group(stream_redis, "eu")
    await stream_redis.xadd(STREAM_NAME, {"event": event(0.27, "2026-11-23T12:00:01")})
    assert len(await read_batch(stream_redis, "eu", "crashed-worker")) == 1
    enable_capture(regional_db, "eu")
    record_spend(regional_db, "local-request", 0.04)
    monkeypatch.setattr(spend_log_replication, "_CLAIM_IDLE_MS", 0)
    stop: Final = asyncio.Event()
    worker: Final = asyncio.create_task(
        spend_log_replication.run_replication(
            regional_db,
            {
                "LITELLM_SPEND_LOG_REGION": "eu",
                "LITELLM_SPEND_LOG_REQUEST_ID_KEY": _REQUEST_ID_KEY,
                "LITELLM_SPEND_LOG_REDIS_URL": f"redis://{stream_server.host}:{stream_server.port}",
            },
            stop,
        )
    )
    try:
        async with asyncio.timeout(5):
            while await stream_redis.xlen(STREAM_NAME) != 2:
                if worker.done():
                    await worker
                    pytest.fail("Replication worker stopped before publishing the outbox")
                await asyncio.sleep(0.01)
    finally:
        stop.set()
        await worker

    assert regional_db.execute('SELECT request_id, spend FROM "LiteLLM_ReplicatedSpendLogs"').fetchall() == [
        ("remote-request", 0.27)
    ]
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []
    assert await stream_redis.xpending_range(STREAM_NAME, "eu", "-", "+", 10) == []


async def test_stream_retention_keeps_unread_and_pending_events_in_each_region(
    stream_redis: Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    await ensure_group(stream_redis, "eu")
    await ensure_group(stream_redis, "us")
    monkeypatch.setattr(spend_log_replication, "_RETENTION_MS", 0)
    for offset in range(1, 205):
        await stream_redis.xadd(STREAM_NAME, {"event": "sample"}, id=f"{offset}-0")
    first_eu: Final = await read_batch(stream_redis, "eu", "worker")
    await stream_redis.xack(STREAM_NAME, "eu", *(entry_id for entry_id, _ in first_eu[1:]))
    for _ in range(3):
        us_entries: Final = await read_batch(stream_redis, "us", "worker")
        await stream_redis.xack(STREAM_NAME, "us", *(entry_id for entry_id, _ in us_entries))

    await trim_stream(stream_redis)
    assert await stream_redis.xlen(STREAM_NAME) == 204
    await stream_redis.xack(STREAM_NAME, "eu", first_eu[0][0])
    await trim_stream(stream_redis)
    second_eu: Final = await read_batch(stream_redis, "eu", "worker")
    assert tuple(entry_id for entry_id, _ in second_eu) == tuple(f"{offset}-0" for offset in range(101, 201))
    await stream_redis.xack(STREAM_NAME, "eu", *(entry_id for entry_id, _ in second_eu))
    await trim_stream(stream_redis)
    assert 4 <= await stream_redis.xlen(STREAM_NAME) < 204


def test_replay_keeps_distinct_origins_and_request_times(regional_db: Connection[TupleRow]) -> None:
    original: Final = event(0.1, "2026-11-23T12:00:00")
    another_region: Final = event(0.2, "2026-11-23T12:00:00", region="ap")
    another_request: Final = event(0.3, "2026-11-23T12:00:01", start_time="2026-11-23T12:00:01")
    import_batch(regional_db, (original, another_region, another_request, original), "eu")

    assert regional_db.execute(
        'SELECT source_region, spend FROM "LiteLLM_SpendLogReporting" ORDER BY spend'
    ).fetchall() == [("us", 0.1), ("ap", 0.2), ("us", 0.3)]


async def test_local_event_is_skipped_and_acknowledged_even_when_its_row_is_absent(
    regional_db: Connection[TupleRow], stream_redis: Redis
) -> None:
    enable_capture(regional_db, "us")
    await ensure_group(stream_redis, "us")
    await stream_redis.xadd(STREAM_NAME, {"event": event(0.13, "2026-11-23T12:00:00")})
    entries: Final = await read_batch(stream_redis, "us", "worker")
    assert len(entries) == 1
    await consume_batch(regional_db, stream_redis, entries, "us")
    assert regional_db.execute('SELECT request_id FROM "LiteLLM_SpendLogReporting"').fetchall() == []
    assert await stream_redis.xpending_range(STREAM_NAME, "us", "-", "+", 10) == []


def test_replay_after_retention_preserves_pre_activation_local_budget_precedence(
    regional_db: Connection[TupleRow],
) -> None:
    record_spend(regional_db, "remote-request", 0.13)
    enable_capture(regional_db, "eu")
    import_batch(regional_db, (event(0.13, "2026-11-23T12:00:00"),), "eu")

    regional_db.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = %s', ("remote-request",))
    import_batch(regional_db, (event(0.27, "2026-11-23T12:00:01"),), "eu")

    assert regional_db.execute(
        'SELECT request_id, source_region, spend FROM "LiteLLM_FleetBudgetSpend"'
    ).fetchall() == [("remote-request", None, 0.0)]


def test_first_import_after_retention_preserves_pre_activation_local_budget_precedence(
    regional_db: Connection[TupleRow],
) -> None:
    record_spend(regional_db, "remote-request", 0.13)
    enable_capture(regional_db, "eu")
    regional_db.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = %s', ("remote-request",))
    import_batch(regional_db, (event(0.13, "2026-11-23T12:00:00"),), "eu")

    assert regional_db.execute(
        'SELECT request_id, source_region, spend FROM "LiteLLM_FleetBudgetSpend"'
    ).fetchall() == [("remote-request", None, 0.0)]
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []


async def test_first_import_after_partition_retention_preserves_local_budget_precedence(
    regional_db: Connection[TupleRow], fleet_prisma: PrismaClient
) -> None:
    execute_sql_file(regional_db, _PARTITION_SCRIPT)
    regional_db.execute('ALTER TABLE "LiteLLM_SpendLogs" DETACH PARTITION "LiteLLM_SpendLogs_pdefault"')
    regional_db.execute(
        'CREATE TABLE "old_partition" PARTITION OF "LiteLLM_SpendLogs" '
        "FOR VALUES FROM ('2025-01-01') TO ('2025-02-01')"
    )
    with regional_db.transaction():
        regional_db.execute('ALTER TABLE "LiteLLM_SpendLogs" DISABLE TRIGGER litellm_capture_spend_log_insert')
        regional_db.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, spend, "startTime", created_at, updated_at) '
            "VALUES ('remote-request', 0.13, '2025-01-01', '2025-01-01', '2025-01-01'),"
            " ('tail', 0.14, '2025-01-01', '2025-01-01', '2025-01-01')"
        )
        regional_db.execute('ALTER TABLE "LiteLLM_SpendLogs" ENABLE TRIGGER litellm_capture_spend_log_insert')

    with regional_db.transaction(), connect_regional_db(os.environ["DATABASE_URL"]) as writer:
        assert regional_db.execute("SELECT litellm_drop_spend_log_partition('old_partition', 1)").fetchone() == (False,)
        assert regional_db.execute(
            'SELECT request_id FROM "LiteLLM_FleetBudgetSpend" ORDER BY request_id'
        ).fetchall() == [("remote-request",)]
        schema: Final = regional_db.execute("SELECT current_schema()").fetchone()
        assert schema is not None and isinstance(schema[0], str)
        writer.execute("SELECT set_config('search_path', quote_ident(%s), false)", (schema[0],))
        writer.execute("SET lock_timeout = '1s'")
        writer.execute("SET litellm.spend_log_replica = 'on'")
        writer.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, spend, "startTime", created_at, updated_at) '
            "VALUES ('late-backfill', 0.15, '2025-01-01', '2025-01-01', '2025-01-01')"
        )

    dropped: Final = await SpendLogsPartitionManager().drop_partitions_older_than(
        fleet_prisma, datetime(2025, 3, 1, tzinfo=timezone.utc), lambda: 5000
    )
    assert dropped == ["old_partition"]
    assert regional_db.execute("SELECT to_regclass('old_partition')").fetchone() == (None,)
    import_batch(
        regional_db,
        (event(0.13, "2025-01-01T00:00:00", start_time="2025-01-01T00:00:00"),),
        "eu",
    )
    assert regional_db.execute(
        'SELECT request_id, source_region, spend FROM "LiteLLM_FleetBudgetSpend" ORDER BY request_id'
    ).fetchall() == [("late-backfill", None, 0.0), ("remote-request", None, 0.0), ("tail", None, 0.0)]
    assert regional_db.execute('SELECT partition_oid FROM "LiteLLM_SpendLogPartitionDrop"').fetchall() == []


class _GroupedLog(TypedDict):
    source_region: str | None
    spend: float


async def test_grouped_reporting_pages_preserve_region_and_timestamp_identity(
    regional_db: Connection[TupleRow], monkeypatch: pytest.MonkeyPatch
) -> None:
    import_batch(
        regional_db,
        (
            event(1.0, "2026-11-23T12:00:00", region="us"),
            event(2.0, "2026-11-23T12:00:00", region="apac"),
            event(3.0, "2026-11-23T12:00:00", region="us", start_time="2026-11-23T11:00:00"),
        ),
        "eu",
    )
    current: Final = regional_db.execute("SELECT current_schema()").fetchone()
    assert current is not None and isinstance(current[0], str)
    parsed: Final = urlsplit(os.environ["DATABASE_URL"])
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": current[0]}))
    )
    monkeypatch.setenv("DATABASE_URL", scoped_url)
    prisma: Final = PrismaClient(scoped_url, ProxyLogging(user_api_key_cache=UserApiKeyCache()))
    await prisma.connect()
    try:

        async def page(cursor: str | None) -> Mapping[str, object]:
            return await _ui_session_grouped_spend_logs(
                prisma, (), (), 1, "success", 1, 1, True, cursor, relation='"LiteLLM_SpendLogReporting"'
            )

        first: Final = await page(None)
        assert first["total"] == 3
        first_cursor: Final = first["next_session_cursor"]
        assert isinstance(first_cursor, str)
        second: Final = await page(first_cursor)
        second_cursor: Final = second["next_session_cursor"]
        assert isinstance(second_cursor, str)
        third: Final = await page(second_cursor)
        assert third["next_session_cursor"] is None
        logs: Final = TypeAdapter(list[_GroupedLog])
        assert [logs.validate_python(result["data"]) for result in (first, second, third)] == [
            [{"source_region": "us", "spend": 1.0}],
            [{"source_region": "apac", "spend": 2.0}],
            [{"source_region": "us", "spend": 3.0}],
        ]
    finally:
        await prisma.disconnect()


async def test_admin_log_filters_match_local_and_encoded_imported_ids(
    regional_db: Connection[TupleRow], fleet_prisma: PrismaClient
) -> None:
    entity_id: Final = "sha256:customer"
    api_key: Final = "a" * 64
    record_spend(regional_db, "local-request", 1.0)
    regional_db.execute(
        """UPDATE "LiteLLM_SpendLogs" SET "user" = %s, team_id = %s, end_user = %s, api_key = %s""",
        (entity_id, entity_id, entity_id, api_key),
    )
    remote: Final = event(2.0, "2026-11-23T12:00:00")
    remote_payload: Final = SpendLogEvent.model_validate_json(remote)
    encoded: Final = replicated_budget_entity_id(entity_id)
    encoded_payload: Final = remote_payload.model_copy(
        update={"row": remote_payload.row.model_copy(update={"user": encoded, "team_id": encoded, "end_user": encoded})}
    ).model_dump_json()
    import_batch(regional_db, (encoded_payload,), "eu")

    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for field in ("team_id", "user_id", "end_user", "search"):
            response = await client.get(
                "/spend/logs/ui",
                params={
                    field: entity_id,
                    "api_key": api_key,
                    "start_date": "2026-11-23 00:00:00",
                    "end_date": "2026-11-24 00:00:00",
                },
            )
            assert response.status_code == 200, response.text
            assert {row["request_id"] for row in response.json()["data"]} == {"local-request", "remote-request"}, field


def test_worker_connects_to_the_prisma_schema(regional_db: Connection[TupleRow]) -> None:
    current: Final = regional_db.execute("SELECT current_schema()").fetchone()
    assert current is not None and isinstance(current[0], str)
    parsed: Final = urlsplit(os.environ["DATABASE_URL"])
    query: Final = {
        **dict(parse_qsl(parsed.query)),
        "schema": current[0],
        "connection_limit": "4",
        "pool_timeout": "15",
        "options": "-c statement_timeout=3000",
    }
    database_url: Final = urlunsplit(parsed._replace(query=urlencode(query, quote_via=quote)))
    with connect_regional_db(database_url) as worker_db:
        enable_capture(worker_db, "eu")
        assert worker_db.execute("SHOW statement_timeout").fetchone() == ("3s",)
    assert regional_db.execute('SELECT region FROM "LiteLLM_SpendLogReplication"').fetchall() == [("eu",)]


async def test_secure_redis_url_requires_tls_certificate_and_hostname_verification() -> None:
    async with replication_redis("rediss://redis.example:6380") as redis:
        options: Final = redis.connection_pool.connection_kwargs
        assert options["ssl_cert_reqs"] == "required"
        assert options["ssl_check_hostname"] is True


@pytest.mark.parametrize(
    "options",
    (
        "ssl_cert_reqs=none",
        "ssl_check_hostname=false",
        "ssl_cert_reqs=none&ssl_cert_reqs=required",
        "ssl_check_hostname=false&ssl_check_hostname=true",
    ),
)
def test_worker_rejects_unsafe_redis_tls_settings(options: str) -> None:
    with pytest.raises(ValueError, match="requires certificate and hostname verification"):
        replication_redis(f"rediss://redis.example:6380?{options}")


def test_replay_survives_a_partition_swap_on_the_same_connection(regional_db: Connection[TupleRow]) -> None:
    record_spend(regional_db, "local-before-setup", 0.12)
    enable_capture(regional_db, "eu")
    original: Final = event(0.13, "2026-11-23T12:00:00")
    updated: Final = event(0.27, "2026-11-23T12:00:01")
    import_batch(regional_db, (original,), "eu")
    assert regional_db.execute('SELECT spend FROM "LiteLLM_ReplicatedSpendLogs"').fetchall() == [(0.13,)]

    execute_sql_file(regional_db, _PARTITION_SCRIPT)
    with regional_db.transaction():
        regional_db.execute("SET LOCAL litellm.spend_log_replica = 'on'")
        regional_db.execute('INSERT INTO "LiteLLM_SpendLogs" SELECT * FROM "LiteLLM_SpendLogs_legacy"')
    import_batch(regional_db, (original, updated, original), "eu")
    execute_sql_file(regional_db, _UNPARTITION_SCRIPT)
    import_batch(regional_db, (original,), "eu")

    assert regional_db.execute(
        'SELECT request_id, spend FROM "LiteLLM_SpendLogReporting" ORDER BY request_id'
    ).fetchall() == [("local-before-setup", 0.12), ("remote-request", 0.27)]
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []
    record_spend(regional_db, "local-after-swap", 0.04)
    assert regional_db.execute('SELECT "row"->>\'request_id\' FROM "LiteLLM_SpendLogOutbox"').fetchall() == [
        ("local-after-swap",)
    ]


def test_pause_survives_worker_restart_and_allows_remote_rows(regional_db: Connection[TupleRow]) -> None:
    enable_capture(regional_db, "eu")
    record_spend(regional_db, "before-pause", 0.05)
    regional_db.execute('UPDATE "LiteLLM_SpendLogReplication" SET enabled = false WHERE id = 1')
    enable_capture(regional_db, "eu")
    record_spend(regional_db, "during-pause", 0.13)

    import_batch(regional_db, (event(0.13, "2026-11-23T12:00:00"),), "eu")
    regional_db.execute('UPDATE "LiteLLM_SpendLogReplication" SET enabled = true WHERE id = 1')
    record_spend(regional_db, "after-resuming", 0.27)

    assert regional_db.execute('SELECT request_id FROM "LiteLLM_SpendLogReporting" ORDER BY request_id').fetchall() == [
        ("after-resuming",),
        ("before-pause",),
        ("during-pause",),
        ("remote-request",),
    ]
    assert regional_db.execute(
        'SELECT "row"->>\'request_id\' FROM "LiteLLM_SpendLogOutbox" ORDER BY id'
    ).fetchall() == [
        ("before-pause",),
        ("after-resuming",),
    ]


async def test_unreachable_redis_leaves_committed_spend_and_retries_outbox(
    regional_db: Connection[TupleRow], stream_redis: Redis, stream_server: OwnedRedis
) -> None:
    record_spend(regional_db, "before-enabling", 0.05)
    enable_capture(regional_db, "us")
    record_spend(regional_db, "after-enabling", 0.13)
    stream_server.stop()
    with pytest.raises(RedisConnectionError):
        await publish_batch(regional_db, stream_redis, "us")

    assert regional_db.execute('SELECT request_id, spend FROM "LiteLLM_SpendLogs" ORDER BY request_id').fetchall() == [
        ("after-enabling", 0.13),
        ("before-enabling", 0.05),
    ]
    assert regional_db.execute('SELECT "row"->>\'request_id\' FROM "LiteLLM_SpendLogOutbox"').fetchall() == [
        ("after-enabling",)
    ]

    stream_server.start()
    assert await publish_batch(regional_db, stream_redis, "us")
    assert tuple(_JSON_VALUE.validate_json(payload) for payload in await stream_payloads(stream_redis)) == (
        expected_payload("after-enabling"),
    )
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []


async def test_oversized_event_is_removed_from_outbox_without_blocking_other_rows(
    regional_db: Connection[TupleRow], stream_redis: Redis
) -> None:
    enable_capture(regional_db, "us")
    record_spend(regional_db, "oversized", 0.13)
    regional_db.execute(
        """UPDATE "LiteLLM_SpendLogOutbox" SET "row" = "row" ||
           jsonb_build_object('request_id', repeat('a', 256 * 1024))"""
    )
    record_spend(regional_db, "normal", 0.27)
    assert await publish_batch(regional_db, stream_redis, "us")
    assert not await publish_batch(regional_db, stream_redis, "us")
    records: Final = tuple(
        SpendLogEvent.model_validate_json(payload) for payload in await stream_payloads(stream_redis)
    )
    assert tuple(record.row.request_id for record in records) == ("normal",)
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []
    assert regional_db.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (2,)


async def test_serialized_stream_event_size_is_bounded(regional_db: Connection[TupleRow], stream_redis: Redis) -> None:
    enable_capture(regional_db, "us")
    record_spend(regional_db, "oversized", 0.13)
    regional_db.execute("UPDATE \"LiteLLM_SpendLogs\" SET model = repeat('é', 150000)")

    assert await publish_batch(regional_db, stream_redis, "us")
    recorded: Final = tuple(
        SpendLogEvent.model_validate_json(payload).row for payload in await stream_payloads(stream_redis)
    )
    assert [(row.model, row.spend) for row in recorded] == [("", 0.13), ("", 0.13)]
    assert regional_db.execute('SELECT id FROM "LiteLLM_SpendLogOutbox"').fetchall() == []


async def test_large_end_user_identifier_preserves_all_fleet_budget_charges(
    regional_db: Connection[TupleRow], fleet_prisma: PrismaClient, stream_redis: Redis
) -> None:
    end_user: Final = "large-end-user-" + "é" * 150_000
    api_key: Final = "a" * 64
    regional_db.execute(
        """INSERT INTO "LiteLLM_SpendLogOutbox" ("row") VALUES (
            %s::jsonb || jsonb_build_object('end_user', %s::text, 'api_key', %s::text,
                'team_id', 'fleet-team', 'user', 'fleet-user', 'spend', 4)
        )""",
        (json.dumps(expected_payload("large-id")["row"]), end_user, api_key),
    )
    assert await publish_batch(regional_db, stream_redis, "us")
    (payload,) = await stream_payloads(stream_redis)
    await ensure_group(stream_redis, "eu")
    await stream_redis.xadd(STREAM_NAME, {"event": payload})
    await consume_batch(regional_db, stream_redis, await read_batch(stream_redis, "eu", "worker"), "eu")
    assert await proxy_server.get_current_spend(f"spend:key:{api_key}", 7.0, max_budget=10.0) == 11.0
    assert await proxy_server.get_current_spend("spend:team:fleet-team", 7.0, max_budget=10.0) == 11.0
    assert await proxy_server.get_current_spend("spend:user:fleet-user", 7.0, max_budget=10.0) == 11.0
    assert await read_replica_budget_spend(fleet_prisma, f"spend:end_user:{end_user}") == 4.0


async def test_retryable_rows_remain_queued_when_an_oversized_event_is_removed(
    regional_db: Connection[TupleRow], stream_redis: Redis, stream_server: OwnedRedis
) -> None:
    enable_capture(regional_db, "us")
    record_spend(regional_db, "oversized", 0.13)
    regional_db.execute(
        """UPDATE "LiteLLM_SpendLogOutbox" SET "row" = "row" ||
           jsonb_build_object('request_id', repeat('a', 256 * 1024))"""
    )
    record_spend(regional_db, "retryable", 0.27)
    stream_server.stop()
    with pytest.raises(RedisConnectionError):
        await publish_batch(regional_db, stream_redis, "us")

    assert regional_db.execute('SELECT "row"->>\'request_id\' FROM "LiteLLM_SpendLogOutbox"').fetchall() == [
        ("retryable",)
    ]
    assert regional_db.execute('SELECT request_id, spend FROM "LiteLLM_SpendLogs" ORDER BY request_id').fetchall() == [
        ("oversized", 0.13),
        ("retryable", 0.27),
    ]


def expected_payload(request_id: str) -> Mapping[str, JsonValue]:
    return {
        "region": "us",
        "row": {
            "request_id": request_id,
            "call_type": "acompletion",
            "api_key": "",
            "spend": 0.13,
            "total_tokens": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "startTime": "2026-11-23T12:00:00",
            "endTime": "2026-11-23T12:00:01",
            "request_duration_ms": None,
            "completionStartTime": None,
            "model": "",
            "model_id": None,
            "model_group": None,
            "custom_llm_provider": None,
            "user": None,
            "end_user": None,
            "team_id": None,
            "organization_id": None,
            "status": None,
            "created_at": "2026-11-23T12:00:00",
            "updated_at": "2026-11-23T12:00:00",
        },
    }


async def test_stream_payload_excludes_all_request_and_response_content(
    regional_db: Connection[TupleRow], stream_redis: Redis
) -> None:
    enable_capture(regional_db, "us")
    record_spend(regional_db, "content-probe", 0.13)
    regional_db.execute(
        """UPDATE "LiteLLM_SpendLogs"
           SET messages = '{"content":"private prompt"}', response = '{"output":"private output"}',
               proxy_server_request = '{"body":{"input":"private input"},"headers":{"private":"header"}}',
               metadata = '{"arbitrary":"private metadata"}', request_tags = '["private tag"]',
               api_key = 'private credential', litellm_call_id = 'client correlation ID'
           WHERE request_id = 'content-probe'"""
    )
    assert regional_db.execute(
        'SELECT litellm_call_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', ("content-probe",)
    ).fetchall() == [("client correlation ID",)]
    expected: Final = expected_payload("content-probe")
    assert regional_db.execute('SELECT "row" FROM "LiteLLM_SpendLogOutbox" ORDER BY id').fetchall() == [
        (expected["row"],),
        (expected["row"],),
    ]

    regional_db.execute(
        """UPDATE "LiteLLM_SpendLogOutbox" SET "row" = "row" ||
           '{"messages":{"content":"stale outbox prompt"},"litellm_call_id":"client correlation ID"}'"""
    )
    assert await publish_batch(regional_db, stream_redis, "us")
    assert tuple(_JSON_VALUE.validate_json(payload) for payload in await stream_payloads(stream_redis)) == (
        expected,
        expected,
    )


async def test_header_derived_request_id_stays_local(regional_db: Connection[TupleRow], stream_redis: Redis) -> None:
    enable_capture(regional_db, "us")
    regional_db.execute(
        """INSERT INTO "LiteLLM_SpendLogs" (request_id, litellm_call_id, spend, created_at, updated_at)
           VALUES ('caller header_video_cost', 'caller header', 0.13, '2026-11-23', '2026-11-23')"""
    )
    regional_db.execute('UPDATE "LiteLLM_SpendLogs" SET spend = 0.23')
    assert await publish_batch(regional_db, stream_redis, "us")
    records: Final = await stream_payloads(stream_redis)
    assert len(records) == 2
    digest: Final = hmac.digest(bytes.fromhex(_REQUEST_ID_KEY), b"caller header_video_cost", "sha256")
    replica_id: Final = f"replica:{digest.hex()}"
    published: Final = tuple(SpendLogEvent.model_validate_json(record).row for record in records)
    assert tuple((row.request_id, row.spend) for row in published) == ((replica_id, 0.13), (replica_id, 0.23))
    assert all("caller header" not in record for record in records)

    replicas: Final = tuple(SpendLogEvent(region=region, row=published[1]).model_dump_json() for region in ("eu", "ap"))
    import_batch(regional_db, replicas, "us")
    assert regional_db.execute(
        'SELECT request_id, source_region, spend FROM "LiteLLM_FleetBudgetSpend"'
    ).fetchall() == [(replica_id, None, 0.0)]
    assert regional_db.execute('SELECT request_id, source_region FROM "LiteLLM_SpendLogReporting"').fetchall() == [
        ("caller header_video_cost", None)
    ]
    assert regional_db.execute('SELECT request_id, litellm_call_id, spend FROM "LiteLLM_SpendLogs"').fetchall() == [
        ("caller header_video_cost", "caller header", 0.23)
    ]

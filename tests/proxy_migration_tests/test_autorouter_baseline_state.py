"""Idempotent journal migration and primary transactional ownership."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.baseline_accounting import BaselineAccountingStore
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.utils import PrismaClient, ProxyLogging

_MIGRATION: Final = Path(__file__).parents[2] / (
    "litellm-proxy-extras/litellm_proxy_extras/migrations/20260915010000_add_autorouter_baseline_state/migration.sql"
)


@pytest.fixture
def database() -> Iterator[tuple[str, psycopg.Connection[tuple[object, ...]]]]:
    base: Final = os.environ["DATABASE_URL"].split("?")[0]
    schema: Final = f"baseline_{uuid4().hex}"
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        try:
            connection.execute(_MIGRATION.read_bytes())
            connection.execute(_MIGRATION.read_bytes())
            yield f"{base}?schema={schema}", connection
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@asynccontextmanager
async def _client(env: pytest.MonkeyPatch, url: str, replica: str | None = None) -> AsyncGenerator[PrismaClient]:
    with env.context() as context:
        context.setenv("DATABASE_URL", url)
        context.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        if replica is not None:
            context.setenv("DATABASE_URL_READ_REPLICA", replica)
        client: Final = PrismaClient(url, ProxyLogging(UserApiKeyCache()))
        try:
            await client.db.connect(timeout=timedelta(seconds=1))
            yield client
        finally:
            await client.db.disconnect()


@pytest.mark.asyncio
async def test_migration_and_projector_use_the_primary_across_clients(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, connection = database
    connection.execute('CREATE TABLE "LiteLLM_SpendLogs" (request_id TEXT PRIMARY KEY)')
    connection.execute('INSERT INTO "LiteLLM_AutoRouterBaselineComparison" '
        '(scope,api_key,session_id,router_name,initial_equivalent,revision) '
        "VALUES ('test','key','session','router',TRUE,1)")
    async with _client(monkeypatch, url, url.split("?")[0]) as first:
        assert await BaselineAccountingStore.for_client(first).project("test") == "published"
    async with _client(monkeypatch, url) as restarted:
        assert await BaselineAccountingStore.for_client(restarted).project("test") == "unchanged"
    assert connection.execute('SELECT revision=published_revision FROM "LiteLLM_AutoRouterBaselineComparison"').fetchone() == (True,)


@pytest.mark.asyncio
async def test_primary_outage_and_missing_table_are_unavailable(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, connection = database
    async with _client(monkeypatch, "postgresql://unused:unused@127.0.0.1:1/unreachable", url) as degraded:
        assert isinstance(degraded.db, RoutingPrismaWrapper) and degraded.db.writer_unavailable
        assert await BaselineAccountingStore.for_client(degraded).project("scope") == "unavailable"
    connection.execute('DROP TABLE "LiteLLM_AutoRouterBaselineComparison"')
    async with _client(monkeypatch, url) as missing:
        assert await BaselineAccountingStore.for_client(missing).project("scope") == "unavailable"


@pytest.mark.asyncio
async def test_locked_projection_is_bounded_and_cancellation_propagates(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, connection = database
    async with _client(monkeypatch, url) as client:
        store: Final = BaselineAccountingStore.for_client(client)
        with connection.transaction():
            connection.execute('LOCK TABLE "LiteLLM_AutoRouterBaselineComparison" IN ACCESS EXCLUSIVE MODE')
            started: Final = time.monotonic()
            assert await store.project("scope") == "unavailable"
            assert time.monotonic() - started < 2
            pending: Final = asyncio.create_task(store.project("scope"))
            await asyncio.sleep(0.01)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert await store.project("scope") == "unchanged"


_COVERAGE_MIGRATION: Final = Path(__file__).parents[2] / (
    "litellm-proxy-extras/litellm_proxy_extras/migrations/20260923000000_add_daily_autorouter_costs/migration.sql"
)


@pytest.mark.parametrize("repair", (False, True))
def test_daily_coverage_install_and_old_publisher_preserve_history(
    database: tuple[str, psycopg.Connection[tuple[object, ...]]], repair: bool,
) -> None:
    _, connection = database
    connection.execute("""CREATE TABLE "LiteLLM_DailyUserSpend" (
        id TEXT PRIMARY KEY, user_id TEXT, date TEXT, api_key TEXT, model TEXT, custom_llm_provider TEXT,
        mcp_namespaced_tool_name TEXT, endpoint TEXT, model_group TEXT, updated_at TIMESTAMP,
        spend FLOAT8 DEFAULT 0, autorouter_savings_spend FLOAT8 DEFAULT 0,
        UNIQUE (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint));
        INSERT INTO "LiteLLM_DailyUserSpend"
        (id,user_id,date,api_key,model,custom_llm_provider,mcp_namespaced_tool_name,endpoint,spend,autorouter_savings_spend)
        VALUES ('recorded','owner','2026-09-23','key','','','','',10,30)""")
    connection.execute(
        'INSERT INTO "LiteLLM_AutoRouterBaselineObservation" (request_id,scope,started_at,revision,data) '
        "VALUES ('request','scope',0,1,%s)",
        (json.dumps({"daily": {"date": "2026-09-23", "api_key": "key",
                              "targets": [{"entity": "user", "entity_id": "owner"}]}}),),
    )
    connection.execute(_COVERAGE_MIGRATION.read_text())
    if repair:
        connection.execute('ALTER TABLE "LiteLLM_AutoRouterBaselineObservation" '
                           'DISABLE TRIGGER litellm_update_daily_autorouter_coverage')
    connection.execute(_COVERAGE_MIGRATION.read_text())
    for status, expected_count, expected_actual in (
        ("estimated", 1, 2), ("estimated", 1, 2), ("unknown", 0, 0), ("estimated", 1, 2),
    ):
        connection.execute(
            'UPDATE "LiteLLM_AutoRouterBaselineObservation" SET publication=%s WHERE request_id=%s',
            (json.dumps({"status": status, "actual_spend": 2, "baseline_spend": 2}), "request"),
        )
        assert connection.execute(
            "SELECT id,spend,autorouter_savings_spend,autorouter_estimated_requests,autorouter_estimated_actual_spend "
            'FROM "LiteLLM_DailyUserSpend"'
        ).fetchall() == [("recorded", 10, 30, expected_count, expected_actual)]
    connection.execute('DELETE FROM "LiteLLM_AutoRouterBaselineObservation"')
    assert connection.execute(
        'SELECT autorouter_estimated_requests,autorouter_estimated_actual_spend FROM "LiteLLM_DailyUserSpend"'
    ).fetchall() == [(1, 2)]

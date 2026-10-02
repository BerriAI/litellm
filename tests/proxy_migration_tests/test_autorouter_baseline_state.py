"""Idempotent journal migration and primary transactional ownership."""

import asyncio
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

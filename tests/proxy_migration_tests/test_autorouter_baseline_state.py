"""Idempotent journal migration and primary transactional ownership."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import ThreadPoolExecutor
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


@pytest.fixture
def _coverage_postgresql() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    base: Final = os.environ["DATABASE_URL"].split("?")[0]
    name: Final = f"coverage_{uuid4().hex}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            with psycopg.connect(base, dbname=name) as connection:
                yield connection
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


_COVERAGE_BASE: Final = """
CREATE TABLE "LiteLLM_DailyUserSpend" (
    id TEXT PRIMARY KEY, user_id TEXT, date TEXT NOT NULL, api_key TEXT NOT NULL,
    model TEXT, custom_llm_provider TEXT, mcp_namespaced_tool_name TEXT, endpoint TEXT,
    model_group TEXT, spend DOUBLE PRECISION DEFAULT 0, autorouter_savings_spend DOUBLE PRECISION DEFAULT 0,
    updated_at TIMESTAMP,
    UNIQUE (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
);
CREATE TABLE "LiteLLM_AutoRouterBaselineObservation" (
    request_id TEXT PRIMARY KEY, data TEXT, publication TEXT
);
INSERT INTO "LiteLLM_DailyUserSpend"
    (id,user_id,date,api_key,model,custom_llm_provider,mcp_namespaced_tool_name,endpoint,spend,autorouter_savings_spend)
    VALUES ('recorded','owner','2026-09-23','key','','','','',10,30)
"""


def _coverage_tables(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    conn.execute(_COVERAGE_BASE)
    conn.execute(
        'INSERT INTO "LiteLLM_AutoRouterBaselineObservation" (request_id,data) VALUES (%s,%s)',
        (
            "request",
            json.dumps(
                {
                    "daily": {
                        "date": "2026-09-23",
                        "api_key": "key",
                        "targets": [{"entity": "user", "entity_id": "owner"}],
                    }
                }
            ),
        ),
    )
    conn.commit()


def _coverage_install(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    conn.execute(_COVERAGE_MIGRATION.read_text())


def _coverage_publish(conn: psycopg.Connection[tuple[object, ...]]) -> tuple[object, ...] | None:
    conn.execute(
        'UPDATE "LiteLLM_AutoRouterBaselineObservation" SET publication=%s WHERE request_id=%s',
        (json.dumps({"status": "estimated", "actual_spend": 2, "baseline_spend": 2}), "request"),
    )
    row: Final = conn.execute(
        "SELECT id,spend,autorouter_savings_spend,autorouter_estimated_requests,autorouter_estimated_actual_spend "
        'FROM "LiteLLM_DailyUserSpend"'
    ).fetchone()
    conn.commit()
    return row


def _coverage_repeat_during_traffic(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    with (
        psycopg.connect(conn.info.dsn, password=conn.info.password) as traffic,
        psycopg.connect(conn.info.dsn, password=conn.info.password, autocommit=True) as installer,
    ):
        traffic.execute('SELECT * FROM "LiteLLM_DailyUserSpend"')
        traffic.execute('UPDATE "LiteLLM_AutoRouterBaselineObservation" SET data=data')
        with ThreadPoolExecutor(max_workers=1) as pool:
            repeated: Final = pool.submit(_coverage_install, installer)
            try:
                repeated.result(timeout=2)
            finally:
                installer.cancel()
                traffic.rollback()


def test_coverage_install_preserves_history_and_repeat_does_not_wait_for_table_traffic(
    _coverage_postgresql: psycopg.Connection[tuple[object, ...]],
) -> None:
    _coverage_tables(_coverage_postgresql)
    _coverage_install(_coverage_postgresql)
    _coverage_repeat_during_traffic(_coverage_postgresql)

    assert _coverage_publish(_coverage_postgresql) == ("recorded", 10, 30, 1, 2)
    assert _coverage_publish(_coverage_postgresql) == ("recorded", 10, 30, 1, 2)


@pytest.mark.parametrize("damage", ("missing", "disabled", "wrong-function", "wrong-event", "wrong-body"))
def test_coverage_installer_repairs_trigger_before_publishing_without_rewriting_history(
    _coverage_postgresql: psycopg.Connection[tuple[object, ...]],
    damage: str,
) -> None:
    _coverage_tables(_coverage_postgresql)
    _coverage_install(_coverage_postgresql)
    if damage == "disabled":
        _coverage_postgresql.execute(
            'ALTER TABLE "LiteLLM_AutoRouterBaselineObservation" '
            "DISABLE TRIGGER litellm_update_daily_autorouter_coverage"
        )
    elif damage == "wrong-body":
        _coverage_postgresql.execute(
            "CREATE OR REPLACE FUNCTION litellm_update_daily_autorouter_coverage() "
            "RETURNS TRIGGER LANGUAGE plpgsql AS 'BEGIN RETURN NEW; END;'"
        )
    else:
        _coverage_postgresql.execute(
            'DROP TRIGGER litellm_update_daily_autorouter_coverage ON "LiteLLM_AutoRouterBaselineObservation"'
        )
        if damage == "wrong-function":
            _coverage_postgresql.execute(
                "CREATE FUNCTION coverage_noop() RETURNS TRIGGER LANGUAGE plpgsql AS 'BEGIN RETURN NEW; END;'"
            )
            _coverage_postgresql.execute(
                "CREATE TRIGGER litellm_update_daily_autorouter_coverage BEFORE UPDATE OF publication "
                'ON "LiteLLM_AutoRouterBaselineObservation" FOR EACH ROW EXECUTE FUNCTION coverage_noop()'
            )
        elif damage == "wrong-event":
            _coverage_postgresql.execute(
                "CREATE TRIGGER litellm_update_daily_autorouter_coverage BEFORE UPDATE OF data "
                'ON "LiteLLM_AutoRouterBaselineObservation" FOR EACH ROW '
                "EXECUTE FUNCTION litellm_update_daily_autorouter_coverage()"
            )
    _coverage_postgresql.commit()

    _coverage_install(_coverage_postgresql)

    assert _coverage_publish(_coverage_postgresql) == ("recorded", 10, 30, 1, 2)


@pytest.mark.parametrize("version", (None, 2))
def test_coverage_installer_retains_compatible_unversioned_and_newer_functions(
    _coverage_postgresql: psycopg.Connection[tuple[object, ...]],
    version: int | None,
) -> None:
    _coverage_tables(_coverage_postgresql)
    _coverage_install(_coverage_postgresql)
    before: Final = _coverage_postgresql.execute(
        "SELECT prosrc FROM pg_proc WHERE oid='litellm_update_daily_autorouter_coverage()'::regprocedure"
    ).fetchone()
    assert before is not None and isinstance(before[0], str)
    body: Final = before[0] if version is None else "\n" + before[0]
    _coverage_postgresql.execute(
        sql.SQL(
            "CREATE OR REPLACE FUNCTION litellm_update_daily_autorouter_coverage() "
            "RETURNS TRIGGER LANGUAGE plpgsql AS {}"
        ).format(sql.Literal(body))
    )
    _coverage_postgresql.execute(
        sql.SQL("COMMENT ON FUNCTION litellm_update_daily_autorouter_coverage() IS {}").format(
            sql.Literal(None if version is None else f"litellm:autorouter_daily_coverage:{version}")
        )
    )
    _coverage_postgresql.commit()

    _coverage_repeat_during_traffic(_coverage_postgresql)

    after: Final = _coverage_postgresql.execute(
        "SELECT prosrc,obj_description(oid,'pg_proc') FROM pg_proc "
        "WHERE oid='litellm_update_daily_autorouter_coverage()'::regprocedure"
    ).fetchone()
    assert after == (body, None if version is None else f"litellm:autorouter_daily_coverage:{version}")
    assert _coverage_publish(_coverage_postgresql) == ("recorded", 10, 30, 1, 2)


@pytest.mark.parametrize(
    "incompatible,reason",
    (
        (
            'ALTER TABLE "LiteLLM_DailyUserSpend" ALTER COLUMN autorouter_estimated_requests TYPE INTEGER',
            "incompatible column definitions",
        ),
        (
            'ALTER TABLE "LiteLLM_DailyUserSpend" ALTER COLUMN autorouter_estimated_requests SET DEFAULT 99',
            "incompatible column definitions",
        ),
        (
            "COMMENT ON FUNCTION litellm_update_daily_autorouter_coverage() IS 'litellm:autorouter_daily_coverage:2';"
            "ALTER FUNCTION litellm_update_daily_autorouter_coverage() IMMUTABLE",
            "Newer daily auto-router coverage is incompatible",
        ),
        (
            "COMMENT ON FUNCTION litellm_update_daily_autorouter_coverage() IS 'litellm:autorouter_daily_coverage:2';"
            'ALTER TABLE "LiteLLM_DailyUserSpend" DROP COLUMN autorouter_estimated_actual_spend',
            "Newer daily auto-router coverage is incompatible",
        ),
        pytest.param(
            "COMMENT ON FUNCTION litellm_update_daily_autorouter_coverage() IS 'litellm:autorouter_daily_coverage:2';"
            'DROP TRIGGER litellm_update_daily_autorouter_coverage ON "LiteLLM_AutoRouterBaselineObservation";'
            "CREATE TRIGGER litellm_update_daily_autorouter_coverage BEFORE UPDATE OF data "
            'ON "LiteLLM_AutoRouterBaselineObservation" FOR EACH ROW '
            "EXECUTE FUNCTION litellm_update_daily_autorouter_coverage()",
            "Newer daily auto-router coverage is incompatible",
            id="newer-trigger-event",
        ),
        pytest.param(
            "COMMENT ON FUNCTION litellm_update_daily_autorouter_coverage() IS 'litellm:autorouter_daily_coverage:2';"
            'ALTER TABLE "LiteLLM_AutoRouterBaselineObservation" '
            "DISABLE TRIGGER litellm_update_daily_autorouter_coverage",
            "Newer daily auto-router coverage is incompatible",
            id="newer-trigger-disabled",
        ),
        pytest.param(
            "COMMENT ON FUNCTION litellm_update_daily_autorouter_coverage() IS 'litellm:autorouter_daily_coverage:2';"
            'DROP TRIGGER litellm_update_daily_autorouter_coverage ON "LiteLLM_AutoRouterBaselineObservation"',
            "Newer daily auto-router coverage is incompatible",
            id="newer-trigger-missing",
        ),
    ),
)
def test_coverage_installer_rejects_incompatible_state_without_replacing_it(
    _coverage_postgresql: psycopg.Connection[tuple[object, ...]],
    incompatible: str,
    reason: str,
) -> None:
    _coverage_tables(_coverage_postgresql)
    _coverage_install(_coverage_postgresql)
    _coverage_postgresql.execute(incompatible)
    state_query: Final = (
        "SELECT p.prosrc,p.provolatile,obj_description(p.oid,'pg_proc'),t.oid,pg_get_triggerdef(t.oid),t.tgenabled "
        "FROM pg_proc p LEFT JOIN pg_trigger t "
        "ON t.tgrelid='\"LiteLLM_AutoRouterBaselineObservation\"'::regclass "
        "AND t.tgname='litellm_update_daily_autorouter_coverage' "
        "WHERE p.oid='litellm_update_daily_autorouter_coverage()'::regprocedure"
    )
    before: Final = _coverage_postgresql.execute(state_query).fetchone()
    assert before is not None
    _coverage_postgresql.commit()

    with pytest.raises(psycopg.errors.RaiseException, match=reason):
        _coverage_install(_coverage_postgresql)

    _coverage_postgresql.rollback()
    assert _coverage_postgresql.execute(state_query).fetchone() == before
    assert _coverage_postgresql.execute(
        'SELECT id,spend,autorouter_savings_spend FROM "LiteLLM_DailyUserSpend"'
    ).fetchall() == [("recorded", 10, 30)]


def test_coverage_installer_does_not_trust_or_modify_public_schema_homonyms(
    _coverage_postgresql: psycopg.Connection[tuple[object, ...]],
) -> None:
    _coverage_tables(_coverage_postgresql)
    _coverage_install(_coverage_postgresql)
    _coverage_postgresql.execute('CREATE SCHEMA "tenant coverage"')
    _coverage_postgresql.execute('SET search_path TO "tenant coverage", public')
    _coverage_tables(_coverage_postgresql)

    _coverage_install(_coverage_postgresql)

    assert _coverage_publish(_coverage_postgresql) == ("recorded", 10, 30, 1, 2)
    assert _coverage_postgresql.execute(
        'SELECT spend,autorouter_savings_spend,autorouter_estimated_requests FROM public."LiteLLM_DailyUserSpend"'
    ).fetchall() == [(10, 30, 0)]
    assert _coverage_postgresql.execute(
        'SELECT daily_costs_publication FROM public."LiteLLM_AutoRouterBaselineObservation"'
    ).fetchall() == [(None,)]


def _coverage_waiting(conn: psycopg.Connection[tuple[object, ...]], pid: int, kind: str) -> None:
    deadline: Final = time.monotonic() + 3
    while time.monotonic() < deadline:
        if conn.execute(
            "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype=%s AND NOT granted)", (pid, kind)
        ).fetchone() == (True,):
            return
        time.sleep(0.01)
    pytest.fail(f"Installer did not reach its expected {kind} wait")


def test_waiting_coverage_installer_rechecks_after_peer_finishes_before_taking_table_locks(
    _coverage_postgresql: psycopg.Connection[tuple[object, ...]],
) -> None:
    _coverage_tables(_coverage_postgresql)
    _coverage_postgresql.autocommit = True
    info: Final = _coverage_postgresql.info
    with (
        psycopg.connect(info.dsn, password=info.password) as original_reader,
        psycopg.connect(info.dsn, password=info.password) as next_reader,
        psycopg.connect(info.dsn, password=info.password, autocommit=True) as first,
        psycopg.connect(info.dsn, password=info.password, autocommit=True) as second,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        original_reader.execute('SELECT * FROM "LiteLLM_DailyUserSpend"')
        installing: Final = pool.submit(_coverage_install, first)
        try:
            _coverage_waiting(_coverage_postgresql, first.info.backend_pid, "relation")
            queued_read: Final = pool.submit(next_reader.execute, 'SELECT * FROM "LiteLLM_DailyUserSpend"')
            _coverage_waiting(_coverage_postgresql, next_reader.info.backend_pid, "relation")
            waiting: Final = pool.submit(_coverage_install, second)
            _coverage_waiting(_coverage_postgresql, second.info.backend_pid, "advisory")
            original_reader.rollback()
            installing.result(timeout=2)
            queued_read.result(timeout=2)
            waiting.result(timeout=2)
        finally:
            first.cancel()
            second.cancel()
            next_reader.cancel()
            original_reader.rollback()
            next_reader.rollback()

    assert _coverage_publish(_coverage_postgresql) == ("recorded", 10, 30, 1, 2)

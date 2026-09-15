"""Tests for the LiteLLM_DailyGlobalSpend reconcile job (LIT-7818)."""

import pathlib
import re
from contextlib import asynccontextmanager
from datetime import date
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row
from pytest_postgresql import factories

from litellm.constants import DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM
from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    build_bulk_upsert_with_global_rollup,
    merge_by_conflict_key,
)
from litellm.proxy.spend_tracking.daily_global_spend_rollup import (
    RECONCILE_DAY_SQL,
    reconciled_through,
    run_daily_global_spend_reconcile,
    run_scheduled_daily_global_spend_reconcile,
)
from litellm.proxy.utils import evict_config_param

USER_TABLE: Final = DAILY_SPEND_TABLES["user"]
TODAY: Final = date(2026, 9, 15)


class _FakeConfigRow:
    def __init__(self, param_name: str, param_value: object) -> None:
        self.param_name = param_name
        self.param_value = param_value


class _FakeConfigTable:
    def __init__(self) -> None:
        self.rows: dict[str, object] = {}

    async def upsert(self, *, where: dict[str, str], data: dict[str, dict[str, str]]) -> _FakeConfigRow:
        self.rows[where["param_name"]] = data["update"]["param_value"]
        return _FakeConfigRow(where["param_name"], data["update"]["param_value"])


class _FakeTransaction:
    def __init__(self, prisma: "_FakePrisma") -> None:
        self._prisma = prisma

    async def execute_raw(self, sql: str, *params: str) -> int:
        if "LOCK TABLE" in sql:
            self._prisma.locks_taken += 1
            return 0
        (day,) = params
        if day in self._prisma.failing_days:
            raise RuntimeError(f"day {day} exploded")
        self._prisma.reconciled.append(day)
        return 1


class _FakeDb:
    def __init__(self, prisma: "_FakePrisma") -> None:
        self._prisma = prisma
        self.litellm_config = _FakeConfigTable()

    async def query_raw(self, sql: str, *params: str) -> list[dict[str, str]]:
        first, last = params
        return [{"date": d} for d in sorted(self._prisma.user_days) if first <= d <= last]

    @asynccontextmanager
    async def tx(self, timeout: object):
        yield _FakeTransaction(self._prisma)


class _FakePrisma:
    """Enough of PrismaClient for the reconcile: per-key dates, a config table, and a transaction."""

    def __init__(self, user_days: tuple[str, ...], failing_days: frozenset[str] = frozenset()) -> None:
        self.user_days = user_days
        self.failing_days = failing_days
        self.reconciled: list[str] = []
        self.locks_taken = 0
        self.db = _FakeDb(self)

    async def get_generic_data(self, key: str, value: str, table_name: str) -> _FakeConfigRow | None:
        stored = self.db.litellm_config.rows.get(value)
        return None if stored is None else _FakeConfigRow(value, stored)


@pytest.fixture(autouse=True)
async def _fresh_marker_cache():
    await evict_config_param(DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM)
    yield
    await evict_config_param(DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM)


@pytest.mark.asyncio
async def test_first_run_rolls_up_every_historical_day_and_today_then_marks_today():
    """Before any marker exists, every day with per-key rows is rolled up, plus today even
    with no rows yet, so reads for ranges ending today can switch to the global table."""
    prisma = _FakePrisma(user_days=("2026-09-01", "2026-09-03", "2026-09-14"))

    result = await run_daily_global_spend_reconcile(prisma, today=TODAY)

    assert result.days_reconciled == ("2026-09-01", "2026-09-03", "2026-09-14", "2026-09-15")
    assert result.failed_day is None
    assert result.reconciled_through == "2026-09-15"
    assert await reconciled_through(prisma) == "2026-09-15"
    assert prisma.locks_taken == 4


@pytest.mark.asyncio
async def test_later_run_replays_the_marker_day_and_the_day_before_only():
    """Days older than marker-1 are settled; the marker day and its predecessor are replayed so
    rows a pre-writer pod flushed around midnight during a rolling deploy get folded in."""
    prisma = _FakePrisma(user_days=("2026-09-01", "2026-09-12", "2026-09-13", "2026-09-14"))
    await run_daily_global_spend_reconcile(prisma, today=date(2026, 9, 13))
    prisma.reconciled.clear()

    result = await run_daily_global_spend_reconcile(prisma, today=TODAY)

    assert result.days_reconciled == ("2026-09-12", "2026-09-13", "2026-09-14", "2026-09-15")
    assert "2026-09-01" not in prisma.reconciled
    assert await reconciled_through(prisma) == "2026-09-15"


@pytest.mark.asyncio
async def test_a_failing_day_stops_the_run_and_leaves_the_marker_on_the_last_good_day():
    """The marker may never claim a day that was not rewritten: reads past it would then trust
    a global table missing that day's spend."""
    prisma = _FakePrisma(user_days=("2026-09-01", "2026-09-02", "2026-09-03"), failing_days=frozenset({"2026-09-02"}))

    result = await run_daily_global_spend_reconcile(prisma, today=TODAY)

    assert result.days_reconciled == ("2026-09-01",)
    assert result.failed_day == "2026-09-02"
    assert result.reconciled_through == "2026-09-01"
    assert prisma.reconciled == ["2026-09-01"]
    assert await reconciled_through(prisma) == "2026-09-01"


@pytest.mark.asyncio
async def test_the_next_run_resumes_from_the_failed_day():
    prisma = _FakePrisma(user_days=("2026-09-01", "2026-09-02", "2026-09-03"), failing_days=frozenset({"2026-09-02"}))
    await run_daily_global_spend_reconcile(prisma, today=TODAY)
    prisma.failing_days = frozenset()

    result = await run_daily_global_spend_reconcile(prisma, today=TODAY)

    assert result.days_reconciled == ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-15")
    assert await reconciled_through(prisma) == "2026-09-15"


@pytest.mark.asyncio
async def test_a_failure_with_nothing_done_reports_the_previous_marker_and_alerts():
    """A pre-writer pod flushing rows for the day before the marker is exactly the replay case;
    when that replay fails the marker must stay put and the operator must hear about it."""
    prisma = _FakePrisma(user_days=("2026-09-13",))
    await run_daily_global_spend_reconcile(prisma, today=date(2026, 9, 13))
    prisma.user_days = ("2026-09-12", "2026-09-13")
    prisma.failing_days = frozenset({"2026-09-12"})
    alert = AsyncMock()

    result = await run_scheduled_daily_global_spend_reconcile(prisma, pod_lock_manager=None, alert=alert, today=TODAY)

    assert result is not None
    assert result.days_reconciled == ()
    assert result.failed_day == "2026-09-12"
    assert result.reconciled_through == "2026-09-13"
    alert.assert_awaited_once()
    assert "2026-09-12" in alert.await_args.args[0]


@pytest.mark.asyncio
async def test_a_clean_run_does_not_alert():
    prisma = _FakePrisma(user_days=("2026-09-13",))
    alert = AsyncMock()

    await run_scheduled_daily_global_spend_reconcile(prisma, pod_lock_manager=None, alert=alert, today=TODAY)

    alert.assert_not_awaited()


def _pod_lock(acquired: bool) -> MagicMock:
    lock = MagicMock()
    lock.redis_cache = MagicMock()
    lock.redis_cache.async_get_cache = AsyncMock(return_value="other-pod")
    lock.get_redis_lock_key = MagicMock(return_value="lock-key")
    lock.acquire_lock = AsyncMock(return_value=acquired)
    lock.release_lock = AsyncMock()
    return lock


@pytest.mark.asyncio
async def test_scheduled_run_skips_when_another_pod_holds_the_lock():
    prisma = _FakePrisma(user_days=("2026-09-13",))
    lock = _pod_lock(acquired=False)

    result = await run_scheduled_daily_global_spend_reconcile(prisma, pod_lock_manager=lock, today=TODAY)

    assert result is None
    assert prisma.reconciled == []
    lock.release_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_run_runs_and_releases_the_lock_when_it_wins():
    prisma = _FakePrisma(user_days=("2026-09-13",))
    lock = _pod_lock(acquired=True)

    result = await run_scheduled_daily_global_spend_reconcile(prisma, pod_lock_manager=lock, today=TODAY)

    assert result is not None and result.days_reconciled == ("2026-09-13", "2026-09-15")
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_run_proceeds_when_the_lock_cannot_be_acquired_or_read():
    """A Redis outage must not stall the backfill: the day rewrite is idempotent, so running
    twice is only wasted effort while skipping forever leaves usage on the slow path."""
    prisma = _FakePrisma(user_days=("2026-09-13",))
    lock = _pod_lock(acquired=False)
    lock.redis_cache.async_get_cache = AsyncMock(side_effect=ConnectionError("redis down"))

    result = await run_scheduled_daily_global_spend_reconcile(prisma, pod_lock_manager=lock, today=TODAY)

    assert result is not None and result.days_reconciled == ("2026-09-13", "2026-09-15")
    lock.release_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_marker_is_read_back_from_the_json_string_the_config_table_stores():
    prisma = _FakePrisma(user_days=())
    prisma.db.litellm_config.rows[DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM] = '{"reconciled_through": "2026-09-10"}'

    assert await reconciled_through(prisma) == "2026-09-10"


@pytest.mark.asyncio
async def test_an_unparseable_marker_reads_as_never_reconciled():
    prisma = _FakePrisma(user_days=())
    prisma.db.litellm_config.rows[DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM] = '{"something_else": 1}'

    assert await reconciled_through(prisma) is None


_rollup_postgresql_proc: Final = factories.postgresql_proc()
_rollup_postgresql: Final = factories.postgresql("_rollup_postgresql_proc")

_MIGRATIONS_DIR: Final = (
    pathlib.Path(__file__).resolve().parents[4] / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
)
_GLOBAL_SPEND_MIGRATION: Final = _MIGRATIONS_DIR / "20260915000000_add_daily_global_spend" / "migration.sql"

_DAILY_USER_SPEND_DDL: Final = """
    CREATE TABLE "LiteLLM_DailyUserSpend" (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        date TEXT NOT NULL,
        api_key TEXT NOT NULL,
        model TEXT,
        model_group TEXT,
        custom_llm_provider TEXT,
        mcp_namespaced_tool_name TEXT,
        endpoint TEXT,
        prompt_tokens BIGINT DEFAULT 0,
        completion_tokens BIGINT DEFAULT 0,
        cache_read_input_tokens BIGINT DEFAULT 0,
        cache_creation_input_tokens BIGINT DEFAULT 0,
        compression_saved_tokens BIGINT DEFAULT 0,
        compression_savings_spend DOUBLE PRECISION DEFAULT 0,
        prompt_caching_savings_spend DOUBLE PRECISION DEFAULT 0,
        gateway_injected_caching_savings_spend DOUBLE PRECISION DEFAULT 0,
        autorouter_savings_spend DOUBLE PRECISION DEFAULT 0,
        spend DOUBLE PRECISION DEFAULT 0,
        api_requests BIGINT DEFAULT 0,
        successful_requests BIGINT DEFAULT 0,
        failed_requests BIGINT DEFAULT 0,
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP,
        UNIQUE (user_id, date, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
    )
"""

_PER_KEY_SUMS_SQL: Final = """
    SELECT COALESCE(model, '') AS model, COALESCE(model_group, '') AS model_group,
           COALESCE(custom_llm_provider, '') AS custom_llm_provider,
           SUM(spend) AS spend, SUM(prompt_tokens) AS prompt_tokens, SUM(api_requests) AS api_requests
    FROM "LiteLLM_DailyUserSpend" WHERE date = %s
    GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
"""
_GLOBAL_ROWS_SQL: Final = """
    SELECT model, model_group, custom_llm_provider, spend, prompt_tokens, api_requests
    FROM "LiteLLM_DailyGlobalSpend" WHERE date = %s ORDER BY 1, 2, 3
"""


def _execute_dollar_sql(conn: psycopg.Connection, sql: str, params: tuple[object, ...]) -> None:
    converted: Final = re.sub(r"\$(\d+)", r"%(p\1)s", sql)
    conn.execute(
        converted,  # pyright: ignore[reportArgumentType]  # psycopg stubs want a literal-typed query
        {f"p{i}": v for i, v in enumerate(params, start=1)},
    )
    conn.commit()


def _user_txn(**overrides):
    return {
        "user_id": "u-1",
        "date": "2026-09-14",
        "api_key": "sk-1",
        "model": "gpt-5",
        "model_group": "gpt-5",
        "custom_llm_provider": "openai",
        "mcp_namespaced_tool_name": "",
        "endpoint": "/chat/completions",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "spend": 1.0,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        **overrides,
    }


def _normalized(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    return [
        (
            r["model"],
            r["model_group"],
            r["custom_llm_provider"],
            float(r["spend"]),
            int(r["prompt_tokens"]),
            int(r["api_requests"]),
        )  # pyright: ignore[reportArgumentType]  # dict_row values are untyped
        for r in rows
    ]


def test_reconcile_day_sql_makes_the_global_day_equal_the_per_key_sums(_rollup_postgresql: psycopg.Connection):
    """Against real Postgres and the shipped migration: rows the writer never saw (a
    pre-writer pod's flush, NULL and '' dimension spellings) end up folded into the global
    day, running the day twice changes nothing, and other days are left alone."""
    conn: Final = _rollup_postgresql
    conn.execute(_DAILY_USER_SPEND_DDL)  # pyright: ignore[reportArgumentType]  # DDL literal
    conn.execute(_GLOBAL_SPEND_MIGRATION.read_text())  # pyright: ignore[reportArgumentType]  # DDL literal
    conn.commit()

    written_batch = merge_by_conflict_key(
        USER_TABLE,
        (_user_txn(api_key="sk-1", spend=1.0), _user_txn(api_key="sk-2", user_id="u-2", spend=2.0, prompt_tokens=20)),
    )
    _execute_dollar_sql(conn, *build_bulk_upsert_with_global_rollup(USER_TABLE, written_batch))

    conn.execute(
        """
        INSERT INTO "LiteLLM_DailyUserSpend"
            (id, user_id, date, api_key, model, model_group, custom_llm_provider, mcp_namespaced_tool_name,
             endpoint, prompt_tokens, spend, api_requests)
        VALUES
            ('legacy-1', 'u-9', '2026-09-14', 'sk-9', 'gpt-5', NULL, 'openai', NULL, NULL, 5, 4.0, 1),
            ('legacy-2', 'u-9', '2026-09-14', 'sk-9', 'gpt-5', '', 'openai', '', '', 5, 8.0, 1),
            ('legacy-3', 'u-9', '2026-09-13', 'sk-9', 'claude', '', 'anthropic', '', '', 7, 16.0, 1)
        """
    )
    conn.commit()

    _execute_dollar_sql(conn, RECONCILE_DAY_SQL, ("2026-09-14",))
    _execute_dollar_sql(conn, RECONCILE_DAY_SQL, ("2026-09-14",))

    with conn.cursor(row_factory=dict_row) as cur:
        global_rows = cur.execute(_GLOBAL_ROWS_SQL, ("2026-09-14",)).fetchall()
        per_key = cur.execute(_PER_KEY_SUMS_SQL, ("2026-09-14",)).fetchall()
        untouched = cur.execute(_GLOBAL_ROWS_SQL, ("2026-09-13",)).fetchall()

    assert _normalized(global_rows) == _normalized(per_key)
    assert sum(float(r["spend"]) for r in global_rows) == pytest.approx(15.0)  # pyright: ignore[reportArgumentType]  # dict_row values are untyped
    assert [(r["model"], r["model_group"]) for r in global_rows] == [("gpt-5", ""), ("gpt-5", "gpt-5")]
    assert untouched == []

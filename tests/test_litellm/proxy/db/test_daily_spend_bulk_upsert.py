"""Tests for the single-statement daily spend upsert (LIT-5291)."""

import pathlib
import re
from typing import Final

import psycopg
import pytest
from psycopg.rows import dict_row
from pytest_postgresql import factories

from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    GLOBAL_SPEND_TABLE,
    build_bulk_upsert,
    build_bulk_upsert_with_global_rollup,
    conflict_key,
    merge_by_conflict_key,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

TAG_TABLE = DAILY_SPEND_TABLES["tag"]
USER_TABLE = DAILY_SPEND_TABLES["user"]

# Every nullable member of the unique constraint, so a test that only varied the provider
# cannot pass while a sibling column still leaks a NULL into the conflict target.
NULLABLE_KEY_COLUMNS = ("model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint")


def tag_txn(**overrides):
    return {
        "tag": "team-a",
        "date": "2026-08-10",
        "api_key": "sk-hash",
        "model": "gpt-4o-mini",
        "model_group": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "mcp_namespaced_tool_name": "",
        "endpoint": "/chat/completions",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "spend": 0.25,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "request_id": "req-1",
        **overrides,
    }


@pytest.mark.parametrize("column", NULLABLE_KEY_COLUMNS)
def test_conflict_key_normalizes_every_nullable_key_column(column):
    """A NULL member can never match itself in a unique index, so the row would be
    re-inserted on every flush. Each nullable key column must arrive as ''."""
    key = conflict_key(TAG_TABLE, tag_txn(**{column: None}))

    assert "" in key
    assert None not in key
    assert key == conflict_key(TAG_TABLE, tag_txn(**{column: ""}))


@pytest.mark.parametrize("order", [("null_first"), ("empty_first")])
def test_null_and_empty_provider_merge_into_one_row(order):
    """Two queue entries differing only in NULL versus '' arbitrate to the same row.
    Postgres rejects one statement touching a row twice, so they must be folded first.
    Asserted under both input orders: a single ordering would prove nothing here."""
    null_entry = tag_txn(custom_llm_provider=None, spend=0.25, api_requests=1)
    empty_entry = tag_txn(custom_llm_provider="", spend=0.75, api_requests=3)
    transactions = (null_entry, empty_entry) if order == "null_first" else (empty_entry, null_entry)

    merged = merge_by_conflict_key(TAG_TABLE, transactions)

    assert len(merged) == 1
    _, folded = merged[0]
    assert folded["spend"] == pytest.approx(1.0)
    assert folded["api_requests"] == 4


def test_distinct_keys_are_not_merged_and_are_ordered_deterministically():
    unordered = (tag_txn(tag="z-team"), tag_txn(tag="a-team"), tag_txn(tag="m-team"))

    merged = merge_by_conflict_key(TAG_TABLE, unordered)

    assert [txn["tag"] for _, txn in merged] == ["a-team", "m-team", "z-team"]
    assert merged == merge_by_conflict_key(TAG_TABLE, tuple(reversed(unordered)))


def test_one_statement_carries_every_row_in_the_batch():
    batch = merge_by_conflict_key(TAG_TABLE, tuple(tag_txn(tag=f"team-{i}") for i in range(100)))

    sql, params = build_bulk_upsert(TAG_TABLE, batch)

    assert sql.count("INSERT INTO") == 1
    assert len(re.findall(r"ON CONFLICT", sql)) == 1
    # 23 bound columns per row plus the inlined updated_at, so the row count is what
    # separates one multi-row statement from a hundred single-row ones.
    assert len(params) == 100 * 23
    assert "$2300::text" in sql
    assert sql.count("(NOW() AT TIME ZONE 'UTC')") == 100 + 1


def test_conflict_target_is_the_full_unique_constraint():
    sql, _ = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(),)))

    conflict_target = re.search(r"ON CONFLICT \(([^)]*)\)", sql)
    assert conflict_target is not None
    assert conflict_target.group(1) == (
        '"tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"'
    )


@pytest.mark.parametrize(
    "column",
    ["prompt_tokens", "completion_tokens", "spend", "api_requests", "successful_requests", "failed_requests"],
)
def test_counters_increment_rather_than_overwrite(column):
    """An overwrite would silently discard every earlier flush's spend for that row."""
    sql, _ = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(),)))

    assert f'"{column}" = "LiteLLM_DailyTagSpend"."{column}" + EXCLUDED."{column}"' in sql


def test_request_id_is_preserved_when_a_later_batch_carries_none():
    sql, params = build_bulk_upsert(
        TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(request_id=None),))
    )

    assert '"request_id" = COALESCE(EXCLUDED."request_id", "LiteLLM_DailyTagSpend"."request_id")' in sql
    assert None in params


def test_non_tag_tables_carry_no_request_id_column():
    user_txn = {**tag_txn(), "user_id": "u-1"}
    del user_txn["tag"]

    sql, _ = build_bulk_upsert(USER_TABLE, merge_by_conflict_key(USER_TABLE, (user_txn,)))

    assert "request_id" not in sql
    assert '"user_id"' in sql


class _RecordingDb:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        self.statements.append((query, args))
        return len(args)


class _RecordingPrismaClient:
    def __init__(self) -> None:
        self.db = _RecordingDb()


@pytest.mark.asyncio
async def test_writer_issues_one_statement_per_batch_not_one_per_key():
    """The whole point of LIT-5291: 250 aggregated keys must not become 250 statements."""
    prisma_client = _RecordingPrismaClient()
    transactions = {f"k{i}": tag_txn(tag=f"team-{i}") for i in range(250)}

    await DBSpendUpdateWriter.update_daily_tag_spend(
        n_retry_times=0,
        prisma_client=prisma_client,
        proxy_logging_obj=None,
        daily_spend_transactions=transactions,
    )

    # 250 keys at a batch size of 100 is three statements, one per batch.
    assert len(prisma_client.db.statements) == 3
    assert [statement.count("ON CONFLICT") for statement, _ in prisma_client.db.statements] == [1, 1, 1]
    assert transactions == {}


@pytest.mark.asyncio
async def test_writer_survives_a_transaction_whose_key_columns_are_null():
    """A NULL key column used to raise out of prisma and drop the whole batch's spend."""
    prisma_client = _RecordingPrismaClient()
    transactions = {
        "mcp": tag_txn(model=None, custom_llm_provider=None, mcp_namespaced_tool_name="server/tool"),
        "chat": tag_txn(),
    }

    await DBSpendUpdateWriter.update_daily_tag_spend(
        n_retry_times=0,
        prisma_client=prisma_client,
        proxy_logging_obj=None,
        daily_spend_transactions=transactions,
    )

    assert len(prisma_client.db.statements) == 1
    _, params = prisma_client.db.statements[0]
    assert None not in params[:9]
    assert transactions == {}


def user_txn(**overrides):
    txn = {**tag_txn(), "user_id": "u-1", **overrides}
    del txn["tag"]
    del txn["request_id"]
    return txn


def _bound_rows(insert_sql: str, params: tuple[object, ...]) -> list[dict[str, object]]:
    """Each VALUES row of one INSERT as a column -> bound value mapping, consuming params in order."""
    header = re.search(r"INSERT INTO \"[A-Za-z_]+\" \(([^)]*)\)", insert_sql)
    assert header is not None, insert_sql
    columns = [c.strip('"') for c in header.group(1).split(", ") if c != '"updated_at"']
    row_count = insert_sql.split("ON CONFLICT", 1)[0].count("(NOW() AT TIME ZONE 'UTC'))")
    return [dict(zip(columns, params[i * len(columns) : (i + 1) * len(columns)])) for i in range(row_count)]


def test_global_rollup_folds_every_key_and_user_into_one_row_per_dimension_tuple():
    """The global table has no api_key or user_id, so a batch spread over many keys and
    users must collapse to one row per (date, model, group, provider, mcp, endpoint)."""
    batch = merge_by_conflict_key(
        USER_TABLE,
        tuple(user_txn(user_id=f"u-{i}", api_key=f"sk-{i}", spend=1.0, api_requests=1) for i in range(5))
        + (user_txn(user_id="u-0", api_key="sk-0", model="claude", spend=10.0, api_requests=3),),
    )

    sql, params = build_bulk_upsert_with_global_rollup(USER_TABLE, batch)

    entity_insert, global_insert = sql.split("RETURNING 1)")
    entity_rows = _bound_rows(entity_insert, params)
    global_rows = _bound_rows(global_insert, params[len(entity_rows) * len(entity_rows[0]) :])
    assert len(entity_rows) == 6
    assert 'INSERT INTO "LiteLLM_DailyGlobalSpend"' in global_insert
    assert [(r["model"], r["spend"], r["api_requests"]) for r in global_rows] == [
        ("claude", 10.0, 3),
        ("gpt-4o-mini", 5.0, 5),
    ]
    assert all("api_key" not in r and "user_id" not in r for r in global_rows)
    conflict = re.search(r"ON CONFLICT \(([^)]*)\)", global_insert)
    assert conflict is not None
    assert conflict.group(1) == ", ".join(f'"{c}"' for c in GLOBAL_SPEND_TABLE.key_columns)


def test_global_rollup_params_follow_the_entity_params_in_one_placeholder_sequence():
    """Both inserts bind from one flat tuple, so the global arm's placeholders must start
    exactly where the entity arm's stop or every value lands one column off."""
    batch = merge_by_conflict_key(USER_TABLE, (user_txn(),))

    sql, params = build_bulk_upsert_with_global_rollup(USER_TABLE, batch)

    placeholders = [int(n) for n in re.findall(r"\$(\d+)::", sql)]
    assert placeholders == list(range(1, len(params) + 1))


_bulk_upsert_postgresql_proc: Final = factories.postgresql_proc()
_bulk_upsert_postgresql: Final = factories.postgresql("_bulk_upsert_postgresql_proc")

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


def _execute_dollar_sql(conn: psycopg.Connection, sql: str, params: tuple[object, ...]) -> None:
    converted: Final = re.sub(r"\$(\d+)", r"%(p\1)s", sql)
    conn.execute(
        converted,  # pyright: ignore[reportArgumentType]  # psycopg stubs want a literal-typed query
        {f"p{i}": v for i, v in enumerate(params, start=1)},
    )
    conn.commit()


def test_global_rollup_equals_the_per_key_sums_after_repeated_flushes(_bulk_upsert_postgresql: psycopg.Connection):
    """Against real Postgres and the shipped migration: two flushes of a mixed batch leave
    the global table exactly equal to the per-key table summed over user and key, with the
    NULL and '' spellings of a dimension folded into one row."""
    conn: Final = _bulk_upsert_postgresql
    conn.execute(_DAILY_USER_SPEND_DDL)  # pyright: ignore[reportArgumentType]  # DDL literal
    conn.execute(_GLOBAL_SPEND_MIGRATION.read_text())  # pyright: ignore[reportArgumentType]  # DDL literal
    conn.commit()

    batch = merge_by_conflict_key(
        USER_TABLE,
        (
            user_txn(user_id="u-1", api_key="sk-1", spend=1.0, prompt_tokens=10),
            user_txn(user_id="u-2", api_key="sk-2", spend=2.0, prompt_tokens=20),
            user_txn(user_id="u-1", api_key="sk-3", model=None, custom_llm_provider=None, spend=4.0),
            user_txn(user_id="u-3", api_key="sk-4", model="", custom_llm_provider="", spend=8.0),
        ),
    )
    sql, params = build_bulk_upsert_with_global_rollup(USER_TABLE, batch)
    _execute_dollar_sql(conn, sql, params)
    _execute_dollar_sql(conn, sql, params)

    with conn.cursor(row_factory=dict_row) as cur:
        global_rows = cur.execute(
            'SELECT model, spend, prompt_tokens, api_requests FROM "LiteLLM_DailyGlobalSpend" ORDER BY model'
        ).fetchall()
        per_key = cur.execute(
            """
            SELECT COALESCE(model, '') AS model, SUM(spend) AS spend, SUM(prompt_tokens) AS prompt_tokens,
                   SUM(api_requests) AS api_requests
            FROM "LiteLLM_DailyUserSpend" GROUP BY COALESCE(model, '') ORDER BY 1
            """
        ).fetchall()

    assert [row["model"] for row in global_rows] == ["", "gpt-4o-mini"]
    assert [(r["model"], r["spend"], int(r["prompt_tokens"]), int(r["api_requests"])) for r in global_rows] == [
        (r["model"], float(r["spend"]), int(r["prompt_tokens"]), int(r["api_requests"])) for r in per_key
    ]
    assert global_rows[0]["spend"] == pytest.approx(24.0)
    assert global_rows[1]["spend"] == pytest.approx(6.0)

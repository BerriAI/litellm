"""Tests for the single-statement daily spend upsert (LIT-5291)."""

import re

import pytest

from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    build_bulk_upsert,
    conflict_key,
    merge_by_conflict_key,
)

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


@pytest.mark.parametrize("batch_size", [100, 250])
def test_one_statement_carries_every_row_in_the_batch(batch_size):
    batch = merge_by_conflict_key(TAG_TABLE, tuple(tag_txn(tag=f"team-{i}") for i in range(batch_size)))

    sql, params = build_bulk_upsert(TAG_TABLE, batch)

    assert sql.count("INSERT INTO") == 1
    assert len(re.findall(r"ON CONFLICT", sql)) == 1
    # 25 bound columns per row plus the inlined updated_at, so the row count is what
    # separates one multi-row statement from a hundred single-row ones.
    assert len(params) == batch_size * 25
    assert f"${batch_size * 25}::text" in sql
    assert sql.count("(NOW() AT TIME ZONE 'UTC')") == batch_size + 1


def test_conflict_target_is_the_full_unique_constraint():
    sql, _ = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(),)))

    conflict_target = re.search(r"ON CONFLICT \(([^)]*)\)", sql)
    assert conflict_target is not None
    assert conflict_target.group(1) == (
        '"tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"'
    )


@pytest.mark.parametrize(
    "column",
    [
        "prompt_tokens",
        "completion_tokens",
        "spend",
        "api_requests",
        "successful_requests",
        "failed_requests",
        "total_response_time_ms",
        "timed_requests",
    ],
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


@pytest.mark.parametrize("column,position", [("model", 4), ("custom_llm_provider", 5), ("mcp_namespaced_tool_name", 6), ("endpoint", 7)])
def test_bulk_upsert_binds_empty_strings_for_null_conflict_columns(column, position):
    statement, params = build_bulk_upsert(TAG_TABLE, merge_by_conflict_key(TAG_TABLE, (tag_txn(**{column: None}),)))

    assert params[position] == ""
    assert statement.count("ON CONFLICT") == 1

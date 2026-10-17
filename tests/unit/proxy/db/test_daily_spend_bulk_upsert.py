"""Tests for the single-statement daily spend upsert (LIT-5291)."""

import re
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest

from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    build_bulk_upsert,
    conflict_key,
    merge_by_conflict_key,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.rollup_lock_timeout import ROLLUP_LOCK_TIMEOUT_SQL

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
    # 25 bound columns per row plus the inlined updated_at, so the row count is what
    # separates one multi-row statement from a hundred single-row ones.
    assert len(params) == 100 * 25
    assert "$2500::text" in sql
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


class _RecordingDb:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.session_settings: list[tuple[str, tuple[object, ...]]] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        if query == ROLLUP_LOCK_TIMEOUT_SQL:
            self.session_settings.append((query, args))
            return 0
        self.statements.append((query, args))
        return len(args)

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator["_RecordingDb"]:
        yield self

    def tx(self, timeout: object = None) -> AbstractAsyncContextManager["_RecordingDb"]:
        return self._tx()


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

def test_unpriced_savings_survive_the_conflict_merge_as_none():
    """#46677: a row whose request saved tokens the cost map cannot price must
    not come out of the merge as a measured $0.00."""
    merged = merge_by_conflict_key(
        TAG_TABLE,
        (
            tag_txn(
                compression_savings_spend=None,
                prompt_caching_savings_spend=None,
                gateway_injected_caching_savings_spend=None,
            ),
        ),
    )
    _, folded = merged[0]
    assert folded["compression_savings_spend"] is None
    assert folded["prompt_caching_savings_spend"] is None
    assert folded["gateway_injected_caching_savings_spend"] is None


@pytest.mark.parametrize("order", ["priced_first", "unpriced_first"])
def test_unpriced_savings_contribute_no_dollars_to_a_priced_row(order):
    """An unpriced contribution must not dilute, wipe, or zero a priced figure."""
    unpriced = tag_txn(
        compression_savings_spend=None,
        prompt_caching_savings_spend=None,
        gateway_injected_caching_savings_spend=None,
    )
    priced = tag_txn(
        compression_savings_spend=0.11,
        prompt_caching_savings_spend=0.22,
        gateway_injected_caching_savings_spend=0.33,
    )
    transactions = (priced, unpriced) if order == "priced_first" else (unpriced, priced)
    merged = merge_by_conflict_key(TAG_TABLE, transactions)
    _, folded = merged[0]
    assert folded["compression_savings_spend"] == pytest.approx(0.11)
    assert folded["prompt_caching_savings_spend"] == pytest.approx(0.22)
    assert folded["gateway_injected_caching_savings_spend"] == pytest.approx(0.33)


def test_two_unpriced_rows_stay_unpriced_not_zero():
    merged = merge_by_conflict_key(
        TAG_TABLE,
        (
            tag_txn(compression_savings_spend=None, prompt_caching_savings_spend=None, gateway_injected_caching_savings_spend=None),
            tag_txn(compression_savings_spend=None, prompt_caching_savings_spend=None, gateway_injected_caching_savings_spend=None),
        ),
    )
    _, folded = merged[0]
    assert folded["compression_savings_spend"] is None
    assert folded["prompt_caching_savings_spend"] is None
    assert folded["gateway_injected_caching_savings_spend"] is None


def test_rows_without_savings_keys_still_merge_to_zero():
    """Rows that never carried the savings keys keep their old measured zero."""
    merged = merge_by_conflict_key(TAG_TABLE, (tag_txn(), tag_txn()))
    _, folded = merged[0]
    assert folded["compression_savings_spend"] == 0.0
    assert folded["prompt_caching_savings_spend"] == 0.0
    assert folded["gateway_injected_caching_savings_spend"] == 0.0


def test_unpriced_savings_bind_as_null_and_merge_null_aware():
    """NULL must bind for an unpriced row, and the increment must be NULL-aware:
    NULL + x reads x on either side, so an unpriced request neither wipes a
    priced row nor turns an all-unpriced row into a measured zero."""
    sql, params = build_bulk_upsert(
        TAG_TABLE,
        merge_by_conflict_key(
            TAG_TABLE,
            (tag_txn(compression_savings_spend=None, prompt_caching_savings_spend=None, gateway_injected_caching_savings_spend=None),),
        ),
    )
    for column in (
        "compression_savings_spend",
        "prompt_caching_savings_spend",
        "gateway_injected_caching_savings_spend",
    ):
        assert (
            f'"{column}" = COALESCE("LiteLLM_DailyTagSpend"."{column}" + EXCLUDED."{column}", '
            f'"LiteLLM_DailyTagSpend"."{column}", EXCLUDED."{column}")' in sql
        )
    assert '"spend" = "LiteLLM_DailyTagSpend"."spend" + EXCLUDED."spend"' in sql
    assert params.count(None) == 3


def test_priced_savings_still_bind_dollars():
    sql, params = build_bulk_upsert(
        TAG_TABLE,
        merge_by_conflict_key(
            TAG_TABLE,
            (tag_txn(compression_savings_spend=0.11, prompt_caching_savings_spend=0.22, gateway_injected_caching_savings_spend=0.33),),
        ),
    )
    assert None not in params
    assert 0.11 in params
    assert 0.22 in params
    assert 0.33 in params

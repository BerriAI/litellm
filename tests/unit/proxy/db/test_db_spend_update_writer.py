import asyncio
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from prisma.errors import DataError as PrismaDataError
from redis.exceptions import DataError

import litellm
from litellm.proxy.db.db_spend_update_writer import (
    _TEAM_ADVISORY_LOCK_SQL,
    _TEAM_MEMBER_SPEND_SQL,
    DBSpendUpdateWriter,
)
from litellm.responses.streaming_iterator import drain_logging_tasks
from tests.unit.proxy.db.fake_prisma_engine import engine_call






def _tool_call_response(*names: str) -> object:
    from types import SimpleNamespace

    tool_calls = [SimpleNamespace(function=SimpleNamespace(name=name)) for name in names]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))])


def _tool_usage_prisma() -> MagicMock:
    prisma = MagicMock()
    prisma.tool_usage_transactions = []
    prisma._tool_usage_transactions_lock = asyncio.Lock()
    prisma.spend_log_transactions = []
    prisma._spend_log_transactions_lock = asyncio.Lock()
    return prisma


def _minimal_spend_payload() -> dict:
    return {
        "request_id": "req-tool-1",
        "agent_id": None,
        "model_id": None,
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "metadata": "{}",

        "startTime": datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
        "endTime": datetime(2026, 7, 25, 10, 0, 1, tzinfo=timezone.utc),
        "spend": 0.0,
        "total_tokens": 42,
        "mcp_namespaced_tool_name": None,
    }


@pytest.mark.asyncio
async def test_update_database_enqueues_tool_usage_for_invoked_tools():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "gpt-4"},
            completion_response=_tool_call_response("get_weather"),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.1,
        )
        await asyncio.sleep(0)

    assert len(prisma.tool_usage_transactions) == 1
    transaction = prisma.tool_usage_transactions[0]
    assert transaction.request_id == "req-tool-1"
    assert transaction.tool_names == ("get_weather",)
    assert transaction.spend == 0.1
    assert transaction.total_tokens == 42
    assert transaction.date == "2026-07-25"


@pytest.mark.asyncio
async def test_update_database_enqueues_realtime_tool_usage():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={
                "model": "gpt-realtime",
                "realtime_tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "rt_tool", "arguments": "{}"}}
                ],
            },
            completion_response=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.2,
        )
        await asyncio.sleep(0)

    assert len(prisma.tool_usage_transactions) == 1
    assert prisma.tool_usage_transactions[0].tool_names == ("rt_tool",)


def test_enqueue_tool_registry_upsert_reads_every_choice():
    from types import SimpleNamespace as NS

    db_writer = DBSpendUpdateWriter()
    db_writer.tool_discovery_queue = MagicMock()
    response = NS(
        choices=[
            NS(message=NS(tool_calls=[NS(function=NS(name="tool_alpha"))])),
            NS(message=NS(tool_calls=[NS(function=NS(name="tool_beta"))])),
        ]
    )

    db_writer._enqueue_tool_registry_upsert(kwargs={}, completion_response=response)

    enqueued = [c.args[0]["tool_name"] for c in db_writer.tool_discovery_queue.add_update.call_args_list]
    assert enqueued == ["tool_alpha", "tool_beta"]


@pytest.mark.asyncio
async def test_update_database_skips_tool_usage_when_spend_logs_disabled():
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=_minimal_spend_payload(),
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "gpt-4"},
            completion_response=_tool_call_response("get_weather"),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.1,
        )
        await asyncio.sleep(0)

    assert prisma.tool_usage_transactions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_spend_logs", [True, False])
@pytest.mark.parametrize("session_id", ["session-1", None])
async def test_a_routed_request_reaches_the_auto_router_rollup_whether_or_not_spend_logs_are_kept(
    disable_spend_logs: bool, session_id: str | None
) -> None:
    db_writer = DBSpendUpdateWriter()
    db_writer._insert_spend_log_to_db = AsyncMock()
    prisma = _tool_usage_prisma()
    prisma.autorouter_turn_transactions = []
    prisma._autorouter_turn_transactions_lock = asyncio.Lock()
    routed_payload: Final = {
        **_minimal_spend_payload(),
        "status": "success",
        "api_key": "hashed-key",
        "user": "u1",
        "session_id": session_id,
        "model": "claude-haiku-4-5",
        "model_group": "smart-router",
        "spend": 0.25,
        "startTime": "2026-07-25T10:00:00+00:00",
        "metadata": json.dumps(
            {
                "routing_decision": {"router_model_name": "smart-router", "router_type": "complexity"},
                "autorouter_savings": 1.5,
            }
        ),
    }

    with (
        patch(
            "litellm.proxy.proxy_server.disable_spend_logs", disable_spend_logs
        ),  # test-quality-ok: update_database reads this proxy_server module global at call time; no injection seam
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
        patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils.get_logging_payload",
            return_value=routed_payload,
        ),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="u1",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={"model": "smart-router"},
            completion_response=_tool_call_response("get_weather"),
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=0.25,
        )

    (turn,) = prisma.autorouter_turn_transactions
    stored_session: Final = session_id if session_id and not disable_spend_logs else ""
    assert (turn.router_name, turn.router_type, turn.session_id) == ("smart-router", "complexity", stored_session)
    assert (turn.spend, turn.saved_spend) == (0.25, 1.5)
    assert (prisma.tool_usage_transactions == []) is disable_spend_logs


Statement = tuple[str, tuple[object, ...]]






def _row_values(statement: Statement, column: str) -> list[object]:
    """Every row's value for one column, read out of the flat parameter tuple."""
    sql, params = statement
    header = re.search(r"INSERT INTO \"[A-Za-z_]+\" \(([^)]*)\)", sql)
    assert header is not None, sql
    columns = header.group(1).split(", ")
    stride = len(columns) - 1  # updated_at is inlined, not bound
    offset = columns.index(f'"{column}"')
    return [params[row * stride + offset] for row in range(len(params) // stride)]




def _daily_txn(user_id: str = "user1") -> dict:
    return {
        "user_id": user_id,
        "date": "2024-01-01",
        "api_key": "test-api-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "spend": 0.1,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }
















# Tag Spend Tracking Tests


















@pytest.mark.asyncio
async def test_commit_spend_updates_to_db_increments_agent_spend():
    """
    Test that _commit_spend_updates_to_db calls litellm_agentstable.update_many
    with spend increment when agent_list_transactions is present.
    """
    db_writer = DBSpendUpdateWriter()

    mock_batcher = MagicMock()
    mock_batcher.litellm_verificationtoken = MagicMock()
    mock_batcher.litellm_verificationtoken.update_many = MagicMock()
    mock_batcher.litellm_usertable = MagicMock()
    mock_batcher.litellm_usertable.update_many = MagicMock()
    mock_batcher.litellm_teamtable = MagicMock()
    mock_batcher.litellm_teamtable.update_many = MagicMock()
    mock_batcher.litellm_teammembership = MagicMock()
    mock_batcher.litellm_teammembership.update_many = MagicMock()
    mock_batcher.litellm_organizationtable = MagicMock()
    mock_batcher.litellm_organizationtable.update_many = MagicMock()
    mock_batcher.litellm_tagtable = MagicMock()
    mock_batcher.litellm_tagtable.update_many = MagicMock()
    mock_batcher.litellm_agentstable = MagicMock()
    mock_batcher.litellm_agentstable.update_many = MagicMock()

    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    mock_transaction.batch_ = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_batcher),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=mock_transaction)

    mock_proxy_logging = MagicMock()

    agent_id = "agent-789"
    response_cost = 0.25
    db_spend_update_transactions = {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {},
        "team_list_transactions": {},
        "team_member_list_transactions": {},
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {agent_id: response_cost},
    }

    with patch("litellm.proxy.utils.raise_failed_update_spend_exception"):
        await db_writer._commit_spend_updates_to_db(
            prisma_client=mock_prisma_client,
            n_retry_times=0,
            proxy_logging_obj=mock_proxy_logging,
            db_spend_update_transactions=db_spend_update_transactions,
        )

    mock_batcher.litellm_agentstable.update_many.assert_called_once()
    call_kwargs = mock_batcher.litellm_agentstable.update_many.call_args[1]
    assert call_kwargs["where"] == {"agent_id": agent_id}
    assert call_kwargs["data"] == {"spend": {"increment": response_cost}}


def _team_member_flush_fixtures() -> tuple[AsyncMock, MagicMock]:
    """A transaction and prisma client that record the raw statement the member spend flush runs."""
    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    mock_transaction.execute_raw = AsyncMock(return_value=1)

    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=mock_transaction)
    return mock_transaction, mock_prisma_client


def _team_member_only_transactions(spend_by_member_key: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {},
        "team_list_transactions": {},
        "team_member_list_transactions": spend_by_member_key,
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {},
    }


@pytest.mark.asyncio
async def test_commit_spend_updates_to_db_writes_team_member_spend_in_one_roster_checked_upsert():
    """
    Regression (LIT-5502): members added without a budget had no membership row, and the
    previous update_many matched zero rows, so their spend was silently dropped.

    The flush now takes the same per-team advisory lock the team endpoints hold, then runs
    one INSERT ... ON CONFLICT statement for the whole batch that adds the cost to both spend
    and total_spend and creates the missing row for a user still on the team roster, so no
    per-team read can fail or time out ahead of the writes.
    """
    db_writer = DBSpendUpdateWriter()
    team_id = "team-abc"
    user_id = "user-xyz"
    response_cost = 0.75
    mock_transaction, mock_prisma_client = _team_member_flush_fixtures()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.call_details.get = MagicMock(return_value=None)

    await db_writer._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=0,
        proxy_logging_obj=mock_proxy_logging,
        db_spend_update_transactions=_team_member_only_transactions(
            {f"team_id::{team_id}::user_id::{user_id}": response_cost}
        ),
    )

    _lock_timeout_call, lock_call, spend_call = mock_transaction.execute_raw.await_args_list
    lock_statement, locked_team_id = lock_call.args
    assert lock_statement is _TEAM_ADVISORY_LOCK_SQL
    assert locked_team_id == team_id
    assert "pg_advisory_xact_lock(hashtext($1))" in lock_statement
    statement, members = spend_call.args
    assert statement is _TEAM_MEMBER_SPEND_SQL
    assert json.loads(members) == [{"user_id": user_id, "team_id": team_id, "cost": response_cost}]
    assert 'INSERT INTO "LiteLLM_TeamMembership"' in statement
    assert "members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', member.user_id))" in statement
    assert "ON CONFLICT (user_id, team_id) DO UPDATE" in statement
    assert 'spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend' in statement
    assert (
        'total_spend = GREATEST("LiteLLM_TeamMembership".total_spend, "LiteLLM_TeamMembership".spend)'
        " + EXCLUDED.total_spend"
    ) in statement


@pytest.mark.asyncio
async def test_commit_spend_updates_to_db_orders_team_member_rows_by_team_then_user():
    """
    The member spend statement touches rows in the order of its input rows, so the batch
    is handed over sorted by (team_id, user_id), with each cost kept next to its member, and
    each distinct team is locked once, in `sorted(team_ids)` order, the order /team/delete
    locks in, so a concurrent flush and delete cannot deadlock. `eng` and `eng2` pin that:
    sorting the composite keys instead would lock `eng2` first because `2` < `:`.
    """
    db_writer = DBSpendUpdateWriter()
    mock_transaction, mock_prisma_client = _team_member_flush_fixtures()

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.call_details.get = MagicMock(return_value=None)

    await db_writer._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=0,
        proxy_logging_obj=mock_proxy_logging,
        db_spend_update_transactions=_team_member_only_transactions(
            {
                "team_id::eng2::user_id::user_x": 0.1,
                "team_id::eng::user_id::user_y": 0.2,
                "team_id::eng::user_id::user_x": 0.3,
                "team_id::eng-b::user_id::user_x": 0.4,
            }
        ),
    )

    _lock_timeout_call, *lock_calls, spend_call = mock_transaction.execute_raw.await_args_list
    _statement, members = spend_call.args
    assert [lock_call.args for lock_call in lock_calls] == [
        (_TEAM_ADVISORY_LOCK_SQL, "eng"),
        (_TEAM_ADVISORY_LOCK_SQL, "eng-b"),
        (_TEAM_ADVISORY_LOCK_SQL, "eng2"),
    ]
    assert [(row["team_id"], row["user_id"], row["cost"]) for row in json.loads(members)] == [
        ("eng", "user_x", 0.3),
        ("eng", "user_y", 0.2),
        ("eng-b", "user_x", 0.4),
        ("eng2", "user_x", 0.1),
    ]














































@pytest.mark.asyncio
async def test_commit_key_spend_updates_includes_last_active():
    """
    Test that _commit_spend_updates_to_db sets last_active alongside spend
    when updating the key table.
    """
    db_writer = DBSpendUpdateWriter()

    # Create mock prisma client with transaction support
    mock_batcher = MagicMock()
    mock_batcher.litellm_verificationtoken = MagicMock()
    mock_batcher.litellm_verificationtoken.update_many = MagicMock()

    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    mock_transaction.batch_ = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_batcher),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=mock_transaction)

    # Also mock the other table batchers to avoid errors
    mock_batcher.litellm_usertable = MagicMock()
    mock_batcher.litellm_usertable.update_many = MagicMock()
    mock_batcher.litellm_teamtable = MagicMock()
    mock_batcher.litellm_teamtable.update_many = MagicMock()
    mock_batcher.litellm_organizationtable = MagicMock()
    mock_batcher.litellm_organizationtable.update_many = MagicMock()
    mock_batcher.litellm_agentstable = MagicMock()
    mock_batcher.litellm_agentstable.update_many = MagicMock()

    mock_proxy_logging = MagicMock()

    db_spend_update_transactions = {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {"hashed_token_abc": 0.05},
        "team_list_transactions": {},
        "team_member_list_transactions": {},
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {},
    }

    before_call = datetime.now(timezone.utc)

    with patch("litellm.proxy.utils.raise_failed_update_spend_exception"):
        await db_writer._commit_spend_updates_to_db(
            prisma_client=mock_prisma_client,
            n_retry_times=0,
            proxy_logging_obj=mock_proxy_logging,
            db_spend_update_transactions=db_spend_update_transactions,
        )

    after_call = datetime.now(timezone.utc)

    # Verify update_many was called on the key table
    mock_batcher.litellm_verificationtoken.update_many.assert_called_once()
    call_kwargs = mock_batcher.litellm_verificationtoken.update_many.call_args[1]

    # Verify the where clause targets the correct token
    assert call_kwargs["where"] == {"token": "hashed_token_abc"}

    # Verify data includes both spend increment and last_active
    assert call_kwargs["data"]["spend"] == {"increment": 0.05}
    assert "last_active" in call_kwargs["data"]

    # Verify last_active is a datetime within the expected range
    last_active = call_kwargs["data"]["last_active"]
    assert isinstance(last_active, datetime)
    assert before_call <= last_active <= after_call


@pytest.mark.asyncio
async def test_commit_spend_updates_to_db_increments_key_total_spend_alongside_spend():
    """
    The key table write must increment the lifetime total_spend by the same amount as the
    resettable spend, in the same update so the two cannot drift.
    """
    db_writer = DBSpendUpdateWriter()

    mock_batcher = MagicMock()
    mock_batcher.litellm_verificationtoken = MagicMock()
    mock_batcher.litellm_verificationtoken.update_many = MagicMock()

    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    mock_transaction.batch_ = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_batcher),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=mock_transaction)

    db_spend_update_transactions = {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {"hashed_token_abc": 0.05, "hashed_token_def": 1.25},
        "team_list_transactions": {},
        "team_member_list_transactions": {},
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {},
    }

    await db_writer._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=0,
        proxy_logging_obj=MagicMock(),
        db_spend_update_transactions=db_spend_update_transactions,
    )

    calls = mock_batcher.litellm_verificationtoken.update_many.call_args_list
    assert [c.kwargs["where"] for c in calls] == [{"token": "hashed_token_abc"}, {"token": "hashed_token_def"}]
    for call, expected_cost in zip(calls, (0.05, 1.25)):
        assert call.kwargs["data"]["spend"] == {"increment": expected_cost}
        assert call.kwargs["data"]["total_spend"] == call.kwargs["data"]["spend"]




@pytest.mark.asyncio
@pytest.mark.parametrize("response_cost", [float("nan"), float("inf"), float("-inf")])
async def test_update_database_records_a_non_finite_response_cost_as_zero(response_cost: float) -> None:
    db_writer: Final = DBSpendUpdateWriter()
    prisma: Final = _tool_usage_prisma()
    prisma.get_request_status = MagicMock(return_value="success")
    response: Final = litellm.ModelResponse(
        model="gpt-5.6-luna",
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"),
    ):
        await db_writer.update_database(
            token="test-token",
            user_id="test-user",
            end_user_id=None,
            team_id="test-team",
            org_id=None,
            kwargs={
                "model": "gpt-5.6-luna",
                "custom_llm_provider": "openai",
                "call_type": "acompletion",
                "litellm_call_id": "req-non-finite-cost",
                "response_cost": response_cost,
                "litellm_params": {"metadata": {"user_api_key": "hashed-key", "user_api_key_team_id": "test-team"}},
            },
            completion_response=response,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            response_cost=response_cost,
        )
        await drain_logging_tasks()

    assert [row["spend"] for row in prisma.spend_log_transactions] == [0]
    from litellm.accounting.projection import project

    contributions = project(prisma.spend_log_transactions[0])
    daily = [dict(item.increments) for item in contributions if item.table in {"LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend"}]
    assert len(daily) == 2
    assert all((row["api_requests"], row["spend"]) == (1, 0) for row in daily)

























@pytest.mark.parametrize(
    "bucket_name,input_dict,table_attr,method_name,where_key,expected_order",
    [
        pytest.param(
            "user_list_transactions",
            {"user_c": 0.1, "user_a": 0.2, "user_b": 0.3},
            "litellm_usertable",
            "update_many",
            "user_id",
            ["user_a", "user_b", "user_c"],
            id="user",
        ),
        pytest.param(
            "key_list_transactions",
            {"tok_c": 0.1, "tok_a": 0.2, "tok_b": 0.3},
            "litellm_verificationtoken",
            "update_many",
            "token",
            ["tok_a", "tok_b", "tok_c"],
            id="key",
        ),
        pytest.param(
            "team_list_transactions",
            {"team_c": 0.1, "team_a": 0.2, "team_b": 0.3},
            "litellm_teamtable",
            "update_many",
            "team_id",
            ["team_a", "team_b", "team_c"],
            id="team",
        ),
        pytest.param(
            "org_list_transactions",
            {"org_c": 0.1, "org_a": 0.2, "org_b": 0.3},
            "litellm_organizationtable",
            "update_many",
            "organization_id",
            ["org_a", "org_b", "org_c"],
            id="org",
        ),
        pytest.param(
            "end_user_list_transactions",
            {"eu_c": 0.1, "eu_a": 0.2, "eu_b": 0.3},
            "litellm_endusertable",
            "upsert",
            "user_id",
            ["eu_a", "eu_b", "eu_c"],
            id="end_user",
        ),
        pytest.param(
            "tag_list_transactions",
            {"prod": 0.1, "customer-x": 0.2, "test": 0.3},
            "litellm_tagtable",
            "update_many",
            "tag_name",
            ["customer-x", "prod", "test"],
            id="tag",
        ),
        pytest.param(
            "agent_list_transactions",
            {"agent_c": 0.1, "agent_a": 0.2, "agent_b": 0.3},
            "litellm_agentstable",
            "update_many",
            "agent_id",
            ["agent_a", "agent_b", "agent_c"],
            id="agent",
        ),
    ],
)
@pytest.mark.asyncio
async def test_commit_spend_updates_iterates_in_sorted_order(
    bucket_name, input_dict, table_attr, method_name, where_key, expected_order
):
    """
    Every spend-bucket code path in _commit_spend_updates_to_db must iterate
    in sorted order so concurrent pods acquire row locks in the same order
    and avoid PostgreSQL deadlocks. Covers the 5 direct loops (user/key/team/
    team_member/org), the end_user path in ProxyUpdateSpend.update_end_user_spend,
    and the shared _update_entity_spend_in_db helper (tag, agent).
    """
    db_writer = DBSpendUpdateWriter()

    captured_where_values = []

    def capture(*, where, data):
        captured_where_values.append(where[where_key])

    mock_batcher = MagicMock()
    table_mock = MagicMock()
    setattr(table_mock, method_name, MagicMock(side_effect=capture))
    setattr(mock_batcher, table_attr, table_mock)

    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    mock_transaction.batch_ = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_batcher),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_transaction.query_raw = AsyncMock(return_value=[])

    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=mock_transaction)

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.call_details = {}

    buckets = {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {},
        "team_list_transactions": {},
        "team_member_list_transactions": {},
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {},
    }
    buckets[bucket_name] = input_dict

    await db_writer._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=3,
        proxy_logging_obj=mock_proxy_logging,
        db_spend_update_transactions=buckets,
    )

    assert captured_where_values == expected_order




@pytest.mark.asyncio
async def test_spend_update_path_never_queries_user_cache_with_none_user_id():
    from litellm.accounting.projection import project

    writer = DBSpendUpdateWriter()
    prisma = _tool_usage_prisma()
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(side_effect=DataError("Invalid None cache key"))
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    with patch("litellm.proxy.proxy_server.prisma_client", prisma), patch("litellm.proxy.proxy_server.user_api_key_cache", cache), patch("litellm.proxy.proxy_server.disable_spend_logs", True):
        assert await writer.update_database(token=None, user_id=None, end_user_id="end-user-1", team_id=None, org_id=None, kwargs={"model": "ledger-test", "litellm_call_id": "anonymous"}, completion_response=litellm.ModelResponse(model="ledger-test"), start_time=now, end_time=now, response_cost=0.25)
    cache.async_get_cache.assert_not_called()
    contributions = project(prisma.spend_log_transactions[0])
    end_users = [item for item in contributions if item.table == "LiteLLM_EndUserTable"]
    assert [(dict(item.key)["user_id"], dict(item.increments)["spend"]) for item in end_users] == [("end-user-1", 0.25)]
    assert not any(item.table == "LiteLLM_UserTable" for item in contributions)


@pytest.mark.asyncio
async def test_daily_transaction_carries_compression_saved_tokens():
    """
    The daily transaction built from a SpendLog payload must carry
    compression_saved_tokens summed from both the native compression_savings
    metadata key and Headroom guardrail entries, alongside the cache token
    fields extracted from usage_object.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    metadata = {
        "usage_object": {"cache_read_input_tokens": 40, "cache_creation_input_tokens": 15},
        "litellm_gateway_injected_cache": "dep-of-the-compression-row",
        "compression_savings": {
            "tokens_before": 12000,
            "tokens_after": 5000,
            "tokens_saved": 7000,
            "source": "compression_interception",
        },
        "guardrail_information": [
            {
                "guardrail_name": "headroom-compressor",
                "guardrail_provider": "headroom",
                "guardrail_status": "success",
                "guardrail_response": {"tokens_before": 1000, "tokens_after": 400, "tokens_saved": 600},
            }
        ],
    }
    payload = {
        "request_id": "req-compression-1",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "model_id": "dep-of-the-compression-row",
        "call_type": "anthropic_messages",
        "prompt_tokens": 5000,
        "completion_tokens": 10,
        "spend": 0.05,
        "metadata": json.dumps(metadata),
    }

    transaction = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=payload,
        prisma_client=mock_prisma,
        type="user",
    )

    assert transaction is not None
    assert transaction["compression_saved_tokens"] == 7600
    assert transaction["cache_read_input_tokens"] == 40
    assert transaction["cache_creation_input_tokens"] == 15

    model_info = litellm.get_model_info(model="claude-sonnet-5", custom_llm_provider="anthropic")
    input_cost = model_info["input_cost_per_token"] or 0.0
    cache_read_cost = model_info.get("cache_read_input_token_cost") or input_cost
    cache_write_cost = model_info.get("cache_creation_input_token_cost") or input_cost
    assert transaction["compression_savings_spend"] == pytest.approx(7600 * input_cost)
    assert transaction["prompt_caching_savings_spend"] == pytest.approx(
        40 * max(input_cost - cache_read_cost, 0.0) - 15 * (cache_write_cost - input_cost)
    )
    assert transaction["compression_savings_spend"] > 0
    assert transaction["prompt_caching_savings_spend"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "estimate, recorded_savings, expected",
    [
        pytest.param(None, None, -0.005, id="plain-classifier-cost"),
        pytest.param({"version": 1, "status": "unknown"}, None, 0.0, id="unknown"),
        pytest.param({"version": 2, "status": "unknown"}, None, 0.0, id="unknown-v2"),
        pytest.param({"version": 1, "status": "unknown"}, -0.003, 0.0, id="unknown-stale-value"),
        pytest.param({"version": 0, "status": "estimated"}, -0.003, 0.0, id="unsupported-version"),
        pytest.param({"version": 1, "status": "estimated"}, -0.003, -0.003, id="estimated"),
        pytest.param(None, -0.003, -0.003, id="legacy"),
    ],
)
async def test_daily_transaction_compression_saved_tokens_zero_when_absent(
    estimate: dict[str, object] | None,
    recorded_savings: float | None,
    expected: float,
) -> None:
    """Requests without any compression metadata produce a zero count."""
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-no-compression",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "call_type": "anthropic_messages",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "spend": 0.01,
        "metadata": json.dumps(
            {
                "usage_object": {"prompt_tokens": 100, "completion_tokens": 10},
                "routing_decision": {"savings_baseline_model": "anthropic/claude-sonnet-5", "classifier_cost": 0.005},
                "autorouter_savings": recorded_savings,
                "autorouter_savings_estimate": estimate,
            }
        ),
    }

    transaction = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=payload,
        prisma_client=mock_prisma,
        type="user",
    )

    assert transaction is not None
    assert transaction["compression_saved_tokens"] == 0
    assert transaction["compression_savings_spend"] == 0
    assert transaction["prompt_caching_savings_spend"] == 0
    assert transaction["spend"] == 0.01
    assert transaction["autorouter_savings_spend"] == expected


# ---------------------------------------------------------------------------
# Budget window spend flush (LiteLLM_BudgetWindowSpend)
# ---------------------------------------------------------------------------


class _WindowSpendFakeBatcher:
    def __init__(self):
        self.calls = []

    def execute_raw(self, query, *args):
        self.calls.append((query, args))


class _WindowSpendFakeDB:
    """Minimal prisma_client.db that records the raw statements it is handed."""

    def __init__(self, existing_rows=None):
        self.existing_rows = existing_rows or []
        self.query_raw_calls = []
        self.batcher = _WindowSpendFakeBatcher()

    async def query_raw(self, query, *args):
        self.query_raw_calls.append((query, args))
        if "LiteLLM_BudgetWindowSpend" in query:
            return self.existing_rows
        return []

    @asynccontextmanager
    async def _tx(self):
        yield self

    def tx(self, timeout=None):
        return self._tx()

    @asynccontextmanager
    async def _batch(self):
        yield self.batcher

    def batch_(self):
        return self._batch()


class _WindowSpendFakePrisma:
    def __init__(self, db):
        self.db = db














class _DailySpendFakeDB(_WindowSpendFakeDB):
    """Records the daily rollup upserts it is handed and fails the ones aimed at one table."""

    def __init__(self, failing_table: str | None, failure: Exception | None = None) -> None:
        super().__init__()
        self.failing_table = failing_table
        self.failure = failure
        self.execute_raw_calls: list[Statement] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        if self.failing_table is not None and self.failing_table in query:
            raise self.failure if self.failure is not None else Exception("connection reset")
        self.execute_raw_calls.append((query, args))
        return len(args)


def _daily_upserts(db: _DailySpendFakeDB, table: str) -> list[Statement]:
    return [statement for statement in db.execute_raw_calls if table in statement[0]]
























@pytest.mark.asyncio
async def test_commit_spend_updates_to_db_does_not_stamp_key_settings_updated_at():
    """Spend flushes must leave settings_updated_at alone, or it decays into
    another `updated_at` and stops being an audit signal."""
    db_writer = DBSpendUpdateWriter()

    mock_batcher = MagicMock()
    mock_batcher.litellm_verificationtoken = MagicMock()
    mock_batcher.litellm_verificationtoken.update_many = MagicMock()
    mock_batcher.litellm_usertable = MagicMock()
    mock_batcher.litellm_usertable.update_many = MagicMock()
    mock_batcher.litellm_teamtable = MagicMock()
    mock_batcher.litellm_teamtable.update_many = MagicMock()
    mock_batcher.litellm_teammembership = MagicMock()
    mock_batcher.litellm_teammembership.update_many = MagicMock()
    mock_batcher.litellm_organizationtable = MagicMock()
    mock_batcher.litellm_organizationtable.update_many = MagicMock()
    mock_batcher.litellm_tagtable = MagicMock()
    mock_batcher.litellm_tagtable.update_many = MagicMock()
    mock_batcher.litellm_agentstable = MagicMock()
    mock_batcher.litellm_agentstable.update_many = MagicMock()

    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=False)
    mock_transaction.batch_ = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_batcher),
            __aexit__=AsyncMock(return_value=False),
        )
    )

    mock_prisma_client = MagicMock()
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=mock_transaction)

    token = "hashed-token-abc"
    response_cost = 0.25
    db_spend_update_transactions = {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {token: response_cost},
        "team_list_transactions": {},
        "team_member_list_transactions": {},
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {},
    }

    with patch("litellm.proxy.utils.raise_failed_update_spend_exception"):
        await db_writer._commit_spend_updates_to_db(
            prisma_client=mock_prisma_client,
            n_retry_times=0,
            proxy_logging_obj=MagicMock(),
            db_spend_update_transactions=db_spend_update_transactions,
        )

    mock_batcher.litellm_verificationtoken.update_many.assert_called_once()
    call_kwargs = mock_batcher.litellm_verificationtoken.update_many.call_args[1]
    assert call_kwargs["where"] == {"token": token}
    assert set(call_kwargs["data"]) == {"spend", "total_spend", "last_active"}
    assert call_kwargs["data"]["spend"] == {"increment": response_cost}


@pytest.mark.asyncio
async def test_commit_spend_updates_to_db_reports_each_completed_table():
    db_writer = DBSpendUpdateWriter()
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(return_value=_good_tx(MagicMock()))
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.call_details = {}
    on_table_committed = MagicMock()

    await db_writer._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=0,
        proxy_logging_obj=proxy_logging_obj,
        db_spend_update_transactions=_empty_spend_transactions(
            org_member_list_transactions={},
            project_list_transactions={},
            model_access_group_list_transactions={},
        ),
        on_table_committed=on_table_committed,
    )

    assert on_table_committed.call_args_list == [
        call("user_list_transactions"),
        call("end_user_list_transactions"),
        call("key_list_transactions"),
        call("team_list_transactions"),
        call("team_member_list_transactions"),
        call("org_list_transactions"),
        call("org_member_list_transactions"),
        call("project_list_transactions"),
        call("tag_list_transactions"),
        call("model_access_group_list_transactions"),
        call("agent_list_transactions"),
    ]




@pytest.mark.asyncio
async def test_daily_transaction_internal_call_keeps_spend_but_not_request_counts():
    """Internal sub-calls (auto-router classifier, shadow eval's shadow and judge) bill
    spend and tokens to the key but are not requests the caller made: api_requests,
    successful_requests, and autorouter_savings_spend must all stay zero for them."""
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    def _payload(metadata: dict) -> dict:
        return {
            "request_id": "req-internal-1",
            "user": "test-user",
            "startTime": "2026-08-11T00:00:00",
            "api_key": "test-key",
            "model": "claude-sonnet-5",
            "custom_llm_provider": "anthropic",
            "model_group": "claude-sonnet-5",
            "call_type": "acompletion",
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "spend": 0.05,
            "metadata": json.dumps(metadata),
        }

    internal = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=_payload({"internal_call_origin": "shadow_eval_judge"}),
        prisma_client=mock_prisma,
        type="user",
    )
    user_sent = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=_payload({}),
        prisma_client=mock_prisma,
        type="user",
    )

    assert internal is not None and user_sent is not None
    assert internal["spend"] == 0.05
    assert internal["prompt_tokens"] == 100
    assert internal["api_requests"] == 0
    assert internal["successful_requests"] == 0
    assert internal["failed_requests"] == 0
    assert internal["autorouter_savings_spend"] == 0.0
    assert user_sent["api_requests"] == 1
    assert user_sent["successful_requests"] == 1


def _response_time_payload(request_duration_ms: object, metadata: dict | None = None) -> dict:
    return {
        "request_id": "req-timed-1",
        "user": "test-user",
        "startTime": "2026-09-15T00:00:00",
        "api_key": "test-key",
        "model": "gpt-5.5",
        "custom_llm_provider": "openai",
        "model_group": "gpt-5.5",
        "call_type": "acompletion",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.01,
        "request_duration_ms": request_duration_ms,
        "metadata": json.dumps(metadata or {}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("request_duration_ms", [1234, 0])
async def test_daily_transaction_rolls_up_response_time_for_successful_requests(request_duration_ms: int):
    """A successful user-sent request contributes its request_duration_ms to the daily
    response-time sum and counts as one timed request, including a 0 ms duration."""
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    transaction = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=_response_time_payload(request_duration_ms),
        prisma_client=mock_prisma,
        type="user",
    )

    assert transaction is not None
    assert transaction["total_response_time_ms"] == request_duration_ms
    assert transaction["timed_requests"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_status", "request_duration_ms", "metadata"),
    [
        ("failure", 1234, {}),
        ("success", None, {}),
        ("success", -5, {}),
        ("success", "1234", {}),
        ("success", 1234, {"internal_call_origin": "shadow_eval_judge"}),
    ],
    ids=["failed", "missing", "negative", "non_int", "internal_call"],
)
async def test_daily_transaction_excludes_untimed_requests_from_response_time(
    request_status: str, request_duration_ms: object, metadata: dict
):
    """Failed, internal, and missing/invalid-duration requests never enter the response-time
    average: both the duration sum and the timed_requests denominator stay at zero."""
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value=request_status)

    transaction = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=_response_time_payload(request_duration_ms, metadata),
        prisma_client=mock_prisma,
        type="user",
    )

    assert transaction is not None
    assert transaction["total_response_time_ms"] == 0
    assert transaction["timed_requests"] == 0


def _deadlock_error():
    from prisma.errors import RawQueryError

    return RawQueryError(
        data={"user_facing_error": {"error_code": "P2034", "meta": {"table": "LiteLLM_VerificationToken"}}}
    )


def _empty_spend_transactions(**overrides):
    base = {
        "user_list_transactions": {},
        "end_user_list_transactions": {},
        "key_list_transactions": {},
        "team_list_transactions": {},
        "team_member_list_transactions": {},
        "org_list_transactions": {},
        "tag_list_transactions": {},
        "agent_list_transactions": {},
    }
    return {**base, **overrides}


def _good_tx(mock_batcher):
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = engine_call(False)
    tx.query_raw = engine_call([])
    tx.batch_ = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_batcher),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return tx


def _failing_tx(error):
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(side_effect=error)
    tx.__aexit__ = AsyncMock(return_value=False)
    return tx


@pytest.mark.asyncio
async def test_commit_spend_updates_retries_deadlock_then_commits(monkeypatch):
    """Regression: a deadlock on the key-spend UPDATE is retried and commits the increment exactly once."""
    slept = []
    monkeypatch.setattr(
        "litellm.proxy.db.db_spend_update_writer.asyncio.sleep",
        AsyncMock(side_effect=lambda s: slept.append(s)),
    )

    mock_batcher = MagicMock()
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(side_effect=[_failing_tx(_deadlock_error()), _good_tx(mock_batcher)])

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    await DBSpendUpdateWriter()._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=3,
        proxy_logging_obj=proxy_logging,
        db_spend_update_transactions=_empty_spend_transactions(key_list_transactions={"sk-abc": 0.5}),
    )

    assert mock_prisma_client.db.tx.call_count == 2
    mock_batcher.litellm_verificationtoken.update_many.assert_called_once()
    call_kwargs = mock_batcher.litellm_verificationtoken.update_many.call_args[1]
    assert call_kwargs["where"] == {"token": "sk-abc"}
    assert call_kwargs["data"]["spend"] == {"increment": 0.5}
    assert len(slept) == 1
    proxy_logging.failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_commit_spend_updates_raises_after_exhausting_deadlock_retries(monkeypatch):
    """A deadlock that never clears must surface after the retry budget is spent, not loop or swallow."""
    monkeypatch.setattr("litellm.proxy.db.db_spend_update_writer.asyncio.sleep", AsyncMock(return_value=None))

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(side_effect=lambda *a, **k: _failing_tx(_deadlock_error()))

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    from prisma.errors import RawQueryError

    with pytest.raises(RawQueryError):
        await DBSpendUpdateWriter()._commit_spend_updates_to_db(
            prisma_client=mock_prisma_client,
            n_retry_times=2,
            proxy_logging_obj=proxy_logging,
            db_spend_update_transactions=_empty_spend_transactions(key_list_transactions={"sk-abc": 0.5}),
        )

    assert mock_prisma_client.db.tx.call_count == 3


@pytest.mark.asyncio
async def test_commit_spend_updates_does_not_retry_non_deadlock_data_error(monkeypatch):
    """A non-retryable data-layer error raises on the first attempt, never retried against the increment."""
    monkeypatch.setattr("litellm.proxy.db.db_spend_update_writer.asyncio.sleep", AsyncMock(return_value=None))

    from prisma.errors import UniqueViolationError

    non_deadlock = UniqueViolationError(
        data={"user_facing_error": {"error_code": "P2002", "meta": {"table": "LiteLLM_VerificationToken"}}}
    )
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(side_effect=lambda *a, **k: _failing_tx(non_deadlock))

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    with pytest.raises(UniqueViolationError):
        await DBSpendUpdateWriter()._commit_spend_updates_to_db(
            prisma_client=mock_prisma_client,
            n_retry_times=3,
            proxy_logging_obj=proxy_logging,
            db_spend_update_transactions=_empty_spend_transactions(key_list_transactions={"sk-abc": 0.5}),
        )

    mock_prisma_client.db.tx.assert_called_once()




@pytest.mark.parametrize(
    "transactions_key, sample_key",
    [
        ("user_list_transactions", "user-1"),
        ("team_list_transactions", "team-1"),
        ("team_member_list_transactions", "team_id::team-1::user_id::user-1"),
        ("org_list_transactions", "org-1"),
        ("org_member_list_transactions", "organization_id::org-1::user_id::user-1"),
        ("tag_list_transactions", "tag-1"),
        ("agent_list_transactions", "agent-1"),
    ],
)
@pytest.mark.asyncio
async def test_commit_spend_updates_retries_deadlock_on_every_entity_path(monkeypatch, transactions_key, sample_key):
    """Every per-entity spend path, not just keys, retries a deadlock instead of dropping the increment."""
    monkeypatch.setattr("litellm.proxy.db.db_spend_update_writer.asyncio.sleep", AsyncMock(return_value=None))

    mock_batcher = MagicMock()
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(side_effect=[_failing_tx(_deadlock_error()), _good_tx(mock_batcher)])

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.call_details = {}

    await DBSpendUpdateWriter()._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=3,
        proxy_logging_obj=proxy_logging,
        db_spend_update_transactions=_empty_spend_transactions(**{transactions_key: {sample_key: 0.5}}),
    )

    assert mock_prisma_client.db.tx.call_count == 2
    proxy_logging.failure_handler.assert_not_called()


_SPEND_UPDATE_PATHS: Final = [
    ("key_list_transactions", "sk-abc"),
    ("user_list_transactions", "user-1"),
    ("team_list_transactions", "team-1"),
    ("team_member_list_transactions", "team_id::team-1::user_id::user-1"),
    ("org_list_transactions", "org-1"),
    ("org_member_list_transactions", "organization_id::org-1::user_id::user-1"),
    ("tag_list_transactions", "tag-1"),
    ("agent_list_transactions", "agent-1"),
]


@pytest.mark.parametrize("transactions_key, sample_key", _SPEND_UPDATE_PATHS)
@pytest.mark.asyncio
async def test_commit_spend_updates_retries_a_rollup_lock_timeout_then_commits(
    monkeypatch, transactions_key, sample_key
):
    """A spend UPDATE cancelled under ``lock_timeout`` (55P03) never took its row lock, so the
    transaction rolled back with nothing applied. Every entity path retries it like a deadlock
    and lands the increment exactly once, instead of raising and dropping the drained batch."""
    slept = []
    monkeypatch.setattr(
        "litellm.proxy.db.db_spend_update_writer.asyncio.sleep",
        AsyncMock(side_effect=lambda s: slept.append(s)),
    )

    mock_batcher = MagicMock()
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(side_effect=[_failing_tx(_lock_timeout_error()), _good_tx(mock_batcher)])

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.call_details = {}

    await DBSpendUpdateWriter()._commit_spend_updates_to_db(
        prisma_client=mock_prisma_client,
        n_retry_times=3,
        proxy_logging_obj=proxy_logging,
        db_spend_update_transactions=_empty_spend_transactions(**{transactions_key: {sample_key: 0.5}}),
    )

    assert {
        "transactions_opened": mock_prisma_client.db.tx.call_count,
        "backoff_sleeps": len(slept),
        "failure_handler_calls": proxy_logging.failure_handler.await_count,
    } == {"transactions_opened": 2, "backoff_sleeps": 1, "failure_handler_calls": 0}


@pytest.mark.asyncio
async def test_commit_spend_updates_raises_after_exhausting_lock_timeout_retries(monkeypatch):
    """A row that stays locked past every retry still surfaces as a failure rather than
    looping forever or being swallowed; the budget is the same one deadlocks get."""
    monkeypatch.setattr("litellm.proxy.db.db_spend_update_writer.asyncio.sleep", AsyncMock(return_value=None))

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.tx = MagicMock(side_effect=lambda *a, **k: _failing_tx(_lock_timeout_error()))

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    with pytest.raises(PrismaDataError, match="lock timeout"):
        await DBSpendUpdateWriter()._commit_spend_updates_to_db(
            prisma_client=mock_prisma_client,
            n_retry_times=2,
            proxy_logging_obj=proxy_logging,
            db_spend_update_transactions=_empty_spend_transactions(key_list_transactions={"sk-abc": 0.5}),
        )

    assert mock_prisma_client.db.tx.call_count == 3


def _responses_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "native",
                "litellm_params": {"model": "openai/gpt-5.1", "api_key": "sk-test"},
                "model_info": {"id": "dep-native"},
            },
            {
                "model_name": "bridged",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                "model_info": {"id": "dep-bridged"},
            },
            {
                "model_name": "forced-bridge",
                "litellm_params": {"model": "openai/gpt-5.1", "api_key": "sk-test", "use_chat_completions_api": True},
                "model_info": {"id": "dep-forced-bridge"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, model_id, expects_flush",
    [
        ("aresponses", "dep-bridged", True),
        ("responses", "dep-bridged", True),
        ("aresponses", "dep-forced-bridge", True),
        ("aresponses", None, True),
        ("aresponses", "dep-native", False),
        ("responses", "dep-native", False),
        ("acompletion", "dep-bridged", False),
    ],
)
async def test_insert_spend_log_asks_for_an_immediate_flush_only_on_rows_the_bridge_reads_back(
    call_type: str, model_id: str | None, expects_flush: bool
):
    """
    A `previous_response_id` chained straight off a bridged turn reads that turn's row from the
    DB, so it cannot sit in this worker's queue until the monitor's next poll. A native Responses
    provider holds the conversation itself, so its row waits for the batch write like a chat row.
    """
    db_writer = DBSpendUpdateWriter()
    prisma = _tool_usage_prisma()
    prisma.spend_log_flush_requested = asyncio.Event()
    payload = {"request_id": "req-1", "call_type": call_type, "model_id": model_id}

    await db_writer._insert_spend_log_to_db(payload=payload, prisma_client=prisma, llm_router=_responses_router())

    assert prisma.spend_log_transactions == [payload]
    assert prisma.spend_log_flush_requested.is_set() is expects_flush


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_chatgpt_responses_row_queues_without_a_default_chatgpt_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "no-default-login"))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    own_login: Final = tmp_path / "team" / "auth.json"
    own_login.parent.mkdir()
    own_login.write_text(json.dumps({"access_token": "at-team", "account_id": "acct-team"}))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "chatgpt-team",
                "litellm_params": {"model": "chatgpt/gpt-5.6-terra", "chatgpt_auth_file": str(own_login)},
                "model_info": {"id": "dep-chatgpt"},
            }
        ]
    )
    prisma: Final = _tool_usage_prisma()
    prisma.spend_log_flush_requested = asyncio.Event()
    payload: Final = {"request_id": "req-1", "call_type": "aresponses", "model_id": "dep-chatgpt"}

    await DBSpendUpdateWriter()._insert_spend_log_to_db(payload=payload, prisma_client=prisma, llm_router=router)

    assert prisma.spend_log_transactions == [payload]
    assert prisma.spend_log_flush_requested.is_set() is False




















_BATCH_CLAIM_FIELDS = {"request_id", "call_type", "status", "spend", "startTime", "endTime", "accounting_path"}












@pytest.mark.asyncio
@pytest.mark.parametrize(
    "injected_deployment, attributed",
    [
        pytest.param("dep-of-this-row", True, id="this-deployment-injected"),
        pytest.param("dep-of-a-sibling-leg", False, id="a-sibling-deployment-injected"),
        pytest.param("", True, id="injected-before-a-deployment-was-chosen"),
    ],
)
async def test_caching_savings_are_attributed_to_the_deployment_that_was_injected(injected_deployment, attributed):
    """Retries, same-group failover and cross-model-group fallbacks all reuse one metadata
    bucket and one litellm_call_id, so a marker written by the leg that injected is
    visible to every sibling and nothing request-scoped can tell them apart.

    Naming the deployment it injected for is what keeps the credit on that leg: a row
    billed for a different deployment reads it as no injection, so no seam has to strip
    it and a deployment that injected nothing is never credited for the one that did.

    An injection that ran before any deployment was chosen, which is what the proxy does
    for prompt templates, is written into the payload every leg goes on to send, so it
    marks the request for all of them and each leg keeps the credit.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-fallback-leg",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "model_id": "dep-of-this-row",
        "call_type": "anthropic_messages",
        "prompt_tokens": 5000,
        "completion_tokens": 10,
        "spend": 0.05,
        "metadata": json.dumps(
            {
                "usage_object": {"cache_read_input_tokens": 4242, "cache_creation_input_tokens": 1111},
                "litellm_gateway_injected_cache": injected_deployment,
            }
        ),
    }

    transaction = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=payload,
        prisma_client=mock_prisma,
        type="user",
    )

    assert transaction is not None
    assert transaction["prompt_caching_savings_spend"] != 0.0
    assert (transaction["gateway_injected_caching_savings_spend"] != 0.0) is attributed


@pytest.mark.asyncio
async def test_daily_transaction_attributes_caching_savings_only_with_an_injection_marker():
    """Cached usage with no litellm_gateway_injected_cache marker is still a real saving.

    Client-sent cache_control and implicit provider caching leave no marker, so the row
    keeps the total the customer actually got while the gateway-attributed column stays
    empty, which is what separates what caching saved from what litellm can claim.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-ungated-caching",
        "user": "test-user",
        "startTime": "2026-07-17T00:00:00",
        "api_key": "test-key",
        "model": "claude-sonnet-5",
        "custom_llm_provider": "anthropic",
        "model_group": "claude-sonnet-5",
        "call_type": "anthropic_messages",
        "prompt_tokens": 5000,
        "completion_tokens": 10,
        "spend": 0.05,
        "metadata": json.dumps(
            {"usage_object": {"cache_read_input_tokens": 4242, "cache_creation_input_tokens": 1111}}
        ),
    }

    transaction = await writer._common_add_spend_log_transaction_to_daily_transaction(
        payload=payload,
        prisma_client=mock_prisma,
        type="user",
    )

    assert transaction is not None
    assert transaction["cache_read_input_tokens"] == 4242
    assert transaction["cache_creation_input_tokens"] == 1111
    assert transaction["prompt_caching_savings_spend"] != 0.0
    assert transaction["gateway_injected_caching_savings_spend"] == 0.0




































def _lock_timeout_error() -> PrismaDataError:
    return PrismaDataError(
        data={
            "user_facing_error": {
                "is_panic": False,
                "message": "Error querying the database: canceling statement due to lock timeout",
                "meta": {"code": "55P03", "message": "canceling statement due to lock timeout"},
            }
        }
    )






def test_record_request_error_feeds_the_failure_rollup_with_the_logged_status() -> None:
    from litellm.proxy.db import db_spend_update_writer as writer_module
    from litellm.proxy.db.request_error_tracking import RequestErrorAccumulator
    from litellm.types.proxy.request_errors import RequestErrorKey

    accumulator: Final = RequestErrorAccumulator()
    prisma_client: Final = MagicMock()
    prisma_client.get_request_status.side_effect = lambda payload: json.loads(payload["metadata"])["status"]
    failed_payload: Final[dict[str, object]] = {
        "startTime": datetime(2026, 10, 8, 23, 59, tzinfo=timezone.utc),
        "api_key": "hash-1",
        "team_id": "team-a",
        "user": "user-a",
        "model_group": "gpt-4o",
        "metadata": json.dumps({"status": "failure", "error_information": {"error_code": "503"}}),
    }
    ok_payload: Final[dict[str, object]] = {**failed_payload, "metadata": json.dumps({"status": "success"})}
    with patch.object(writer_module, "request_error_accumulator", accumulator):
        DBSpendUpdateWriter._record_request_error(payload=failed_payload, prisma_client=prisma_client)
        DBSpendUpdateWriter._record_request_error(payload=ok_payload, prisma_client=prisma_client)
    assert accumulator.drain() == {
        RequestErrorKey(
            date="2026-10-08",
            api_key="hash-1",
            team_id="team-a",
            user_id="user-a",
            model_group="gpt-4o",
            status_code=503,
        ): 1
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_error_logs", [False, True])
@pytest.mark.parametrize("status", ["success", "failure"])
async def test_completed_spend_survives_error_log_suppression(
    monkeypatch: pytest.MonkeyPatch, disable_error_logs: bool, status: str
) -> None:
    from litellm.proxy import proxy_server
    from litellm.types.utils import ModelResponse, Usage

    writer: Final = DBSpendUpdateWriter()
    prisma: Final = _tool_usage_prisma()
    prisma.get_request_status.return_value = status
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "general_settings", {"disable_error_logs": disable_error_logs})
    now: Final = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    result: Final = ModelResponse(
        model="gpt-4o-mini", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert await writer.update_database(
        token="test-token",
        user_id="test-user",
        end_user_id=None,
        team_id=None,
        org_id=None,
        kwargs={
            "model": "gpt-4o-mini",
            "litellm_call_id": "completed-call",
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test-token",
                    "user_api_key_user_id": "test-user",
                    "status": status,
                }
            },
        },
        completion_response=result,
        start_time=now,
        end_time=now,
        response_cost=0.23,
    )
    await drain_logging_tasks()
    rows: Final = prisma.spend_log_transactions
    assert len(rows) == 1
    assert rows[0]["spend"] == pytest.approx(0.23)
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (10, 5)


@pytest.mark.parametrize("ledger_enabled", [False, True])
@pytest.mark.asyncio
async def test_request_error_counts_survive_the_ledger_path_switch(monkeypatch: pytest.MonkeyPatch, ledger_enabled: bool):
    from litellm.proxy import proxy_server
    from litellm.proxy.db import db_spend_update_writer as writer_module
    from litellm.proxy.db.request_error_tracking import RequestErrorAccumulator
    from litellm.types.proxy.request_errors import RequestErrorKey

    accumulator = RequestErrorAccumulator()
    prisma = _tool_usage_prisma()
    prisma.get_request_status.return_value = "failure"
    monkeypatch.setattr(writer_module, "request_error_accumulator", accumulator)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "general_settings", {"use_spend_logs_ledger": ledger_enabled})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    await DBSpendUpdateWriter().update_database(
        token="key", user_id="user", team_id=None, org_id=None, end_user_id=None,
        kwargs={
            "model": "ledger-test", "call_type": "acompletion", "litellm_call_id": "failed-request",
            "litellm_params": {"metadata": {
                "user_api_key": "key", "user_api_key_user_id": "user", "status": "failure",
                "error_information": {"error_code": "503"},
            }},
        },
        completion_response=None, start_time=now, end_time=now, response_cost=0,
    )
    await drain_logging_tasks()
    assert accumulator.drain() == {RequestErrorKey(
        date="2026-11-02", api_key=prisma.spend_log_transactions[0]["api_key"], team_id="", user_id="user", model_group="ledger-test", status_code=503,
    ): 1}

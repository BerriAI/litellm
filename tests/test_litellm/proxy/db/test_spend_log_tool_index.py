"""
Tests for the tool usage writer: ToolUsageTransaction construction (invoked tools
only) and the flush that writes LiteLLM_SpendLogToolIndex plus the
LiteLLM_DailyToolSpend rollup in one transaction.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.spend_log_tool_index import (
    ToolUsageTransaction,
    build_tool_usage_transaction,
    flush_tool_usage_transactions,
    response_tool_call_names,
)


def _response_with_tool_calls(*names: str) -> SimpleNamespace:
    tool_calls = [SimpleNamespace(function=SimpleNamespace(name=name)) for name in names]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))])


class _FakeBatcher:
    def __init__(self) -> None:
        self.litellm_spendlogtoolindex = MagicMock()
        self.litellm_dailytoolspend = MagicMock()

    async def __aenter__(self) -> "_FakeBatcher":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


def _prisma_with_batcher() -> tuple[MagicMock, _FakeBatcher]:
    batcher = _FakeBatcher()
    prisma = MagicMock()
    prisma.db.batch_ = MagicMock(return_value=batcher)
    return prisma, batcher


class TestBuildToolUsageTransaction:
    def test_declared_tools_never_reach_the_transaction(self):
        # Regression for the inflation bug: the builder's only non-MCP source is
        # the response's tool_calls, so a request declaring N tools while the
        # model invokes one produces exactly one attribution.
        transaction = build_tool_usage_transaction(
            request_id="r1",
            start_time_iso="2026-07-25T10:00:00+00:00",
            mcp_namespaced_tool_name=None,
            spend=0.5,
            total_tokens=100,
            completion_response=_response_with_tool_calls("get_weather"),
        )
        assert transaction is not None
        assert transaction.tool_names == ("get_weather",)

    def test_no_invoked_tools_returns_none(self):
        assert (
            build_tool_usage_transaction(
                request_id="r1",
                start_time_iso="2026-07-25T10:00:00+00:00",
                mcp_namespaced_tool_name=None,
                spend=0.5,
                total_tokens=100,
                completion_response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]),
            )
            is None
        )

    def test_mcp_name_and_response_names_dedupe(self):
        transaction = build_tool_usage_transaction(
            request_id="r1",
            start_time_iso="2026-07-25T10:00:00+00:00",
            mcp_namespaced_tool_name="srv/tool_a",
            spend=0.5,
            total_tokens=100,
            completion_response=_response_with_tool_calls("srv/tool_a", "tool_b", "tool_b"),
        )
        assert transaction is not None
        assert transaction.tool_names == ("srv/tool_a", "tool_b")

    def test_date_matches_daily_spend_writer_derivation(self):
        # The daily spend writer derives its date bucket as
        # payload["startTime"].split("T")[0] (db_spend_update_writer.py), i.e. the
        # timestamp's own calendar date, NOT the astimezone-UTC date. A non-UTC
        # isoformat pins the difference: 2026-07-25T22:00:00-07:00 is 2026-07-26
        # in UTC but must bucket as 2026-07-25 to match LiteLLM_DailyUserSpend.
        start_time_iso = "2026-07-25T22:00:00-07:00"
        transaction = build_tool_usage_transaction(
            request_id="r1",
            start_time_iso=start_time_iso,
            mcp_namespaced_tool_name="srv/tool_a",
            spend=0.5,
            total_tokens=100,
            completion_response=None,
        )
        assert transaction is not None
        assert transaction.date == start_time_iso.split("T")[0] == "2026-07-25"

    def test_realtime_tool_calls_reach_the_transaction(self):
        # Realtime sessions carry invoked tools in kwargs["realtime_tool_calls"]
        # (OpenAI tool_calls dict shape, built in realtime_streaming.py), not on a
        # response object; they must land in the rollup like any other invocation.
        realtime_tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "rt_get_weather", "arguments": "{}"}},
        ]
        transaction = build_tool_usage_transaction(
            request_id="r1",
            start_time_iso="2026-07-25T10:00:00+00:00",
            mcp_namespaced_tool_name=None,
            spend=0.5,
            total_tokens=100,
            completion_response=None,
            realtime_tool_calls=realtime_tool_calls,
        )
        assert transaction is not None
        assert transaction.tool_names == ("rt_get_weather",)

    def test_realtime_names_dedupe_against_response_names(self):
        realtime_tool_calls = [{"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
        transaction = build_tool_usage_transaction(
            request_id="r1",
            start_time_iso="2026-07-25T10:00:00+00:00",
            mcp_namespaced_tool_name=None,
            spend=0.5,
            total_tokens=100,
            completion_response=_response_with_tool_calls("get_weather"),
            realtime_tool_calls=realtime_tool_calls,
        )
        assert transaction is not None
        assert transaction.tool_names == ("get_weather",)

    def test_n_greater_than_one_tools_from_every_choice_reach_the_transaction(self):
        # Regression: an n>1 request pays for every choice, and a tool invoked
        # only in a later choice really ran; it must not be dropped because the
        # extractor read choices[0] alone.
        from types import SimpleNamespace as NS

        response = NS(
            choices=[
                NS(message=NS(tool_calls=[NS(function=NS(name="tool_alpha"))])),
                NS(message=NS(tool_calls=[NS(function=NS(name="tool_beta"))])),
            ]
        )
        transaction = build_tool_usage_transaction(
            request_id="r1",
            start_time_iso="2026-07-25T10:00:00+00:00",
            mcp_namespaced_tool_name=None,
            spend=0.5,
            total_tokens=100,
            completion_response=response,
        )
        assert transaction is not None
        assert transaction.tool_names == ("tool_alpha", "tool_beta")

    def test_unparseable_start_time_returns_none(self):
        assert (
            build_tool_usage_transaction(
                request_id="r1",
                start_time_iso="not-a-timestamp",
                mcp_namespaced_tool_name="srv/tool_a",
                spend=0.5,
                total_tokens=100,
                completion_response=None,
            )
            is None
        )


class TestResponseToolCallNames:
    def test_unrecognized_shapes_yield_nothing(self):
        assert response_tool_call_names(None) == ()
        assert response_tool_call_names(SimpleNamespace()) == ()
        assert response_tool_call_names(ValueError("boom")) == ()

    def test_blank_names_are_dropped(self):
        assert response_tool_call_names(_response_with_tool_calls("  ", "real_tool")) == ("real_tool",)

    def test_responses_api_output_function_calls(self):
        # Regression: /v1/responses carries invocations in output[] items of
        # type function_call, not in choices; they must reach the rollup.
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="function_call", name="get_weather", call_id="c1", arguments="{}"),
                SimpleNamespace(type="message", name=None, call_id=None, arguments=None),
            ]
        )
        assert response_tool_call_names(response) == ("get_weather",)

    def test_anthropic_messages_tool_use_blocks(self):
        response = {
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "ant_get_weather", "input": {"city": "Paris"}},
            ]
        }
        assert response_tool_call_names(response) == ("ant_get_weather",)


def _transaction(
    request_id: str,
    date: str = "2026-07-25",
    tool_names: tuple = ("tool_a",),
    spend: float = 1.0,
    total_tokens: int = 10,
) -> ToolUsageTransaction:
    from datetime import datetime, timezone

    return ToolUsageTransaction(
        request_id=request_id,
        date=date,
        start_time=datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc),
        tool_names=tool_names,
        spend=spend,
        total_tokens=total_tokens,
    )


class TestFlushToolUsageTransactions:
    @pytest.mark.asyncio
    async def test_multi_tool_request_attributes_full_spend_to_each_tool(self):
        prisma, batcher = _prisma_with_batcher()
        await flush_tool_usage_transactions(
            prisma_client=prisma,
            transactions=[_transaction("r1", tool_names=("tool_a", "tool_b"), spend=0.10, total_tokens=100)],
        )
        index_rows = batcher.litellm_spendlogtoolindex.create_many.call_args.kwargs["data"]
        assert [(r["request_id"], r["tool_name"]) for r in index_rows] == [("r1", "tool_a"), ("r1", "tool_b")]
        assert batcher.litellm_spendlogtoolindex.create_many.call_args.kwargs["skip_duplicates"] is True

        upserts = {
            c.kwargs["where"]["date_tool_name"]["tool_name"]: c.kwargs["data"]
            for c in batcher.litellm_dailytoolspend.upsert.call_args_list
        }
        assert set(upserts) == {"tool_a", "tool_b"}
        for data in upserts.values():
            assert data["create"]["spend"] == 0.10
            assert data["create"]["request_count"] == 1
            assert data["update"]["spend"] == {"increment": 0.10}
            assert data["update"]["request_count"] == {"increment": 1}

    @pytest.mark.asyncio
    async def test_same_day_same_tool_aggregates_within_batch(self):
        prisma, batcher = _prisma_with_batcher()
        await flush_tool_usage_transactions(
            prisma_client=prisma,
            transactions=[
                _transaction("r1", spend=0.10, total_tokens=100),
                _transaction("r2", spend=0.30, total_tokens=200),
            ],
        )
        assert batcher.litellm_dailytoolspend.upsert.call_count == 1
        data = batcher.litellm_dailytoolspend.upsert.call_args.kwargs["data"]
        assert data["create"] == {
            "date": "2026-07-25",
            "tool_name": "tool_a",
            "spend": pytest.approx(0.40),
            "total_tokens": 300,
            "request_count": 2,
        }
        assert data["update"]["spend"] == {"increment": pytest.approx(0.40)}
        assert data["update"]["total_tokens"] == {"increment": 300}
        assert data["update"]["request_count"] == {"increment": 2}

    @pytest.mark.asyncio
    async def test_index_rows_and_rollup_share_one_transaction(self):
        # Both writes go through the same batch_() so a failed flush cannot leave
        # index rows without their rollup increments (or vice versa); increments
        # are not idempotent, so partial states must be unreachable.
        prisma, batcher = _prisma_with_batcher()
        await flush_tool_usage_transactions(
            prisma_client=prisma,
            transactions=[_transaction("r1")],
        )
        prisma.db.batch_.assert_called_once()
        batcher.litellm_spendlogtoolindex.create_many.assert_called_once()
        batcher.litellm_dailytoolspend.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_batch_touches_nothing(self):
        prisma, _ = _prisma_with_batcher()
        await flush_tool_usage_transactions(prisma_client=prisma, transactions=[])
        prisma.db.batch_.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_errors_retry_and_succeed(self, monkeypatch):
        # A failed batch commits nothing, so retrying a connection error cannot
        # double-count; the flush must retry rather than drop the batch.
        import httpx

        batcher = _FakeBatcher()
        prisma = MagicMock()
        prisma.db.batch_ = MagicMock(side_effect=[httpx.ConnectError("down"), batcher])
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("litellm.proxy.db.spend_log_tool_index.asyncio.sleep", fake_sleep)
        await flush_tool_usage_transactions(prisma_client=prisma, transactions=[_transaction("r1")])
        assert prisma.db.batch_.call_count == 2
        assert len(sleeps) == 1
        batcher.litellm_dailytoolspend.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_errors_exhaust_retries_then_raise(self, monkeypatch):
        import httpx

        prisma = MagicMock()
        prisma.db.batch_ = MagicMock(side_effect=httpx.ConnectError("down"))

        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr("litellm.proxy.db.spend_log_tool_index.asyncio.sleep", fake_sleep)
        with pytest.raises(httpx.ConnectError):
            await flush_tool_usage_transactions(
                prisma_client=prisma, transactions=[_transaction("r1")], n_retry_times=2
            )
        assert prisma.db.batch_.call_count == 3

    @pytest.mark.asyncio
    async def test_non_connection_errors_do_not_retry(self):
        prisma = MagicMock()
        prisma.db.batch_ = MagicMock(side_effect=ValueError("bad data"))
        with pytest.raises(ValueError):
            await flush_tool_usage_transactions(prisma_client=prisma, transactions=[_transaction("r1")])
        prisma.db.batch_.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ambiguous_error", ["ReadTimeout", "ReadError"])
    async def test_post_send_ambiguous_errors_drop_without_retry(self, ambiguous_error):
        # A ReadTimeout means the statements were sent and the outcome is
        # unknown; the engine can leave the transaction open on the pooled
        # connection, so a retry's statements would stack into it and one
        # commit would apply both increment sets. These must never retry.
        import httpx

        error = getattr(httpx, ambiguous_error)("ambiguous")
        prisma = MagicMock()
        prisma.db.batch_ = MagicMock(side_effect=error)
        with pytest.raises((httpx.ReadTimeout, httpx.ReadError)):
            await flush_tool_usage_transactions(prisma_client=prisma, transactions=[_transaction("r1")])
        prisma.db.batch_.assert_called_once()

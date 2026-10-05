import asyncio
from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from prisma.errors import PrismaError

from litellm._service_logger import ServiceTypes
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.log_db_metrics import record_db_io


@pytest.fixture
def service_hooks() -> Iterator[tuple[AsyncMock, AsyncMock]]:
    success: Final = AsyncMock()
    failure: Final = AsyncMock()
    service_logging: Final = MagicMock(async_service_success_hook=success, async_service_failure_hook=failure)
    with patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock(service_logging_obj=service_logging)):
        yield success, failure


@pytest.mark.asyncio
async def test_a_completed_write_emits_one_db_event_named_for_the_call_and_table(
    service_hooks: tuple[AsyncMock, AsyncMock],
) -> None:
    success, failure = service_hooks

    async with db_span("commit_spend_updates", "LiteLLM_UserTable"):
        record_db_io()
    await asyncio.sleep(0)

    event: Final = success.await_args.kwargs
    assert (event["service"], event["call_type"], event["event_metadata"]) == (
        ServiceTypes.DB,
        "commit_spend_updates",
        {"table_name": "LiteLLM_UserTable"},
    )
    assert event["duration"] == pytest.approx((event["end_time"] - event["start_time"]).total_seconds())
    assert failure.await_count == 0


@pytest.mark.asyncio
async def test_a_prisma_error_inside_the_write_emits_a_db_failure_event_and_propagates(
    service_hooks: tuple[AsyncMock, AsyncMock],
) -> None:
    success, failure = service_hooks

    with pytest.raises(PrismaError):
        async with db_span("insert_spend_logs", "LiteLLM_SpendLogs"):
            raise PrismaError("connection reset")
    await asyncio.sleep(0)

    event: Final = failure.await_args.kwargs
    assert (event["service"], event["call_type"], event["event_metadata"], str(event["error"])) == (
        ServiceTypes.DB,
        "insert_spend_logs",
        {"table_name": "LiteLLM_SpendLogs"},
        "connection reset",
    )
    assert success.await_count == 0


@pytest.mark.asyncio
async def test_a_dropped_query_engine_connection_emits_a_db_failure_event(
    service_hooks: tuple[AsyncMock, AsyncMock],
) -> None:
    success, failure = service_hooks

    with pytest.raises(httpx.ReadError):
        async with db_span("write_tool_spend", "LiteLLM_DailyToolSpend"):
            raise httpx.ReadError("peer closed connection")
    await asyncio.sleep(0)

    event: Final = failure.await_args.kwargs
    assert (event["call_type"], event["event_metadata"], str(event["error"])) == (
        "write_tool_spend",
        {"table_name": "LiteLLM_DailyToolSpend"},
        "peer closed connection",
    )
    assert success.await_count == 0


@pytest.mark.asyncio
async def test_a_non_database_error_inside_the_write_emits_no_db_event(
    service_hooks: tuple[AsyncMock, AsyncMock],
) -> None:
    success, failure = service_hooks

    with pytest.raises(ValueError, match="bad row"):
        async with db_span("insert_spend_logs", "LiteLLM_SpendLogs"):
            raise ValueError("bad row")
    await asyncio.sleep(0)

    assert (success.await_count, failure.await_count) == (0, 0)


@pytest.mark.asyncio
async def test_a_raising_failure_hook_never_replaces_the_prisma_error(
    service_hooks: tuple[AsyncMock, AsyncMock],
) -> None:
    success, failure = service_hooks
    failure.side_effect = RuntimeError("exporter down")

    with pytest.raises(PrismaError):
        async with db_span("commit_spend_updates", "LiteLLM_UserTable"):
            raise PrismaError("connection reset")

    assert failure.await_count == 1
    assert success.await_count == 0


@pytest.mark.asyncio
async def test_a_block_whose_prisma_client_never_reached_the_engine_emits_no_db_event(
    service_hooks: tuple[AsyncMock, AsyncMock],
) -> None:
    success, failure = service_hooks

    async with db_span("team_user_spend", "LiteLLM_SpendLogs"):
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert (success.await_count, failure.await_count) == (0, 0)

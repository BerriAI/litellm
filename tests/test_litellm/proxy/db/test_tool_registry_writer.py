"""
Unit tests for tool_registry_writer.py — tests use a mock prisma client
to avoid requiring a real DB connection.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.db.tool_registry_writer import (
    batch_upsert_tools,
    get_tool,
    get_tools_by_names,
    list_tools,
    update_tool_policy,
)


def _make_prisma(rows=None):
    """Return a minimal mock prisma_client."""
    now = datetime.now(timezone.utc)
    default_row = MagicMock(
        tool_id="uuid-1",
        tool_name="my_tool",
        origin="user_defined",
        call_policy="untrusted",
        assignments={},
        created_at=now,
        updated_at=now,
        created_by=None,
        updated_by=None,
    )
    table = MagicMock()
    table.create_many = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=rows if rows is not None else [default_row])
    table.find_unique = AsyncMock(return_value=default_row)
    table.upsert = AsyncMock(return_value=default_row)

    prisma = MagicMock()
    prisma.db.litellm_tooltable = table
    return prisma


@pytest.mark.asyncio
async def test_batch_upsert_tools_calls_create_many():
    prisma = _make_prisma()
    items = [{"tool_name": "tool_a", "origin": "mcp_server", "created_by": None}]
    await batch_upsert_tools(prisma, items)
    prisma.db.litellm_tooltable.create_many.assert_awaited_once()
    call_kwargs = prisma.db.litellm_tooltable.create_many.call_args
    assert call_kwargs.kwargs["skip_duplicates"] is True


@pytest.mark.asyncio
async def test_batch_upsert_tools_empty_list():
    prisma = _make_prisma()
    await batch_upsert_tools(prisma, [])
    prisma.db.litellm_tooltable.create_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_tools_skips_empty_names():
    prisma = _make_prisma()
    items = [{"tool_name": "", "origin": None}, {"tool_name": None}]  # type: ignore[list-item]
    await batch_upsert_tools(prisma, items)
    prisma.db.litellm_tooltable.create_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_tools_no_filter():
    now = datetime.now(timezone.utc)
    row = MagicMock(
        tool_id="id1",
        tool_name="tool_a",
        origin="mcp",
        call_policy="untrusted",
        assignments={},
        created_at=now,
        updated_at=now,
        created_by=None,
        updated_by=None,
    )
    prisma = _make_prisma(rows=[row])
    result = await list_tools(prisma)
    assert len(result) == 1
    assert result[0].tool_name == "tool_a"


@pytest.mark.asyncio
async def test_list_tools_with_policy_filter():
    now = datetime.now(timezone.utc)
    row = MagicMock(
        tool_id="id1",
        tool_name="blocked_tool",
        origin=None,
        call_policy="blocked",
        assignments=None,
        created_at=now,
        updated_at=now,
        created_by=None,
        updated_by=None,
    )
    prisma = _make_prisma(rows=[row])
    result = await list_tools(prisma, call_policy="blocked")
    assert result[0].call_policy == "blocked"
    call_kwargs = prisma.db.litellm_tooltable.find_many.call_args
    assert call_kwargs.kwargs["where"] == {"call_policy": "blocked"}


@pytest.mark.asyncio
async def test_get_tool_found():
    prisma = _make_prisma()
    result = await get_tool(prisma, "my_tool")
    assert result is not None
    assert result.tool_name == "my_tool"


@pytest.mark.asyncio
async def test_get_tool_not_found():
    prisma = _make_prisma()
    prisma.db.litellm_tooltable.find_unique = AsyncMock(return_value=None)
    result = await get_tool(prisma, "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_tool_policy_upsert():
    prisma = _make_prisma()
    result = await update_tool_policy(prisma, "my_tool", "blocked", "admin")
    assert result is not None
    prisma.db.litellm_tooltable.upsert.assert_awaited_once()
    upsert_call = prisma.db.litellm_tooltable.upsert.call_args
    assert upsert_call.kwargs["data"]["update"]["call_policy"] == "blocked"
    assert upsert_call.kwargs["data"]["update"]["updated_by"] == "admin"


@pytest.mark.asyncio
async def test_get_tools_by_names_returns_policy_map():
    now = datetime.now(timezone.utc)
    rows = [
        MagicMock(tool_name="tool_a", call_policy="trusted"),
        MagicMock(tool_name="tool_b", call_policy="blocked"),
    ]
    prisma = _make_prisma(rows=rows)
    result = await get_tools_by_names(prisma, ["tool_a", "tool_b"])
    assert result == {"tool_a": "trusted", "tool_b": "blocked"}


@pytest.mark.asyncio
async def test_get_tools_by_names_empty_list():
    prisma = _make_prisma()
    result = await get_tools_by_names(prisma, [])
    assert result == {}
    prisma.db.litellm_tooltable.find_many.assert_not_awaited()

"""
Unit tests for tool_registry_writer.py — uses a mock prisma client
that exposes execute_raw / query_raw (matching the actual raw-SQL implementation).
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


def _make_prisma(query_rows=None):
    """Return a minimal mock prisma_client with execute_raw / query_raw."""
    default_row = {
        "tool_id": "uuid-1",
        "tool_name": "my_tool",
        "origin": "user_defined",
        "call_policy": "untrusted",
        "call_count": 1,
        "assignments": {},
        "key_hash": None,
        "team_id": None,
        "key_alias": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "created_by": None,
        "updated_by": None,
    }
    rows = query_rows if query_rows is not None else [default_row]

    prisma = MagicMock()
    prisma.db.execute_raw = AsyncMock(return_value=None)
    prisma.db.query_raw = AsyncMock(return_value=rows)
    return prisma


@pytest.mark.asyncio
async def test_batch_upsert_tools_calls_execute_raw():
    prisma = _make_prisma()
    items = [{"tool_name": "tool_a", "origin": "mcp_server", "created_by": None}]
    await batch_upsert_tools(prisma, items)
    prisma.db.execute_raw.assert_awaited_once()
    call_args = prisma.db.execute_raw.call_args
    sql = call_args.args[0]
    assert "LiteLLM_ToolTable" in sql
    assert "ON CONFLICT" in sql


@pytest.mark.asyncio
async def test_batch_upsert_tools_empty_list():
    prisma = _make_prisma()
    await batch_upsert_tools(prisma, [])
    prisma.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_tools_skips_empty_names():
    prisma = _make_prisma()
    items = [{"tool_name": "", "origin": None}, {"tool_name": None}]  # type: ignore[list-item]
    await batch_upsert_tools(prisma, items)
    prisma.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_multiple_tools_calls_execute_raw_per_tool():
    prisma = _make_prisma()
    items = [
        {"tool_name": "tool_a", "origin": "mcp_server", "created_by": None},
        {"tool_name": "tool_b", "origin": "user_defined", "created_by": "alice"},
    ]
    await batch_upsert_tools(prisma, items)
    assert prisma.db.execute_raw.await_count == 2


@pytest.mark.asyncio
async def test_list_tools_no_filter():
    row = {
        "tool_id": "id1",
        "tool_name": "tool_a",
        "origin": "mcp",
        "call_policy": "untrusted",
        "call_count": 5,
        "assignments": {},
        "key_hash": None,
        "team_id": None,
        "key_alias": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "created_by": None,
        "updated_by": None,
    }
    prisma = _make_prisma(query_rows=[row])
    result = await list_tools(prisma)
    assert len(result) == 1
    assert result[0].tool_name == "tool_a"
    assert result[0].call_count == 5
    prisma.db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_tools_with_policy_filter():
    row = {
        "tool_id": "id1",
        "tool_name": "blocked_tool",
        "origin": None,
        "call_policy": "blocked",
        "call_count": 2,
        "assignments": None,
        "key_hash": None,
        "team_id": None,
        "key_alias": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "created_by": None,
        "updated_by": None,
    }
    prisma = _make_prisma(query_rows=[row])
    result = await list_tools(prisma, call_policy="blocked")
    assert result[0].call_policy == "blocked"
    call_args = prisma.db.query_raw.call_args
    sql = call_args.args[0]
    assert "WHERE call_policy" in sql


@pytest.mark.asyncio
async def test_get_tool_found():
    prisma = _make_prisma()
    result = await get_tool(prisma, "my_tool")
    assert result is not None
    assert result.tool_name == "my_tool"
    prisma.db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_tool_not_found():
    prisma = _make_prisma(query_rows=[])
    result = await get_tool(prisma, "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_tool_policy_calls_execute_raw():
    row = {
        "tool_id": "uuid-1",
        "tool_name": "my_tool",
        "origin": "user_defined",
        "call_policy": "blocked",
        "call_count": 1,
        "assignments": {},
        "key_hash": None,
        "team_id": None,
        "key_alias": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "created_by": None,
        "updated_by": "admin",
    }
    prisma = _make_prisma(query_rows=[row])
    result = await update_tool_policy(prisma, "my_tool", "blocked", "admin")
    assert result is not None
    assert result.call_policy == "blocked"
    prisma.db.execute_raw.assert_awaited_once()
    call_args = prisma.db.execute_raw.call_args
    sql = call_args.args[0]
    assert "ON CONFLICT" in sql
    assert "call_policy" in sql


@pytest.mark.asyncio
async def test_get_tools_by_names_returns_policy_map():
    rows = [
        {"tool_name": "tool_a", "call_policy": "trusted"},
        {"tool_name": "tool_b", "call_policy": "blocked"},
    ]
    prisma = _make_prisma(query_rows=rows)
    result = await get_tools_by_names(prisma, ["tool_a", "tool_b"])
    assert result == {"tool_a": "trusted", "tool_b": "blocked"}


@pytest.mark.asyncio
async def test_get_tools_by_names_empty_list():
    prisma = _make_prisma()
    result = await get_tools_by_names(prisma, [])
    assert result == {}
    prisma.db.query_raw.assert_not_awaited()

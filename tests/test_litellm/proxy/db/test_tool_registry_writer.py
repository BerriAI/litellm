"""
Unit tests for tool_registry_writer.py — uses a mock prisma client
that exposes litellm_tooltable.upsert / find_many / find_unique.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.db.tool_registry_writer import (batch_upsert_tools,
                                                   get_tool,
                                                   get_tools_by_names,
                                                   list_tools,
                                                   update_tool_policy)


def _mock_row(**kwargs):
    """Build a row-like object with real attributes (no MagicMock) for _row_to_model."""

    class Row:
        pass

    default = {
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
    default.update(kwargs)
    row = Row()
    for k, v in default.items():
        setattr(row, k, v)
    return row


def _make_prisma(
    *,
    upsert_return=None,
    find_many_rows=None,
    find_unique_row=None,
):
    """Return a mock prisma_client with litellm_tooltable.upsert, find_many, find_unique."""
    prisma = MagicMock()
    prisma.db.litellm_tooltable = MagicMock()
    prisma.db.litellm_tooltable.upsert = AsyncMock(return_value=upsert_return)
    prisma.db.litellm_tooltable.find_many = AsyncMock(
        return_value=find_many_rows if find_many_rows is not None else []
    )
    prisma.db.litellm_tooltable.find_unique = AsyncMock(
        return_value=find_unique_row
    )
    return prisma


@pytest.mark.asyncio
async def test_batch_upsert_tools_calls_upsert():
    prisma = _make_prisma()
    items = [{"tool_name": "tool_a", "origin": "mcp_server", "created_by": None}]
    await batch_upsert_tools(prisma, items)
    prisma.db.litellm_tooltable.upsert.assert_awaited_once()
    call_kw = prisma.db.litellm_tooltable.upsert.call_args.kwargs
    assert call_kw["where"] == {"tool_name": "tool_a"}
    assert call_kw["data"]["create"]["tool_name"] == "tool_a"
    assert call_kw["data"]["create"]["origin"] == "mcp_server"
    assert call_kw["data"]["create"]["call_policy"] == "untrusted"
    assert call_kw["data"]["create"]["call_count"] == 1
    assert call_kw["data"]["update"]["call_count"] == {"increment": 1}
    assert "updated_at" in call_kw["data"]["update"]


@pytest.mark.asyncio
async def test_batch_upsert_tools_empty_list():
    prisma = _make_prisma()
    await batch_upsert_tools(prisma, [])
    prisma.db.litellm_tooltable.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_tools_skips_empty_names():
    prisma = _make_prisma()
    items = [{"tool_name": "", "origin": None}, {"tool_name": None}]  # type: ignore[list-item]
    await batch_upsert_tools(prisma, items)
    prisma.db.litellm_tooltable.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_upsert_multiple_tools_calls_upsert_per_tool():
    prisma = _make_prisma()
    items = [
        {"tool_name": "tool_a", "origin": "mcp_server", "created_by": None},
        {"tool_name": "tool_b", "origin": "user_defined", "created_by": "alice"},
    ]
    await batch_upsert_tools(prisma, items)
    assert prisma.db.litellm_tooltable.upsert.await_count == 2
    calls = prisma.db.litellm_tooltable.upsert.call_args_list
    assert calls[0].kwargs["where"]["tool_name"] == "tool_a"
    assert calls[1].kwargs["where"]["tool_name"] == "tool_b"


@pytest.mark.asyncio
async def test_list_tools_no_filter():
    row = _mock_row(
        tool_id="id1",
        tool_name="tool_a",
        origin="mcp",
        call_policy="untrusted",
        call_count=5,
    )
    prisma = _make_prisma(find_many_rows=[row])
    result = await list_tools(prisma)
    assert len(result) == 1
    assert result[0].tool_name == "tool_a"
    assert result[0].call_count == 5
    prisma.db.litellm_tooltable.find_many.assert_awaited_once()
    call_kw = prisma.db.litellm_tooltable.find_many.call_args.kwargs
    assert call_kw["where"] == {}
    assert call_kw["order"] == {"created_at": "desc"}


@pytest.mark.asyncio
async def test_list_tools_with_policy_filter():
    row = _mock_row(
        tool_id="id1",
        tool_name="blocked_tool",
        origin=None,
        call_policy="blocked",
        call_count=2,
        assignments=None,
    )
    prisma = _make_prisma(find_many_rows=[row])
    result = await list_tools(prisma, call_policy="blocked")
    assert result[0].call_policy == "blocked"
    call_kw = prisma.db.litellm_tooltable.find_many.call_args.kwargs
    assert call_kw["where"] == {"call_policy": "blocked"}


@pytest.mark.asyncio
async def test_get_tool_found():
    row = _mock_row(tool_name="my_tool")
    prisma = _make_prisma(find_unique_row=row)
    result = await get_tool(prisma, "my_tool")
    assert result is not None
    assert result.tool_name == "my_tool"
    prisma.db.litellm_tooltable.find_unique.assert_awaited_once_with(
        where={"tool_name": "my_tool"}
    )


@pytest.mark.asyncio
async def test_get_tool_not_found():
    prisma = _make_prisma(find_unique_row=None)
    result = await get_tool(prisma, "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_tool_policy_calls_upsert_then_get_tool():
    row = _mock_row(
        tool_name="my_tool",
        call_policy="blocked",
        updated_by="admin",
    )
    prisma = _make_prisma(find_unique_row=row)
    result = await update_tool_policy(prisma, "my_tool", "blocked", "admin")
    assert result is not None
    assert result.call_policy == "blocked"
    prisma.db.litellm_tooltable.upsert.assert_awaited_once()
    call_kw = prisma.db.litellm_tooltable.upsert.call_args.kwargs
    assert call_kw["where"] == {"tool_name": "my_tool"}
    assert call_kw["data"]["update"]["call_policy"] == "blocked"
    assert call_kw["data"]["update"]["updated_by"] == "admin"
    prisma.db.litellm_tooltable.find_unique.assert_awaited_with(
        where={"tool_name": "my_tool"}
    )


@pytest.mark.asyncio
async def test_get_tools_by_names_returns_policy_map():
    rows = [
        _mock_row(tool_name="tool_a", call_policy="trusted"),
        _mock_row(tool_name="tool_b", call_policy="blocked"),
    ]
    prisma = _make_prisma(find_many_rows=rows)
    result = await get_tools_by_names(prisma, ["tool_a", "tool_b"])
    assert result == {"tool_a": "trusted", "tool_b": "blocked"}
    prisma.db.litellm_tooltable.find_many.assert_awaited_once_with(
        where={"tool_name": {"in": ["tool_a", "tool_b"]}}
    )


@pytest.mark.asyncio
async def test_get_tools_by_names_empty_list():
    prisma = _make_prisma()
    result = await get_tools_by_names(prisma, [])
    assert result == {}
    prisma.db.litellm_tooltable.find_many.assert_not_awaited()

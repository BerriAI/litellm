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

from litellm.proxy.db.tool_registry_writer import (
    ToolPolicyRegistry,
    batch_upsert_tools,
    get_tool,
    get_tool_policy_registry,
    get_tools_by_names,
    list_tools,
    update_tool_policy,
)


def _mock_row(**kwargs):
    """Build a row-like object with real attributes (no MagicMock) for _row_to_model."""

    class Row:
        pass

    default = {
        "tool_id": "uuid-1",
        "tool_name": "my_tool",
        "origin": "user_defined",
        "input_policy": "untrusted",
        "output_policy": "untrusted",
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
    prisma.db.litellm_tooltable.find_unique = AsyncMock(return_value=find_unique_row)
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
    assert call_kw["data"]["create"]["input_policy"] == "untrusted"
    assert call_kw["data"]["create"]["output_policy"] == "untrusted"
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
        input_policy="untrusted",
        output_policy="untrusted",
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
async def test_list_tools_with_input_policy_filter():
    row = _mock_row(
        tool_id="id1",
        tool_name="blocked_tool",
        origin=None,
        input_policy="blocked",
        output_policy="untrusted",
        call_count=2,
        assignments=None,
    )
    prisma = _make_prisma(find_many_rows=[row])
    result = await list_tools(prisma, input_policy="blocked")
    assert result[0].input_policy == "blocked"
    call_kw = prisma.db.litellm_tooltable.find_many.call_args.kwargs
    assert call_kw["where"] == {"input_policy": "blocked"}


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
        input_policy="blocked",
        output_policy="untrusted",
        updated_by="admin",
    )
    prisma = _make_prisma(find_unique_row=row)
    result = await update_tool_policy(
        prisma, "my_tool", updated_by="admin", input_policy="blocked"
    )
    assert result is not None
    assert result.input_policy == "blocked"
    prisma.db.litellm_tooltable.upsert.assert_awaited_once()
    call_kw = prisma.db.litellm_tooltable.upsert.call_args.kwargs
    assert call_kw["where"] == {"tool_name": "my_tool"}
    assert call_kw["data"]["update"]["input_policy"] == "blocked"
    assert call_kw["data"]["update"]["updated_by"] == "admin"
    prisma.db.litellm_tooltable.find_unique.assert_awaited_with(
        where={"tool_name": "my_tool"}
    )


@pytest.mark.asyncio
async def test_get_tools_by_names_returns_policy_map():
    rows = [
        _mock_row(
            tool_name="tool_a", input_policy="trusted", output_policy="untrusted"
        ),
        _mock_row(
            tool_name="tool_b", input_policy="blocked", output_policy="untrusted"
        ),
    ]
    prisma = _make_prisma(find_many_rows=rows)
    result = await get_tools_by_names(prisma, ["tool_a", "tool_b"])
    assert result == {
        "tool_a": ("trusted", "untrusted"),
        "tool_b": ("blocked", "untrusted"),
    }
    prisma.db.litellm_tooltable.find_many.assert_awaited_once_with(
        where={"tool_name": {"in": ["tool_a", "tool_b"]}}
    )


@pytest.mark.asyncio
async def test_get_tools_by_names_empty_list():
    prisma = _make_prisma()
    result = await get_tools_by_names(prisma, [])
    assert result == {}
    prisma.db.litellm_tooltable.find_many.assert_not_awaited()


# --- ToolPolicyRegistry ---


def _mock_tool_row(
    tool_name: str,
    input_policy: str = "untrusted",
    output_policy: str = "untrusted",
):
    row = MagicMock()
    row.tool_name = tool_name
    row.input_policy = input_policy
    row.output_policy = output_policy
    return row


def _mock_perm_row(object_permission_id: str, blocked_tools: list):
    row = MagicMock()
    row.object_permission_id = object_permission_id
    row.blocked_tools = blocked_tools
    return row


@pytest.mark.asyncio
async def test_tool_policy_registry_sync_and_get_effective_policies():
    """Registry syncs from DB; get_effective_policies returns merged blocked + global."""
    prisma = MagicMock()
    prisma.db.litellm_tooltable.find_many = AsyncMock(
        return_value=[
            _mock_tool_row("tool_a", input_policy="trusted"),
            _mock_tool_row("tool_b", input_policy="blocked"),
            _mock_tool_row("tool_c", input_policy="untrusted"),
        ]
    )
    prisma.db.litellm_objectpermissiontable.find_many = AsyncMock(
        return_value=[
            _mock_perm_row("op-key-1", ["tool_a"]),
            _mock_perm_row("op-team-1", ["tool_c"]),
        ]
    )
    registry = get_tool_policy_registry()
    await registry.sync_tool_policy_from_db(prisma)
    assert registry.is_initialized()
    # Key blocked: tool_a. Team blocked: tool_c. Global: tool_b blocked.
    result = registry.get_effective_policies(
        ["tool_a", "tool_b", "tool_c"],
        object_permission_id="op-key-1",
        team_object_permission_id="op-team-1",
    )
    assert result["tool_a"] == "blocked"
    assert result["tool_b"] == "blocked"
    assert result["tool_c"] == "blocked"
    # No op ids: only global
    result_global = registry.get_effective_policies(["tool_a", "tool_b", "tool_c"])
    assert result_global["tool_a"] == "trusted"
    assert result_global["tool_b"] == "blocked"
    assert result_global["tool_c"] == "untrusted"


@pytest.mark.asyncio
async def test_tool_policy_registry_not_initialized_returns_untrusted():
    """When not synced, get_effective_policies still returns untrusted for unknown tools."""
    registry = ToolPolicyRegistry()
    assert not registry.is_initialized()
    result = registry.get_effective_policies(["unknown_tool"])
    assert result == {"unknown_tool": "untrusted"}


@pytest.mark.asyncio
async def test_sync_tool_policy_from_db_retries_on_transport_error_first_read():
    """`ToolPolicyRegistry.sync_tool_policy_from_db` self-heals across one
    ClientNotConnectedError on the tools read — the perms read still fires
    after the recovery and the registry initializes cleanly."""
    import prisma as prisma_pkg

    registry = ToolPolicyRegistry()
    invocations: list = []

    async def _flaky_find_many():
        invocations.append(None)
        if len(invocations) == 1:
            raise prisma_pkg.errors.ClientNotConnectedError()
        return []

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_tooltable.find_many = AsyncMock(
        side_effect=_flaky_find_many
    )
    mock_prisma_client.db.litellm_objectpermissiontable.find_many = AsyncMock(
        return_value=[]
    )
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    mock_prisma_client._db_auth_reconnect_timeout_seconds = 2.0
    mock_prisma_client._db_auth_reconnect_lock_timeout_seconds = 0.1

    await registry.sync_tool_policy_from_db(mock_prisma_client)

    assert len(invocations) == 2
    mock_prisma_client.attempt_db_reconnect.assert_awaited_once()
    reconnect_kwargs = mock_prisma_client.attempt_db_reconnect.await_args.kwargs
    assert (
        reconnect_kwargs["reason"]
        == "sync_tool_policy_from_db_tools_lookup_failure"
    )
    assert registry.is_initialized()


@pytest.mark.asyncio
async def test_sync_tool_policy_from_db_retries_on_transport_error_second_read():
    """Same as above but the blip happens on the perms read — distinct reason
    tag in telemetry confirms the second wrap is wired separately."""
    import prisma as prisma_pkg

    registry = ToolPolicyRegistry()
    perms_invocations: list = []

    async def _flaky_perms_find_many():
        perms_invocations.append(None)
        if len(perms_invocations) == 1:
            raise prisma_pkg.errors.ClientNotConnectedError()
        return []

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_tooltable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_objectpermissiontable.find_many = AsyncMock(
        side_effect=_flaky_perms_find_many
    )
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    mock_prisma_client._db_auth_reconnect_timeout_seconds = 2.0
    mock_prisma_client._db_auth_reconnect_lock_timeout_seconds = 0.1

    await registry.sync_tool_policy_from_db(mock_prisma_client)

    assert len(perms_invocations) == 2
    mock_prisma_client.attempt_db_reconnect.assert_awaited_once()
    reconnect_kwargs = mock_prisma_client.attempt_db_reconnect.await_args.kwargs
    assert (
        reconnect_kwargs["reason"]
        == "sync_tool_policy_from_db_perms_lookup_failure"
    )

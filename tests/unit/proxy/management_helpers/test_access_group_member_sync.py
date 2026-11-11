from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_helpers.access_group_member_sync import (
    sync_access_groups_for_deleted_agent,
    sync_access_groups_for_deleted_mcp_server,
)

_INVALIDATE = "litellm.proxy.management_helpers.access_group_member_sync.invalidate_access_group_caches"


def _routed_prisma_client(touched_group_ids, writer_unavailable: bool = False):
    rows = [{"access_group_id": group_id} for group_id in touched_group_ids]
    writer_inner = MagicMock(name="writer_prisma")
    reader_inner = MagicMock(name="reader_prisma")
    writer_inner.query_raw = AsyncMock(return_value=rows)
    reader_inner.query_raw = AsyncMock(return_value=rows)
    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    routing._writer_unavailable = writer_unavailable
    return SimpleNamespace(db=routing), writer_inner, reader_inner


@pytest.mark.asyncio
async def test_deleted_agent_is_removed_from_every_group_that_lists_it():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(["ag-1", "ag-2"])

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_agent(prisma_client, agent_id="agent-123")

    (update_call,) = writer_inner.query_raw.await_args_list
    sql = update_call.args[0]
    assert sql.startswith('UPDATE "LiteLLM_AccessGroupTable"')
    assert 'array_remove("access_agent_ids", $1)' in sql
    assert '$1 = ANY("access_agent_ids")' in sql
    assert "access_mcp_server_ids" not in sql
    assert update_call.args[1:] == ("agent-123",)
    invalidate.assert_awaited_once_with(("ag-1", "ag-2"))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_mcp_server_is_removed_from_the_id_list_and_the_tool_permissions():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(["ag-1"])

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_mcp_server(prisma_client, server_id="server-9")

    (update_call,) = writer_inner.query_raw.await_args_list
    sql = update_call.args[0]
    assert sql.startswith('UPDATE "LiteLLM_AccessGroupTable"')
    assert 'array_remove("access_mcp_server_ids", $1)' in sql
    assert '"mcp_tool_permissions" - $1::text' in sql
    # A group can scope tools for a server it no longer lists, so the key alone must match too.
    assert '$1 = ANY("access_mcp_server_ids")' in sql
    assert 'jsonb_exists("mcp_tool_permissions", $1::text)' in sql
    assert "access_agent_ids" not in sql
    assert update_call.args[1:] == ("server-9",)
    invalidate.assert_awaited_once_with(("ag-1",))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_server_removal_leaves_a_json_null_tool_permissions_column_alone():
    prisma_client, writer_inner, _ = _routed_prisma_client(["ag-1"])

    with patch(_INVALIDATE, new=AsyncMock()):
        await sync_access_groups_for_deleted_mcp_server(prisma_client, server_id="server-9")

    sql = writer_inner.query_raw.await_args.args[0]
    # `jsonb - text` raises on a scalar; groups without tool scopes store a JSON null.
    assert "WHEN jsonb_typeof(\"mcp_tool_permissions\") = 'object' THEN" in sql
    assert 'ELSE "mcp_tool_permissions" END' in sql


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sync, kwargs, member_id",
    [
        (sync_access_groups_for_deleted_agent, {"agent_id": "agent-123"}, "agent-123"),
        (sync_access_groups_for_deleted_mcp_server, {"server_id": "server-9"}, "server-9"),
    ],
)
async def test_removal_stays_on_the_writer_while_writer_flagged_unavailable(sync, kwargs, member_id):
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(["ag-1"], writer_unavailable=True)

    with patch(_INVALIDATE, new=AsyncMock()):
        await sync(prisma_client, **kwargs)

    (update_call,) = writer_inner.query_raw.await_args_list
    assert update_call.args[1:] == (member_id,)
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_cache_is_dropped_when_no_group_listed_the_deleted_resource():
    prisma_client, writer_inner, _ = _routed_prisma_client([])

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_agent(prisma_client, agent_id="agent-unlisted")

    (update_call,) = writer_inner.query_raw.await_args_list
    assert update_call.args[1:] == ("agent-unlisted",)
    assert invalidate.await_args.args == ((),)

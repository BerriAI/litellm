from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_helpers.access_group_member_prune import (
    remove_deleted_agent_from_access_groups,
    remove_deleted_mcp_server_from_access_groups,
)

_INVALIDATE = "litellm.proxy.management_helpers.access_group_member_prune.invalidate_access_group_caches"


def _routed_prisma_client(touched_group_ids: tuple[str, ...], writer_unavailable: bool = False):
    writer_inner = MagicMock(name="writer_prisma")
    reader_inner = MagicMock(name="reader_prisma")
    rows = [{"access_group_id": group_id} for group_id in touched_group_ids]
    writer_inner.query_raw = AsyncMock(return_value=rows)
    reader_inner.query_raw = AsyncMock(return_value=rows)
    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    routing._writer_unavailable = writer_unavailable
    return SimpleNamespace(db=routing), writer_inner, reader_inner


@pytest.mark.asyncio
async def test_deleted_agent_is_removed_from_every_group_listing_it_and_those_caches_drop():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(("ag-1", "ag-2"))

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await remove_deleted_agent_from_access_groups(prisma_client, "agent-1")

    (update_call,) = writer_inner.query_raw.await_args_list
    assert 'array_remove("access_agent_ids", $1)' in update_call.args[0]
    assert update_call.args[1:] == ("agent-1",)
    invalidate.assert_awaited_once_with(("ag-1", "ag-2"))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_mcp_server_is_removed_from_server_ids_and_tool_permissions():
    prisma_client, writer_inner, _ = _routed_prisma_client(("ag-1",))

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await remove_deleted_mcp_server_from_access_groups(prisma_client, "srv-1")

    (update_call,) = writer_inner.query_raw.await_args_list
    sql = update_call.args[0]
    assert 'array_remove("access_mcp_server_ids", $1)' in sql
    assert '"mcp_tool_permissions" = "mcp_tool_permissions" - $1' in sql
    assert 'jsonb_exists("mcp_tool_permissions", $1)' in sql
    assert update_call.args[1:] == ("srv-1",)
    invalidate.assert_awaited_once_with(("ag-1",))


@pytest.mark.asyncio
async def test_prune_stays_on_the_writer_while_writer_flagged_unavailable():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client((), writer_unavailable=True)

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await remove_deleted_agent_from_access_groups(prisma_client, "agent-1")

    writer_inner.query_raw.assert_awaited_once()
    reader_inner.query_raw.assert_not_awaited()
    invalidate.assert_awaited_once_with(())

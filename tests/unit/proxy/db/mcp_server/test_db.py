from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


from litellm.proxy._experimental.mcp_server.db import (
    approve_mcp_server,
    get_mcp_servers_by_team,
    reject_mcp_server,
)


def _prisma_client_returning(team_record: object) -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_record)
    return prisma_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_record, expected",
    [
        (None, []),
        (SimpleNamespace(object_permission=None), []),
        (SimpleNamespace(object_permission=SimpleNamespace(mcp_servers=None)), []),
        (SimpleNamespace(object_permission=SimpleNamespace(mcp_servers=[])), []),
        (
            SimpleNamespace(
                object_permission=SimpleNamespace(mcp_servers=["server_a", "server_b"])
            ),
            ["server_a", "server_b"],
        ),
    ],
)
async def test_fetch_mcp_servers_by_team(team_record, expected):
    prisma_client = _prisma_client_returning(team_record)

    assert await get_mcp_servers_by_team(prisma_client, "team-123") == expected

    prisma_client.db.litellm_teamtable.find_unique.assert_awaited_once_with(
        where={"team_id": "team-123"},
        include={"object_permission": True},
    )


def _prisma_client_with_missing_mcp_server_row() -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.update = AsyncMock(return_value=None)
    return prisma_client


@pytest.mark.asyncio
async def test_approve_mcp_server_raises_value_error_when_row_missing():
    prisma_client = _prisma_client_with_missing_mcp_server_row()

    with pytest.raises(ValueError, match=r"^MCP server not found, passed server_id=server-gone$"):
        await approve_mcp_server(prisma_client, "server-gone", touched_by="admin")


@pytest.mark.asyncio
async def test_reject_mcp_server_raises_value_error_when_row_missing():
    prisma_client = _prisma_client_with_missing_mcp_server_row()

    with pytest.raises(ValueError, match=r"^MCP server not found, passed server_id=server-gone$"):
        await reject_mcp_server(
            prisma_client,
            "server-gone",
            touched_by="admin",
            review_notes="spam",
        )

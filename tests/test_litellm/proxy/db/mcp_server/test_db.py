from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


from litellm.proxy._experimental.mcp_server.db import get_mcp_servers_by_team


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

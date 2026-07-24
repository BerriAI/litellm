import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.db import get_mcp_servers_by_team
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable


def _make_prisma_with_team(team_record):
    """Build a MagicMock prisma_client whose team table returns ``team_record``
    for find_unique (the only call get_mcp_servers_by_team makes)."""
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_record)
    return prisma


@pytest.mark.asyncio
async def test_fetch_mcp_servers_by_team():
    team_mcp_servers = ["github-mcp", "slack-mcp"]
    team_record = LiteLLM_TeamTable(team_id="team-1", object_permission_id="perm-1")
    team_record.object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-1",
        mcp_servers=team_mcp_servers,
    )
    prisma = _make_prisma_with_team(team_record)

    result = await get_mcp_servers_by_team(prisma, "team-1")

    assert result == team_mcp_servers
    prisma.db.litellm_teamtable.find_unique.assert_awaited_once_with(
        where={"team_id": "team-1"},
        include={"object_permission": True},
    )


@pytest.mark.asyncio
async def test_fetch_mcp_servers_by_team_no_team():
    prisma = _make_prisma_with_team(None)

    result = await get_mcp_servers_by_team(prisma, "missing-team")

    assert result == []


@pytest.mark.asyncio
async def test_fetch_mcp_servers_by_team_no_object_permission():
    team_record = LiteLLM_TeamTable(team_id="team-1")

    prisma = _make_prisma_with_team(team_record)

    result = await get_mcp_servers_by_team(prisma, "team-1")

    assert result == []

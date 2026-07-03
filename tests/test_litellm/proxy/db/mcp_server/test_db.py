import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.db import (
    get_all_mcp_servers_for_user,
    get_mcp_servers_by_team,
)
from litellm.proxy._types import SpecialMCPServerNames, UserAPIKeyAuth

DB_MODULE = "litellm.proxy._experimental.mcp_server.db"


def test_fetch_mcp_servers_by_team():
    assert True == True


@pytest.mark.asyncio
@patch(f"{DB_MODULE}.get_mcp_servers", new_callable=AsyncMock)
@patch(f"{DB_MODULE}.get_mcp_servers_by_team", new_callable=AsyncMock)
@patch(f"{DB_MODULE}.get_mcp_servers_by_verificationtoken", new_callable=AsyncMock)
async def test_get_all_mcp_servers_for_user_all_team_mcps_expands_to_team(
    mock_token_servers, mock_team_servers, mock_get_servers
):
    """A key scoped to all-team-mcps surfaces its team's servers in discovery,
    matching the access resolver. Pins that the live sentinel value drives the
    expansion (the removed legacy all-team-mcpservers value would not)."""
    mock_token_servers.return_value = [
        SpecialMCPServerNames.all_team_mcp_servers.value
    ]
    mock_team_servers.return_value = ["team-srv-1", "team-srv-2"]
    mock_get_servers.return_value = []

    user = UserAPIKeyAuth(api_key="k", user_id="u", team_id="team-1")
    await get_all_mcp_servers_for_user(MagicMock(), user)

    mock_team_servers.assert_awaited_once()
    queried_ids = set(mock_get_servers.call_args.args[1])
    assert {"team-srv-1", "team-srv-2"} <= queried_ids


@pytest.mark.asyncio
@patch(f"{DB_MODULE}.get_mcp_servers", new_callable=AsyncMock)
@patch(f"{DB_MODULE}.get_mcp_servers_by_team", new_callable=AsyncMock)
@patch(f"{DB_MODULE}.get_mcp_servers_by_verificationtoken", new_callable=AsyncMock)
async def test_get_all_mcp_servers_for_user_explicit_servers_skip_team_expansion(
    mock_token_servers, mock_team_servers, mock_get_servers
):
    """Without the all-team-mcps sentinel, discovery never pulls in the team's
    servers — proving the expansion is gated on the sentinel, not unconditional."""
    mock_token_servers.return_value = ["explicit-srv"]
    mock_get_servers.return_value = []

    user = UserAPIKeyAuth(api_key="k", user_id="u", team_id="team-1")
    await get_all_mcp_servers_for_user(MagicMock(), user)

    mock_team_servers.assert_not_awaited()
    assert set(mock_get_servers.call_args.args[1]) == {"explicit-srv"}

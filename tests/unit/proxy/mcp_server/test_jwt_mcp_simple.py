"""
Simple test to validate MCP permissions are enforced when calling MCP routes with JWT.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)


@pytest.mark.asyncio
async def test_simple_jwt_mcp_permissions_enforced():
    """
    Simple test: Call MCP route with JWT, verify team's MCP servers are returned.

    Setup:
    - Team "my-team" has MCP servers: ["github-mcp", "slack-mcp"]
    - JWT user belongs to "my-team"

    Expected: Only ["github-mcp", "slack-mcp"] should be allowed
    """
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    # 1. Create a user authenticated via JWT with team_id set
    user_auth = UserAPIKeyAuth(
        api_key=None,  # JWT auth doesn't have api_key
        user_id="jwt-user-123",
        team_id="my-team",  # This is set by JWT auth when team is in groups
    )

    # 2. Team's MCP permissions
    team_mcp_servers = ["github-mcp", "slack-mcp"]
    team_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-123",
        mcp_servers=team_mcp_servers,
    )
    team_obj = LiteLLM_TeamTable(
        team_id="my-team",
        access_group_ids=[],
        object_permission_id="perm-123",
    )
    team_obj.object_permission = team_object_permission

    # 3. Mock the team object lookup (object_permission attached) and prisma_client
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=team_obj,
        ) as mock_get_team,
        patch.object(
            MCPRequestHandler,
            "_get_key_object_permission",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            MCPRequestHandler,
            "get_mcp_servers_from_access_groups",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        # 4. Call get_allowed_mcp_servers - this is what MCP routes use
        allowed = await MCPRequestHandler.get_allowed_mcp_servers(user_auth)

        # 5. Verify only team's MCP servers are returned
        assert sorted(allowed) == sorted(team_mcp_servers), f"Expected {team_mcp_servers}, got {allowed}"

        # Verify team was looked up
        mock_get_team.assert_called()


@pytest.mark.asyncio
async def test_simple_jwt_no_team_no_mcp_servers():
    """
    Simple test: JWT user with no team should get no MCP servers.
    """
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    # User with no team_id (JWT didn't have teams in groups)
    user_auth = UserAPIKeyAuth(
        api_key=None,
        user_id="jwt-user-no-team",
        team_id=None,  # No team
    )

    # _get_allowed_mcp_servers_for_team returns [] when team_id is None
    allowed = await MCPRequestHandler._get_allowed_mcp_servers_for_team(user_auth)

    assert allowed == [], f"Expected [], got {allowed}"


@pytest.mark.asyncio
async def test_simple_jwt_team_id_required_for_mcp_permissions():
    """
    Simple test: Verify that team_id must be set for team MCP permissions to work.

    This is the key insight - if JWT auth doesn't set team_id,
    team MCP permissions won't be enforced.
    """
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    # Case 1: team_id is set -> team permissions should be checked
    user_with_team = UserAPIKeyAuth(
        api_key=None,
        user_id="user-1",
        team_id="team-abc",
    )

    team_mcp_servers = ["server-1", "server-2"]
    team_perm = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-1",
        mcp_servers=team_mcp_servers,
    )
    team_obj = LiteLLM_TeamTable(
        team_id="team-abc",
        access_group_ids=[],
        object_permission_id="perm-1",
    )
    team_obj.object_permission = team_perm

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=team_obj,
        ) as mock_get_team,
        patch.object(
            MCPRequestHandler,
            "get_mcp_servers_from_access_groups",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(user_with_team)

        assert sorted(result) == sorted(team_mcp_servers)
        mock_get_team.assert_called()  # Team WAS looked up

    # Case 2: team_id is None -> team permissions NOT checked
    user_without_team = UserAPIKeyAuth(
        api_key=None,
        user_id="user-2",
        team_id=None,
    )

    result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(user_without_team)
    assert result == []  # No permissions returned

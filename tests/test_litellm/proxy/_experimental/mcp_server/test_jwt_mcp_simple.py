"""
Simple test to validate MCP permissions are enforced when calling MCP routes with JWT.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_TeamTable,
    LiteLLM_ObjectPermissionTable,
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
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
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
    
    # 3. Mock the team permission lookup
    with patch.object(
        MCPRequestHandler, "_get_team_object_permission", new_callable=AsyncMock
    ) as mock_team_perm:
        mock_team_perm.return_value = team_object_permission
        
        # Mock key permissions (empty - user has no key-level MCP permissions)
        with patch.object(
            MCPRequestHandler, "_get_key_object_permission", new_callable=AsyncMock
        ) as mock_key_perm:
            mock_key_perm.return_value = None
            
            # Mock access groups (empty)
            with patch.object(
                MCPRequestHandler, "_get_mcp_servers_from_access_groups", new_callable=AsyncMock
            ) as mock_access_groups:
                mock_access_groups.return_value = []
                
                # 4. Call get_allowed_mcp_servers - this is what MCP routes use
                allowed = await MCPRequestHandler.get_allowed_mcp_servers(user_auth)
                
                # 5. Verify only team's MCP servers are returned
                assert sorted(allowed) == sorted(team_mcp_servers), (
                    f"Expected {team_mcp_servers}, got {allowed}"
                )
                
                # Verify team permission was looked up
                mock_team_perm.assert_called_once_with(user_auth)


@pytest.mark.asyncio
async def test_simple_jwt_no_team_no_mcp_servers():
    """
    Simple test: JWT user with no team should get no MCP servers.
    """
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
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
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
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
    
    with patch.object(
        MCPRequestHandler, "_get_team_object_permission", new_callable=AsyncMock
    ) as mock_perm:
        mock_perm.return_value = team_perm
        
        with patch.object(
            MCPRequestHandler, "_get_mcp_servers_from_access_groups", new_callable=AsyncMock
        ) as mock_groups:
            mock_groups.return_value = []
            
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(user_with_team)
            
            assert sorted(result) == sorted(team_mcp_servers)
            mock_perm.assert_called_once()  # Permission WAS checked
    
    # Case 2: team_id is None -> team permissions NOT checked
    user_without_team = UserAPIKeyAuth(
        api_key=None,
        user_id="user-2",
        team_id=None,
    )
    
    result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(user_without_team)
    assert result == []  # No permissions returned


@pytest.mark.asyncio 
async def test_jwt_auth_sets_team_id_for_mcp_route():
    """
    Test that JWT auth properly sets team_id when accessing MCP routes.
    
    This is the critical test - when user calls /mcp/tools/list with JWT,
    the team_id from JWT groups must be set on UserAPIKeyAuth.
    """
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    
    # Setup
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_ids_jwt_field="groups",  # Teams come from "groups" field in JWT
    )
    
    # Team exists with models
    team = LiteLLM_TeamTable(
        team_id="team-from-jwt",
        models=["gpt-4"],
    )
    
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
    
    # Mock JWT token with team in groups
    jwt_payload = {
        "sub": "user-123",
        "groups": ["team-from-jwt"],
        "scope": "",
    }
    
    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth:
        mock_auth.return_value = jwt_payload
        
        with patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_team:
            mock_get_team.return_value = team
            
            # Simulate calling MCP route
            result = await JWTAuthManager.auth_builder(
                api_key="jwt-token",
                jwt_handler=jwt_handler,
                request_data={},
                general_settings={},
                route="/mcp/tools/list",  # MCP route
                prisma_client=None,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
            )
            
            # THE KEY ASSERTION: team_id must be set
            assert result["team_id"] == "team-from-jwt", (
                f"team_id should be 'team-from-jwt' but got '{result['team_id']}'. "
                "This means JWT auth is not properly setting team_id for MCP routes!"
            )


@pytest.mark.asyncio 
async def test_mcp_route_without_model_still_returns_team_id():
    """
    Test that MCP routes (which don't specify a model) still get team_id assigned.
    
    Key insight: MCP routes don't require a model in the request, but the JWT auth
    flow must still assign a team_id so that team MCP permissions are enforced.
    
    The flow is:
    1. JWT token contains team in "groups" field
    2. find_team_with_model_access() is called with requested_model=None
    3. Since `not requested_model` is True, model check passes
    4. Route check passes because "mcp_routes" is in team_allowed_routes
    5. team_id is returned and set on UserAPIKeyAuth
    """
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    
    # Setup
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_ids_jwt_field="groups",
    )
    
    # Team exists - note: models is a list (can be empty or have values)
    # The key is that when no model is requested, model check is skipped
    team = LiteLLM_TeamTable(
        team_id="my-team",
        models=["gpt-4", "gpt-3.5-turbo"],  # Team has models, but MCP request won't specify one
    )
    
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
    
    # JWT with team in groups
    jwt_payload = {
        "sub": "user-abc",
        "groups": ["my-team"],
        "scope": "",
    }
    
    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth:
        mock_auth.return_value = jwt_payload
        
        with patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_team:
            mock_get_team.return_value = team
            
            # Call MCP route with NO MODEL in request_data
            result = await JWTAuthManager.auth_builder(
                api_key="jwt-token",
                jwt_handler=jwt_handler,
                request_data={},  # <-- NO MODEL SPECIFIED
                general_settings={},
                route="/mcp/tools/list",  # MCP route
                prisma_client=None,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
            )
            
            # Team ID must still be set even though no model was requested
            assert result["team_id"] == "my-team", (
                f"Expected team_id='my-team' but got '{result['team_id']}'. "
                "MCP routes without model should still get team_id from JWT!"
            )

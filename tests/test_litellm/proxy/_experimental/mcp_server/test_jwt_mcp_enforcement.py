"""
Test to verify Team MCP permissions are enforced when using JWT authentication.

Scenario:
1. Team "ABC" exists with models configured and MCPs assigned
2. User JWT has team "ABC" in groups (via team_ids_jwt_field)
3. Call MCP list endpoint
4. EXPECTED: Team MCP permissions should be enforced
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_TeamTable,
    LiteLLM_ObjectPermissionTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager


@pytest.mark.asyncio
async def test_reproduce_jwt_mcp_enforcement_issue(monkeypatch):
    """
    Reproduce the bug where Team MCP permissions are NOT enforced when using JWT.
    
    Setup:
    - Team "ABC" has models ["gpt-4"] and MCPs ["mcp-server-1"] assigned
    - JWT has team "ABC" in groups field
    - User calls MCP list endpoint (no model requested)
    
    Expected: team_id should be set to "ABC" so MCP permissions are enforced
    Actual (BUG): team_id is None because route check fails for MCP routes
    """
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

    # Setup mock router
    router = Router(model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])
    import sys
    import types
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    # Team "ABC" has models configured AND MCPs assigned
    team_with_mcp = LiteLLM_TeamTable(
        team_id="ABC",
        models=["gpt-4"],  # Team HAS models
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-123",
            mcp_servers=["mcp-server-1"],  # Team has MCPs assigned
        ),
    )

    async def mock_get_team_object(*args, **kwargs):
        team_id = kwargs.get("team_id") or args[0]
        if team_id == "ABC":
            return team_with_mcp
        return None

    monkeypatch.setattr(
        "litellm.proxy.auth.handle_jwt.get_team_object", mock_get_team_object
    )

    # Setup JWT handler with team_ids_jwt_field (groups)
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_ids_jwt_field="groups",  # Use groups field for teams
        # NOTE: team_allowed_routes defaults to ["openai_routes", "info_routes"]
        # which does NOT include "mcp_routes"
    )

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    # Simulate JWT payload with team in groups
    jwt_token = {
        "sub": "user-123",
        "groups": ["ABC"],  # Team "ABC" is in groups
        "scope": "",
    }

    # Mock auth_jwt to return our token
    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt:
        mock_auth_jwt.return_value = jwt_token

        # Call auth_builder for MCP route (like /mcp/tools/list)
        result = await JWTAuthManager.auth_builder(
            api_key="test-jwt-token",
            jwt_handler=jwt_handler,
            request_data={},  # No model in request (MCP endpoint)
            general_settings={},
            route="/mcp/tools/list",  # MCP route
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # THIS IS THE BUG: team_id should be "ABC" but it's None!
        print(f"Result team_id: {result['team_id']}")
        print(f"Result team_object: {result['team_object']}")
        
        # The test should FAIL if the bug exists (team_id is None)
        # If the fix is applied, team_id should be "ABC"
        assert result["team_id"] == "ABC", (
            f"BUG: team_id should be 'ABC' but got '{result['team_id']}'. "
            f"This happens because default team_allowed_routes does not include 'mcp_routes', "
            f"so allowed_routes_check() fails and the team is skipped in find_team_with_model_access()."
        )


@pytest.mark.asyncio  
async def test_verify_mcp_routes_in_default_team_allowed_routes():
    """
    Verify that mcp_routes IS in the default team_allowed_routes.
    This is required for team MCP permissions to work with JWT auth.
    """
    default_jwt_auth = LiteLLM_JWTAuth()
    
    print(f"Default team_allowed_routes: {default_jwt_auth.team_allowed_routes}")
    
    # mcp_routes must be in defaults for team MCP permissions to work
    assert "mcp_routes" in default_jwt_auth.team_allowed_routes, (
        "mcp_routes must be in default team_allowed_routes for JWT MCP enforcement to work"
    )


@pytest.mark.asyncio
async def test_mcp_route_check_passes_for_team():
    """
    Verify that allowed_routes_check returns True for MCP routes with default settings.
    This is required for teams to access MCP endpoints with JWT auth.
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.auth_checks import allowed_routes_check
    
    jwt_auth = LiteLLM_JWTAuth()  # Use defaults
    
    # Check if MCP route is allowed for TEAM role
    is_allowed = allowed_routes_check(
        user_role=LitellmUserRoles.TEAM,
        user_route="/mcp/tools/list",
        litellm_proxy_roles=jwt_auth,
    )
    
    print(f"Is /mcp/tools/list allowed for TEAM with defaults? {is_allowed}")
    
    # MCP routes should be allowed by default for teams
    assert is_allowed is True, (
        "MCP routes must be allowed by default for teams for JWT MCP enforcement to work"
    )


@pytest.mark.asyncio
async def test_e2e_jwt_team_mcp_permissions_enforced(monkeypatch):
    """
    End-to-end test verifying that team MCP permissions are properly enforced
    when using JWT authentication with teams in groups.
    
    This test verifies the complete flow:
    1. JWT token contains team "ABC" in groups field
    2. Team "ABC" exists with MCP servers ["mcp-server-1", "mcp-server-2"] assigned
    3. JWT auth properly sets team_id on UserAPIKeyAuth
    4. MCPRequestHandler.get_allowed_mcp_servers() returns team's MCP servers
    """
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

    # Setup mock router
    router = Router(model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])
    import sys
    import types
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    proxy_server_module.prisma_client = MagicMock()  # Mock prisma client
    proxy_server_module.user_api_key_cache = DualCache()
    proxy_server_module.proxy_logging_obj = MagicMock()
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    # Team "ABC" has MCP servers assigned via object_permission
    team_mcp_servers = ["mcp-server-1", "mcp-server-2"]
    team_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-abc-123",
        mcp_servers=team_mcp_servers,
        mcp_access_groups=[],
        vector_stores=[],
    )
    
    team_with_mcp = LiteLLM_TeamTable(
        team_id="ABC",
        models=["gpt-4"],
        object_permission=team_object_permission,
        object_permission_id="perm-abc-123",
    )

    async def mock_get_team_object(*args, **kwargs):
        team_id = kwargs.get("team_id") or (args[0] if args else None)
        if team_id == "ABC":
            return team_with_mcp
        return None

    monkeypatch.setattr(
        "litellm.proxy.auth.handle_jwt.get_team_object", mock_get_team_object
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_team_object", mock_get_team_object
    )

    # Setup JWT handler with team_ids_jwt_field (groups)
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_ids_jwt_field="groups",
    )

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    # Simulate JWT payload with team in groups
    jwt_token = {
        "sub": "user-123",
        "groups": ["ABC"],
        "scope": "",
    }

    # Step 1: Verify JWT auth returns correct team_id
    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt:
        mock_auth_jwt.return_value = jwt_token

        result = await JWTAuthManager.auth_builder(
            api_key="test-jwt-token",
            jwt_handler=jwt_handler,
            request_data={},
            general_settings={},
            route="/mcp/tools/list",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Verify team_id is set correctly
        assert result["team_id"] == "ABC", f"Expected team_id='ABC', got '{result['team_id']}'"
        assert result["team_object"] is not None, "team_object should not be None"
        
        # Step 2: Create UserAPIKeyAuth with the team_id from JWT auth
        user_api_key_auth = UserAPIKeyAuth(
            api_key=None,
            team_id=result["team_id"],
            user_id=result["user_id"],
        )
        
        # Step 3: Verify MCPRequestHandler returns team's MCP servers
        # Mock _get_team_object_permission to return our team's object_permission
        with patch.object(
            MCPRequestHandler, "_get_team_object_permission"
        ) as mock_get_team_perm:
            mock_get_team_perm.return_value = team_object_permission
            
            # Mock _get_allowed_mcp_servers_for_key to return empty (no key-level permissions)
            with patch.object(
                MCPRequestHandler, "_get_allowed_mcp_servers_for_key"
            ) as mock_key_servers:
                mock_key_servers.return_value = []
                
                # Mock _get_mcp_servers_from_access_groups to return empty
                with patch.object(
                    MCPRequestHandler, "_get_mcp_servers_from_access_groups"
                ) as mock_access_groups:
                    mock_access_groups.return_value = []
                    
                    allowed_servers = await MCPRequestHandler.get_allowed_mcp_servers(
                        user_api_key_auth
                    )
                    
                    print(f"Allowed MCP servers: {allowed_servers}")
                    
                    # Verify team's MCP servers are returned
                    assert set(allowed_servers) == set(team_mcp_servers), (
                        f"Expected team MCP servers {team_mcp_servers}, got {allowed_servers}"
                    )


@pytest.mark.asyncio
async def test_e2e_jwt_without_team_no_mcp_servers(monkeypatch):
    """
    End-to-end test verifying that when JWT has no teams, no MCP servers are returned.
    
    This ensures:
    1. JWT token with no groups returns no team_id
    2. MCPRequestHandler.get_allowed_mcp_servers() returns empty list
    """
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

    # Setup mock router
    router = Router(model_list=[])
    import sys
    import types
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    async def mock_get_team_object(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.proxy.auth.handle_jwt.get_team_object", mock_get_team_object
    )

    # Setup JWT handler
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_ids_jwt_field="groups",
    )

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    # JWT payload with empty groups
    jwt_token = {
        "sub": "user-123",
        "groups": [],  # No teams
        "scope": "",
    }

    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt:
        mock_auth_jwt.return_value = jwt_token

        result = await JWTAuthManager.auth_builder(
            api_key="test-jwt-token",
            jwt_handler=jwt_handler,
            request_data={},
            general_settings={},
            route="/mcp/tools/list",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Verify no team_id is set
        assert result["team_id"] is None, f"Expected team_id=None, got '{result['team_id']}'"
        
        # Create UserAPIKeyAuth without team_id
        user_api_key_auth = UserAPIKeyAuth(
            api_key=None,
            team_id=None,
            user_id=result["user_id"],
        )
        
        # Verify no MCP servers are returned when there's no team
        allowed_servers = await MCPRequestHandler._get_allowed_mcp_servers_for_team(
            user_api_key_auth
        )
        
        assert allowed_servers == [], f"Expected empty list, got {allowed_servers}"


@pytest.mark.asyncio
async def test_e2e_jwt_team_mcp_key_intersection(monkeypatch):
    """
    End-to-end test verifying MCP permission intersection between key and team.
    
    Scenario:
    - Team has MCP servers: ["server-1", "server-2", "server-3"]
    - Key has MCP servers: ["server-2", "server-4"]
    - Result should be intersection: ["server-2"]
    """
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

    # Setup mock router
    router = Router(model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])
    import sys
    import types
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    proxy_server_module.prisma_client = MagicMock()
    proxy_server_module.user_api_key_cache = DualCache()
    proxy_server_module.proxy_logging_obj = MagicMock()
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    # Team MCP servers
    team_mcp_servers = ["server-1", "server-2", "server-3"]
    team_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="team-perm",
        mcp_servers=team_mcp_servers,
    )
    
    team_with_mcp = LiteLLM_TeamTable(
        team_id="TEAM-X",
        models=["gpt-4"],
        object_permission=team_object_permission,
    )

    # Key MCP servers
    key_mcp_servers = ["server-2", "server-4"]
    key_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="key-perm",
        mcp_servers=key_mcp_servers,
    )

    async def mock_get_team_object(*args, **kwargs):
        team_id = kwargs.get("team_id") or (args[0] if args else None)
        if team_id == "TEAM-X":
            return team_with_mcp
        return None

    monkeypatch.setattr(
        "litellm.proxy.auth.handle_jwt.get_team_object", mock_get_team_object
    )

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_ids_jwt_field="groups")

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_token = {"sub": "user-123", "groups": ["TEAM-X"], "scope": ""}

    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt:
        mock_auth_jwt.return_value = jwt_token

        result = await JWTAuthManager.auth_builder(
            api_key="test-jwt-token",
            jwt_handler=jwt_handler,
            request_data={},
            general_settings={},
            route="/mcp/tools/list",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        assert result["team_id"] == "TEAM-X"
        
        user_api_key_auth = UserAPIKeyAuth(
            api_key=None,
            team_id=result["team_id"],
            user_id=result["user_id"],
            object_permission=key_object_permission,  # Key has its own permissions
        )
        
        # Mock the helper methods to return our test data
        with patch.object(
            MCPRequestHandler, "_get_team_object_permission"
        ) as mock_team_perm:
            mock_team_perm.return_value = team_object_permission
            
            with patch.object(
                MCPRequestHandler, "_get_key_object_permission"
            ) as mock_key_perm:
                mock_key_perm.return_value = key_object_permission
                
                with patch.object(
                    MCPRequestHandler, "_get_mcp_servers_from_access_groups"
                ) as mock_access_groups:
                    mock_access_groups.return_value = []
                    
                    allowed_servers = await MCPRequestHandler.get_allowed_mcp_servers(
                        user_api_key_auth
                    )
                    
                    # Should be intersection: only server-2 is in both
                    expected = ["server-2"]
                    assert sorted(allowed_servers) == sorted(expected), (
                        f"Expected intersection {expected}, got {allowed_servers}"
                    )

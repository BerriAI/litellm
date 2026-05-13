import os
import sys
import types
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm._uuid import uuid
from litellm.proxy.management_endpoints import (
    mcp_management_endpoints as mgmt_endpoints,
)

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LitellmUserRoles,
    MCPTransport,
    NewMCPServerRequest,
    UpdateMCPServerRequest,
    UserAPIKeyAuth,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def generate_mock_mcp_server_db_record(
    server_id: Optional[str] = None,
    alias: str = "Test DB Server",
    url: str = "https://db-server.example.com/mcp",
    transport: str = "sse",
    auth_type: Optional[str] = None,
) -> LiteLLM_MCPServerTable:
    """Generate a mock MCP server record from database"""
    now = datetime.now()
    return LiteLLM_MCPServerTable(
        server_id=server_id or str(uuid.uuid4()),
        alias=alias,
        url=url,
        transport=MCPTransport.sse if transport == "sse" else MCPTransport.http,
        auth_type=MCPAuth.api_key if auth_type == "api_key" else None,
        created_at=now,
        updated_at=now,
        created_by="test_user",
        updated_by="test_user",
    )


def generate_mock_mcp_server_config_record(
    server_id: Optional[str] = None,
    name: str = "Test Config Server",
    url: str = "https://config-server.example.com/mcp",
    transport: str = "http",
    auth_type: Optional[str] = None,
) -> MCPServer:
    """Generate a mock MCP server record from config.yaml"""
    return MCPServer(
        server_id=server_id or str(uuid.uuid4()),
        name=name,
        alias=name,  # Set alias to match the name for consistency with tests
        server_name=name,
        url=url,
        transport=MCPTransport.http if transport == "http" else MCPTransport.sse,
        auth_type=MCPAuth.api_key if auth_type == "api_key" else None,
        mcp_info={
            "server_name": name,
            "description": "Config server description",
        },
    )


def _make_mock_request(ip: str = "127.0.0.1"):
    """Create a mock Request for fetch_mcp_server tests (IP used for access control)."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = ip
    req.headers = {}
    return req


def generate_mock_user_api_key_auth(
    user_role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN,
    user_id: str = "test_user_id",
    api_key: str = "test_api_key",
    team_id: Optional[str] = None,
) -> UserAPIKeyAuth:
    """Generate a mock UserAPIKeyAuth object"""
    return UserAPIKeyAuth(
        user_role=user_role,
        user_id=user_id,
        api_key=api_key,
        team_id=team_id,
    )


def generate_mock_team_record(
    team_id: str, team_alias: str, organization_id: str, mcp_servers: List[str]
):
    """Generate a mock team record with object permissions"""
    return MagicMock(
        team_id=team_id,
        team_alias=team_alias,
        organization_id=organization_id,
        members_with_roles=[{"user_id": "test_user_id"}],
        object_permission=MagicMock(mcp_servers=mcp_servers),
    )


def setup_mock_prisma_client(
    mock_prisma_client: MagicMock,
    team_records: List[MagicMock],
    mcp_servers: List[LiteLLM_MCPServerTable],
):
    """Helper to set up a mock prisma client with proper async behavior"""
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_teamtable = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.find_many = AsyncMock(
        return_value=team_records
    )
    mock_prisma_client.db.litellm_mcpservertable = AsyncMock()
    mock_prisma_client.db.litellm_mcpservertable.find_many = AsyncMock(
        return_value=mcp_servers
    )
    return mock_prisma_client


def create_mcp_router_test_client() -> TestClient:
    from litellm.proxy.management_endpoints.mcp_management_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def patch_proxy_general_settings(settings: dict):
    fake_proxy_server_module = types.SimpleNamespace(general_settings=settings)
    return patch.dict(
        sys.modules,
        {"litellm.proxy.proxy_server": fake_proxy_server_module},
    )


class TestListMCPServers:
    """Test suite for list MCP servers functionality"""

    @pytest.mark.asyncio
    async def test_list_mcp_servers_config_yaml_only(self):
        """
        Test 1: Returns MCPs defined on the config.yaml only

        Scenario: No DB MCPs, only config.yaml MCPs
        Expected: Should return only config.yaml MCPs
        """
        # Mock dependencies
        mock_prisma_client = MagicMock()
        mock_prisma_client = setup_mock_prisma_client(
            mock_prisma_client=mock_prisma_client,
            team_records=[
                generate_mock_team_record(
                    team_id="team1",
                    team_alias="Team 1",
                    organization_id="org1",
                    mcp_servers=["config_server_1", "config_server_2"],
                )
            ],
            mcp_servers=[],  # No DB servers in this test
        )
        mock_user_auth = generate_mock_user_api_key_auth()

        # Mock config MCPs
        config_server_1 = generate_mock_mcp_server_config_record(
            server_id="config_server_1",
            name="Zapier MCP",
            url="https://actions.zapier.com/mcp/sse",
            transport="sse",
        )
        config_server_2 = generate_mock_mcp_server_config_record(
            server_id="config_server_2",
            name="DeepWiki MCP",
            url="https://mcp.deepwiki.com/mcp",
            transport="http",
        )

        # Mock global MCP server manager
        mock_manager = MagicMock()
        mock_manager.config_mcp_servers = {
            "config_server_1": config_server_1,
            "config_server_2": config_server_2,
        }
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["config_server_1", "config_server_2"]
        )

        # Mock the new method that returns servers without health check
        mock_servers = [
            generate_mock_mcp_server_db_record(
                server_id="config_server_1",
                alias="Zapier MCP",
                url="https://actions.zapier.com/mcp/sse",
                transport="sse",
            ),
            generate_mock_mcp_server_db_record(
                server_id="config_server_2",
                alias="DeepWiki MCP",
                url="https://mcp.deepwiki.com/mcp",
                transport="http",
            ),
        ]
        mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=mock_servers)

        for idx, server in enumerate(mock_servers):
            server.credentials = {"auth_value": f"secret_{idx}"}

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results
            assert len(result) == 2
            assert all(server.credentials is None for server in result)

            # Check that both config servers are returned
            server_ids = [server.server_id for server in result]
            assert "config_server_1" in server_ids
            assert "config_server_2" in server_ids

            # Check server details
            for server in result:
                if server.server_id == "config_server_1":
                    assert server.alias == "Zapier MCP"
                    assert server.url == "https://actions.zapier.com/mcp/sse"
                    assert server.transport == "sse"
                elif server.server_id == "config_server_2":
                    assert server.alias == "DeepWiki MCP"
                    assert server.url == "https://mcp.deepwiki.com/mcp"
                    assert server.transport == "http"

    @pytest.mark.asyncio
    async def test_list_mcp_servers_view_all_mode(self):
        """Users should see all MCP servers when view_all mode is enabled."""

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER
        )

        mock_servers = [
            generate_mock_mcp_server_db_record(server_id="server-1", alias="One"),
            generate_mock_mcp_server_db_record(server_id="server-2", alias="Two"),
        ]

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_unfiltered = AsyncMock(
            return_value=mock_servers
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="view_all",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            assert len(result) == 2
            assert {server.server_id for server in result} == {"server-1", "server-2"}

    @pytest.mark.asyncio
    async def test_list_mcp_servers_view_all_mode_virtual_key_is_sanitized(self):
        """Issue #20325: virtual keys should get a safe discovery view."""

        mock_user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="test_user_id",
            api_key="test_api_key",
            allowed_routes=["mcp_routes"],
        )

        mock_servers = [
            generate_mock_mcp_server_db_record(server_id="server-1", alias="One"),
            generate_mock_mcp_server_db_record(server_id="server-2", alias="Two"),
        ]
        for idx, server in enumerate(mock_servers):
            server.credentials = {"auth_value": f"secret_{idx}"}
            server.env = {"API_KEY": "super-secret"}
            server.static_headers = {"Authorization": "Bearer super-secret"}
            server.mcp_access_groups = ["group-a"]
            server.teams = [{"team_id": "team-1", "team_alias": "Team 1"}]
            server.command = "bash"
            server.args = ["-lc", "echo hi"]
            server.extra_headers = ["Authorization"]

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_unfiltered = AsyncMock(
            return_value=mock_servers
        )
        mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=mock_servers)

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="view_all",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Ensure we did not bypass filtering via view_all for restricted virtual keys.
            mock_manager.get_all_mcp_servers_unfiltered.assert_not_called()

            assert len(result) == 2
            assert {server.server_id for server in result} == {"server-1", "server-2"}

            for server in result:
                assert server.credentials is None
                assert server.url is None
                assert server.static_headers is None
                assert server.env == {}
                assert server.command is None
                assert server.args == []
                assert server.extra_headers == []
                assert server.allowed_tools == []
                assert server.mcp_access_groups == []
                assert server.teams == []

    @pytest.mark.asyncio
    async def test_list_mcp_servers_combined_config_and_db(self):
        """
        Test 2: If both config.yaml and DB then combines both and returns the result

        Scenario: Both DB and config.yaml have MCPs
        Expected: Should return combined list from both sources without duplicates
        """
        # Mock DB MCPs
        db_server_1 = generate_mock_mcp_server_db_record(
            server_id="db_server_1",
            alias="DB Gmail MCP",
            url="https://gmail-mcp.example.com/mcp",
            transport="sse",
        )
        db_server_2 = generate_mock_mcp_server_db_record(
            server_id="db_server_2",
            alias="DB Slack MCP",
            url="https://slack-mcp.example.com/mcp",
            transport="http",
        )

        # Mock dependencies
        mock_prisma_client = MagicMock()
        mock_prisma_client = setup_mock_prisma_client(
            mock_prisma_client=mock_prisma_client,
            team_records=[
                generate_mock_team_record(
                    team_id="team1",
                    team_alias="Team 1",
                    organization_id="org1",
                    mcp_servers=[
                        "db_server_1",
                        "db_server_2",
                        "config_server_1",
                        "config_server_2",
                    ],
                )
            ],
            mcp_servers=[db_server_1, db_server_2],  # DB servers for this test
        )
        mock_user_auth = generate_mock_user_api_key_auth()

        # Mock config MCPs
        config_server_1 = generate_mock_mcp_server_config_record(
            server_id="config_server_1",
            name="Zapier MCP",
            url="https://actions.zapier.com/mcp/sse",
            transport="sse",
        )
        config_server_2 = generate_mock_mcp_server_config_record(
            server_id="config_server_2",
            name="DeepWiki MCP",
            url="https://mcp.deepwiki.com/mcp",
            transport="http",
        )

        # Mock global MCP server manager
        mock_manager = MagicMock()
        mock_manager.config_mcp_servers = {
            "config_server_1": config_server_1,
            "config_server_2": config_server_2,
        }
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=[
                "db_server_1",
                "db_server_2",
                "config_server_1",
                "config_server_2",
            ]
        )

        # Mock the new method that returns servers without health check
        mock_servers = [
            db_server_1,
            db_server_2,
            generate_mock_mcp_server_db_record(
                server_id="config_server_1",
                alias="Zapier MCP",
                url="https://actions.zapier.com/mcp/sse",
                transport="sse",
            ),
            generate_mock_mcp_server_db_record(
                server_id="config_server_2",
                alias="DeepWiki MCP",
                url="https://mcp.deepwiki.com/mcp",
                transport="http",
            ),
        ]
        mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=mock_servers)

        for idx, server in enumerate(mock_servers):
            server.credentials = {"auth_value": f"secret_{idx}"}

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results
            assert len(result) == 4
            assert all(server.credentials is None for server in result)

            # Check that both DB and config servers are returned
            server_ids = [server.server_id for server in result]
            assert "db_server_1" in server_ids
            assert "db_server_2" in server_ids
            assert "config_server_1" in server_ids
            assert "config_server_2" in server_ids

            # Check server details
            for server in result:
                if server.server_id == "db_server_1":
                    assert server.alias == "DB Gmail MCP"
                    assert server.url == "https://gmail-mcp.example.com/mcp"
                    assert server.transport == "sse"
                elif server.server_id == "db_server_2":
                    assert server.alias == "DB Slack MCP"
                    assert server.url == "https://slack-mcp.example.com/mcp"
                    assert server.transport == "http"
                elif server.server_id == "config_server_1":
                    assert server.alias == "Zapier MCP"
                    assert server.url == "https://actions.zapier.com/mcp/sse"
                    assert server.transport == "sse"
                elif server.server_id == "config_server_2":
                    assert server.alias == "DeepWiki MCP"
                    assert server.url == "https://mcp.deepwiki.com/mcp"
                    assert server.transport == "http"

    @pytest.mark.asyncio
    async def test_list_mcp_servers_non_admin_user_filtered(self):
        """
        Test 3: Non-admin users only see MCPs they have access to

        Scenario: Non-admin user with limited access
        Expected: Should return only MCPs the user has access to
        """
        # Mock DB MCPs - user only has access to one
        db_server_allowed = generate_mock_mcp_server_db_record(
            server_id="db_server_allowed",
            alias="Allowed Gmail MCP",
            url="https://gmail-mcp.example.com/mcp",
        )

        # Mock dependencies
        mock_prisma_client = MagicMock()
        mock_prisma_client = setup_mock_prisma_client(
            mock_prisma_client=mock_prisma_client,
            team_records=[
                generate_mock_team_record(
                    team_id="team1",
                    team_alias="Team 1",
                    organization_id="org1",
                    mcp_servers=["db_server_allowed", "config_server_allowed"],
                )
            ],
            mcp_servers=[db_server_allowed],  # Only the allowed DB server
        )
        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,  # Non-admin user
            team_id="team_123",
        )

        # Mock config MCPs - user has access to one
        config_server_allowed = generate_mock_mcp_server_config_record(
            server_id="config_server_allowed",
            name="Allowed Zapier MCP",
            url="https://actions.zapier.com/mcp/sse",
        )

        # Mock global MCP server manager
        mock_manager = MagicMock()
        mock_manager.config_mcp_servers = {
            "config_server_allowed": config_server_allowed,
            "config_server_not_allowed": generate_mock_mcp_server_config_record(
                server_id="config_server_not_allowed"
            ),
        }
        # User only has access to specific servers
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["db_server_allowed", "config_server_allowed"]
        )

        # Mock the new method that returns servers without health check
        mock_servers = [
            db_server_allowed,
            generate_mock_mcp_server_db_record(
                server_id="config_server_allowed",
                alias="Allowed Zapier MCP",
                url="https://actions.zapier.com/mcp/sse",
            ),
        ]
        mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=mock_servers)

        for idx, server in enumerate(mock_servers):
            server.credentials = {"auth_value": f"secret_{idx}"}

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=False,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results - should only return servers user has access to
            assert len(result) == 2
            assert all(server.credentials is None for server in result)

            # Check that only allowed servers are returned
            server_ids = [server.server_id for server in result]
            assert "db_server_allowed" in server_ids
            assert "config_server_allowed" in server_ids
            assert "config_server_not_allowed" not in server_ids

            # Check server details — non-admin viewers must not see the
            # raw `url` (it can carry bearer tokens for many MCP
            # integrations). Identity fields stay so the UI can list
            # the server.
            for server in result:
                assert server.url is None
                if server.server_id == "db_server_allowed":
                    assert server.alias == "Allowed Gmail MCP"
                elif server.server_id == "config_server_allowed":
                    assert server.alias == "Allowed Zapier MCP"

    @pytest.mark.asyncio
    async def test_admin_user_with_object_permission_respects_mcp_servers(self):
        """
        Test that admin users with explicit object_permission.mcp_servers
        only see the servers specified in object_permission.

        Scenario: Admin user has object_permission.mcp_servers set to specific servers
        Expected: Only those servers are returned, not all servers in the registry
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        # Create mock object permission with specific servers
        mock_object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="test-obj-perm-id",
            mcp_servers=["server-1", "server-2"],  # Only these two servers
            mcp_access_groups=[],
            mcp_tool_permissions={},
            vector_stores=[],
            agents=[],
            agent_access_groups=[],
        )

        # Create admin user with object permission
        mock_user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id="admin_user_id",
            api_key="admin_api_key",
            object_permission=mock_object_permission,
            object_permission_id="test-obj-perm-id",
        )

        # Mock servers that the user should see
        server_1 = generate_mock_mcp_server_db_record(
            server_id="server-1", alias="Server 1", url="https://server1.example.com"
        )
        server_2 = generate_mock_mcp_server_db_record(
            server_id="server-2", alias="Server 2", url="https://server2.example.com"
        )

        # Mock manager
        mock_manager = MagicMock()
        mock_manager.get_all_allowed_mcp_servers = AsyncMock(
            return_value=[server_1, server_2]
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results - should only return the 2 servers in object_permission
            assert len(result) == 2
            server_ids = {server.server_id for server in result}
            assert server_ids == {"server-1", "server-2"}

            # Verify credentials are redacted
            assert all(server.credentials is None for server in result)

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_redacts_credentials(self):
        mock_server = generate_mock_mcp_server_db_record(
            server_id="server-1", alias="Server 1"
        )
        mock_server.credentials = {"auth_value": "top-secret"}

        mock_prisma_client = MagicMock()

        # Mock health check result as LiteLLM_MCPServerTable
        mock_health_result = generate_mock_mcp_server_db_record(
            server_id="server-1", alias="Server 1"
        )
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=mock_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.health_check_server",
                AsyncMock(return_value=mock_health_result),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(),
                server_id="server-1",
                user_api_key_dict=mock_user_auth,
            )

            assert result.server_id == "server-1"
            assert result.credentials is None
            assert mock_server.credentials == {"auth_value": "top-secret"}
            assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_handles_missing_credentials_field(self):
        mock_server = generate_mock_mcp_server_db_record(
            server_id="server-2", alias="Server 2"
        )
        # Simulate ORM object without credentials attribute (e.g., older schema)
        delattr(mock_server, "credentials")

        mock_prisma_client = MagicMock()

        # Mock health check result as LiteLLM_MCPServerTable
        mock_health_result = generate_mock_mcp_server_db_record(
            server_id="server-2", alias="Server 2"
        )
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=mock_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.health_check_server",
                AsyncMock(return_value=mock_health_result),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(),
                server_id="server-2",
                user_api_key_dict=mock_user_auth,
            )

            assert result.server_id == "server-2"
            # credentials attribute should still be absent and no exception raised
            assert not hasattr(result, "credentials")
            assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_from_registry_config_based(self):
        """
        Test that fetch_mcp_server finds config-based servers when not in DB.
        Config servers appear in list via get_registry() but were 404 on fetch.
        """
        config_server = generate_mock_mcp_server_config_record(
            server_id="serper_custom_dev",
            name="Serper MCP",
            url="https://serper.example.com/mcp",
            transport="http",
        )

        mock_health_result = generate_mock_mcp_server_db_record(
            server_id="serper_custom_dev", alias="Serper MCP"
        )
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda sid: (
                config_server if sid == "serper_custom_dev" else None
            )
        )
        mock_manager.get_mcp_server_by_name = MagicMock(return_value=None)
        mock_manager._build_mcp_server_table = MagicMock(
            return_value=generate_mock_mcp_server_db_record(
                server_id="serper_custom_dev",
                alias="Serper MCP",
                url="https://serper.example.com/mcp",
                transport="http",
            )
        )
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["serper_custom_dev"]
        )
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(),
                server_id="serper_custom_dev",
                user_api_key_dict=mock_user_auth,
            )

            assert result.server_id == "serper_custom_dev"
            assert result.status == "healthy"
            mock_manager.get_mcp_server_by_id.assert_called_with("serper_custom_dev")
            mock_manager._build_mcp_server_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_from_registry_by_name_passes_client_ip(self):
        """
        When lookup by server_id fails, fallback to get_mcp_server_by_name.
        Verify client_ip is passed for IP-based access control (security).
        """
        config_server = generate_mock_mcp_server_config_record(
            server_id="serper_custom_dev",
            name="Serper MCP",
            url="https://serper.example.com/mcp",
            transport="http",
        )

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(return_value=None)
        mock_manager.get_mcp_server_by_name = MagicMock(return_value=config_server)
        mock_manager._build_mcp_server_table = MagicMock(
            return_value=generate_mock_mcp_server_db_record(
                server_id="serper_custom_dev",
                alias="Serper MCP",
                url="https://serper.example.com/mcp",
                transport="http",
            )
        )
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["serper_custom_dev"]
        )
        mock_manager.health_check_server = AsyncMock(
            return_value=generate_mock_mcp_server_db_record(
                server_id="serper_custom_dev", alias="Serper MCP"
            )
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(ip="192.168.1.100"),
                server_id="Serper MCP",
                user_api_key_dict=mock_user_auth,
            )

            assert result.server_id == "serper_custom_dev"
            mock_manager.get_mcp_server_by_id.assert_called_with("Serper MCP")
            mock_manager.get_mcp_server_by_name.assert_called_once_with(
                "Serper MCP", client_ip="192.168.1.100"
            )

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_from_registry_non_admin_denied(self):
        """
        Non-admin user: config server NOT in allowed_server_ids -> 403.
        """
        config_server = generate_mock_mcp_server_config_record(
            server_id="restricted_server",
            name="Restricted MCP",
            url="https://restricted.example.com/mcp",
            transport="http",
        )

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda sid: (
                config_server if sid == "restricted_server" else None
            )
        )
        mock_manager.get_mcp_server_by_name = MagicMock(return_value=None)
        mock_manager._build_mcp_server_table = MagicMock(
            return_value=generate_mock_mcp_server_db_record(
                server_id="restricted_server",
                alias="Restricted MCP",
                url="https://restricted.example.com/mcp",
                transport="http",
            )
        )
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["other_server"]  # restricted_server NOT in list
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=False,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            with pytest.raises(HTTPException) as exc_info:
                await fetch_mcp_server(
                    request=_make_mock_request(),
                    server_id="restricted_server",
                    user_api_key_dict=mock_user_auth,
                )

            assert exc_info.value.status_code == 403
            mock_manager.get_allowed_mcp_servers.assert_called_once_with(mock_user_auth)

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_from_registry_non_admin_granted(self):
        """
        Non-admin user: config server IS in allowed_server_ids -> 200.
        """
        config_server = generate_mock_mcp_server_config_record(
            server_id="allowed_config_server",
            name="Allowed MCP",
            url="https://allowed.example.com/mcp",
            transport="http",
        )

        mock_health_result = generate_mock_mcp_server_db_record(
            server_id="allowed_config_server", alias="Allowed MCP"
        )
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda sid: (
                config_server if sid == "allowed_config_server" else None
            )
        )
        mock_manager.get_mcp_server_by_name = MagicMock(return_value=None)
        mock_manager._build_mcp_server_table = MagicMock(
            return_value=generate_mock_mcp_server_db_record(
                server_id="allowed_config_server",
                alias="Allowed MCP",
                url="https://allowed.example.com/mcp",
                transport="http",
            )
        )
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["allowed_config_server"]
        )
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=False,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(),
                server_id="allowed_config_server",
                user_api_key_dict=mock_user_auth,
            )

            assert result.server_id == "allowed_config_server"
            assert result.status == "healthy"
            mock_manager.get_allowed_mcp_servers.assert_called_once_with(mock_user_auth)


class TestTeamScopedMCPServerAccess:
    """Tests for cross-team information disclosure and restricted key bypass fixes."""

    @pytest.mark.asyncio
    async def test_non_member_cannot_query_foreign_team(self):
        """Non-admin user who is NOT a member of the target team should get 403."""
        from litellm.proxy._types import Member

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="attacker_user",
        )

        # Team with a different member
        mock_team_obj = MagicMock()
        mock_team_obj.members_with_roles = [
            Member(user_id="legitimate_user", role="admin"),
        ]

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=False,
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                AsyncMock(return_value=mock_team_obj),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            with pytest.raises(HTTPException) as exc_info:
                await fetch_all_mcp_servers(
                    user_api_key_dict=mock_user_auth, team_id="foreign-team-id"
                )
            assert exc_info.value.status_code == 403
            assert "permission" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_team_member_can_query_own_team(self):
        """User who IS a member of the team should be able to query it."""
        from litellm.proxy._types import Member

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="team_member",
        )

        mock_team_obj = MagicMock()
        mock_team_obj.members_with_roles = [
            Member(user_id="team_member", role="user"),
        ]
        mock_team_obj.object_permission = MagicMock(mcp_servers=["server-1"])

        mock_server = generate_mock_mcp_server_config_record(
            server_id="server-1", name="Team Server"
        )
        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(return_value=mock_server)
        mock_manager._build_mcp_server_table = MagicMock(
            return_value=generate_mock_mcp_server_db_record(server_id="server-1")
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=False,
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                AsyncMock(return_value=mock_team_obj),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_team_scoped_mcp_server_list",
                AsyncMock(
                    return_value=[
                        generate_mock_mcp_server_db_record(server_id="server-1")
                    ]
                ),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(
                user_api_key_dict=mock_user_auth, team_id="my-team-id"
            )
            assert len(result) == 1
            assert result[0].server_id == "server-1"

    @pytest.mark.asyncio
    async def test_admin_can_query_any_team(self):
        """Proxy admins should be able to query any team's MCP servers."""
        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id="admin_user",
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
                return_value=True,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_team_scoped_mcp_server_list",
                AsyncMock(
                    return_value=[
                        generate_mock_mcp_server_db_record(server_id="server-1")
                    ]
                ),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            # Admin should NOT need to be a team member
            result = await fetch_all_mcp_servers(
                user_api_key_dict=mock_user_auth, team_id="any-team-id"
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_restricted_virtual_key_cannot_use_team_id_filter(self):
        """Restricted virtual keys must not bypass access limits via team_id."""
        mock_user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="vkey_user",
            api_key="sk-restricted",
            allowed_routes=["mcp_routes"],
        )

        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            fetch_all_mcp_servers,
        )

        with pytest.raises(HTTPException) as exc_info:
            await fetch_all_mcp_servers(
                user_api_key_dict=mock_user_auth, team_id="some-team"
            )
        assert exc_info.value.status_code == 403
        assert "Restricted virtual key" in str(exc_info.value.detail)


class TestTemporaryMCPSessionEndpoints:
    def test_inherit_credentials_from_existing_server(self):
        payload = NewMCPServerRequest(
            server_id="server-123",
            alias="Temp Server",
            url="https://temp.example.com",
            transport=MCPTransport.http,
        )
        existing_server = MagicMock()
        existing_server.authentication_token = "token-abc"
        existing_server.client_id = "client-123"
        existing_server.client_secret = "secret-xyz"
        existing_server.scopes = ["scope:a", "scope:b"]
        existing_server.aws_access_key_id = None
        existing_server.aws_secret_access_key = None
        existing_server.aws_session_token = None
        existing_server.aws_region_name = None
        existing_server.aws_service_name = None

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = existing_server

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                _inherit_credentials_from_existing_server,
            )

            updated_payload = _inherit_credentials_from_existing_server(payload)

        assert updated_payload.credentials == {
            "auth_value": "token-abc",
            "client_id": "client-123",
            "client_secret": "secret-xyz",
            "scopes": ["scope:a", "scope:b"],
        }
        mock_manager.get_mcp_server_by_id.assert_called_once_with("server-123")

    def test_cache_temporary_mcp_server_stores_entry_with_ttl(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _cache_temporary_mcp_server,
        )

        server = generate_mock_mcp_server_config_record(server_id="temp-cache")
        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._temporary_mcp_servers",
            {},
        ) as cache:
            cached_server = _cache_temporary_mcp_server(server, ttl_seconds=2)

        assert cached_server is server
        assert "temp-cache" in cache
        assert cache["temp-cache"].server is server
        assert cache["temp-cache"].expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_prunes_expired_entries(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _TemporaryMCPServerEntry,
            get_cached_temporary_mcp_server,
        )

        server = generate_mock_mcp_server_config_record(server_id="expired")
        expired_entry = _TemporaryMCPServerEntry(
            server=server,
            expires_at=datetime.utcnow() - timedelta(seconds=30),
        )
        cache = {"expired": expired_entry}
        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._temporary_mcp_servers",
            cache,
        ):
            result = await get_cached_temporary_mcp_server("expired")

        assert result is None
        assert "expired" not in cache

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_or_404(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_cached_temporary_mcp_server_or_404,
        )

        server = generate_mock_mcp_server_config_record(server_id="cached")
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
            return_value=server,
        ) as get_cached:
            result = await _get_cached_temporary_mcp_server_or_404("cached", admin_auth)

        assert result is server
        get_cached.assert_awaited_once_with("cached")

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _get_cached_temporary_mcp_server_or_404("missing", admin_auth)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_non_admin_denied(self):
        """Non-admin without access to the server gets 403, not the server."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_cached_temporary_mcp_server_or_404,
        )

        registry_server = generate_mock_mcp_server_config_record(server_id="server-x")
        non_admin = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = registry_server
        mock_manager.get_mcp_server_by_name.return_value = None
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=[])

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
                return_value=None,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _get_cached_temporary_mcp_server_or_404("server-x", non_admin)

        assert exc_info.value.status_code == 403
        mock_manager.get_allowed_mcp_servers.assert_awaited_once_with(non_admin)

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_non_admin_allowed(self):
        """Non-admin with the server in their allowed set gets the server."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_cached_temporary_mcp_server_or_404,
        )

        registry_server = generate_mock_mcp_server_config_record(server_id="server-x")
        non_admin = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = registry_server
        mock_manager.get_mcp_server_by_name.return_value = None
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server-x"])

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
                return_value=None,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await _get_cached_temporary_mcp_server_or_404(
                "server-x", non_admin
            )

        assert result is registry_server

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_temp_cache_non_admin_denied(self):
        """Servers resolved from the admin-only temp cache reject non-admins."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_cached_temporary_mcp_server_or_404,
        )

        temp_server = generate_mock_mcp_server_config_record(server_id="temp-cache")
        non_admin = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
            return_value=temp_server,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _get_cached_temporary_mcp_server_or_404("temp-cache", non_admin)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_add_session_mcp_server_caches_and_redacts_credentials(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            TEMPORARY_MCP_SERVER_TTL_SECONDS,
            add_session_mcp_server,
        )

        payload = NewMCPServerRequest(
            server_id="temp-server",
            alias="Temporary",
            url="https://temp.example.com",
            transport=MCPTransport.http,
        )
        user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id="admin-user",
        )
        inherited_server = MagicMock(
            authentication_token="token-abc",
            client_id="client-id",
            client_secret="client-secret",
            scopes=["scope1"],
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_region_name=None,
            aws_service_name=None,
        )
        built_server = generate_mock_mcp_server_config_record(server_id="temp-server")
        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = inherited_server
        mock_manager.build_mcp_server_from_table = AsyncMock(return_value=built_server)

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
                MagicMock(),
            ) as validate_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._cache_temporary_mcp_server",
                MagicMock(),
            ) as cache_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._cache_temporary_mcp_server_in_redis",
                AsyncMock(),
            ) as redis_cache_mock,
        ):
            response = await add_session_mcp_server(
                payload=payload,
                user_api_key_dict=user_auth,
            )

        validate_mock.assert_called_once_with(payload)
        mock_manager.build_mcp_server_from_table.assert_awaited_once()
        cache_mock.assert_called_once_with(
            built_server, ttl_seconds=TEMPORARY_MCP_SERVER_TTL_SECONDS
        )
        redis_cache_mock.assert_awaited_once_with(
            built_server, ttl_seconds=TEMPORARY_MCP_SERVER_TTL_SECONDS
        )

        args, _ = mock_manager.build_mcp_server_from_table.call_args
        temp_record = args[0]
        assert temp_record.credentials == {
            "auth_value": "token-abc",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": ["scope1"],
        }
        assert response.credentials is None

    @pytest.mark.asyncio
    async def test_add_session_mcp_server_rejects_non_admins(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_session_mcp_server,
        )

        payload = NewMCPServerRequest(
            alias="Temporary",
            server_id="temp-server",
            url="https://temp.example.com",
            transport=MCPTransport.http,
        )
        non_admin = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            MagicMock(),
        ):
            with pytest.raises(Exception) as exc_info:
                await add_session_mcp_server(
                    payload=payload,
                    user_api_key_dict=non_admin,
                )

        assert "permission" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_mcp_oauth_user_api_key_auth_falls_back_to_token_cookie(self):
        """
        When the Authorization header is absent but a valid 'token' cookie is
        present (browser navigation), _mcp_oauth_user_api_key_auth should
        decode the cookie JWT and authenticate via the API key stored in it.
        """
        import jwt

        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _mcp_oauth_user_api_key_auth,
        )

        master_key = "test-master-key"
        api_key_in_cookie = "sk-test-cookie-key"
        token_cookie = jwt.encode(
            {
                "user_id": "user@example.com",
                "key": api_key_in_cookie,
                "user_role": "proxy_admin",
                "login_method": "sso",
            },
            master_key,
            algorithm="HS256",
        )

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {"token": token_cookie}

        expected_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key=api_key_in_cookie
        )
        fake_proxy_server = types.SimpleNamespace(master_key=master_key)

        with (
            patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_api_key_auth_builder",
                AsyncMock(return_value=expected_auth),
            ) as auth_builder_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._read_request_body",
                AsyncMock(return_value={}),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.populate_request_with_path_params",
                side_effect=lambda request_data, request: request_data,
            ),
        ):
            result = await _mcp_oauth_user_api_key_auth(mock_request)

        assert result is expected_auth
        _, call_kwargs = auth_builder_mock.call_args
        assert call_kwargs["api_key"] == f"Bearer {api_key_in_cookie}"

    @pytest.mark.asyncio
    async def test_mcp_oauth_user_api_key_auth_uses_authorization_header_when_present(
        self,
    ):
        """When Authorization header is present it takes priority over the cookie."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _mcp_oauth_user_api_key_auth,
        )

        expected_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer sk-header-key"}
        mock_request.cookies = {}

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_api_key_auth_builder",
                AsyncMock(return_value=expected_auth),
            ) as auth_builder_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._read_request_body",
                AsyncMock(return_value={}),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.populate_request_with_path_params",
                side_effect=lambda request_data, request: request_data,
            ),
        ):
            result = await _mcp_oauth_user_api_key_auth(mock_request)

        assert result is expected_auth
        _, call_kwargs = auth_builder_mock.call_args
        assert call_kwargs["api_key"] == "Bearer sk-header-key"

    @pytest.mark.asyncio
    async def test_mcp_oauth_user_api_key_auth_requires_oauth2_for_delegate_bypass(
        self,
    ):
        """Non-oauth2 servers must not get anonymous access from the delegate flag."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _mcp_oauth_user_api_key_auth,
        )

        expected_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {}
        mock_request.path_params = {"server_id": "server-1"}
        non_oauth_server = MagicMock()
        non_oauth_server.auth_type = MCPAuth.api_key
        non_oauth_server.delegate_auth_to_upstream = True
        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = non_oauth_server
        mock_manager.get_mcp_server_by_name.return_value = None
        fake_proxy_server = types.SimpleNamespace(master_key=None)

        with (
            patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_api_key_auth_builder",
                AsyncMock(return_value=expected_auth),
            ) as auth_builder_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._read_request_body",
                AsyncMock(return_value={}),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.populate_request_with_path_params",
                side_effect=lambda request_data, request: request_data,
            ),
        ):
            result = await _mcp_oauth_user_api_key_auth(mock_request)

        assert result is expected_auth
        auth_builder_mock.assert_awaited_once()
        _, call_kwargs = auth_builder_mock.call_args
        assert call_kwargs["api_key"] == ""

    @pytest.mark.asyncio
    async def test_mcp_oauth_user_api_key_auth_requires_public_server_for_delegate_bypass(
        self,
    ):
        """Internal-only delegate servers must still require LiteLLM auth."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _mcp_oauth_user_api_key_auth,
        )

        expected_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {}
        mock_request.path_params = {"server_id": "server-1"}
        internal_server = MagicMock()
        internal_server.auth_type = MCPAuth.oauth2
        internal_server.delegate_auth_to_upstream = True
        internal_server.available_on_public_internet = False
        internal_server.has_client_credentials = False
        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = internal_server
        mock_manager.get_mcp_server_by_name.return_value = None
        fake_proxy_server = types.SimpleNamespace(master_key=None)

        with (
            patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._user_api_key_auth_builder",
                AsyncMock(return_value=expected_auth),
            ) as auth_builder_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._read_request_body",
                AsyncMock(return_value={}),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.populate_request_with_path_params",
                side_effect=lambda request_data, request: request_data,
            ),
        ):
            result = await _mcp_oauth_user_api_key_auth(mock_request)

        assert result is expected_auth
        auth_builder_mock.assert_awaited_once()
        _, call_kwargs = auth_builder_mock.call_args
        assert call_kwargs["api_key"] == ""

    @pytest.mark.asyncio
    async def test_mcp_authorize_proxies_to_discoverable_endpoint(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_authorize,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        authorize_response = MagicMock()
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ) as get_server,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.authorize_with_server",
                AsyncMock(return_value=authorize_response),
            ) as authorize_mock,
        ):
            result = await mcp_authorize(
                request=request,
                server_id="server-1",
                user_api_key_dict=admin_auth,
                client_id="client-id",
                redirect_uri="https://example.com/callback",
                state="state123",
                code_challenge="challenge",
                code_challenge_method="S256",
                response_type="code",
                scope="scope1",
            )

        assert result is authorize_response
        get_server.assert_awaited_once_with("server-1", admin_auth, request=request)
        authorize_mock.assert_awaited_once_with(
            request=request,
            mcp_server=server,
            client_id="client-id",
            redirect_uri="https://example.com/callback",
            state="state123",
            code_challenge="challenge",
            code_challenge_method="S256",
            response_type="code",
            scope="scope1",
        )

    @pytest.mark.asyncio
    async def test_mcp_token_proxies_to_exchange_endpoint(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_token,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        exchange_response = {"access_token": "token"}
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ) as get_server,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.exchange_token_with_server",
                AsyncMock(return_value=exchange_response),
            ) as exchange_mock,
        ):
            result = await mcp_token(
                request=request,
                server_id="server-1",
                user_api_key_dict=admin_auth,
                grant_type="authorization_code",
                code="code-123",
                redirect_uri="https://example.com/callback",
                client_id="client",
                client_secret="secret",
                code_verifier="verifier",
                refresh_token=None,
                scope=None,
            )

        assert result is exchange_response
        get_server.assert_awaited_once_with("server-1", admin_auth, request=request)
        exchange_mock.assert_awaited_once_with(
            request=request,
            mcp_server=server,
            grant_type="authorization_code",
            code="code-123",
            redirect_uri="https://example.com/callback",
            client_id="client",
            client_secret="secret",
            code_verifier="verifier",
            refresh_token=None,
            scope=None,
        )

    @pytest.mark.asyncio
    async def test_mcp_token_proxies_refresh_token_grant(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_token,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        exchange_response = {"access_token": "new-token", "refresh_token": "new-rt"}
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ) as get_server,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.exchange_token_with_server",
                AsyncMock(return_value=exchange_response),
            ) as exchange_mock,
        ):
            result = await mcp_token(
                request=request,
                server_id="server-1",
                user_api_key_dict=admin_auth,
                grant_type="refresh_token",
                code=None,
                redirect_uri=None,
                client_id="client",
                client_secret="secret",
                code_verifier=None,
                refresh_token="rt-123",
                scope=None,
            )

        assert result is exchange_response
        get_server.assert_awaited_once_with("server-1", admin_auth, request=request)
        exchange_mock.assert_awaited_once_with(
            request=request,
            mcp_server=server,
            grant_type="refresh_token",
            code=None,
            redirect_uri=None,
            client_id="client",
            client_secret="secret",
            code_verifier=None,
            refresh_token="rt-123",
            scope=None,
        )

    @pytest.mark.asyncio
    async def test_mcp_register_proxies_request_body(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_register,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        register_response = {"client_id": "generated"}
        request_body = {
            "client_name": "LiteLLM",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_basic",
        }
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ) as get_server,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._read_request_body",
                AsyncMock(return_value=request_body),
            ) as read_body,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.register_client_with_server",
                AsyncMock(return_value=register_response),
            ) as register_mock,
        ):
            result = await mcp_register(
                request=request,
                server_id="server-1",
                user_api_key_dict=admin_auth,
            )

        assert result is register_response
        get_server.assert_awaited_once_with("server-1", admin_auth, request=request)
        read_body.assert_awaited_once_with(request=request)
        register_mock.assert_awaited_once_with(
            request=request,
            mcp_server=server,
            client_name="LiteLLM",
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            fallback_client_id="server-1",
        )

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_falls_back_to_redis(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_cached_temporary_mcp_server,
        )

        server = generate_mock_mcp_server_config_record(server_id="from-redis")
        serialized = json.dumps(server.model_dump(mode="json"))
        mock_cache_backend = SimpleNamespace(
            async_get_cache=AsyncMock(return_value="encrypted-payload")
        )
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with (
                patch(
                    "litellm.proxy.management_endpoints.mcp_management_endpoints._temporary_mcp_servers",
                    {},
                ),
                patch(
                    "litellm.proxy.management_endpoints.mcp_management_endpoints.decrypt_value_helper",
                    return_value=serialized,
                ),
            ):
                result = await get_cached_temporary_mcp_server("from-redis")
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        assert result is not None
        assert result.server_id == "from-redis"
        mock_cache_backend.async_get_cache.assert_awaited_once_with(
            key="litellm:mcp:temporary_server:from-redis"
        )

    @pytest.mark.asyncio
    async def test_cache_temporary_mcp_server_in_redis_uses_ttl_and_key(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _cache_temporary_mcp_server_in_redis,
        )

        server = generate_mock_mcp_server_config_record(server_id="to-redis")
        mock_cache_backend = SimpleNamespace(async_set_cache=AsyncMock())
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.encrypt_value_helper",
                return_value="encrypted-payload",
            ):
                await _cache_temporary_mcp_server_in_redis(server, ttl_seconds=123)
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        mock_cache_backend.async_set_cache.assert_awaited_once()
        call_kwargs = mock_cache_backend.async_set_cache.await_args.kwargs
        assert call_kwargs["key"] == "litellm:mcp:temporary_server:to-redis"
        assert call_kwargs["ttl"] == 123

    @pytest.mark.asyncio
    async def test_cache_temporary_mcp_server_in_redis_encrypts_payload(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _cache_temporary_mcp_server_in_redis,
        )

        server = generate_mock_mcp_server_config_record(server_id="to-redis-encrypted")
        mock_cache_backend = SimpleNamespace(async_set_cache=AsyncMock())
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.encrypt_value_helper",
                return_value="encrypted-payload",
            ) as encrypt_mock:
                await _cache_temporary_mcp_server_in_redis(server, ttl_seconds=60)
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        encrypt_mock.assert_called_once()
        call_kwargs = mock_cache_backend.async_set_cache.await_args.kwargs
        assert call_kwargs["value"] == "encrypted-payload"

    @pytest.mark.asyncio
    async def test_get_temporary_mcp_server_from_redis_decrypts_payload(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_temporary_mcp_server_from_redis,
        )

        server = generate_mock_mcp_server_config_record(
            server_id="from-redis-encrypted"
        )
        serialized = json.dumps(server.model_dump(mode="json"))
        mock_cache_backend = SimpleNamespace(
            async_get_cache=AsyncMock(return_value="encrypted-payload")
        )
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.decrypt_value_helper",
                return_value=serialized,
            ) as decrypt_mock:
                result = await _get_temporary_mcp_server_from_redis(
                    "from-redis-encrypted"
                )
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        assert result is not None
        assert result.server_id == "from-redis-encrypted"
        decrypt_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_temporary_mcp_server_in_redis_skips_on_encrypt_failure(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _cache_temporary_mcp_server_in_redis,
        )

        server = generate_mock_mcp_server_config_record(server_id="encrypt-fail")
        mock_cache_backend = SimpleNamespace(async_set_cache=AsyncMock())
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.encrypt_value_helper",
                side_effect=Exception("boom"),
            ):
                await _cache_temporary_mcp_server_in_redis(server, ttl_seconds=60)
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        mock_cache_backend.async_set_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_temporary_mcp_server_in_redis_skips_non_string_encryption_result(
        self,
    ):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _cache_temporary_mcp_server_in_redis,
        )

        server = generate_mock_mcp_server_config_record(server_id="encrypt-non-string")
        mock_cache_backend = SimpleNamespace(async_set_cache=AsyncMock())
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.encrypt_value_helper",
                return_value={"not": "a-string"},
            ):
                await _cache_temporary_mcp_server_in_redis(server, ttl_seconds=60)
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        mock_cache_backend.async_set_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_temporary_mcp_server_from_redis_returns_none_on_invalid_decrypt_json(
        self,
    ):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_temporary_mcp_server_from_redis,
        )

        mock_cache_backend = SimpleNamespace(
            async_get_cache=AsyncMock(return_value="enc")
        )
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.decrypt_value_helper",
                return_value="{not json}",
            ):
                result = await _get_temporary_mcp_server_from_redis("bad-json")
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        assert result is None

    @pytest.mark.asyncio
    async def test_get_temporary_mcp_server_from_redis_returns_none_on_decrypt_none(
        self,
    ):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_temporary_mcp_server_from_redis,
        )

        mock_cache_backend = SimpleNamespace(
            async_get_cache=AsyncMock(return_value="enc")
        )
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.decrypt_value_helper",
                return_value=None,
            ):
                result = await _get_temporary_mcp_server_from_redis("decrypt-none")
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        assert result is None

    @pytest.mark.asyncio
    async def test_get_temporary_mcp_server_from_redis_rejects_plain_dict_payload(self):
        """Plain dict values in Redis are not accepted (write path is encrypted-only)."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_temporary_mcp_server_from_redis,
        )

        server = generate_mock_mcp_server_config_record(server_id="legacy-dict")
        mock_cache_backend = SimpleNamespace(
            async_get_cache=AsyncMock(return_value=server.model_dump(mode="json"))
        )
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            result = await _get_temporary_mcp_server_from_redis("legacy-dict")
        finally:
            mgmt_endpoints.litellm.cache = original_cache

        assert result is None


class TestUpdateMCPServer:
    """Test suite for update MCP server functionality"""

    @pytest.mark.asyncio
    async def test_update_mcp_server_respects_extra_headers(self):
        """
        Test that updating an MCP server with extra_headers properly saves the field.

        This test ensures that extra_headers field in UpdateMCPServerRequest
        is properly handled and persisted when updating an MCP server.
        """
        # Create an existing server
        existing_server = generate_mock_mcp_server_db_record(
            server_id="test-server-1",
            alias="Test Server",
            url="https://test.example.com/mcp",
            transport="http",
        )
        existing_server.extra_headers = []  # Initially empty

        # Create update request with extra_headers
        update_request = UpdateMCPServerRequest(
            server_id="test-server-1",
            alias="Updated Test Server",
            extra_headers=["X-Custom-Header", "X-Another-Header"],
        )

        # Mock the updated server with extra_headers
        updated_server = generate_mock_mcp_server_db_record(
            server_id="test-server-1",
            alias="Updated Test Server",
            url="https://test.example.com/mcp",
            transport="http",
        )
        updated_server.extra_headers = ["X-Custom-Header", "X-Another-Header"]

        # Mock dependencies
        mock_prisma_client = MagicMock()
        mock_prisma_client.db = MagicMock()
        mock_prisma_client.db.litellm_mcpservertable = AsyncMock()
        mock_prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
            return_value=existing_server
        )
        mock_prisma_client.db.litellm_mcpservertable.update = AsyncMock(
            return_value=updated_server
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Mock the update_mcp_server function to capture the call
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.update_mcp_server",
                AsyncMock(return_value=updated_server),
            ) as update_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.add_server",
                AsyncMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.reload_servers_from_database",
                AsyncMock(),
            ),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                edit_mcp_server,
            )

            result = await edit_mcp_server(
                payload=update_request, user_api_key_dict=mock_user_auth
            )

            # Verify that update_mcp_server was called with the correct payload
            update_mock.assert_awaited_once()
            call_args = update_mock.call_args
            # First arg is prisma_client, second is the payload (UpdateMCPServerRequest)
            called_payload = call_args[0][1]
            assert called_payload.server_id == "test-server-1"
            assert called_payload.extra_headers == [
                "X-Custom-Header",
                "X-Another-Header",
            ]
            assert called_payload.alias == "Updated Test Server"

            # Verify the result includes extra_headers
            assert result.extra_headers == ["X-Custom-Header", "X-Another-Header"]
            assert result.alias == "Updated Test Server"


class TestHealthCheckServers:
    """Test suite for health check servers endpoint"""

    @pytest.mark.asyncio
    async def test_health_check_all_servers(self):
        """
        Test health check for all accessible servers

        Scenario: User has access to 2 servers, checks all
        Expected: Returns health status for both servers
        """
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            health_check_servers,
        )

        # Mock user auth
        mock_user_auth = generate_mock_user_api_key_auth()

        # Mock health check results
        mock_health_result_1 = generate_mock_mcp_server_db_record(
            server_id="server-1",
            alias="Server 1",
            url="https://server1.example.com",
        )
        mock_health_result_1.status = "healthy"
        mock_health_result_1.last_health_check = datetime.now()
        mock_health_result_1.health_check_error = None

        mock_health_result_2 = generate_mock_mcp_server_db_record(
            server_id="server-2",
            alias="Server 2",
            url="https://server2.example.com",
        )
        mock_health_result_2.status = "unhealthy"
        mock_health_result_2.last_health_check = datetime.now()
        mock_health_result_2.health_check_error = "Connection timeout"

        # Mock manager
        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=[mock_health_result_1, mock_health_result_2]
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            result = await health_check_servers(
                server_ids=None,
                user_api_key_dict=mock_user_auth,
            )

            # Verify results
            assert len(result) == 2
            assert result[0]["server_id"] == "server-1"
            assert result[0]["status"] == "healthy"
            assert result[1]["server_id"] == "server-2"
            assert result[1]["status"] == "unhealthy"


class TestMCPRegistryEndpoint:
    def test_registry_returns_404_when_flag_missing(self):
        client = create_mcp_router_test_client()

        with patch_proxy_general_settings({}):
            response = client.get("/v1/mcp/registry.json")

        assert response.status_code == 404

    def test_registry_returns_404_when_flag_false(self):
        client = create_mcp_router_test_client()

        with patch_proxy_general_settings({"enable_mcp_registry": False}):
            response = client.get("/v1/mcp/registry.json")

        assert response.status_code == 404

    def test_registry_returns_entries_when_enabled(self):
        client = create_mcp_router_test_client()

        mock_server = generate_mock_mcp_server_config_record(
            server_id="server-123",
            name="zapier",
            url="https://zapier.example.com/mcp",
            transport="http",
        )

        mock_manager = MagicMock()
        mock_manager.get_registry.return_value = {mock_server.server_id: mock_server}
        # The registry endpoint uses get_filtered_registry (filters by client IP)
        mock_manager.get_filtered_registry.return_value = {
            mock_server.server_id: mock_server
        }

        with (
            patch_proxy_general_settings({"enable_mcp_registry": True}),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            response = client.get("/v1/mcp/registry.json")

        assert response.status_code == 200
        data = response.json()
        assert len(data["servers"]) == 2  # built-in + custom server

        builtin_entry = data["servers"][0]["server"]
        assert builtin_entry["name"] == "litellm-mcp-server"
        assert builtin_entry["remotes"][0]["url"].endswith("/mcp")

        custom_entry = data["servers"][1]["server"]
        assert custom_entry["name"] == "zapier"
        assert custom_entry["remotes"][0]["url"].endswith("/zapier/mcp")

    @pytest.mark.asyncio
    async def test_health_check_specific_servers(self):
        """
        Test health check for specific servers

        Scenario: User requests health check for specific server IDs
        Expected: Returns health status only for requested servers
        """
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            health_check_servers,
        )

        # Mock user auth
        mock_user_auth = generate_mock_user_api_key_auth()

        # Mock health check result
        mock_health_result = generate_mock_mcp_server_db_record(
            server_id="server-1",
            alias="Server 1",
            url="https://server1.example.com",
        )
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        # Mock manager
        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=[mock_health_result]
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            result = await health_check_servers(
                server_ids=["server-1"],
                user_api_key_dict=mock_user_auth,
            )

            # Verify results
            assert len(result) == 1
            assert result[0]["server_id"] == "server-1"
            assert result[0]["status"] == "healthy"


class TestManagementPayloadValidation:
    def test_rejects_invalid_alias(self):
        payload = SimpleNamespace(server_name="valid_server", alias="bad/name")

        with pytest.raises(HTTPException) as exc_info:
            mgmt_endpoints.validate_and_normalize_mcp_server_payload(payload)

        assert exc_info.value.status_code == 400
        error_message = exc_info.value.detail["error"]
        assert "bad/name" in error_message

    def test_accepts_valid_names(self):
        payload = SimpleNamespace(server_name="valid_server", alias=None)

        mgmt_endpoints.validate_and_normalize_mcp_server_payload(payload)

        assert payload.alias == "valid_server"

    @pytest.mark.asyncio
    async def test_health_check_view_all_mode(self):
        """view_all mode should return health info for all MCP servers."""

        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            health_check_servers,
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER
        )

        health_result_one = generate_mock_mcp_server_db_record(
            server_id="server-1", alias="One"
        )
        health_result_one.status = "healthy"

        health_result_two = generate_mock_mcp_server_db_record(
            server_id="server-2", alias="Two"
        )
        health_result_two.status = "unhealthy"

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_unfiltered = AsyncMock(
            return_value=[health_result_one, health_result_two]
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="view_all",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await health_check_servers(
                server_ids=None,
                user_api_key_dict=mock_user_auth,
            )

            assert len(result) == 2
            assert result[0]["server_id"] == "server-1"
            assert result[0]["status"] == "healthy"
            assert result[1]["server_id"] == "server-2"
            assert result[1]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_unauthorized_servers(self):
        """
        Test health check with unauthorized servers

        Scenario: User requests health check for servers they don't have access to
        Expected: Only checks accessible servers, unauthorized servers are filtered out
        """
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            health_check_servers,
        )

        # Mock user auth
        mock_user_auth = generate_mock_user_api_key_auth()

        # Mock health check result for authorized server
        mock_health_result = generate_mock_mcp_server_db_record(
            server_id="server-1",
            alias="Server 1",
            url="https://server1.example.com",
        )
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        # Mock manager - server_ids filter is applied inside get_all_mcp_servers_with_health_and_teams
        # So it only returns servers the user has access to
        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=[mock_health_result]  # Only server-1 is returned (accessible)
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            result = await health_check_servers(
                server_ids=["server-1", "server-unauthorized"],
                user_api_key_dict=mock_user_auth,
            )

            # Verify results - only accessible server is returned
            assert len(result) == 1
            assert result[0]["server_id"] == "server-1"
            assert result[0]["status"] == "healthy"


class TestMCPApprovalWorkflow:
    """Tests for BYOM submission: register, list submissions, approve, reject."""

    @pytest.mark.asyncio
    async def test_register_mcp_server_requires_team_key(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            register_mcp_server,
        )

        payload = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
        )
        # No team_id → should raise 400
        user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            team_id=None,
        )
        with pytest.raises(HTTPException) as exc_info:
            await register_mcp_server(payload=payload, user_api_key_dict=user_auth)
        assert exc_info.value.status_code == 400
        assert "team" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_register_mcp_server_rejects_stdio_transport(self):
        # stdio servers spawn a local subprocess on the proxy host. Accepting
        # them from the non-admin submission endpoint would let a team member
        # propose a config that an admin could rubber-stamp into local code
        # execution. Admins use POST /v1/mcp/server or config.yaml instead.
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            register_mcp_server,
        )

        payload = NewMCPServerRequest(
            alias="local",
            transport=MCPTransport.stdio,
            command="python3",
            args=["-m", "mcp_server_filesystem", "/tmp"],
        )
        user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            team_id="team-123",
            user_id="user-abc",
        )
        with pytest.raises(HTTPException) as exc_info:
            await register_mcp_server(payload=payload, user_api_key_dict=user_auth)
        assert exc_info.value.status_code == 400
        assert "stdio" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_register_mcp_server_sets_pending_review(self):
        from litellm.proxy._types import MCPApprovalStatus
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            register_mcp_server,
        )

        payload = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
        )
        user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            team_id="team-123",
            user_id="user-abc",
        )
        created_record = generate_mock_mcp_server_db_record(
            alias="My Server",
            url="https://example.com/mcp",
        )
        created_record.approval_status = MCPApprovalStatus.pending_review
        created_record.submitted_by = "user-abc"

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.create_mcp_server",
                AsyncMock(return_value=created_record),
            ) as mock_create,
        ):
            result = await register_mcp_server(
                payload=payload, user_api_key_dict=user_auth
            )

        # Endpoint sets pending_review before calling create_mcp_server
        call_payload = mock_create.call_args[0][1]
        assert call_payload.approval_status == MCPApprovalStatus.pending_review
        assert call_payload.submitted_by == "user-abc"
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_submissions_non_admin_forbidden(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_mcp_server_submissions,
        )

        non_admin = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_mcp_server_submissions(user_api_key_dict=non_admin)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_submissions_admin_returns_summary(self):
        from litellm.proxy._types import MCPSubmissionsSummary
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_mcp_server_submissions,
        )

        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
        pending = generate_mock_mcp_server_db_record(alias="Pending")
        pending.approval_status = "pending_review"
        summary = MCPSubmissionsSummary(
            total=1, pending_review=1, active=0, rejected=0, items=[pending]
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_submissions",
                AsyncMock(return_value=summary),
            ),
        ):
            result = await get_mcp_server_submissions(user_api_key_dict=admin)

        assert result.total == 1
        assert result.pending_review == 1

    @pytest.mark.asyncio
    async def test_approve_non_pending_server_raises_400(self):
        from litellm.proxy._types import MCPApprovalStatus
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            approve_mcp_server_submission,
        )

        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
        active_server = generate_mock_mcp_server_db_record()
        active_server.approval_status = MCPApprovalStatus.active

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=active_server),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await approve_mcp_server_submission(
                    server_id="server-1", user_api_key_dict=admin
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_approve_pending_server_loads_into_registry(self):
        from litellm.proxy._types import MCPApprovalStatus
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            approve_mcp_server_submission,
        )

        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
        pending_server = generate_mock_mcp_server_db_record()
        pending_server.approval_status = MCPApprovalStatus.pending_review
        approved_server = generate_mock_mcp_server_db_record()
        approved_server.approval_status = MCPApprovalStatus.active

        mock_manager = MagicMock()
        mock_manager.reload_servers_from_database = AsyncMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=pending_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.approve_mcp_server",
                AsyncMock(return_value=approved_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await approve_mcp_server_submission(
                server_id=pending_server.server_id, user_api_key_dict=admin
            )

        mock_manager.reload_servers_from_database.assert_awaited_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_reject_already_rejected_raises_400(self):
        from litellm.proxy._types import MCPApprovalStatus, RejectMCPServerRequest
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            reject_mcp_server_submission,
        )

        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
        rejected_server = generate_mock_mcp_server_db_record()
        rejected_server.approval_status = MCPApprovalStatus.rejected

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=rejected_server),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reject_mcp_server_submission(
                    server_id="server-1",
                    payload=RejectMCPServerRequest(review_notes="duplicate"),
                    user_api_key_dict=admin,
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_reject_active_server_allowed(self):
        """Admin can deactivate an already-approved server via the reject endpoint."""
        from litellm.proxy._types import MCPApprovalStatus, RejectMCPServerRequest
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            reject_mcp_server_submission,
        )

        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
        active_server = generate_mock_mcp_server_db_record()
        active_server.approval_status = MCPApprovalStatus.active
        now_rejected = generate_mock_mcp_server_db_record()
        now_rejected.approval_status = MCPApprovalStatus.rejected

        mock_manager = MagicMock()
        mock_manager.reload_servers_from_database = AsyncMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=active_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.reject_mcp_server",
                AsyncMock(return_value=now_rejected),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await reject_mcp_server_submission(
                server_id=active_server.server_id,
                payload=RejectMCPServerRequest(review_notes="policy violation"),
                user_api_key_dict=admin,
            )
        assert result is not None
        mock_manager.reload_servers_from_database.assert_awaited_once()


class TestValidateMCPRequiredFields:
    """Tests for _validate_mcp_required_fields."""

    def test_missing_required_field_raises_400(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _validate_mcp_required_fields,
        )

        payload = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
            # source_url is absent
        )
        with patch_proxy_general_settings({"mcp_required_fields": ["source_url"]}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_mcp_required_fields(payload)
        assert exc_info.value.status_code == 400
        assert "source_url" in str(exc_info.value.detail)

    def test_auth_type_sentinel_treated_as_absent(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _validate_mcp_required_fields,
        )

        payload = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
            auth_type=MCPAuth.none,  # sentinel value — treated as absent
        )
        with patch_proxy_general_settings({"mcp_required_fields": ["auth_type"]}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_mcp_required_fields(payload)
        assert exc_info.value.status_code == 400
        assert "auth_type" in str(exc_info.value.detail)

    def test_all_required_fields_present_passes(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _validate_mcp_required_fields,
        )

        payload = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
            source_url="https://github.com/org/repo",
            auth_type=MCPAuth.bearer_token,
        )
        with patch_proxy_general_settings(
            {"mcp_required_fields": ["source_url", "auth_type"]}
        ):
            # Should not raise
            _validate_mcp_required_fields(payload)

    def test_no_required_fields_configured_always_passes(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _validate_mcp_required_fields,
        )

        payload = NewMCPServerRequest(
            alias="Minimal",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
        )
        with patch_proxy_general_settings({}):
            # Should not raise when no required fields are configured
            _validate_mcp_required_fields(payload)

    def test_unknown_field_name_in_config_raises_500(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _validate_mcp_required_fields,
        )

        payload = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.sse,
        )
        # "source_Url" is a typo — not a real field on NewMCPServerRequest
        with patch_proxy_general_settings({"mcp_required_fields": ["source_Url"]}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_mcp_required_fields(payload)
        assert exc_info.value.status_code == 500
        assert "source_Url" in str(exc_info.value.detail)


# ── OAuth user credential endpoint unit tests ──────────────────────────────────


def _make_user_auth(user_id: str = "user-abc") -> "UserAPIKeyAuth":
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def _make_prisma_client():
    """Return a minimal mock PrismaClient accepted by get_prisma_client_or_throw."""
    client = MagicMock()
    client.db = MagicMock()
    return client


@pytest.mark.asyncio
async def test_store_mcp_oauth_user_credential_returns_status():
    """store_mcp_oauth_user_credential persists the token and echoes back status."""
    from litellm.proxy._types import (
        MCPOAuthUserCredentialRequest,
        MCPOAuthUserCredentialStatus,
    )

    if not mgmt_endpoints.MCP_AVAILABLE:
        pytest.skip("MCP module not installed")

    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        store_mcp_oauth_user_credential,
    )

    server_id = "srv-1"
    user_id = "user-123"
    stored_payload = {
        "type": "oauth2",
        "access_token": "tok",
        "expires_at": "2099-01-01T00:00:00+00:00",
        "connected_at": "2026-01-01T00:00:00+00:00",
        "server_id": server_id,
    }

    mock_prisma = _make_prisma_client()

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            new=AsyncMock(
                return_value=generate_mock_mcp_server_db_record(server_id=server_id)
            ),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.store_user_oauth_credential",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_user_oauth_credential",
            new=AsyncMock(return_value=stored_payload),
        ),
    ):
        result = await store_mcp_oauth_user_credential(
            server_id=server_id,
            payload=MCPOAuthUserCredentialRequest(
                access_token="tok",
                expires_in=3600,
            ),
            user_api_key_dict=_make_user_auth(user_id),
        )

    assert isinstance(result, MCPOAuthUserCredentialStatus)
    assert result.has_credential is True
    assert result.server_id == server_id
    # expires_at should come from the stored record, not be recomputed
    assert result.expires_at == "2099-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_delete_mcp_oauth_user_credential_only_deletes_oauth():
    """delete_mcp_oauth_user_credential only deletes OAuth2 credentials, not BYOK."""
    from litellm.proxy._types import MCPOAuthUserCredentialStatus

    if not mgmt_endpoints.MCP_AVAILABLE:
        pytest.skip("MCP module not installed")

    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        delete_mcp_oauth_user_credential,
    )

    server_id = "srv-2"
    user_id = "user-456"
    delete_mock = AsyncMock(return_value=None)

    # When get_user_oauth_credential returns None (no OAuth cred), delete should NOT be called.
    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=_make_prisma_client(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_user_oauth_credential",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.delete_user_credential",
            new=delete_mock,
        ),
    ):
        result = await delete_mcp_oauth_user_credential(
            server_id=server_id,
            user_api_key_dict=_make_user_auth(user_id),
        )

    delete_mock.assert_not_called()
    assert isinstance(result, MCPOAuthUserCredentialStatus)
    assert result.has_credential is False


@pytest.mark.asyncio
async def test_list_mcp_user_credentials_batch_server_fetch():
    """list_mcp_user_credentials uses a single batch DB call, not N+1 queries."""
    from litellm.proxy._types import MCPUserCredentialListItem

    if not mgmt_endpoints.MCP_AVAILABLE:
        pytest.skip("MCP module not installed")

    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        list_mcp_user_credentials,
    )

    user_id = "user-789"
    server_id = "srv-3"
    stored_creds = [
        {
            "type": "oauth2",
            "access_token": "tok",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "connected_at": "2026-01-01T00:00:00+00:00",
            "server_id": server_id,
        }
    ]
    mock_server = generate_mock_mcp_server_db_record(
        server_id=server_id, alias="My Server"
    )
    # get_mcp_servers (batch) should be called once; get_mcp_server (single) must not be called.
    batch_mock = AsyncMock(return_value=[mock_server])
    single_mock = AsyncMock(return_value=mock_server)

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=_make_prisma_client(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.list_user_oauth_credentials",
            new=AsyncMock(return_value=stored_creds),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_servers",
            new=batch_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            new=single_mock,
        ),
    ):
        result = await list_mcp_user_credentials(
            user_api_key_dict=_make_user_auth(user_id),
        )

    batch_mock.assert_called_once()
    single_mock.assert_not_called()
    assert len(result) == 1
    assert isinstance(result[0], MCPUserCredentialListItem)
    assert result[0].server_id == server_id
    assert result[0].alias == "My Server"
    # expires_at should always be the raw timestamp (not set to None when expired)
    assert result[0].expires_at == "2099-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# VERIA-8: non-admin viewers must not see the raw MCP server URL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_mcp_servers_non_admin_url_redacted():
    """A standard authenticated user (no admin role, not a restricted
    virtual key) used to receive the raw `url` field, which can contain
    bearer tokens like `https://actions.zapier.com/mcp/<api-key>/sse`.
    They must now get the credential-bearing fields stripped."""
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        fetch_all_mcp_servers,
    )

    user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="alice",
        api_key="sk-alice",
        # NOT allowed_routes — a normal authenticated user.
    )

    server = generate_mock_mcp_server_db_record(
        server_id="zapier-1",
        alias="Zapier",
        url="https://actions.zapier.com/mcp/SUPER-SECRET-TOKEN/sse",
    )
    server.static_headers = {"Authorization": "Bearer SUPER-SECRET-TOKEN"}
    server.env = {"API_KEY": "another-secret"}
    server.extra_headers = ["Authorization"]
    server.command = "npx"
    server.args = ["-y", "@sensitive/mcp"]
    server.authorization_url = "https://oauth.example.com/authorize?token=foo"
    server.token_url = "https://oauth.example.com/token"
    server.registration_url = "https://oauth.example.com/register"

    mock_manager = MagicMock()
    mock_manager.get_all_mcp_servers_unfiltered = AsyncMock(return_value=[server])
    mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=[server])

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
            return_value="view_all",
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
            AsyncMock(return_value=[user]),
        ),
    ):
        result = await fetch_all_mcp_servers(user_api_key_dict=user)

    assert len(result) == 1
    s = result[0]
    # Identity fields stay so the UI can list the server.
    assert s.server_id == "zapier-1"
    assert s.alias == "Zapier"
    # Credential-bearing fields must all be cleared.
    assert s.url is None
    assert s.static_headers is None
    assert s.env == {}
    assert s.extra_headers == []
    assert s.command is None
    assert s.args == []
    assert s.authorization_url is None
    assert s.token_url is None
    assert s.registration_url is None


@pytest.mark.asyncio
async def test_list_mcp_servers_admin_keeps_url():
    """Proxy admins must continue to see the raw URL — the redaction
    only applies to non-admin viewers."""
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        fetch_all_mcp_servers,
    )

    admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="root",
        api_key="sk-admin",
    )

    server = generate_mock_mcp_server_db_record(
        server_id="zapier-1",
        alias="Zapier",
        url="https://actions.zapier.com/mcp/SUPER-SECRET-TOKEN/sse",
    )
    server.static_headers = {"Authorization": "Bearer SUPER-SECRET-TOKEN"}

    mock_manager = MagicMock()
    mock_manager.get_all_mcp_servers_unfiltered = AsyncMock(return_value=[server])
    mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=[server])

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
            return_value="view_all",
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
            AsyncMock(return_value=[admin]),
        ),
    ):
        result = await fetch_all_mcp_servers(user_api_key_dict=admin)

    assert len(result) == 1
    assert result[0].url == "https://actions.zapier.com/mcp/SUPER-SECRET-TOKEN/sse"
    assert result[0].static_headers == {"Authorization": "Bearer SUPER-SECRET-TOKEN"}


def test_sanitize_mcp_server_for_non_admin_clears_credential_fields():
    """Direct unit test on the helper for fast feedback."""
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        _sanitize_mcp_server_for_non_admin,
    )

    server = generate_mock_mcp_server_db_record(
        url="https://example.com/mcp/SUPER-SECRET/sse",
    )
    server.credentials = {"auth_value": "secret"}
    server.static_headers = {"Authorization": "Bearer x"}
    server.spec_path = "https://example.com/specs/openapi-with-token.yaml"
    server.env = {"API_KEY": "y"}
    server.extra_headers = ["Authorization"]
    server.command = "python"
    server.args = ["server.py", "--token", "secret"]
    server.authorization_url = "https://idp/authorize"
    server.token_url = "https://idp/token"
    server.registration_url = "https://idp/register"

    sanitized = _sanitize_mcp_server_for_non_admin(server)

    assert sanitized.credentials is None
    assert sanitized.url is None
    assert sanitized.spec_path is None
    assert sanitized.static_headers is None
    assert sanitized.env == {}
    assert sanitized.extra_headers == []
    assert sanitized.command is None
    assert sanitized.args == []
    assert sanitized.authorization_url is None
    assert sanitized.token_url is None
    assert sanitized.registration_url is None

    # Identity / metadata fields are preserved so the UI can list the
    # server without exposing secrets.
    assert sanitized.server_id == server.server_id
    assert sanitized.alias == server.alias

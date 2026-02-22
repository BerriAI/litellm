import os
import sys
import types
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

            # Check server details
            for server in result:
                if server.server_id == "db_server_allowed":
                    assert server.alias == "Allowed Gmail MCP"
                    assert server.url == "https://gmail-mcp.example.com/mcp"
                elif server.server_id == "config_server_allowed":
                    assert server.alias == "Allowed Zapier MCP"
                    assert server.url == "https://actions.zapier.com/mcp/sse"

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

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
            AsyncMock(return_value=[mock_user_auth]),
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
                server_id="server-1", user_api_key_dict=mock_user_auth
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
                server_id="server-2", user_api_key_dict=mock_user_auth
            )

            assert result.server_id == "server-2"
            # credentials attribute should still be absent and no exception raised
            assert not hasattr(result, "credentials")
            assert result.status == "healthy"


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

    def test_get_cached_temporary_mcp_server_prunes_expired_entries(self):
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
            result = get_cached_temporary_mcp_server("expired")

        assert result is None
        assert "expired" not in cache

    def test_get_cached_temporary_mcp_server_or_404(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_cached_temporary_mcp_server_or_404,
        )

        server = generate_mock_mcp_server_config_record(server_id="cached")

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
            return_value=server,
        ) as get_cached:
            result = _get_cached_temporary_mcp_server_or_404("cached")

        assert result is server
        get_cached.assert_called_once_with("cached")

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                _get_cached_temporary_mcp_server_or_404("missing")

        assert exc_info.value.status_code == 404

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
    async def test_mcp_authorize_proxies_to_discoverable_endpoint(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_authorize,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        authorize_response = MagicMock()

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
                client_id="client-id",
                redirect_uri="https://example.com/callback",
                state="state123",
                code_challenge="challenge",
                code_challenge_method="S256",
                response_type="code",
                scope="scope1",
            )

        assert result is authorize_response
        get_server.assert_called_once_with("server-1")
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
                grant_type="authorization_code",
                code="code-123",
                redirect_uri="https://example.com/callback",
                client_id="client",
                client_secret="secret",
                code_verifier="verifier",
            )

        assert result is exchange_response
        get_server.assert_called_once_with("server-1")
        exchange_mock.assert_awaited_once_with(
            request=request,
            mcp_server=server,
            grant_type="authorization_code",
            code="code-123",
            redirect_uri="https://example.com/callback",
            client_id="client",
            client_secret="secret",
            code_verifier="verifier",
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
            result = await mcp_register(request=request, server_id="server-1")

        assert result is register_response
        get_server.assert_called_once_with("server-1")
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

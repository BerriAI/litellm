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

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

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


def generate_mock_team_record(team_id: str, team_alias: str, organization_id: str, mcp_servers: List[str]):
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
    mock_prisma_client.db.litellm_teamtable.find_many = AsyncMock(return_value=team_records)
    mock_prisma_client.db.litellm_mcpservertable = AsyncMock()
    mock_prisma_client.db.litellm_mcpservertable.find_many = AsyncMock(return_value=mcp_servers)
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


class TestMCPCredentialsTokenExchangeProfile:
    """token_exchange_profile must be a declared MCPCredentials field so the management API can
    persist the entra_obo profile. An undeclared key is silently stripped by pydantic when the
    credentials dict is validated against the TypedDict, so it would never reach the JSON blob."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda creds: NewMCPServerRequest(
                server_name="s", auth_type=MCPAuth.oauth2_token_exchange, credentials=creds
            ),
            lambda creds: UpdateMCPServerRequest(server_id="s", credentials=creds),
        ],
        ids=["new", "update"],
    )
    def test_request_preserves_token_exchange_profile_in_credentials(self, build):
        creds = {
            "client_id": "cid",
            "client_secret": "sec",
            "token_exchange_endpoint": "https://login.microsoftonline.com/tid/oauth2/v2.0/token",
            "scopes": ["api://target/.default"],
            "token_exchange_profile": "entra_obo",
        }
        request = build(creds)
        assert request.credentials is not None
        assert request.credentials.get("token_exchange_profile") == "entra_obo"


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
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["config_server_1", "config_server_2"])

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

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)

        mock_servers = [
            generate_mock_mcp_server_db_record(server_id="server-1", alias="One"),
            generate_mock_mcp_server_db_record(server_id="server-2", alias="Two"),
        ]

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_unfiltered = AsyncMock(return_value=mock_servers)

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
        mock_manager.get_all_mcp_servers_unfiltered = AsyncMock(return_value=mock_servers)
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
            "config_server_not_allowed": generate_mock_mcp_server_config_record(server_id="config_server_not_allowed"),
        }
        # User only has access to specific servers
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["db_server_allowed", "config_server_allowed"])

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
        mock_manager.get_all_allowed_mcp_servers = AsyncMock(return_value=[server_1, server_2])

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
        mock_server = generate_mock_mcp_server_db_record(server_id="server-1", alias="Server 1")
        mock_server.credentials = {"auth_value": "top-secret"}

        mock_prisma_client = MagicMock()

        # Mock health check result as LiteLLM_MCPServerTable
        mock_health_result = generate_mock_mcp_server_db_record(server_id="server-1", alias="Server 1")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

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
        mock_server = generate_mock_mcp_server_db_record(server_id="server-2", alias="Server 2")
        # Simulate ORM object without credentials attribute (e.g., older schema)
        delattr(mock_server, "credentials")

        mock_prisma_client = MagicMock()

        # Mock health check result as LiteLLM_MCPServerTable
        mock_health_result = generate_mock_mcp_server_db_record(server_id="server-2", alias="Server 2")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

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

        mock_health_result = generate_mock_mcp_server_db_record(server_id="serper_custom_dev", alias="Serper MCP")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda sid: config_server if sid == "serper_custom_dev" else None
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
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["serper_custom_dev"])
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

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
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["serper_custom_dev"])
        mock_manager.health_check_server = AsyncMock(
            return_value=generate_mock_mcp_server_db_record(server_id="serper_custom_dev", alias="Serper MCP")
        )

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

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
            mock_manager.get_mcp_server_by_name.assert_called_once_with("Serper MCP", client_ip="192.168.1.100")

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
            side_effect=lambda sid: config_server if sid == "restricted_server" else None
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

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)

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

        mock_health_result = generate_mock_mcp_server_db_record(server_id="allowed_config_server", alias="Allowed MCP")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda sid: config_server if sid == "allowed_config_server" else None
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
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["allowed_config_server"])
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)

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

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_drops_env_vars_for_non_admin(self):
        """A non-admin GET /v1/mcp/server/{id} for a server with env_vars must
        not 500 and must not leak env var config. ``db.get_mcp_server`` returns
        the raw Prisma model whose JSONB ``env_vars`` deserialize to plain
        dicts; it is wrapped in ``LiteLLM_MCPServerTable`` (parsing the dicts
        into ``MCPEnvVar``) before sanitization. The non-admin sanitizer then
        drops ``env_vars`` entirely, since even the names (e.g. GLOBAL_KEY)
        reveal which secrets the admin configured.
        """

        # Mirror what Prisma returns: a model whose JSONB ``env_vars`` are
        # plain dicts, not parsed ``MCPEnvVar`` objects. ``model_construct``
        # skips validation so the dicts survive verbatim.
        raw_prisma_model = LiteLLM_MCPServerTable.model_construct(
            server_id="env-server",
            server_name="Env Server",
            alias="Env Server",
            transport=MCPTransport.http,
            url="https://env.example.com/mcp",
            static_headers={
                "Authorization": "Bearer ${GLOBAL_KEY}",
                "X-User": "${USER_KEY}",
            },
            env_vars=[
                {"name": "GLOBAL_KEY", "value": "super-secret", "scope": "global"},
                {
                    "name": "USER_KEY",
                    "value": "",
                    "scope": "user",
                    "description": "your key",
                },
            ],
        )
        assert isinstance(raw_prisma_model.env_vars[0], dict)

        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=raw_prisma_model)

        mock_health_result = generate_mock_mcp_server_db_record(server_id="env-server", alias="Env Server")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.add_server = AsyncMock()
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
                AsyncMock(return_value=[generate_mock_mcp_server_db_record(server_id="env-server")]),
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
                server_id="env-server",
                user_api_key_dict=mock_user_auth,
            )

        assert result.server_id == "env-server"
        # Non-admin viewers get no env var config at all (not even names).
        assert result.env_vars is None

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_sanitizes_for_view_only_admin(self):
        """PROXY_ADMIN_VIEW_ONLY must NOT see credential-bearing fields.

        It previously passed the _user_has_admin_view gate (which also grants view-only
        admins) and only had the explicit `credentials` field cleared, leaking secrets
        embedded in url/static_headers/env_vars. Only a FULL PROXY_ADMIN may see those.
        This test exercises the real role helpers (no patching of the gate)."""
        mock_server = LiteLLM_MCPServerTable.model_construct(
            server_id="leaky-server",
            server_name="Leaky Server",
            alias="Leaky Server",
            transport=MCPTransport.http,
            url="https://leaky.example.com/mcp?api_key=sk-embedded-in-url",
            static_headers={"Authorization": "Bearer sk-secret-header"},
            env={"UPSTREAM_TOKEN": "sk-secret-env"},
            env_vars=[
                {"name": "GLOBAL_KEY", "value": "super-secret", "scope": "global"},
            ],
            credentials={"auth_value": "sk-explicit-credential"},
        )

        mock_prisma_client = MagicMock()

        mock_health_result = generate_mock_mcp_server_db_record(server_id="leaky-server", alias="Leaky Server")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.add_server = AsyncMock()
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

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
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(),
                server_id="leaky-server",
                user_api_key_dict=mock_user_auth,
            )

        assert result.server_id == "leaky-server"
        assert result.credentials is None
        assert result.url is None
        assert result.static_headers is None
        assert result.env == {}
        assert result.env_vars is None

    @pytest.mark.asyncio
    async def test_fetch_single_mcp_server_full_admin_still_sees_secrets(self):
        """the fix must not over-redact for FULL PROXY_ADMIN,
        who needs url/static_headers/env to populate the edit form."""
        mock_server = LiteLLM_MCPServerTable.model_construct(
            server_id="admin-server",
            server_name="Admin Server",
            alias="Admin Server",
            transport=MCPTransport.http,
            url="https://admin.example.com/mcp",
            static_headers={"Authorization": "Bearer sk-secret-header"},
            credentials={"auth_value": "sk-explicit-credential"},
        )

        mock_prisma_client = MagicMock()

        mock_health_result = generate_mock_mcp_server_db_record(server_id="admin-server", alias="Admin Server")
        mock_health_result.status = "healthy"
        mock_health_result.last_health_check = datetime.now()
        mock_health_result.health_check_error = None

        mock_manager = MagicMock()
        mock_manager.add_server = AsyncMock()
        mock_manager.health_check_server = AsyncMock(return_value=mock_health_result)

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

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
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_mcp_server,
            )

            result = await fetch_mcp_server(
                request=_make_mock_request(),
                server_id="admin-server",
                user_api_key_dict=mock_user_auth,
            )

        # credentials field is always redacted; the rest must survive for full admin.
        assert result.credentials is None
        assert result.url == "https://admin.example.com/mcp"
        assert result.static_headers == {"Authorization": "Bearer sk-secret-header"}


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
                await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth, team_id="foreign-team-id")
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

        mock_server = generate_mock_mcp_server_config_record(server_id="server-1", name="Team Server")
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
                AsyncMock(return_value=[generate_mock_mcp_server_db_record(server_id="server-1")]),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth, team_id="my-team-id")
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
                AsyncMock(return_value=[generate_mock_mcp_server_db_record(server_id="server-1")]),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            # Admin should NOT need to be a team member
            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth, team_id="any-team-id")
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
            await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth, team_id="some-team")
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
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[non_admin]),
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
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[non_admin]),
            ),
        ):
            result = await _get_cached_temporary_mcp_server_or_404("server-x", non_admin)

        assert result is registry_server

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_non_admin_allowed_via_team_access_group(
        self,
    ):
        """Internal user whose only grant to the server flows through a team
        access-group must pass the authorize/token access check. The check has to
        expand the UI session into per-team contexts (build_effective_auth_contexts),
        the same way the server-list grid does; checking only the bare session
        context leaves the team grant invisible and 403s the user."""
        from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _get_cached_temporary_mcp_server_or_404,
        )

        registry_server = generate_mock_mcp_server_config_record(server_id="server-x")
        ui_session_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            team_id=UI_SESSION_TOKEN_TEAM_ID,
        )
        team_context = ui_session_auth.model_copy()
        team_context.team_id = "team-with-mcp-grant"

        mock_manager = MagicMock()
        mock_manager.get_mcp_server_by_id.return_value = registry_server
        mock_manager.get_mcp_server_by_name.return_value = None

        def allowed_for(auth):
            return ["server-x"] if auth.team_id == "team-with-mcp-grant" else []

        mock_manager.get_allowed_mcp_servers = AsyncMock(side_effect=allowed_for)

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_cached_temporary_mcp_server",
                return_value=None,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[ui_session_auth, team_context]),
            ),
        ):
            result = await _get_cached_temporary_mcp_server_or_404("server-x", ui_session_auth)

        assert result is registry_server
        assert mock_manager.get_allowed_mcp_servers.await_count == 2

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
        cache_mock.assert_called_once_with(built_server, ttl_seconds=TEMPORARY_MCP_SERVER_TTL_SECONDS)
        redis_cache_mock.assert_awaited_once_with(built_server, ttl_seconds=TEMPORARY_MCP_SERVER_TTL_SECONDS)

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

        expected_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
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

        expected_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
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
    async def test_mcp_oauth_user_api_key_auth_internal_delegate_bypasses(
        self,
    ):
        """Internal-only delegate servers still get anonymous PKCE /authorize bypass."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _mcp_oauth_user_api_key_auth,
        )

        expected_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {}
        mock_request.path_params = {"server_id": "server-1"}
        # Real path so ``endswith("/token")`` is not fooled by MagicMock truthiness.
        mock_request.url = types.SimpleNamespace(path="/server-1/authorize")
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

        assert isinstance(result, UserAPIKeyAuth)
        auth_builder_mock.assert_not_called()

    def test_mcp_oauth_authorize_token_routes_use_browser_auth_dependency(self):
        from fastapi.routing import APIRoute

        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _mcp_oauth_user_api_key_auth,
            router,
        )

        oauth_routes = {
            route.path: route
            for route in router.routes
            if isinstance(route, APIRoute)
            and route.path
            in {
                "/v1/mcp/server/oauth/{server_id}/authorize",
                "/v1/mcp/server/oauth/{server_id}/token",
            }
        }

        assert set(oauth_routes) == {
            "/v1/mcp/server/oauth/{server_id}/authorize",
            "/v1/mcp/server/oauth/{server_id}/token",
        }
        for route in oauth_routes.values():
            dependency_names = {
                dependant.name
                for dependant in route.dependant.dependencies
                if dependant.call is _mcp_oauth_user_api_key_auth
            }
            assert dependency_names == {None, "user_api_key_dict"}

    @pytest.mark.asyncio
    async def test_mcp_authorize_proxies_to_discoverable_endpoint(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_authorize,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        server.auth_type = MCPAuth.oauth2
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
    async def test_mcp_authorize_rejects_non_oauth2_server(self):
        """mcp_authorize must reject a none-auth server with an accurate 'does not use OAuth'
        400 before the client_id check, never delegating to authorize_with_server."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_authorize,
        )

        server = generate_mock_mcp_server_config_record(server_id="none-server")
        server.auth_type = MCPAuth.none
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.authorize_with_server",
                AsyncMock(),
            ) as authorize_mock,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await mcp_authorize(
                    request=MagicMock(),
                    server_id="none-server",
                    user_api_key_dict=admin_auth,
                    client_id=None,
                    redirect_uri="https://example.com/callback",
                    state="state123",
                )

        assert exc_info.value.status_code == 400
        detail_text = str(exc_info.value.detail)
        assert "does not use OAuth" in detail_text
        assert "missing_client_id" not in detail_text
        authorize_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mcp_token_rejects_non_oauth2_server(self):
        """mcp_token must reject a none-auth server with 'does not use OAuth' 400 before the
        client_id check, never delegating to exchange_token_with_server."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_token,
        )

        server = generate_mock_mcp_server_config_record(server_id="none-server")
        server.auth_type = MCPAuth.none
        admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.exchange_token_with_server",
                AsyncMock(),
            ) as exchange_mock,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await mcp_token(
                    request=MagicMock(),
                    server_id="none-server",
                    user_api_key_dict=admin_auth,
                    grant_type="authorization_code",
                    code="code-123",
                    redirect_uri="https://example.com/callback",
                    client_id=None,
                    client_secret=None,
                    code_verifier="verifier",
                    refresh_token=None,
                    scope=None,
                )

        assert exc_info.value.status_code == 400
        detail_text = str(exc_info.value.detail)
        assert "does not use OAuth" in detail_text
        assert "missing_client_id" not in detail_text
        exchange_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mcp_token_proxies_to_exchange_endpoint(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            mcp_token,
        )

        request = MagicMock()
        server = generate_mock_mcp_server_config_record(server_id="server-1")
        server.auth_type = MCPAuth.oauth2
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
        server.auth_type = MCPAuth.oauth2
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
        server.auth_type = MCPAuth.oauth2
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
            persist_credentials=True,
        )

    @pytest.mark.asyncio
    async def test_mcp_register_does_not_persist_for_non_admin(self):
        """A non-admin caller (who may have access to a real server) must not persist the DCR
        result onto the shared server row. register_client_with_server is invoked with
        persist_credentials=False, so user-side registration returns the DCR response without
        writing shared client credentials. Only a full PROXY_ADMIN establishes the shared client."""
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
        non_admin_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER,
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_cached_temporary_mcp_server_or_404",
                return_value=server,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._read_request_body",
                AsyncMock(return_value=request_body),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.register_client_with_server",
                AsyncMock(return_value=register_response),
            ) as register_mock,
        ):
            result = await mcp_register(
                request=request,
                server_id="server-1",
                user_api_key_dict=non_admin_auth,
            )

        assert result is register_response
        assert register_mock.await_args.kwargs["persist_credentials"] is False

    @pytest.mark.asyncio
    async def test_get_cached_temporary_mcp_server_falls_back_to_redis(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_cached_temporary_mcp_server,
        )

        server = generate_mock_mcp_server_config_record(server_id="from-redis")
        serialized = json.dumps(server.model_dump(mode="json"))
        mock_cache_backend = SimpleNamespace(async_get_cache=AsyncMock(return_value="encrypted-payload"))
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
        mock_cache_backend.async_get_cache.assert_awaited_once_with(key="litellm:mcp:temporary_server:from-redis")

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

        server = generate_mock_mcp_server_config_record(server_id="from-redis-encrypted")
        serialized = json.dumps(server.model_dump(mode="json"))
        mock_cache_backend = SimpleNamespace(async_get_cache=AsyncMock(return_value="encrypted-payload"))
        original_cache = mgmt_endpoints.litellm.cache
        mgmt_endpoints.litellm.cache = SimpleNamespace(cache=mock_cache_backend)
        try:
            with patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.decrypt_value_helper",
                return_value=serialized,
            ) as decrypt_mock:
                result = await _get_temporary_mcp_server_from_redis("from-redis-encrypted")
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

        mock_cache_backend = SimpleNamespace(async_get_cache=AsyncMock(return_value="enc"))
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

        mock_cache_backend = SimpleNamespace(async_get_cache=AsyncMock(return_value="enc"))
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
        mock_cache_backend = SimpleNamespace(async_get_cache=AsyncMock(return_value=server.model_dump(mode="json")))
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
        mock_prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing_server)
        mock_prisma_client.db.litellm_mcpservertable.update = AsyncMock(return_value=updated_server)

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

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

            result = await edit_mcp_server(payload=update_request, user_api_key_dict=mock_user_auth)

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


class TestAddMCPServerAtomicity:
    """A committed MCP server must survive a post-write registry refresh failure.

    Regression: add_mcp_server inserted the row and then reloaded the whole
    registry from the database inside the same try block. One unrelated malformed
    row made the reload raise, so the endpoint returned 500 even though the new
    row was already persisted. Callers assumed failure and retried, creating
    duplicate servers.
    """

    @pytest.mark.asyncio
    async def test_create_succeeds_when_registry_refresh_fails(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_mcp_server,
        )

        payload = NewMCPServerRequest(
            alias="echo",
            url="https://echo.example.com/mcp",
            transport=MCPTransport.http,
        )
        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user")
        created_server = generate_mock_mcp_server_db_record(server_id="created-1", alias="echo")

        mock_manager = MagicMock()
        mock_manager.add_server = AsyncMock()
        mock_manager.reload_servers_from_database = AsyncMock(side_effect=Exception("malformed pre-existing row"))

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
                AsyncMock(return_value=created_server),
            ) as create_mock,
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await add_mcp_server(payload=payload, user_api_key_dict=admin)

        create_mock.assert_awaited_once()
        mock_manager.reload_servers_from_database.assert_awaited_once()
        assert result.server_id == "created-1"

    @pytest.mark.asyncio
    async def test_create_500s_and_skips_registry_when_db_write_fails(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_mcp_server,
        )

        payload = NewMCPServerRequest(
            alias="echo",
            url="https://echo.example.com/mcp",
            transport=MCPTransport.http,
        )
        admin = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user")

        mock_manager = MagicMock()
        mock_manager.add_server = AsyncMock()
        mock_manager.reload_servers_from_database = AsyncMock()

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
                AsyncMock(side_effect=Exception("db down")),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await add_mcp_server(payload=payload, user_api_key_dict=admin)

        assert exc_info.value.status_code == 500
        mock_manager.add_server.assert_not_awaited()
        mock_manager.reload_servers_from_database.assert_not_awaited()


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
        mock_manager.get_filtered_registry.return_value = {mock_server.server_id: mock_server}

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
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(return_value=[mock_health_result])

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

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)

        health_result_one = generate_mock_mcp_server_db_record(server_id="server-1", alias="One")
        health_result_one.status = "healthy"

        health_result_two = generate_mock_mcp_server_db_record(server_id="server-2", alias="Two")
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
            result = await register_mcp_server(payload=payload, user_api_key_dict=user_auth)

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
        summary = MCPSubmissionsSummary(total=1, pending_review=1, active=0, rejected=0, items=[pending])

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
    async def test_get_submissions_sanitizes_for_view_only_admin(self):
        """PROXY_ADMIN_VIEW_ONLY reviewing the submission queue must go through
        the non-admin sanitizer that fetch/list endpoints use: url,
        static_headers, env, env_vars, and credentials are all dropped. A
        mutation swapping the gate back to the old partial-blank pattern (which
        left url/static_headers/env and env-var names intact) would fail this."""
        from litellm.proxy._types import MCPSubmissionsSummary
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_mcp_server_submissions,
        )

        item = _leaky_list_server()
        item.approval_status = "pending_review"
        summary = MCPSubmissionsSummary(total=1, pending_review=1, active=0, rejected=0, items=[item])

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
            result = await get_mcp_server_submissions(
                user_api_key_dict=generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
            )

        assert len(result.items) == 1
        sanitized = result.items[0]
        assert sanitized.url is None
        assert sanitized.static_headers is None
        assert sanitized.env == {}
        assert sanitized.env_vars is None
        assert sanitized.credentials is None

        # The source record must not be mutated by sanitization.
        assert item.url == "https://leaky.example.com/mcp?api_key=sk-embedded-in-url"
        assert item.static_headers == {"Authorization": "Bearer sk-secret-header"}

    @pytest.mark.asyncio
    async def test_get_submissions_full_admin_still_sees_secrets(self):
        """The view-only redaction must not over-redact for a full PROXY_ADMIN,
        who needs url/static_headers/env/env_vars to review the pending
        submission. Only the explicit credentials field is cleared."""
        from litellm.proxy._types import MCPSubmissionsSummary
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_mcp_server_submissions,
        )

        item = _leaky_list_server()
        item.approval_status = "pending_review"
        summary = MCPSubmissionsSummary(total=1, pending_review=1, active=0, rejected=0, items=[item])

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
            result = await get_mcp_server_submissions(
                user_api_key_dict=generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN),
            )

        assert len(result.items) == 1
        raw = result.items[0]
        assert raw.url == "https://leaky.example.com/mcp?api_key=sk-embedded-in-url"
        assert raw.static_headers == {"Authorization": "Bearer sk-secret-header"}
        assert raw.env == {"UPSTREAM_TOKEN": "sk-secret-env"}
        assert raw.credentials is None
        assert raw.env_vars is not None
        assert len(raw.env_vars) == 1
        # ``model_construct`` in ``_leaky_list_server`` skips validation, so
        # env_vars stays as raw dicts; mirror the fixture shape here.
        entry = raw.env_vars[0]
        name = entry["name"] if isinstance(entry, dict) else entry.name
        value = entry["value"] if isinstance(entry, dict) else entry.value
        assert name == "GLOBAL_KEY"
        assert value == "super-secret"

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
                await approve_mcp_server_submission(server_id="server-1", user_api_key_dict=admin)
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
        approved_server.submitted_by = "submitter-user"

        mock_manager = MagicMock()
        mock_manager.invalidate_byom_submitted_servers_cache = AsyncMock()
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
            result = await approve_mcp_server_submission(server_id=pending_server.server_id, user_api_key_dict=admin)

        mock_manager.reload_servers_from_database.assert_awaited_once()
        mock_manager.invalidate_byom_submitted_servers_cache.assert_awaited_once_with("submitter-user")
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
        with patch_proxy_general_settings({"mcp_required_fields": ["source_url", "auth_type"]}):
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
            new=AsyncMock(return_value=generate_mock_mcp_server_db_record(server_id=server_id)),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=True,
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
async def test_store_mcp_oauth_user_credential_invalidates_cached_token():
    """Re-authorizing via the Tools-tab persist drops the v2 per-user token cache entry, so
    egress stops serving the replaced token immediately instead of until its TTL."""
    from litellm.proxy._types import MCPOAuthUserCredentialRequest

    if not mgmt_endpoints.MCP_AVAILABLE:
        pytest.skip("MCP module not installed")

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as manager_module
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        store_mcp_oauth_user_credential,
    )

    server_id = "srv-inv-1"
    user_id = "user-inv-1"
    invalidate_mock = AsyncMock(return_value=None)

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=_make_prisma_client(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            new=AsyncMock(return_value=generate_mock_mcp_server_db_record(server_id=server_id)),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=True,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.store_user_oauth_credential",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_user_oauth_credential",
            new=AsyncMock(return_value={"type": "oauth2", "access_token": "new-tok"}),
        ),
        patch.object(
            manager_module.global_mcp_server_manager,
            "invalidate_user_oauth_token_cache",
            new=invalidate_mock,
        ),
    ):
        await store_mcp_oauth_user_credential(
            server_id=server_id,
            payload=MCPOAuthUserCredentialRequest(access_token="new-tok", expires_in=3600),
            user_api_key_dict=_make_user_auth(user_id),
        )

    invalidate_mock.assert_awaited_once_with(user_id, server_id)


@pytest.mark.asyncio
async def test_delete_mcp_oauth_user_credential_invalidates_cached_token():
    """Revoking a stored OAuth credential drops the v2 per-user token cache entry, so the
    revoked token stops flowing upstream immediately instead of until its TTL."""
    if not mgmt_endpoints.MCP_AVAILABLE:
        pytest.skip("MCP module not installed")

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as manager_module
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        delete_mcp_oauth_user_credential,
    )

    server_id = "srv-inv-2"
    user_id = "user-inv-2"
    invalidate_mock = AsyncMock(return_value=None)

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=_make_prisma_client(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_user_oauth_credential",
            new=AsyncMock(return_value={"type": "oauth2", "access_token": "revoked-tok"}),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.delete_user_credential",
            new=AsyncMock(return_value=None),
        ),
        patch.object(
            manager_module.global_mcp_server_manager,
            "invalidate_user_oauth_token_cache",
            new=invalidate_mock,
        ),
    ):
        result = await delete_mcp_oauth_user_credential(
            server_id=server_id,
            user_api_key_dict=_make_user_auth(user_id),
        )

    invalidate_mock.assert_awaited_once_with(user_id, server_id)
    assert result.has_credential is False


@pytest.mark.asyncio
async def test_delete_mcp_oauth_user_credential_invalidates_when_record_already_gone():
    """A concurrent delete can remove the row between the read and the delete; the cache may
    still hold the revoked token, so the invalidate must fire even on RecordNotFoundError."""
    if not mgmt_endpoints.MCP_AVAILABLE:
        pytest.skip("MCP module not installed")

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as manager_module
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        delete_mcp_oauth_user_credential,
    )

    server_id = "srv-inv-3"
    user_id = "user-inv-3"
    invalidate_mock = AsyncMock(return_value=None)

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=_make_prisma_client(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_user_oauth_credential",
            new=AsyncMock(return_value={"type": "oauth2", "access_token": "revoked-tok"}),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.delete_user_credential",
            new=AsyncMock(side_effect=mgmt_endpoints.RecordNotFoundError({}, message="already gone")),
        ),
        patch.object(
            manager_module.global_mcp_server_manager,
            "invalidate_user_oauth_token_cache",
            new=invalidate_mock,
        ),
    ):
        result = await delete_mcp_oauth_user_credential(
            server_id=server_id,
            user_api_key_dict=_make_user_auth(user_id),
        )

    invalidate_mock.assert_awaited_once_with(user_id, server_id)
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
    mock_server = generate_mock_mcp_server_db_record(server_id=server_id, alias="My Server")
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
    server.token_exchange_endpoint = "https://idp/token-exchange"
    server.audience = "https://upstream/api"
    server.subject_token_type = "urn:ietf:params:oauth:token-type:jwt"
    server.token_exchange_profile = "entra_obo"

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
    # The token-exchange IdP endpoint is as sensitive as token_url; audience names the upstream.
    # subject_token_type is a public RFC 8693 URN, cleared for uniformity: non-admins
    # receive no token-exchange config at all.
    assert sanitized.token_exchange_endpoint is None
    assert sanitized.audience is None
    assert sanitized.subject_token_type is None
    assert sanitized.token_exchange_profile is None

    # Identity / metadata fields are preserved so the UI can list the
    # server without exposing secrets.
    assert sanitized.server_id == server.server_id
    assert sanitized.alias == server.alias


def _server_with_global_and_user_env_vars():
    base = generate_mock_mcp_server_db_record()
    return LiteLLM_MCPServerTable(
        **{
            **base.model_dump(),
            "env_vars": [
                {"name": "ADMIN_API_KEY", "value": "super-secret", "scope": "global"},
                {"name": "USER_TOKEN", "value": "placeholder-hint", "scope": "user"},
            ],
        }
    )


def test_sanitize_non_admin_drops_all_env_vars():
    """The non-admin view drops env vars entirely; even the names are admin
    config metadata (e.g. DB_PASSWORD) that must not leak. Non-admins get the
    per-user vars they need from the /user-env-vars/status endpoint."""
    import litellm.proxy.management_endpoints.mcp_management_endpoints as mgmt

    server = _server_with_global_and_user_env_vars()

    sanitized = mgmt._sanitize_mcp_server_for_non_admin(server)

    assert sanitized.env_vars is None

    # The original object must not be mutated.
    original_by_name = {ev.name: ev for ev in server.env_vars}
    assert original_by_name["ADMIN_API_KEY"].value == "super-secret"


def test_sanitize_virtual_key_drops_all_env_vars():
    """Virtual-key callers get a discovery-only view; env var entries (even the
    names, which are admin config metadata) must be dropped entirely, not just
    have their global values blanked."""
    import litellm.proxy.management_endpoints.mcp_management_endpoints as mgmt

    server = _server_with_global_and_user_env_vars()

    sanitized = mgmt._sanitize_mcp_server_for_virtual_key(server)

    assert sanitized.env_vars is None

    # The original object must not be mutated.
    assert server.env_vars[0].value == "super-secret"


def test_sanitize_virtual_key_clears_token_exchange_endpoint_and_audience():
    """Virtual-key callers must not receive the token-exchange IdP endpoint or audience,
    matching how token_url is scrubbed for the same view."""
    import litellm.proxy.management_endpoints.mcp_management_endpoints as mgmt

    server = generate_mock_mcp_server_db_record()
    server.token_url = "https://idp/token"
    server.token_exchange_endpoint = "https://idp/token-exchange"
    server.audience = "https://upstream/api"
    server.subject_token_type = "urn:ietf:params:oauth:token-type:jwt"
    server.token_exchange_profile = "entra_obo"

    sanitized = mgmt._sanitize_mcp_server_for_virtual_key(server)

    assert sanitized.token_url is None
    assert sanitized.token_exchange_endpoint is None
    assert sanitized.audience is None
    assert sanitized.subject_token_type is None
    assert sanitized.token_exchange_profile is None


def _server_with_env_vars(server_id: str = "srv-env"):
    base = generate_mock_mcp_server_db_record(server_id=server_id)
    return LiteLLM_MCPServerTable(
        **{
            **base.model_dump(),
            "env_vars": [
                {"name": "ADMIN_API_KEY", "value": "super-secret", "scope": "global"},
                {"name": "USER_TOKEN", "value": "placeholder-hint", "scope": "user"},
            ],
        }
    )


@pytest.mark.asyncio
async def test_fetch_single_mcp_server_env_vars_full_admin_vs_view_only():
    """full admins see admin-supplied global env var secrets so the edit
    form can pre-fill; read-only admins now go through the non-admin sanitizer, which
    drops env_vars entirely (the names alone, e.g. ADMIN_API_KEY, leak what secrets the
    admin configured). Previously the view-only case merely blanked the global value
    while keeping the names, which still leaked configuration metadata."""
    server = _server_with_env_vars()

    health_result = generate_mock_mcp_server_db_record(server_id=server.server_id)
    health_result.status = "healthy"
    health_result.last_health_check = datetime.now()
    health_result.health_check_error = None

    async def _fetch(user_role):
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.add_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.health_check_server",
                AsyncMock(return_value=health_result),
            ),
        ):
            return await mgmt_endpoints.fetch_mcp_server(
                request=_make_mock_request(),
                server_id=server.server_id,
                user_api_key_dict=generate_mock_user_api_key_auth(user_role=user_role),
            )

    full_admin = await _fetch(LitellmUserRoles.PROXY_ADMIN)
    by_name = {ev.name: ev for ev in full_admin.env_vars}
    assert by_name["ADMIN_API_KEY"].value == "super-secret"
    assert by_name["USER_TOKEN"].value == "placeholder-hint"

    view_only = await _fetch(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert view_only.env_vars is None

    # The source record must never be mutated.
    assert {ev.name: ev.value for ev in server.env_vars}["ADMIN_API_KEY"] == "super-secret"


@pytest.mark.asyncio
async def test_fetch_all_mcp_servers_env_vars_full_admin_vs_view_only():
    """same posture as the single-server fetch. Full admins
    keep the env var values; view-only admins get env_vars dropped via the non-admin
    sanitizer rather than only having the global value blanked."""
    server = _server_with_env_vars()

    async def _fetch_all(user_role):
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="view_all",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.get_all_mcp_servers_unfiltered",
                AsyncMock(return_value=[server]),
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                None,
            ),
        ):
            return await mgmt_endpoints.fetch_all_mcp_servers(
                user_api_key_dict=generate_mock_user_api_key_auth(user_role=user_role),
            )

    full_admin = await _fetch_all(LitellmUserRoles.PROXY_ADMIN)
    by_name = {ev.name: ev for ev in full_admin[0].env_vars}
    assert by_name["ADMIN_API_KEY"].value == "super-secret"
    assert by_name["USER_TOKEN"].value == "placeholder-hint"

    view_only = await _fetch_all(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert view_only[0].env_vars is None

    assert {ev.name: ev.value for ev in server.env_vars}["ADMIN_API_KEY"] == "super-secret"


def _leaky_list_server() -> "LiteLLM_MCPServerTable":
    """A server whose url/static_headers/env carry embedded secrets, for the
    list-endpoint sanitization tests. ``model_construct`` skips validation so
    the raw values survive verbatim."""
    return LiteLLM_MCPServerTable.model_construct(
        server_id="leaky-list-server",
        server_name="Leaky List Server",
        alias="Leaky List Server",
        transport=MCPTransport.http,
        url="https://leaky.example.com/mcp?api_key=sk-embedded-in-url",
        static_headers={"Authorization": "Bearer sk-secret-header"},
        env={"UPSTREAM_TOKEN": "sk-secret-env"},
        env_vars=[
            {"name": "GLOBAL_KEY", "value": "super-secret", "scope": "global"},
        ],
        credentials={"auth_value": "sk-explicit-credential"},
    )


async def _fetch_all_via_view_all(user_role: LitellmUserRoles):
    """Drive GET /v1/mcp/server in view_all mode for the given role using the
    real role helpers (the full-admin gate is never patched)."""
    server = _leaky_list_server()
    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
            return_value="view_all",
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager.get_all_mcp_servers_unfiltered",
            AsyncMock(return_value=[server]),
        ),
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ),
    ):
        result = await mgmt_endpoints.fetch_all_mcp_servers(
            user_api_key_dict=generate_mock_user_api_key_auth(user_role=user_role),
        )
    return server, result


@pytest.mark.asyncio
async def test_list_mcp_servers_sanitized_for_view_only_admin():
    """PROXY_ADMIN_VIEW_ONLY listing servers must go through the non-admin
    sanitizer: url and static_headers cleared, env emptied, env_vars dropped.
    A mutation swapping _user_is_full_admin() back to _user_has_admin_view()
    (which also grants view-only admins) would return the raw url/headers and
    fail this. The real role helpers are exercised; the gate is not patched."""
    source, result = await _fetch_all_via_view_all(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

    assert len(result) == 1
    sanitized = result[0]
    assert sanitized.server_id == "leaky-list-server"
    assert sanitized.url is None
    assert sanitized.static_headers is None
    assert sanitized.env == {}
    assert sanitized.env_vars is None
    assert sanitized.credentials is None

    # The source record must never be mutated by sanitization.
    assert source.url == "https://leaky.example.com/mcp?api_key=sk-embedded-in-url"
    assert source.static_headers == {"Authorization": "Bearer sk-secret-header"}


@pytest.mark.asyncio
async def test_list_mcp_servers_full_admin_still_sees_secrets():
    """The view-only redaction must not over-redact for a FULL PROXY_ADMIN,
    who needs url/static_headers to populate the edit form. Only the explicit
    credentials field is cleared for full admins on the list endpoint."""
    _, result = await _fetch_all_via_view_all(LitellmUserRoles.PROXY_ADMIN)

    assert len(result) == 1
    raw = result[0]
    assert raw.url == "https://leaky.example.com/mcp?api_key=sk-embedded-in-url"
    assert raw.static_headers == {"Authorization": "Bearer sk-secret-header"}
    assert raw.credentials is None


def _make_env_var_server(
    *,
    server_id: str = "srv-1",
    server_name: str = "DB Server",
    alias: str = "db_server",
    env_vars=None,
    static_headers=None,
):
    """Lightweight server stand-in for the per-user env-var endpoints.

    The handlers only read ``server_id``/``server_name``/``alias``/``env_vars``/
    ``static_headers`` via ``getattr``, so a SimpleNamespace is enough and keeps
    the test decoupled from the full Prisma model.
    """
    return SimpleNamespace(
        server_id=server_id,
        server_name=server_name,
        alias=alias,
        env_vars=env_vars,
        static_headers=static_headers,
    )


# env_vars with two referenced per-user fields, one unreferenced per-user field
# (must NOT be blocking), and a global value.
_ENV_VARS_MIXED = [
    {"name": "DB_PROTOCOL", "value": "postgres", "scope": "global"},
    {
        "name": "CORP_USERNAME",
        "value": "",
        "scope": "user",
        "description": "Your username",
    },
    {"name": "CORP_PASSWORD", "value": "", "scope": "user"},
    {"name": "UNUSED_USER_VAR", "value": "", "scope": "user"},
]
_STATIC_HEADERS_MIXED = {
    "Authorization": "${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@host/db",
}


class TestComputeUserEnvVarStatus:
    """Unit tests for the _compute_user_env_var_status helper."""

    def test_only_referenced_per_user_vars_are_required(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        status = mgmt_endpoints._compute_user_env_var_status(server=server, stored_values={"CORP_USERNAME": "alice"})
        names = {spec.name for spec in status.required}
        # UNUSED_USER_VAR is declared per-user but never referenced -> not blocking.
        assert names == {"CORP_USERNAME", "CORP_PASSWORD"}
        by_name = {spec.name: spec for spec in status.required}
        assert by_name["CORP_USERNAME"].is_set is True
        assert by_name["CORP_USERNAME"].description == "Your username"
        assert by_name["CORP_PASSWORD"].is_set is False
        # Stored credentials are write-only: the secret is never echoed back.
        assert "alice" not in status.model_dump_json()
        assert status.missing_count == 1
        assert status.server_id == "srv-1"
        assert status.server_name == "DB Server"
        assert status.alias == "db_server"
        # required is non-empty -> a setup URL is provided.
        assert status.setup_url and "srv-1" in status.setup_url

    def test_all_filled_has_zero_missing(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        status = mgmt_endpoints._compute_user_env_var_status(
            server=server,
            stored_values={"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"},
        )
        assert status.missing_count == 0
        assert all(spec.is_set for spec in status.required)

    def test_static_headers_as_json_string_is_parsed(self):
        server = _make_env_var_server(
            env_vars=_ENV_VARS_MIXED,
            static_headers='{"Authorization": "${CORP_USERNAME}"}',
        )
        status = mgmt_endpoints._compute_user_env_var_status(server=server, stored_values={})
        # Only CORP_USERNAME is referenced via the JSON-string headers.
        assert {spec.name for spec in status.required} == {"CORP_USERNAME"}
        assert status.missing_count == 1

    def test_static_headers_invalid_json_string_yields_no_required(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers="not-json{")
        status = mgmt_endpoints._compute_user_env_var_status(server=server, stored_values={})
        assert status.required == []
        assert status.missing_count == 0
        # No required fields -> no setup URL.
        assert status.setup_url is None

    def test_no_per_user_vars_referenced_yields_no_required(self):
        server = _make_env_var_server(
            env_vars=[{"name": "DB_PROTOCOL", "value": "postgres", "scope": "global"}],
            static_headers={"Authorization": "${DB_PROTOCOL}://host"},
        )
        status = mgmt_endpoints._compute_user_env_var_status(server=server, stored_values={})
        assert status.required == []
        assert status.setup_url is None

    def test_dual_scope_var_with_global_fallback_is_not_required(self):
        # SHARED_TOKEN is declared both global and user. The global value covers
        # the reference (globals win in _resolve_static_headers_with_env_vars),
        # so the tool-call path never raises a 412 for it. The status endpoint
        # must agree and not report it as required/missing, otherwise it asks the
        # user for a credential the request would never actually need.
        server = _make_env_var_server(
            env_vars=[
                {"name": "SHARED_TOKEN", "value": "global-secret", "scope": "global"},
                {"name": "SHARED_TOKEN", "value": "", "scope": "user"},
            ],
            static_headers={"Authorization": "Bearer ${SHARED_TOKEN}"},
        )
        status = mgmt_endpoints._compute_user_env_var_status(server=server, stored_values={})
        assert status.required == []
        assert status.missing_count == 0
        assert status.setup_url is None

    def test_dual_scope_var_with_empty_global_is_required(self):
        # SHARED_TOKEN is declared both global (empty value) and user. An empty
        # global is not a usable fallback, so _resolve_static_headers_with_env_vars
        # still requires the user value and the tool-call path 412s without it. The
        # status endpoint must agree and report it required, or it would tell the
        # user no credential is needed for a var every call rejects.
        server = _make_env_var_server(
            env_vars=[
                {"name": "SHARED_TOKEN", "value": "", "scope": "global"},
                {"name": "SHARED_TOKEN", "value": "", "scope": "user"},
            ],
            static_headers={"Authorization": "Bearer ${SHARED_TOKEN}"},
        )
        status = mgmt_endpoints._compute_user_env_var_status(server=server, stored_values={})
        assert {spec.name for spec in status.required} == {"SHARED_TOKEN"}
        assert status.missing_count == 1
        assert status.setup_url and "srv-1" in status.setup_url


class TestGetMCPUserEnvVars:
    @pytest.mark.asyncio
    async def test_returns_status_for_server(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints,
                "get_user_env_vars",
                AsyncMock(return_value={"CORP_USERNAME": "alice"}),
            ),
        ):
            result = await mgmt_endpoints.get_mcp_user_env_vars(
                server_id="srv-1",
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
            )
        assert result.server_id == "srv-1"
        assert result.missing_count == 1
        assert {s.name for s in result.required} == {"CORP_USERNAME", "CORP_PASSWORD"}
        # The single-server endpoint reports which credentials are set without
        # ever echoing the decrypted secret back to the caller.
        by_name = {s.name: s for s in result.required}
        assert by_name["CORP_USERNAME"].is_set is True
        assert by_name["CORP_PASSWORD"].is_set is False
        assert "alice" not in result.model_dump_json()

    @pytest.mark.asyncio
    async def test_missing_user_id_raises_400(self):
        with patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.get_mcp_user_env_vars(
                    server_id="srv-1",
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id=""),
                )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_server_raises_404(self):
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.get_mcp_user_env_vars(
                    server_id="missing",
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
                )
        assert exc.value.status_code == 404


class TestStoreMCPUserEnvVars:
    @pytest.mark.asyncio
    async def test_persists_only_allowed_non_empty_values(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        merge_mock = AsyncMock(return_value={"CORP_USERNAME": "alice"})
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(mgmt_endpoints, "merge_user_env_vars", merge_mock),
        ):
            result = await mgmt_endpoints.store_mcp_user_env_vars(
                server_id="srv-1",
                payload=mgmt_endpoints.MCPUserEnvVarsRequest(
                    values={
                        "CORP_USERNAME": "alice",
                        "CORP_PASSWORD": "",  # empty -> dropped
                        "NOT_A_DECLARED_VAR": "x",  # unknown -> dropped
                    }
                ),
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
            )
        # Only the declared, non-empty value reaches the atomic merge, scoped to
        # the admin-declared user vars.
        merge_mock.assert_awaited_once()
        _, _, _, updates, allowed_names = merge_mock.await_args.args
        assert updates == {"CORP_USERNAME": "alice"}
        assert set(allowed_names) == {
            "CORP_USERNAME",
            "CORP_PASSWORD",
            "UNUSED_USER_VAR",
        }
        # CORP_PASSWORD remains unset in the returned status.
        assert result.missing_count == 1

    @pytest.mark.asyncio
    async def test_forwards_only_submitted_updates_and_returns_merged_status(self):
        """The endpoint forwards only the user's submitted (allowed, non-empty)
        update to the atomic merge and reports status from the merged result, so
        a one-field edit never sends the other stored values back through."""
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        merge_mock = AsyncMock(return_value={"CORP_USERNAME": "alice", "CORP_PASSWORD": "new"})
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(mgmt_endpoints, "merge_user_env_vars", merge_mock),
        ):
            result = await mgmt_endpoints.store_mcp_user_env_vars(
                server_id="srv-1",
                payload=mgmt_endpoints.MCPUserEnvVarsRequest(values={"CORP_PASSWORD": "new"}),
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
            )
        merge_mock.assert_awaited_once()
        _, _, _, updates, _ = merge_mock.await_args.args
        assert updates == {"CORP_PASSWORD": "new"}
        # Status reflects the merged set returned by the atomic merge.
        assert result.missing_count == 0

    @pytest.mark.asyncio
    async def test_missing_user_id_raises_400(self):
        with patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.store_mcp_user_env_vars(
                    server_id="srv-1",
                    payload=mgmt_endpoints.MCPUserEnvVarsRequest(values={}),
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id=""),
                )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_server_raises_404(self):
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.store_mcp_user_env_vars(
                    server_id="missing",
                    payload=mgmt_endpoints.MCPUserEnvVarsRequest(values={}),
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
                )
        assert exc.value.status_code == 404


class TestClearMCPUserEnvVars:
    @pytest.mark.asyncio
    async def test_clears_and_returns_empty_status(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        delete_mock = AsyncMock()
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(mgmt_endpoints, "delete_user_env_vars", delete_mock),
        ):
            result = await mgmt_endpoints.clear_mcp_user_env_vars(
                server_id="srv-1",
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
            )
        delete_mock.assert_awaited_once()
        # Everything is now unset.
        assert result.missing_count == 2
        assert all(not spec.is_set for spec in result.required)

    @pytest.mark.asyncio
    async def test_delete_db_error_propagates(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints,
                "delete_user_env_vars",
                AsyncMock(side_effect=Exception("db down")),
            ),
        ):
            # A real DB failure must surface, not be masked as a successful clear.
            with pytest.raises(Exception, match="db down"):
                await mgmt_endpoints.clear_mcp_user_env_vars(
                    server_id="srv-1",
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
                )

    @pytest.mark.asyncio
    async def test_missing_user_id_raises_400(self):
        with patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.clear_mcp_user_env_vars(
                    server_id="srv-1",
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id=""),
                )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_server_raises_404(self):
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.clear_mcp_user_env_vars(
                    server_id="missing",
                    user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice"),
                )
        assert exc.value.status_code == 404


class TestListMCPUserEnvVarStatus:
    @pytest.mark.asyncio
    async def test_no_user_id_returns_empty(self):
        with patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()):
            result = await mgmt_endpoints.list_mcp_user_env_var_status(
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="")
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_accessible_servers_returns_empty(self):
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(
                mgmt_endpoints,
                "_resolve_accessible_mcp_servers",
                AsyncMock(return_value=[]),
            ),
        ):
            result = await mgmt_endpoints.list_mcp_user_env_var_status(
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice")
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_only_servers_with_required_fields_are_returned(self):
        server_with = _make_env_var_server(
            server_id="srv-with",
            env_vars=_ENV_VARS_MIXED,
            static_headers=_STATIC_HEADERS_MIXED,
        )
        # No per-user var is referenced -> contributes no status entry.
        server_without = _make_env_var_server(
            server_id="srv-without",
            env_vars=[{"name": "DB_PROTOCOL", "value": "postgres", "scope": "global"}],
            static_headers={"Authorization": "${DB_PROTOCOL}://host"},
        )
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(
                mgmt_endpoints,
                "_resolve_accessible_mcp_servers",
                AsyncMock(return_value=[server_with, server_without]),
            ),
            patch.object(
                mgmt_endpoints,
                "get_user_env_vars_bulk",
                AsyncMock(return_value={"srv-with": {"CORP_USERNAME": "alice"}}),
            ),
        ):
            result = await mgmt_endpoints.list_mcp_user_env_var_status(
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice")
            )
        assert [s.server_id for s in result] == ["srv-with"]
        assert result[0].missing_count == 1

    @pytest.mark.asyncio
    async def test_bulk_status_omits_stored_credential_values(self):
        """The bulk feed only drives the "fields missing" badge, so it must not
        echo stored credential values back; is_set still reflects presence."""
        server = _make_env_var_server(
            server_id="srv-with",
            env_vars=_ENV_VARS_MIXED,
            static_headers=_STATIC_HEADERS_MIXED,
        )
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(
                mgmt_endpoints,
                "_resolve_accessible_mcp_servers",
                AsyncMock(return_value=[server]),
            ),
            patch.object(
                mgmt_endpoints,
                "get_user_env_vars_bulk",
                AsyncMock(return_value={"srv-with": {"CORP_USERNAME": "alice"}}),
            ),
        ):
            result = await mgmt_endpoints.list_mcp_user_env_var_status(
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="alice")
            )
        by_name = {s.name: s for s in result[0].required}
        assert by_name["CORP_USERNAME"].is_set is True
        assert by_name["CORP_PASSWORD"].is_set is False
        assert "alice" not in result[0].model_dump_json()

    @pytest.mark.asyncio
    async def test_admin_view_all_flags_missing_fields_without_key_grants(self):
        """Regression: the red "user fields missing" card must light up for an
        admin in view_all mode even when their key carries no per-server MCP
        grant. The bulk status feed has to resolve the same server set the
        dashboard grid renders; the old narrow key-scoped listing returned
        nothing for such an admin, leaving every card un-highlighted."""
        server = _make_env_var_server(
            server_id="srv-with",
            env_vars=_ENV_VARS_MIXED,
            static_headers=_STATIC_HEADERS_MIXED,
        )
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(
                mgmt_endpoints,
                "_get_user_mcp_management_mode",
                return_value="view_all",
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_all_mcp_servers_unfiltered",
                AsyncMock(return_value=[server]),
            ),
            patch.object(
                mgmt_endpoints,
                "get_user_env_vars_bulk",
                AsyncMock(return_value={}),
            ),
        ):
            result = await mgmt_endpoints.list_mcp_user_env_var_status(
                user_api_key_dict=generate_mock_user_api_key_auth(
                    user_id="admin",
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                )
            )
        assert [s.server_id for s in result] == ["srv-with"]
        assert result[0].missing_count == 2
        assert {f.name for f in result[0].required} == {
            "CORP_USERNAME",
            "CORP_PASSWORD",
        }


class TestMCPUserEnvVarsAccessControl:
    """Per-server env-var endpoints must enforce the same access gate as
    fetch_mcp_server: a non-admin caller can only touch servers in their
    allowed set."""

    @pytest.mark.asyncio
    async def test_get_forbidden_for_non_admin_without_access(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        get_user_env_vars = AsyncMock(return_value={})
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_allowed_mcp_servers",
                AsyncMock(return_value=["other"]),
            ),
            patch.object(mgmt_endpoints, "get_user_env_vars", get_user_env_vars),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.get_mcp_user_env_vars(
                    server_id="srv-1",
                    user_api_key_dict=generate_mock_user_api_key_auth(
                        user_id="alice",
                        user_role=LitellmUserRoles.INTERNAL_USER,
                    ),
                )
        assert exc.value.status_code == 403
        get_user_env_vars.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_forbidden_for_non_admin_without_access(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        merge_mock = AsyncMock()
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_allowed_mcp_servers",
                AsyncMock(return_value=[]),
            ),
            patch.object(mgmt_endpoints, "merge_user_env_vars", merge_mock),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.store_mcp_user_env_vars(
                    server_id="srv-1",
                    payload=mgmt_endpoints.MCPUserEnvVarsRequest(values={"CORP_USERNAME": "alice"}),
                    user_api_key_dict=generate_mock_user_api_key_auth(
                        user_id="alice",
                        user_role=LitellmUserRoles.INTERNAL_USER,
                    ),
                )
        assert exc.value.status_code == 403
        merge_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_forbidden_for_non_admin_without_access(self):
        server = _make_env_var_server(env_vars=_ENV_VARS_MIXED, static_headers=_STATIC_HEADERS_MIXED)
        delete_mock = AsyncMock()
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_allowed_mcp_servers",
                AsyncMock(return_value=[]),
            ),
            patch.object(mgmt_endpoints, "delete_user_env_vars", delete_mock),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.clear_mcp_user_env_vars(
                    server_id="srv-1",
                    user_api_key_dict=generate_mock_user_api_key_auth(
                        user_id="alice",
                        user_role=LitellmUserRoles.INTERNAL_USER,
                    ),
                )
        assert exc.value.status_code == 403
        delete_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_allowed_for_non_admin_with_access(self):
        server = _make_env_var_server(
            server_id="srv-1",
            env_vars=_ENV_VARS_MIXED,
            static_headers=_STATIC_HEADERS_MIXED,
        )
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_allowed_mcp_servers",
                AsyncMock(return_value=["srv-1"]),
            ),
            patch.object(
                mgmt_endpoints,
                "get_user_env_vars",
                AsyncMock(return_value={"CORP_USERNAME": "alice"}),
            ),
        ):
            result = await mgmt_endpoints.get_mcp_user_env_vars(
                server_id="srv-1",
                user_api_key_dict=generate_mock_user_api_key_auth(
                    user_id="alice",
                    user_role=LitellmUserRoles.INTERNAL_USER,
                ),
            )
        assert result.server_id == "srv-1"
        assert result.missing_count == 1

    @pytest.mark.asyncio
    async def test_admin_bypasses_access_check(self):
        """Proxy admins must not be filtered by the allowed-server check."""
        server = _make_env_var_server(
            server_id="srv-1",
            env_vars=_ENV_VARS_MIXED,
            static_headers=_STATIC_HEADERS_MIXED,
        )
        allowed_mock = AsyncMock(return_value=[])
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=server)),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_allowed_mcp_servers",
                allowed_mock,
            ),
            patch.object(mgmt_endpoints, "get_user_env_vars", AsyncMock(return_value={})),
        ):
            result = await mgmt_endpoints.get_mcp_user_env_vars(
                server_id="srv-1",
                user_api_key_dict=generate_mock_user_api_key_auth(
                    user_id="admin",
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                ),
            )
        assert result.server_id == "srv-1"
        allowed_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_admin_gets_403_not_404_for_inaccessible_server(self):
        """A non-admin cannot distinguish "server does not exist" (404) from
        "server exists but you lack access" (403): both collapse to 403 so server
        ids stay non-enumerable, even when neither the DB nor the registry has the
        server."""
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_mcp_server_by_id",
                MagicMock(return_value=None),
            ),
            patch.object(
                mgmt_endpoints.global_mcp_server_manager,
                "get_allowed_mcp_servers",
                AsyncMock(return_value=[]),
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.get_mcp_user_env_vars(
                    server_id="srv-1",
                    user_api_key_dict=generate_mock_user_api_key_auth(
                        user_id="alice",
                        user_role=LitellmUserRoles.INTERNAL_USER,
                    ),
                )
        assert exc.value.status_code == 403


def test_oauth2_flow_accepted_on_create_request():
    """NewMCPServerRequest carries oauth2_flow through to the persisted dict."""
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data

    payload = NewMCPServerRequest(
        server_name="m2m-server",
        url="https://example.com/mcp",
        transport="http",
        auth_type="oauth2",
        token_url="https://idp.example.com/oauth/token",
        oauth2_flow="client_credentials",
    )
    data_dict = _prepare_mcp_server_data(payload)
    assert data_dict["oauth2_flow"] == "client_credentials"


def test_oauth2_flow_round_trips_on_update_and_response_models():
    """oauth2_flow survives UpdateMCPServerRequest and the LiteLLM_MCPServerTable
    response model. Before the fix these models dropped the field (no attribute),
    which is why a persisted value never round-tripped."""
    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        UpdateMCPServerRequest,
    )

    update = UpdateMCPServerRequest(server_id="srv-1", oauth2_flow="client_credentials")
    assert update.oauth2_flow == "client_credentials"

    row = LiteLLM_MCPServerTable(
        server_id="srv-1",
        transport="http",
        oauth2_flow="client_credentials",
    )
    assert row.oauth2_flow == "client_credentials"


def test_oauth2_flow_defaults_to_none_when_omitted():
    """Omitting oauth2_flow is valid and resolves to None (runtime infers it)."""
    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        UpdateMCPServerRequest,
    )

    assert UpdateMCPServerRequest(server_id="srv-1").oauth2_flow is None
    assert LiteLLM_MCPServerTable(server_id="srv-1", transport="http").oauth2_flow is None


def test_dcr_bridge_rejected_on_create_for_gateway_managed_auth_type():
    from pydantic import ValidationError

    from litellm.proxy._types import NewMCPServerRequest

    with pytest.raises(ValidationError) as exc:
        NewMCPServerRequest(
            server_name="bridge-server",
            url="https://example.com/mcp",
            transport="http",
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            dcr_bridge=True,
        )
    assert "dcr_bridge is only supported" in str(exc.value)


def test_dcr_bridge_rejected_on_create_when_auth_type_omitted():
    from pydantic import ValidationError

    from litellm.proxy._types import NewMCPServerRequest

    with pytest.raises(ValidationError) as exc:
        NewMCPServerRequest(
            server_name="bridge-server",
            url="https://example.com/mcp",
            transport="http",
            dcr_bridge=True,
        )
    assert "dcr_bridge is only supported" in str(exc.value)


@pytest.mark.parametrize("auth_type", ["true_passthrough", "oauth_delegate"])
def test_dcr_bridge_accepted_on_create_for_client_forwarded_modes(auth_type):
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import NewMCPServerRequest

    payload = NewMCPServerRequest(
        server_name="bridge-server",
        url="https://example.com/mcp",
        transport="http",
        auth_type=auth_type,
        dcr_bridge=True,
    )
    data_dict = _prepare_mcp_server_data(payload)
    assert data_dict["dcr_bridge"] is True


def test_dcr_bridge_update_rejected_when_payload_auth_type_not_client_forwarded():
    from pydantic import ValidationError

    from litellm.proxy._types import UpdateMCPServerRequest

    with pytest.raises(ValidationError) as exc:
        UpdateMCPServerRequest(server_id="srv-1", auth_type="oauth2", dcr_bridge=True)
    assert "dcr_bridge is only supported" in str(exc.value)


def test_dcr_bridge_update_without_auth_type_defers_to_endpoint():
    from litellm.proxy._types import UpdateMCPServerRequest

    assert UpdateMCPServerRequest(server_id="srv-1", dcr_bridge=True).dcr_bridge is True


def test_dcr_bridge_round_trips_on_response_model():
    from litellm.proxy._types import LiteLLM_MCPServerTable

    row = LiteLLM_MCPServerTable(server_id="srv-1", transport="http", dcr_bridge=True)
    assert row.dcr_bridge is True
    assert LiteLLM_MCPServerTable(server_id="srv-1", transport="http").dcr_bridge is None


def _edit_endpoint_patches(old_record, update_mock):
    return (
        patch("litellm.proxy.management_endpoints.mcp_management_endpoints.MCP_AVAILABLE", True),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            AsyncMock(side_effect=old_record)
            if isinstance(old_record, Exception)
            else AsyncMock(return_value=old_record),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.update_mcp_server",
            update_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            autospec=True,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_auth_type", ["oauth2", "api_key", "none"])
async def test_edit_mcp_server_rejects_dcr_bridge_when_stored_auth_type_not_client_forwarded(stored_auth_type):
    from litellm.proxy._types import UpdateMCPServerRequest
    from litellm.proxy.management_endpoints.mcp_management_endpoints import edit_mcp_server

    old_record = MagicMock()
    old_record.auth_type = stored_auth_type
    update_mock = AsyncMock()
    p1, p2, p3, p4, p5 = _edit_endpoint_patches(old_record, update_mock)
    with p1, p2, p3, p4, p5:
        payload = UpdateMCPServerRequest(server_id="srv-1", dcr_bridge=True)
        user_auth = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        with pytest.raises(HTTPException) as exc:
            await edit_mcp_server(payload=payload, user_api_key_dict=user_auth)

    assert exc.value.status_code == 400
    assert "dcr_bridge is only supported" in str(exc.value.detail)
    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_edit_mcp_server_rejects_dcr_bridge_when_stored_record_unreadable():
    from litellm.proxy._types import UpdateMCPServerRequest
    from litellm.proxy.management_endpoints.mcp_management_endpoints import edit_mcp_server

    update_mock = AsyncMock()
    p1, p2, p3, p4, p5 = _edit_endpoint_patches(RuntimeError("db down"), update_mock)
    with p1, p2, p3, p4, p5:
        payload = UpdateMCPServerRequest(server_id="srv-1", dcr_bridge=True)
        user_auth = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        with pytest.raises(HTTPException) as exc:
            await edit_mcp_server(payload=payload, user_api_key_dict=user_auth)

    assert exc.value.status_code == 400
    update_mock.assert_not_called()


@pytest.mark.asyncio
async def test_edit_mcp_server_dcr_bridge_on_unknown_server_returns_404_not_400():
    """A dcr_bridge enablement targeting a server_id that does not exist must surface the accurate
    404 from the update path, not a misleading 400 about the stored auth_type: get_mcp_server
    returns None for a missing row without raising, which is distinct from a failed read."""
    from litellm.proxy._types import UpdateMCPServerRequest
    from litellm.proxy.management_endpoints.mcp_management_endpoints import edit_mcp_server

    update_mock = AsyncMock(return_value=None)
    p1, p2, p3, p4, p5 = _edit_endpoint_patches(None, update_mock)
    with p1, p2, p3, p4, p5:
        payload = UpdateMCPServerRequest(server_id="does-not-exist", dcr_bridge=True)
        user_auth = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        with pytest.raises(HTTPException) as exc:
            await edit_mcp_server(payload=payload, user_api_key_dict=user_auth)

    assert exc.value.status_code == 404
    update_mock.assert_called_once()


class TestPerUserCredentialConfigServerResolution:
    """Per-user credential and env-var endpoints must resolve config-defined MCP
    servers, which live only in the in-memory registry and never get a DB row, so
    a user can store their BYOK key / OAuth token / env vars against them. The
    same allowed-server authorization the MCP gateway enforces also gates these
    writes for non-admins.
    """

    # 32-char sha256 stable id, the shape a config.yaml server gets.
    CONFIG_SERVER_ID = "3a6a3f8633340371b49562c8c4682da9"

    def _registry_only_manager(self, *, is_byok: bool = False):
        """A manager mock where the server exists only in the registry (DB miss)."""
        config_server = generate_mock_mcp_server_config_record(server_id=self.CONFIG_SERVER_ID, name="Config Server")
        record = generate_mock_mcp_server_db_record(server_id=self.CONFIG_SERVER_ID).model_copy(
            update={"is_byok": is_byok}
        )
        manager = MagicMock()
        manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda sid: config_server if sid == self.CONFIG_SERVER_ID else None
        )
        manager._build_mcp_server_table = MagicMock(return_value=record)
        manager.get_allowed_mcp_servers = AsyncMock(return_value=[])
        return manager

    @pytest.mark.asyncio
    async def test_store_oauth_credential_resolves_config_server_for_admin(self):
        """OBO token persists for a config-defined server (DB miss, registry hit).
        Before the registry fallback this raised 404 "MCP Server not found"."""
        manager = self._registry_only_manager()
        store_mock = AsyncMock(return_value=None)
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
            patch.object(mgmt_endpoints, "global_mcp_server_manager", manager),
            patch.object(mgmt_endpoints, "store_user_oauth_credential", store_mock),
            patch.object(
                mgmt_endpoints,
                "get_user_oauth_credential",
                AsyncMock(return_value={"expires_at": None}),
            ),
        ):
            result = await mgmt_endpoints.store_mcp_oauth_user_credential(
                server_id=self.CONFIG_SERVER_ID,
                payload=mgmt_endpoints.MCPOAuthUserCredentialRequest(access_token="tok", expires_in=3600),
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="admin"),
            )
        assert result.has_credential is True
        store_mock.assert_awaited_once()
        manager.get_mcp_server_by_id.assert_called_once_with(self.CONFIG_SERVER_ID)

    @pytest.mark.asyncio
    async def test_store_byok_credential_resolves_config_server_for_admin(self):
        """BYOK key persists for a config-defined BYOK server (DB miss, registry
        hit). Before the registry fallback this raised 404."""
        manager = self._registry_only_manager(is_byok=True)
        store_mock = AsyncMock(return_value=None)
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
            patch.object(mgmt_endpoints, "global_mcp_server_manager", manager),
            patch.object(mgmt_endpoints, "store_user_credential", store_mock),
        ):
            result = await mgmt_endpoints.store_mcp_user_credential(
                server_id=self.CONFIG_SERVER_ID,
                payload=mgmt_endpoints.MCPUserCredentialRequest(credential="my-key"),
                user_api_key_dict=generate_mock_user_api_key_auth(user_id="admin"),
            )
        assert result.has_credential is True
        store_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_oauth_credential_forbidden_for_non_admin_without_access(self):
        """A non-admin storing a credential for a server not in their allowed set
        gets 403 and no row is written (the store endpoints had no authz before)."""
        manager = self._registry_only_manager()
        manager.get_allowed_mcp_servers = AsyncMock(return_value=[])
        store_mock = AsyncMock(return_value=None)
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
            patch.object(mgmt_endpoints, "global_mcp_server_manager", manager),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(mgmt_endpoints, "store_user_oauth_credential", store_mock),
        ):
            with pytest.raises(HTTPException) as exc:
                await mgmt_endpoints.store_mcp_oauth_user_credential(
                    server_id=self.CONFIG_SERVER_ID,
                    payload=mgmt_endpoints.MCPOAuthUserCredentialRequest(access_token="tok", expires_in=3600),
                    user_api_key_dict=generate_mock_user_api_key_auth(
                        user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER
                    ),
                )
        assert exc.value.status_code == 403
        store_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_oauth_credential_allowed_for_non_admin_with_access(self):
        """A non-admin with the config server in their allowed set persists the
        token; proves the non-admin authz uses the registry-aware allowed set."""
        manager = self._registry_only_manager()
        manager.get_allowed_mcp_servers = AsyncMock(return_value=[self.CONFIG_SERVER_ID])
        store_mock = AsyncMock(return_value=None)
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
            patch.object(mgmt_endpoints, "global_mcp_server_manager", manager),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(mgmt_endpoints, "store_user_oauth_credential", store_mock),
            patch.object(
                mgmt_endpoints,
                "get_user_oauth_credential",
                AsyncMock(return_value={"expires_at": None}),
            ),
        ):
            result = await mgmt_endpoints.store_mcp_oauth_user_credential(
                server_id=self.CONFIG_SERVER_ID,
                payload=mgmt_endpoints.MCPOAuthUserCredentialRequest(access_token="tok", expires_in=3600),
                user_api_key_dict=generate_mock_user_api_key_auth(
                    user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER
                ),
            )
        assert result.has_credential is True
        store_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_env_vars_resolves_config_server_for_non_admin_with_access(
        self,
    ):
        """Per-user env vars persist for a config server a non-admin may access.
        The non-admin path previously used a DB-only access list that never
        included config servers, so this 403'd before the fix."""
        env_var_server = _make_env_var_server(
            server_id=self.CONFIG_SERVER_ID,
            env_vars=_ENV_VARS_MIXED,
            static_headers=_STATIC_HEADERS_MIXED,
        )
        manager = MagicMock()
        manager.get_mcp_server_by_id = MagicMock(
            return_value=generate_mock_mcp_server_config_record(server_id=self.CONFIG_SERVER_ID)
        )
        manager._build_mcp_server_table = MagicMock(return_value=env_var_server)
        manager.get_allowed_mcp_servers = AsyncMock(return_value=[self.CONFIG_SERVER_ID])
        merge_mock = AsyncMock(return_value={"CORP_USERNAME": "alice"})
        with (
            patch.object(mgmt_endpoints, "get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(mgmt_endpoints, "get_mcp_server", AsyncMock(return_value=None)),
            patch.object(mgmt_endpoints, "global_mcp_server_manager", manager),
            patch.object(
                mgmt_endpoints,
                "build_effective_auth_contexts",
                AsyncMock(return_value=[object()]),
            ),
            patch.object(mgmt_endpoints, "merge_user_env_vars", merge_mock),
        ):
            result = await mgmt_endpoints.store_mcp_user_env_vars(
                server_id=self.CONFIG_SERVER_ID,
                payload=mgmt_endpoints.MCPUserEnvVarsRequest(values={"CORP_USERNAME": "alice"}),
                user_api_key_dict=generate_mock_user_api_key_auth(
                    user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER
                ),
            )
        merge_mock.assert_awaited_once()
        _, _, _, updates, _ = merge_mock.await_args.args
        assert updates == {"CORP_USERNAME": "alice"}
        assert result.server_id == self.CONFIG_SERVER_ID


def _oauth2_create_payload(**overrides):
    base = dict(
        server_name="stamp_test_server",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type="oauth2",
    )
    base.update(overrides)
    return NewMCPServerRequest(**base)


def test_stamp_oauth2_flow_bare_oauth2_defaults_to_authorization_code():
    """A bare oauth2 create (no endpoints, no creds) is interactive: stamping it
    authorization_code matches how needs_user_oauth_token treats a null flow."""
    payload = _oauth2_create_payload()
    mgmt_endpoints.stamp_omitted_oauth2_flow(payload)
    assert payload.oauth2_flow == "authorization_code"


def test_stamp_oauth2_flow_marks_m2m_shape_client_credentials():
    """token_url + full client credentials and no authorization_url is the M2M shape;
    the stamp mirrors the legacy inference in _resolve_oauth2_flow so REST-created M2M
    servers persist the flow instead of relying on read-time inference."""
    payload = _oauth2_create_payload(
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    mgmt_endpoints.stamp_omitted_oauth2_flow(payload)
    assert payload.oauth2_flow == "client_credentials"


def test_stamp_oauth2_flow_authorization_url_wins_over_m2m_shape():
    """An authorization endpoint means interactive even when client creds + token_url
    are present (GitHub Enterprise style); M2M never has an authorization endpoint."""
    payload = _oauth2_create_payload(
        authorization_url="https://idp.example.com/authorize",
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    mgmt_endpoints.stamp_omitted_oauth2_flow(payload)
    assert payload.oauth2_flow == "authorization_code"


def test_stamp_oauth2_flow_respects_explicit_value():
    """An explicit oauth2_flow from the caller must never be overridden by the stamp."""
    payload = _oauth2_create_payload(
        oauth2_flow="authorization_code",
        token_url="https://idp.example.com/token",
        credentials={"client_id": "cid", "client_secret": "csecret"},
    )
    mgmt_endpoints.stamp_omitted_oauth2_flow(payload)
    assert payload.oauth2_flow == "authorization_code"


def test_stamp_oauth2_flow_ignores_non_oauth2():
    payload = _oauth2_create_payload(auth_type="none")
    mgmt_endpoints.stamp_omitted_oauth2_flow(payload)
    assert payload.oauth2_flow is None


class TestHealthCheckServersIncludesNames:
    @pytest.mark.asyncio
    async def test_health_view_all_includes_server_name_and_alias(self):
        server = generate_mock_mcp_server_db_record(server_id="server-1", alias="server_one")
        server.server_name = "server_one"
        server.status = "healthy"

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_unfiltered = AsyncMock(return_value=[server])

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.PROXY_ADMIN)

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="view_all",
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_servers,
            )

            result = await health_check_servers(server_ids=None, user_api_key_dict=mock_user_auth)

        assert result == [
            {
                "server_id": "server-1",
                "server_name": "server_one",
                "alias": "server_one",
                "status": "healthy",
            }
        ]

    @pytest.mark.asyncio
    async def test_health_restricted_virtual_key_does_not_get_unfiltered_names(self):
        scoped = generate_mock_mcp_server_db_record(server_id="server-1", alias="scoped_one")
        scoped.server_name = "scoped_one"
        scoped.status = "healthy"
        hidden = generate_mock_mcp_server_db_record(server_id="server-2", alias="hidden_two")
        hidden.server_name = "hidden_two"
        hidden.status = "healthy"

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_unfiltered = AsyncMock(return_value=[scoped, hidden])
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(return_value=[scoped])

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)
        mock_user_auth.allowed_routes = ["/v1/mcp/server/health"]

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="view_all",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_servers,
            )

            result = await health_check_servers(server_ids=None, user_api_key_dict=mock_user_auth)

        mock_manager.get_all_mcp_servers_with_health_unfiltered.assert_not_awaited()
        assert [entry["server_id"] for entry in result] == ["server-1"]
        assert all("hidden_two" not in str(entry.values()) for entry in result)

    @pytest.mark.asyncio
    async def test_health_scoped_mode_includes_server_name_and_alias(self):
        server_a = generate_mock_mcp_server_db_record(server_id="server-1", alias="cit")
        server_a.server_name = "cit"
        server_a.status = "unknown"
        server_b = generate_mock_mcp_server_db_record(server_id="server-2", alias="server-two-alias")
        server_b.server_name = "server_two"
        server_b.status = "healthy"

        mock_manager = MagicMock()
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(return_value=[server_a, server_b])

        mock_user_auth = generate_mock_user_api_key_auth(user_role=LitellmUserRoles.INTERNAL_USER)

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                mock_manager,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_user_mcp_management_mode",
                return_value="self_serve",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.build_effective_auth_contexts",
                AsyncMock(return_value=[mock_user_auth]),
            ),
        ):
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_servers,
            )

            result = await health_check_servers(server_ids=None, user_api_key_dict=mock_user_auth)

        by_id = {entry["server_id"]: entry for entry in result}
        assert by_id["server-1"]["server_name"] == "cit"
        assert by_id["server-1"]["alias"] == "cit"
        assert by_id["server-1"]["status"] == "unknown"
        assert by_id["server-2"]["server_name"] == "server_two"
        assert by_id["server-2"]["alias"] == "server-two-alias"
        assert by_id["server-2"]["status"] == "healthy"


async def _run_edit(old_record, updated_record, purge_mock=None):
    from litellm.proxy.management_endpoints.mcp_management_endpoints import edit_mcp_server

    server_id = updated_record.server_id
    with (
        patch("litellm.proxy.management_endpoints.mcp_management_endpoints.MCP_AVAILABLE", True),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            AsyncMock(side_effect=old_record)
            if isinstance(old_record, Exception)
            else AsyncMock(return_value=old_record),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.update_mcp_server",
            AsyncMock(return_value=updated_record),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            autospec=True,
        ),
        patch("litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager") as mock_manager,
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.purge_user_oauth_credentials_for_server",
            purge_mock if purge_mock is not None else AsyncMock(return_value=1),
        ) as mock_purge,
    ):
        mock_manager.update_server = AsyncMock()
        mock_manager.reload_servers_from_database = AsyncMock()
        payload = UpdateMCPServerRequest(server_id=server_id, alias=updated_record.alias, url=updated_record.url)
        user_auth = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        result = await edit_mcp_server(payload=payload, user_api_key_dict=user_auth)
        return result, mock_purge


@pytest.mark.asyncio
async def test_edit_mcp_server_purges_user_tokens_on_mint_relevant_change():
    server_id = str(uuid.uuid4())
    old = generate_mock_mcp_server_db_record(server_id=server_id, url="https://old.example.com/mcp")
    updated = generate_mock_mcp_server_db_record(server_id=server_id, url="https://new.example.com/mcp")

    result, mock_purge = await _run_edit(old, updated)

    assert result.server_id == server_id
    mock_purge.assert_awaited_once()
    assert mock_purge.await_args.args[1] == server_id


@pytest.mark.asyncio
async def test_edit_mcp_server_skips_purge_when_identity_unchanged():
    server_id = str(uuid.uuid4())
    old = generate_mock_mcp_server_db_record(server_id=server_id, alias="Before")
    updated = generate_mock_mcp_server_db_record(server_id=server_id, alias="After")

    result, mock_purge = await _run_edit(old, updated)

    assert result.server_id == server_id
    mock_purge.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_mcp_server_purge_failure_does_not_fail_the_edit():
    """The purge is best-effort: a purge exception after a successful update must be swallowed and
    logged, never turned into an error response for an edit whose primary job already succeeded."""
    server_id = str(uuid.uuid4())
    old = generate_mock_mcp_server_db_record(server_id=server_id, url="https://old.example.com/mcp")
    updated = generate_mock_mcp_server_db_record(server_id=server_id, url="https://new.example.com/mcp")

    result, mock_purge = await _run_edit(old, updated, purge_mock=AsyncMock(side_effect=RuntimeError("db down")))

    assert result.server_id == server_id
    mock_purge.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_mcp_server_snapshot_failure_skips_purge_but_edit_succeeds():
    """The pre-update snapshot read is advisory (it only feeds the purge decision); a read failure
    must skip the stale-token check with a warning, never fail the edit itself."""
    server_id = str(uuid.uuid4())
    updated = generate_mock_mcp_server_db_record(server_id=server_id, url="https://new.example.com/mcp")

    result, mock_purge = await _run_edit(RuntimeError("db read failed"), updated)

    assert result.server_id == server_id
    mock_purge.assert_not_awaited()

import json
import os
import sys
from litellm._uuid import uuid
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from typing import Optional

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LitellmUserRoles,
    MCPTransport,
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

        # Mock the new method that returns servers with health and team data
        mock_servers_with_health = [
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
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=mock_servers_with_health
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=True,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results
            assert len(result) == 2

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

        # Mock the new method that returns servers with health and team data
        mock_servers_with_health = [
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
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=mock_servers_with_health
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=True,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results
            assert len(result) == 4

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

        # Mock the new method that returns servers with health and team data
        mock_servers_with_health = [
            db_server_allowed,
            generate_mock_mcp_server_db_record(
                server_id="config_server_allowed",
                alias="Allowed Zapier MCP",
                url="https://actions.zapier.com/mcp/sse",
            ),
        ]
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=mock_servers_with_health
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=False,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify results - should only return servers user has access to
            assert len(result) == 2

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


class TestMCPHealthCheckEndpoints:
    """Test MCP health check endpoints"""

    @pytest.mark.asyncio
    async def test_health_check_mcp_server_success(self):
        """Test successful health check for a specific MCP server"""
        # Mock server
        mock_server = generate_mock_mcp_server_db_record(
            server_id="test-server", alias="Test Server"
        )

        # Mock dependencies
        mock_prisma_client = MagicMock()

        # Mock global MCP server manager
        mock_manager = MagicMock()
        mock_manager.health_check_server = AsyncMock(
            return_value={
                "server_id": "test-server",
                "status": "healthy",
                "tools_count": 3,
                "last_health_check": "2024-01-01T12:00:00",
                "response_time_ms": 150.5,
                "error": None,
            }
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=True,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            AsyncMock(return_value=mock_server),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_mcp_server,
            )

            result = await health_check_mcp_server(
                server_id="test-server", user_api_key_dict=mock_user_auth
            )

            # Verify results
            assert result["server_id"] == "test-server"
            assert result["status"] == "healthy"
            assert result["tools_count"] == 3
            assert result["response_time_ms"] == 150.5
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_health_check_mcp_server_not_found(self):
        """Test health check for a server that doesn't exist"""
        # Mock dependencies
        mock_prisma_client = MagicMock()

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            AsyncMock(return_value=None),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_mcp_server,
            )

            # Should raise HTTPException
            with pytest.raises(Exception) as exc_info:
                await health_check_mcp_server(
                    server_id="non-existent-server", user_api_key_dict=mock_user_auth
                )

            assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_health_check_mcp_server_unauthorized(self):
        """Test health check for a server user doesn't have access to"""
        # Mock server
        mock_server = generate_mock_mcp_server_db_record(
            server_id="test-server", alias="Test Server"
        )

        # Mock dependencies
        mock_prisma_client = MagicMock()

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER  # Non-admin user
        )

        # Mock user doesn't have access to this server
        mock_user_servers = []

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=False,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            return_value=mock_user_servers,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            AsyncMock(return_value=mock_server),
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_mcp_server,
            )

            # Should raise HTTPException
            with pytest.raises(Exception) as exc_info:
                await health_check_mcp_server(
                    server_id="test-server", user_api_key_dict=mock_user_auth
                )

            assert "permission" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_health_check_all_mcp_servers(self):
        """Test health check for all accessible MCP servers"""
        # Mock team records
        team_records = [
            generate_mock_team_record(
                team_id="team1",
                team_alias="Team 1",
                organization_id="org1",
                mcp_servers=["server1", "server2"],
            )
        ]

        # Mock DB servers
        db_servers = [
            generate_mock_mcp_server_db_record(server_id="server1"),
            generate_mock_mcp_server_db_record(server_id="server2"),
        ]

        # Mock dependencies
        mock_prisma_client = MagicMock()
        mock_prisma_client = setup_mock_prisma_client(
            mock_prisma_client=mock_prisma_client,
            team_records=team_records,
            mcp_servers=db_servers,
        )

        # Mock global MCP server manager
        mock_manager = MagicMock()
        mock_manager.health_check_allowed_servers = AsyncMock(
            return_value={
                "server1": {
                    "server_id": "server1",
                    "status": "healthy",
                    "tools_count": 2,
                    "last_health_check": "2024-01-01T12:00:00",
                    "response_time_ms": 100.0,
                    "error": None,
                },
                "server2": {
                    "server_id": "server2",
                    "status": "unhealthy",
                    "last_health_check": "2024-01-01T12:00:00",
                    "response_time_ms": 5000.0,
                    "error": "Connection timeout",
                },
            }
        )
        mock_manager.get_allowed_mcp_servers = AsyncMock(
            return_value=["server1", "server2"]
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.INTERNAL_USER
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=False,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                health_check_all_mcp_servers,
            )

            result = await health_check_all_mcp_servers(
                user_api_key_dict=mock_user_auth
            )

            # Verify results
            assert result["total_servers"] == 2
            assert result["healthy_count"] == 1
            assert result["unhealthy_count"] == 1
            assert result["unknown_count"] == 0
            assert "server1" in result["servers"]
            assert "server2" in result["servers"]

            # Check individual server results
            assert result["servers"]["server1"]["status"] == "healthy"
            assert result["servers"]["server1"]["tools_count"] == 2
            assert result["servers"]["server2"]["status"] == "unhealthy"
            assert result["servers"]["server2"]["error"] == "Connection timeout"

    @pytest.mark.asyncio
    async def test_fetch_all_mcp_servers_with_health_status(self):
        """Test that fetch_all_mcp_servers includes health check status"""
        # Mock server with health status
        mock_server = generate_mock_mcp_server_db_record(
            server_id="test-server", alias="Test Server"
        )
        # Add health status to the mock server
        mock_server.status = "healthy"
        mock_server.last_health_check = datetime.now()
        mock_server.health_check_error = None

        # Mock dependencies
        mock_prisma_client = MagicMock()
        mock_prisma_client = setup_mock_prisma_client(
            mock_prisma_client=mock_prisma_client,
            team_records=[],
            mcp_servers=[],  # Don't add servers here since we're mocking get_all_mcp_servers
        )

        # Mock global MCP server manager
        mock_manager = MagicMock()
        mock_manager.config_mcp_servers = {}
        mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=[])
        mock_manager.get_all_mcp_servers_with_health_and_teams = AsyncMock(
            return_value=[mock_server]
        )

        mock_user_auth = generate_mock_user_api_key_auth(
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=mock_prisma_client,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._user_has_admin_view",
            return_value=True,
        ), patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ):
            # Import and call the function
            from litellm.proxy.management_endpoints.mcp_management_endpoints import (
                fetch_all_mcp_servers,
            )

            result = await fetch_all_mcp_servers(user_api_key_dict=mock_user_auth)

            # Verify health check status is included
            assert len(result) == 1
            server = result[0]
            assert server.server_id == "test-server"
            assert server.status == "healthy"
            assert server.last_health_check is not None
            assert server.health_check_error is None

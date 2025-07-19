import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import Request, FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from starlette.datastructures import Headers

from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._types import SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


@pytest.mark.asyncio
class TestMCPRequestHandler:

    @pytest.mark.parametrize(
        "user_api_key_auth,object_permission_id,prisma_client_available,db_result,expected_result",
        [
            # Test case 1: user_api_key_auth is None
            (None, None, True, None, []),
            # Test case 2: object_permission_id is None
            (UserAPIKeyAuth(), None, True, None, []),
            # Test case 3: prisma_client is None
            (
                UserAPIKeyAuth(object_permission_id="test-id"),
                "test-id",
                False,
                None,
                [],
            ),
            # Test case 4: Database query returns None
            (UserAPIKeyAuth(object_permission_id="test-id"), "test-id", True, None, []),
            # Test case 5: Database query returns object with mcp_servers
            (
                UserAPIKeyAuth(object_permission_id="test-id"),
                "test-id",
                True,
                MagicMock(mcp_servers=["server1", "server2"]),
                ["server1", "server2"],
            ),
            # Test case 6: Database query returns object with None mcp_servers
            (
                UserAPIKeyAuth(object_permission_id="test-id"),
                "test-id",
                True,
                MagicMock(mcp_servers=None),
                [],
            ),
            # Test case 7: Database query returns object with empty mcp_servers
            (
                UserAPIKeyAuth(object_permission_id="test-id"),
                "test-id",
                True,
                MagicMock(mcp_servers=[]),
                [],
            ),
        ],
    )
    async def test_get_allowed_mcp_servers_for_key(
        self,
        user_api_key_auth,
        object_permission_id,
        prisma_client_available,
        db_result,
        expected_result,
    ):
        """Test _get_allowed_mcp_servers_for_key with various scenarios"""

        # Setup user_api_key_auth object_permission_id if provided
        if user_api_key_auth and object_permission_id:
            user_api_key_auth.object_permission_id = object_permission_id

            # Mock prisma_client
        mock_prisma_client = MagicMock() if prisma_client_available else None
        mock_find_unique = None

        if mock_prisma_client:
            # Mock the database query
            mock_find_unique = AsyncMock(return_value=db_result)
            mock_prisma_client.db.litellm_objectpermissiontable.find_unique = (
                mock_find_unique
            )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            # Call the method
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(
                user_api_key_auth
            )

            # Assert the result (order-independent comparison)
            assert sorted(result) == sorted(expected_result)

            # Verify database call was made correctly when expected
            if (
                user_api_key_auth
                and user_api_key_auth.object_permission_id
                and prisma_client_available
                and mock_find_unique
            ):
                mock_find_unique.assert_called_once_with(
                    where={
                        "object_permission_id": user_api_key_auth.object_permission_id
                    }
                )
            elif mock_find_unique:
                # If prisma_client exists but conditions aren't met, no call should be made
                if not user_api_key_auth or not user_api_key_auth.object_permission_id:
                    mock_find_unique.assert_not_called()

    @pytest.mark.parametrize(
        "headers,expected_api_key,expected_mcp_auth_header",
        [
            # Test case 1: x-litellm-api-key header present
            (
                [(b"x-litellm-api-key", b"test-api-key-123")],
                "test-api-key-123",
                None,
            ),
            # Test case 2: Authorization header present (fallback)
            (
                [(b"authorization", b"Bearer test-auth-token")],
                "Bearer test-auth-token",
                None,
            ),
            # Test case 3: Both headers present (primary should win)
            (
                [
                    (b"x-litellm-api-key", b"primary-key"),
                    (b"authorization", b"Bearer fallback-token"),
                ],
                "primary-key",
                None,
            ),
            # Test case 4: Case insensitive headers
            (
                [(b"X-LITELLM-API-KEY", b"case-insensitive-key")],
                "case-insensitive-key",
                None,
            ),
            # Test case 5: No relevant headers
            (
                [(b"content-type", b"application/json")],
                "",
                None,
            ),
            # Test case 6: Empty headers
            ([], "", None),
            # Test case 7: MCP auth header present
            (
                [
                    (b"x-litellm-api-key", b"test-api-key-123"),
                    (b"x-mcp-auth", b"mcp-auth-token"),
                ],
                "test-api-key-123",
                "mcp-auth-token",
            ),
            # Test case 8: Only MCP auth header present (no API key)
            (
                [(b"x-mcp-auth", b"mcp-auth-token")],
                "",
                "mcp-auth-token",
            ),
        ],
    )
    async def test_process_mcp_request(self, headers, expected_api_key, expected_mcp_auth_header):
        """Test process_mcp_request method with various header scenarios"""

        # Create ASGI scope with headers
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": headers,
        }

        # Create an async mock for user_api_key_auth
        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(
                token="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" if api_key else None,
                api_key=api_key,
                user_id="test-user-id" if api_key else None,
                team_id="test-team-id" if api_key else None,
                user_role=None,
                request_route=None
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ) as mock_auth:
            # Call the method
            auth_result, mcp_auth_header, mcp_servers = await MCPRequestHandler.process_mcp_request(scope)

            # Assert the results
            assert auth_result.api_key == expected_api_key
            assert auth_result.user_id == ("test-user-id" if expected_api_key else None)
            assert auth_result.team_id == ("test-team-id" if expected_api_key else None)
            assert mcp_auth_header == expected_mcp_auth_header
            # For these tests, mcp_servers should be None
            assert mcp_servers is None

    @pytest.mark.parametrize(
        "headers,expected_result",
        [
            # Test case 1: All headers present
            (
                [
                    (b"x-litellm-api-key", b"test-api-key"),
                    (b"x-mcp-auth", b"test-mcp-auth"),
                    (b"x-mcp-servers", b"server1,server2"),
                ],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": "test-mcp-auth",
                    "mcp_servers": ["server1", "server2"],
                }
            ),
            # Test case 2: Only API key present
            (
                [(b"x-litellm-api-key", b"test-api-key")],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": None,
                }
            ),
            # Test case 3: Invalid format in mcp_servers
            (
                [
                    (b"x-litellm-api-key", b"test-api-key"),
                    (b"x-mcp-servers", b"[invalid,format]"),
                ],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": ["[invalid", "format]"],
                }
            ),
            # Test case 4: Single server
            (
                [
                    (b"x-litellm-api-key", b"test-api-key"),
                    (b"x-mcp-servers", b"server1"),
                ],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": ["server1"],
                }
            ),
            # Test case 5: Empty server string
            (
                [
                    (b"x-litellm-api-key", b"test-api-key"),
                    (b"x-mcp-servers", b""),
                ],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": [],
                }
            ),
            # Test case 6: Using Authorization header instead of x-litellm-api-key
            (
                [
                    (b"authorization", b"Bearer test-api-key"),
                    (b"x-mcp-servers", b"server1"),
                ],
                {
                    "api_key": "Bearer test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": ["server1"],
                }
            ),
            # Test case 7: Case insensitive header names
            (
                [
                    (b"X-LITELLM-API-KEY", b"test-api-key"),
                    (b"X-MCP-AUTH", b"test-mcp-auth"),
                    (b"X-MCP-SERVERS", b"server1"),
                ],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": "test-mcp-auth",
                    "mcp_servers": ["server1"],
                }
            ),
            # Test case 8: Multiple servers with spaces
            (
                [
                    (b"x-litellm-api-key", b"test-api-key"),
                    (b"x-mcp-servers", b"server1, server2,  server3"),
                ],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": ["server1", "server2", "server3"],
                }
            ),
        ]
    )
    async def test_header_extraction(self, headers, expected_result):
        """Test header extraction and processing from ASGI scope"""
        
        # Create ASGI scope with headers
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": headers,
        }

        # Get headers using the internal method
        extracted_headers = MCPRequestHandler._safe_get_headers_from_scope(scope)

        # Verify API key extraction
        api_key = MCPRequestHandler.get_litellm_api_key_from_headers(extracted_headers)
        assert api_key == expected_result["api_key"]

        # Verify MCP auth header
        mcp_auth = extracted_headers.get(SpecialHeaders.mcp_auth.value)
        assert mcp_auth == expected_result["mcp_auth"]

        # Verify MCP servers
        mcp_servers_header = extracted_headers.get(SpecialHeaders.mcp_servers.value)
        mcp_servers = None
        if mcp_servers_header is not None:  # Changed from 'if mcp_servers_header:' to handle empty strings
            try:
                # First try to parse as JSON array for backward compatibility
                try:
                    mcp_servers = json.loads(mcp_servers_header)
                    if not isinstance(mcp_servers, list):
                        mcp_servers = None
                except (json.JSONDecodeError, TypeError, ValueError):
                    # If JSON parsing fails, treat as comma-separated list
                    mcp_servers = [s.strip() for s in mcp_servers_header.split(",") if s.strip()]
            except Exception:
                mcp_servers = None

            # If we got an empty string or parsing resulted in no servers, return empty list
            if mcp_servers_header == "" or (mcp_servers is not None and len(mcp_servers) == 0):
                mcp_servers = []

        assert mcp_servers == expected_result["mcp_servers"]

        # Test the full process_mcp_request method
        mock_auth_result = UserAPIKeyAuth(
            api_key=expected_result["api_key"],
            user_id="test-user-id",
            team_id="test-team-id",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth"
        ) as mock_user_api_key_auth:
            mock_user_api_key_auth.return_value = mock_auth_result

            # Call the method
            auth_result, mcp_auth_header, mcp_servers_result = await MCPRequestHandler.process_mcp_request(scope)
            assert auth_result == mock_auth_result
            assert mcp_auth_header == expected_result["mcp_auth"]
            assert mcp_servers_result == expected_result["mcp_servers"]


class TestMCPCustomHeaderName:
    """Test suite for custom MCP authentication header name functionality"""

    @pytest.mark.parametrize(
        "env_var,general_setting,expected_header_name",
        [
            # Test case 1: Default behavior (no custom settings)
            (None, None, "x-mcp-auth"),
            # Test case 2: Environment variable set
            ("custom-mcp-header", None, "custom-mcp-header"),
            # Test case 3: General setting set (env var takes precedence)
            (None, "settings-mcp-header", "settings-mcp-header"),
            # Test case 4: Both set (env var takes precedence)
            ("env-mcp-header", "settings-mcp-header", "env-mcp-header"),
            # Test case 5: Empty env var (should fallback to default due to 'or' logic)
            ("", "settings-mcp-header", "x-mcp-auth"),
            # Test case 6: Empty general setting (should fallback to default)
            (None, "", "x-mcp-auth"),
        ],
    )
    def test_get_mcp_client_side_auth_header_name(
        self, env_var, general_setting, expected_header_name
    ):
        """Test that custom header name configuration works correctly"""
        
        # Mock the secret manager and general settings
        with patch("litellm.secret_managers.main.get_secret_str") as mock_get_secret:
            with patch("litellm.proxy.proxy_server.general_settings") as mock_general_settings:
                
                # Configure mocks
                mock_get_secret.return_value = env_var
                mock_general_settings.get.return_value = general_setting
                
                # Call the method
                result = MCPRequestHandler._get_mcp_client_side_auth_header_name()
                
                # Assert the result
                assert result == expected_header_name
                
                # Verify secret manager was called (the function calls it twice)
                expected_secret_calls = 2 if env_var is not None else 1
                assert mock_get_secret.call_count == expected_secret_calls
                
                # Verify all calls were with the correct parameter
                for call in mock_get_secret.call_args_list:
                    assert call.args == ("LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME",)
                
                # Verify general settings was called based on env var value
                if env_var is None:
                    # When env var is None, general settings should be checked (twice if not None)
                    expected_general_calls = 2 if general_setting is not None else 1
                    assert mock_general_settings.get.call_count == expected_general_calls
                    for call in mock_general_settings.get.call_args_list:
                        assert call.args == ("mcp_client_side_auth_header_name",)
                else:
                    # If env var is set (even empty string), general settings shouldn't be checked
                    mock_general_settings.get.assert_not_called()

    @pytest.mark.parametrize(
        "custom_header_name,headers,expected_auth_header",
        [
            # Test case 1: Default header name
            (
                "x-mcp-auth", 
                [(b"x-mcp-auth", b"default-auth-token")],
                "default-auth-token"
            ),
            # Test case 2: Custom header name
            (
                "custom-auth-header",
                [(b"custom-auth-header", b"custom-auth-token")],
                "custom-auth-token"
            ),
            # Test case 3: Custom header name with case insensitive
            (
                "Custom-Auth-Header",
                [(b"custom-auth-header", b"case-insensitive-token")],
                "case-insensitive-token"
            ),
            # Test case 4: Header not present
            (
                "missing-header",
                [(b"x-mcp-auth", b"wrong-header-token")],
                None
            ),
            # Test case 5: Multiple headers, only custom one should be used
            (
                "my-custom-auth",
                [
                    (b"x-mcp-auth", b"default-token"),
                    (b"my-custom-auth", b"custom-token")
                ],
                "custom-token"
            ),
        ],
    )
    def test_get_mcp_auth_header_from_headers_with_custom_name(
        self, custom_header_name, headers, expected_auth_header
    ):
        """Test that MCP auth header extraction uses custom header name"""
        
        # Mock the header name method
        with patch.object(
            MCPRequestHandler, 
            '_get_mcp_client_side_auth_header_name',
            return_value=custom_header_name
        ):
            # Create headers from the test data
            scope = {
                "type": "http",
                "method": "POST", 
                "path": "/test",
                "headers": headers,
            }
            extracted_headers = MCPRequestHandler._safe_get_headers_from_scope(scope)
            
            # Call the method
            result = MCPRequestHandler._get_mcp_auth_header_from_headers(extracted_headers)
            
            # Assert the result
            assert result == expected_auth_header

    @pytest.mark.asyncio
    async def test_process_mcp_request_with_custom_auth_header(self):
        """Test that process_mcp_request works with custom authentication header"""
        
        custom_header_name = "x-custom-mcp-auth"
        custom_auth_token = "custom-auth-token-123"
        api_key = "test-api-key"
        
        # Create ASGI scope with custom header
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [
                (b"x-litellm-api-key", api_key.encode()),
                (custom_header_name.encode(), custom_auth_token.encode()),
            ],
        }
        
        # Mock the custom header name method
        with patch.object(
            MCPRequestHandler,
            '_get_mcp_client_side_auth_header_name',
            return_value=custom_header_name
        ):
            # Mock user_api_key_auth
            mock_auth_result = UserAPIKeyAuth(
                api_key=api_key,
                user_id="test-user-id",
                team_id="test-team-id",
            )
            
            with patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth"
            ) as mock_user_api_key_auth:
                mock_user_api_key_auth.return_value = mock_auth_result
                
                # Call the method
                auth_result, mcp_auth_header, mcp_servers = await MCPRequestHandler.process_mcp_request(scope)
                
                # Assert the results
                assert auth_result == mock_auth_result
                assert mcp_auth_header == custom_auth_token
                assert mcp_servers is None
                
                # Verify user_api_key_auth was called with correct API key
                mock_user_api_key_auth.assert_called_once()
                call_args = mock_user_api_key_auth.call_args
                assert call_args.kwargs["api_key"] == api_key

class TestMCPAccessGroupsE2E:
    """Simple e2e tests for MCP access groups functionality"""

    @pytest.mark.asyncio 
    async def test_mcp_access_group_resolution_e2e(self):
        """E2E test: Pass access group in x-mcp-servers header, verify it resolves to server IDs"""
        
        # Mock servers that belong to the access group
        mock_server_1 = MagicMock()
        mock_server_1.server_id = "gmail-server"
        mock_server_2 = MagicMock()  
        mock_server_2.server_id = "slack-server"
        
        # Mock prisma client
        mock_prisma_client = MagicMock()
        mock_find_many = AsyncMock(return_value=[mock_server_1, mock_server_2])
        mock_prisma_client.db.litellm_mcpservertable.find_many = mock_find_many
        
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            # Test that when we pass an access group, it calls _get_mcp_servers_from_access_groups
            result = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                access_groups=["dev-group"]
            )
            
            # Verify it unfurls the MCPs from that group
            assert sorted(result) == sorted(["gmail-server", "slack-server"])
            
            # Verify the correct database query was made
            mock_find_many.assert_called_once_with(
                where={
                    "mcp_access_groups": {
                        "hasSome": ["dev-group"]
                    }
                }
            )

    @pytest.mark.asyncio
    async def test_mcp_header_with_mixed_servers_and_groups(self):
        """E2E test: x-mcp-servers header with both server names and access groups"""
        
        # Create ASGI scope with mixed servers and access groups
        scope = {
            "type": "http", 
            "method": "POST",
            "path": "/test",
            "headers": [
                (b"x-litellm-api-key", b"test-api-key"),
                (b"x-mcp-servers", b"zapier-server,dev-group"),  # Mix of server name and access group
            ],
        }
        
        # Mock user authentication
        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(
                token="test-token",
                api_key=api_key,
                user_id="test-user-id",
                team_id="test-team-id",
                user_role=None,
                request_route=None
            )
        
        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ):
            # Call process_mcp_request 
            auth_result, mcp_auth_header, mcp_servers = await MCPRequestHandler.process_mcp_request(scope)
            
            # Verify the header parsing worked correctly
            assert auth_result.api_key == "test-api-key"
            assert mcp_servers == ["zapier-server", "dev-group"]  # Should contain both server name and access group


@pytest.mark.asyncio
def test_mcp_path_based_server_segregation(monkeypatch):
    # Import the MCP server FastAPI app and context getter
    from litellm.proxy._experimental.mcp_server.server import app, get_auth_context

    captured_mcp_servers = {}

    # Patch the session manager to send a dummy response and capture context
    async def dummy_handle_request(scope, receive, send):
        from litellm.proxy._experimental.mcp_server.server import get_auth_context
        _, _, mcp_servers = get_auth_context()
        captured_mcp_servers[id(scope)] = mcp_servers
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"ok": true}',
        })

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.session_manager",
        MagicMock(handle_request=dummy_handle_request)
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.initialize_session_managers",
        AsyncMock()
    )

    # Patch user_api_key_auth to always return a dummy user
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
        AsyncMock(return_value=UserAPIKeyAuth(api_key="test", user_id="user"))
    )

    # Use TestClient to make a request to /mcp/zapier,group1/tools
    client = TestClient(app)
    response = client.get("/mcp/zapier,group1/tools", headers={"x-litellm-api-key": "test"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    # The context should have mcp_servers set to ["zapier", "group1"]
    assert list(captured_mcp_servers.values())[0] == ["zapier", "group1"]

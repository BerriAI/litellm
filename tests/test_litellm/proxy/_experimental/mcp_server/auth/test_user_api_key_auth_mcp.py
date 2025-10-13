import json
import os
import sys
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import FastAPI, Request
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
        "key_servers,team_servers,expected_result,scenario",
        [
            # Test case 1: No key servers, no team servers
            ([], [], [], "no_permissions"),
            # Test case 2: Key has servers, no team servers
            (["server1", "server2"], [], ["server1", "server2"], "key_only"),
            # Test case 3: No key servers, team has servers (inherit from team)
            (
                [],
                ["team_server1", "team_server2"],
                ["team_server1", "team_server2"],
                "inherit_from_team",
            ),
            # Test case 4: Key and team both have servers (intersection)
            (
                ["server1", "server2"],
                ["server1", "team_server"],
                ["server1"],
                "intersection",
            ),
            # Test case 5: Key and team have no overlap (empty result)
            (
                ["server1", "server2"],
                ["team_server1", "team_server2"],
                [],
                "no_overlap",
            ),
            # Test case 6: Key and team have complete overlap
            (
                ["server1", "server2"],
                ["server1", "server2"],
                ["server1", "server2"],
                "complete_overlap",
            ),
        ],
    )
    async def test_get_allowed_mcp_servers(
        self,
        key_servers,
        team_servers,
        expected_result,
        scenario,
    ):
        """Test get_allowed_mcp_servers with various key/team permission scenarios"""

        # Create a mock user
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
        )

        # Mock the helper methods instead of database calls
        with patch.object(
            MCPRequestHandler, "_get_allowed_mcp_servers_for_key"
        ) as mock_key_servers:
            with patch.object(
                MCPRequestHandler, "_get_allowed_mcp_servers_for_team"
            ) as mock_team_servers:
                # Set up return values
                mock_key_servers.return_value = key_servers
                mock_team_servers.return_value = team_servers

                # Call the method
                result = await MCPRequestHandler.get_allowed_mcp_servers(
                    user_api_key_auth=mock_user_auth
                )

                # Assert the result (order-independent comparison)
                assert sorted(result) == sorted(expected_result)

                # Verify helper methods were called
                mock_key_servers.assert_called_once_with(mock_user_auth)
                mock_team_servers.assert_called_once_with(mock_user_auth)

    @pytest.mark.parametrize(
        "team_servers,key_servers,expected_servers,scenario",
        [
            # Test case 1: Key has no permissions, should inherit from team
            (["server1", "server2"], [], ["server1", "server2"], "inherit_from_team"),
            # Test case 2: Key has permissions, should use intersection with team
            (
                ["server1", "server2", "server3"],
                ["server2", "server4"],
                ["server2"],
                "intersection_logic",
            ),
            # Test case 3: Key has permissions but no overlap with team
            (["server1", "server2"], ["server3", "server4"], [], "no_overlap"),
            # Test case 4: Team has no permissions, use key permissions
            ([], ["server1", "server2"], ["server1", "server2"], "no_team_permissions"),
            # Test case 5: Both team and key have no permissions
            ([], [], [], "no_permissions"),
            # Test case 6: Team has permissions, key has subset
            (
                ["server1", "server2", "server3"],
                ["server1", "server3"],
                ["server1", "server3"],
                "key_subset",
            ),
            # Test case 7: Team has permissions, key has superset (intersection should limit)
            (
                ["server1", "server2"],
                ["server1", "server2", "server3"],
                ["server1", "server2"],
                "key_superset",
            ),
        ],
    )
    async def test_get_allowed_mcp_servers_inheritance_logic(
        self, team_servers, key_servers, expected_servers, scenario
    ):
        """Test the inheritance and intersection logic in get_allowed_mcp_servers"""

        # Create mock user_api_key_auth
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team" if team_servers else None,
            object_permission_id="test-permission" if key_servers else None,
        )

        # Mock the helper functions
        with patch.object(
            MCPRequestHandler, "_get_allowed_mcp_servers_for_key"
        ) as mock_key_servers:
            with patch.object(
                MCPRequestHandler, "_get_allowed_mcp_servers_for_team"
            ) as mock_team_servers:
                # Configure mocks to return the test data
                mock_key_servers.return_value = key_servers
                mock_team_servers.return_value = team_servers

                # Call the method
                result = await MCPRequestHandler.get_allowed_mcp_servers(
                    user_api_key_auth
                )

                # Assert the result (order-independent comparison)
                assert sorted(result) == sorted(expected_servers)

                # Verify the mock functions were called correctly
                mock_key_servers.assert_called_once_with(user_api_key_auth)
                mock_team_servers.assert_called_once_with(user_api_key_auth)

    async def test_permission_inheritance_edge_cases(self):
        """Test edge cases in permission inheritance"""

        # Test case: None values in database
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_objectpermissiontable.find_unique.return_value = (
            None
        )
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = None

        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            object_permission_id="test-permission",
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)
            assert result == []

        # Test case: Exception handling
        mock_prisma_client.db.litellm_objectpermissiontable.find_unique.side_effect = (
            Exception("DB Error")
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)
            assert result == []  # Should handle exception gracefully

    @pytest.mark.parametrize(
        "headers,expected_api_key,expected_mcp_auth_header,expected_server_auth_headers",
        [
            # Test case 1: x-litellm-api-key header present
            (
                [(b"x-litellm-api-key", b"test-api-key-123")],
                "test-api-key-123",
                None,
                {},
            ),
            # Test case 2: Authorization header present (fallback)
            (
                [(b"authorization", b"Bearer test-auth-token")],
                "Bearer test-auth-token",
                None,
                {},
            ),
            # Test case 3: Both headers present (primary should win)
            (
                [
                    (b"x-litellm-api-key", b"primary-key"),
                    (b"authorization", b"Bearer fallback-token"),
                ],
                "primary-key",
                None,
                {},
            ),
            # Test case 4: Case insensitive headers
            (
                [(b"X-LITELLM-API-KEY", b"case-insensitive-key")],
                "case-insensitive-key",
                None,
                {},
            ),
            # Test case 5: No relevant headers
            (
                [(b"content-type", b"application/json")],
                "",
                None,
                {},
            ),
            # Test case 6: Empty headers
            ([], "", None, {}),
            # Test case 7: Legacy MCP auth header present
            (
                [
                    (b"x-litellm-api-key", b"test-api-key-123"),
                    (b"x-mcp-auth", b"mcp-auth-token"),
                ],
                "test-api-key-123",
                "mcp-auth-token",
                {},
            ),
            # Test case 8: Only legacy MCP auth header present (no API key)
            (
                [(b"x-mcp-auth", b"mcp-auth-token")],
                "",
                "mcp-auth-token",
                {},
            ),
            # Test case 9: Server-specific auth headers present
            (
                [
                    (b"x-litellm-api-key", b"test-api-key-123"),
                    (b"x-mcp-github-authorization", b"Bearer github-token"),
                    (b"x-mcp-zapier_x_api-key", b"zapier-api-key"),
                ],
                "test-api-key-123",
                None,
                {
                    "github": {"Authorization": "Bearer github-token"},
                    "zapier_x_api": {"key": "zapier-api-key"},
                },
            ),
            # Test case 10: Both legacy and server-specific auth headers
            (
                [
                    (b"x-litellm-api-key", b"test-api-key-123"),
                    (b"x-mcp-auth", b"legacy-token"),
                    (b"x-mcp-github-authorization", b"Bearer github-token"),
                ],
                "test-api-key-123",
                "legacy-token",
                {"github": {"Authorization": "Bearer github-token"}},
            ),
            # Test case 11: Server-specific auth headers with different header types
            (
                [
                    (b"x-litellm-api-key", b"test-api-key-123"),
                    (b"x-mcp-deepwiki-authorization", b"Basic base64-encoded"),
                    (b"x-mcp-custom_x_custom-header", b"custom-value"),
                ],
                "test-api-key-123",
                None,
                {
                    "deepwiki": {"Authorization": "Basic base64-encoded"},
                    "custom_x_custom": {"header": "custom-value"},
                },
            ),
            # Test case 12: Case insensitive server-specific headers
            (
                [
                    (b"x-litellm-api-key", b"test-api-key-123"),
                    (b"X-MCP-GITHUB-AUTHORIZATION", b"Bearer github-token"),
                ],
                "test-api-key-123",
                None,
                {"github": {"Authorization": "Bearer github-token"}},
            ),
        ],
    )
    async def test_process_mcp_request_with_server_auth_headers(
        self,
        headers,
        expected_api_key,
        expected_mcp_auth_header,
        expected_server_auth_headers,
    ):
        """Test process_mcp_request method with server-specific auth headers"""

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
                token=(
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    if api_key
                    else None
                ),
                api_key=api_key,
                user_id="test-user-id" if api_key else None,
                team_id="test-team-id" if api_key else None,
                user_role=None,
                request_route=None,
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ) as mock_auth:
            # Call the method
            (
                auth_result,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            # Assert the results
            assert auth_result.api_key == expected_api_key
            assert auth_result.user_id == ("test-user-id" if expected_api_key else None)
            assert auth_result.team_id == ("test-team-id" if expected_api_key else None)
            assert mcp_auth_header == expected_mcp_auth_header
            assert mcp_server_auth_headers == expected_server_auth_headers
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
                },
            ),
            # Test case 2: Only API key present
            (
                [(b"x-litellm-api-key", b"test-api-key")],
                {
                    "api_key": "test-api-key",
                    "mcp_auth": None,
                    "mcp_servers": None,
                },
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
                },
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
                },
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
                },
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
                },
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
                },
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
                },
            ),
        ],
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
        if (
            mcp_servers_header is not None
        ):  # Changed from 'if mcp_servers_header:' to handle empty strings
            try:
                # First try to parse as JSON array for backward compatibility
                try:
                    mcp_servers = json.loads(mcp_servers_header)
                    if not isinstance(mcp_servers, list):
                        mcp_servers = None
                except (json.JSONDecodeError, TypeError, ValueError):
                    # If JSON parsing fails, treat as comma-separated list
                    mcp_servers = [
                        s.strip() for s in mcp_servers_header.split(",") if s.strip()
                    ]
            except Exception:
                mcp_servers = None

            # If we got an empty string or parsing resulted in no servers, return empty list
            if mcp_servers_header == "" or (
                mcp_servers is not None and len(mcp_servers) == 0
            ):
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
            (
                auth_result,
                mcp_auth_header,
                mcp_servers_result,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)
            assert auth_result == mock_auth_result
            assert mcp_auth_header == expected_result["mcp_auth"]
            assert mcp_servers_result == expected_result["mcp_servers"]
            # For these tests, mcp_server_auth_headers should be empty
            assert mcp_server_auth_headers == {}


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
            with patch(
                "litellm.proxy.proxy_server.general_settings"
            ) as mock_general_settings:
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
                    assert (
                        mock_general_settings.get.call_count == expected_general_calls
                    )
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
                "default-auth-token",
            ),
            # Test case 2: Custom header name
            (
                "custom-auth-header",
                [(b"custom-auth-header", b"custom-auth-token")],
                "custom-auth-token",
            ),
            # Test case 3: Custom header name with case insensitive
            (
                "Custom-Auth-Header",
                [(b"custom-auth-header", b"case-insensitive-token")],
                "case-insensitive-token",
            ),
            # Test case 4: Header not present
            ("missing-header", [(b"x-mcp-auth", b"wrong-header-token")], None),
            # Test case 5: Multiple headers, only custom one should be used
            (
                "my-custom-auth",
                [
                    (b"x-mcp-auth", b"default-token"),
                    (b"my-custom-auth", b"custom-token"),
                ],
                "custom-token",
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
            "_get_mcp_client_side_auth_header_name",
            return_value=custom_header_name,
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
            result = MCPRequestHandler._get_mcp_auth_header_from_headers(
                extracted_headers
            )

            # Assert the result
            assert result == expected_auth_header

    @pytest.mark.asyncio
    async def test_process_mcp_request_with_custom_auth_header(self):
        """Test process_mcp_request with custom auth header name"""

        # Mock the custom header name
        with patch.object(
            MCPRequestHandler,
            "_get_mcp_client_side_auth_header_name",
            return_value="custom-auth-header",
        ):
            # Create ASGI scope with custom header
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/test",
                "headers": [
                    (b"x-litellm-api-key", b"test-api-key"),
                    (b"custom-auth-header", b"custom-auth-token"),
                ],
            }

            # Create an async mock for user_api_key_auth
            async def mock_user_api_key_auth(api_key, request):
                return UserAPIKeyAuth(
                    token="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    api_key=api_key,
                    user_id="test-user-id",
                    team_id="test-team-id",
                    user_role=None,
                    request_route=None,
                )

            with patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth,
            ) as mock_auth:
                # Call the method
                (
                    auth_result,
                    mcp_auth_header,
                    mcp_servers,
                    mcp_server_auth_headers,
                    oauth2_headers,
                    raw_headers,
                ) = await MCPRequestHandler.process_mcp_request(scope)

                # Assert the results
                assert auth_result.api_key == "test-api-key"
                assert mcp_auth_header == "custom-auth-token"
                assert mcp_servers is None
                assert mcp_server_auth_headers == {}

                # Verify the mock was called
                mock_auth.assert_called_once()
                call_args = mock_auth.call_args
                assert call_args.kwargs["api_key"] == "test-api-key"

    def test_get_mcp_server_auth_headers_from_headers(self):
        """Test _get_mcp_server_auth_headers_from_headers method"""
        from starlette.datastructures import Headers

        # Test case 1: No server-specific headers
        headers = Headers(
            {"x-litellm-api-key": "test-key", "content-type": "application/json"}
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {}

        # Test case 2: Single server-specific header
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "x-mcp-github-authorization": "Bearer github-token",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {"github": {"Authorization": "Bearer github-token"}}

        # Test case 3: Multiple server-specific headers
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "x-mcp-github-authorization": "Bearer github-token",
                "x-mcp-zapier_x_api-key": "zapier-api-key",
                "x-mcp-deepwiki-authorization": "Basic base64-encoded",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        expected = {
            "github": {"Authorization": "Bearer github-token"},
            "zapier_x_api": {"key": "zapier-api-key"},
            "deepwiki": {"Authorization": "Basic base64-encoded"},
        }
        assert result == expected

        # Test case 4: Case insensitive headers
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "X-MCP-GITHUB-AUTHORIZATION": "Bearer github-token",
                "x-mcp-ZAPIER-x-api-key": "zapier-api-key",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        expected = {
            "github": {"Authorization": "Bearer github-token"},
            "zapier": {"x-api-key": "zapier-api-key"},
        }
        assert result == expected

        # Test case 5: Invalid format headers (should be ignored)
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "x-mcp-invalid": "should-be-ignored",
                "x-mcp-github": "should-be-ignored",
                "x-mcp-github-authorization": "Bearer github-token",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {"github": {"Authorization": "Bearer github-token"}}

        # Test case 6: Edge case - header with multiple hyphens in server alias
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "x-mcp-github_mcp-authorization": "Bearer github-mcp-token",
                "x-mcp-gh_mcp2-authorization": "Bearer gh-mcp2-token",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        expected = {
            "github_mcp": {"Authorization": "Bearer github-mcp-token"},
            "gh_mcp2": {"Authorization": "Bearer gh-mcp2-token"},
        }
        assert result == expected

        # Test case 7: Edge case - header with underscore in server alias
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "x-mcp-github_mcp-authorization": "Bearer github-mcp-token",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {"github_mcp": {"Authorization": "Bearer github-mcp-token"}}

        # Test case 8: Edge case - empty header value
        headers = Headers(
            {"x-litellm-api-key": "test-key", "x-mcp-github-authorization": ""}
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {"github": {"Authorization": ""}}

        # Test case 9: Edge case - very long header value
        long_token = "Bearer " + "x" * 1000
        headers = Headers(
            {"x-litellm-api-key": "test-key", "x-mcp-github-authorization": long_token}
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {"github": {"Authorization": long_token}}

        # Test case 10: Edge case - special characters in server alias
        headers = Headers(
            {
                "x-litellm-api-key": "test-key",
                "x-mcp-github-123-authorization": "Bearer github-123-token",
                "x-mcp-github_test-authorization": "Bearer github-test-token",
            }
        )
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        expected = {
            "github": {"123-authorization": "Bearer github-123-token"},
            "github_test": {"Authorization": "Bearer github-test-token"},
        }
        assert result == expected


class TestMCPAccessGroupsE2E:
    """Simple e2e tests for MCP access groups functionality"""

    @pytest.mark.asyncio
    async def test_mcp_access_group_resolution_e2e(self):
        """Test that MCP access groups are properly resolved from headers"""

        # Create ASGI scope with access groups header
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [
                (b"x-litellm-api-key", b"test-api-key"),
                (b"x-mcp-access-groups", b"dev_group,prod_group"),
            ],
        }

        # Create an async mock for user_api_key_auth
        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(
                token="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                api_key=api_key,
                user_id="test-user-id",
                team_id="test-team-id",
                user_role=None,
                request_route=None,
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ) as mock_auth:
            # Call the method
            (
                auth_result,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            # Assert the results
            assert auth_result.api_key == "test-api-key"
            assert mcp_auth_header is None
            assert (
                mcp_servers is None
            )  # x-mcp-access-groups is not parsed as mcp_servers
            assert mcp_server_auth_headers == {}

            # Verify the mock was called
            mock_auth.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_header_with_mixed_servers_and_groups(self):
        """Test that MCP headers work with mixed servers and access groups"""

        # Create ASGI scope with mixed servers and groups
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [
                (b"x-litellm-api-key", b"test-api-key"),
                (b"x-mcp-servers", b"server1,dev_group,server2"),
            ],
        }

        # Create an async mock for user_api_key_auth
        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(
                token="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                api_key=api_key,
                user_id="test-user-id",
                team_id="test-team-id",
                user_role=None,
                request_route=None,
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ) as mock_auth:
            # Call the method
            (
                auth_result,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            # Assert the results
            assert auth_result.api_key == "test-api-key"
            assert mcp_auth_header is None
            assert mcp_servers == ["server1", "dev_group", "server2"]
            assert mcp_server_auth_headers == {}

            # Verify the mock was called
            mock_auth.assert_called_once()


@pytest.mark.asyncio
def test_mcp_path_based_server_segregation(monkeypatch):
    # Import the MCP server FastAPI app and context getter
    from litellm.proxy._experimental.mcp_server.server import app, get_auth_context

    captured_mcp_servers = {}

    # Patch the session manager to send a dummy response and capture context
    async def dummy_handle_request(scope, receive, send):
        """Dummy handler for testing"""
        # Get auth context
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
        ) = get_auth_context()

        # Capture the MCP servers for testing
        captured_mcp_servers["servers"] = mcp_servers

        # Send response
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"status": "ok"}',
            }
        )

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.session_manager",
        MagicMock(handle_request=dummy_handle_request),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.initialize_session_managers",
        AsyncMock(),
    )

    # Patch user_api_key_auth to always return a dummy user
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
        AsyncMock(return_value=UserAPIKeyAuth(api_key="test", user_id="user")),
    )

    # Use TestClient to make a request to /mcp/zapier,group1/tools
    client = TestClient(app)
    response = client.get(
        "/mcp/zapier,group1/tools", headers={"x-litellm-api-key": "test"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # The context should have mcp_servers set to ["zapier", "group1"]
    assert list(captured_mcp_servers.values())[0] == ["zapier", "group1"]


@pytest.mark.parametrize(
    "headers,expected_result",
    [
        (
            Headers(
                {
                    "x-litellm-api-key": "test-key",
                    "x-mcp-github-authorization": "Bearer github-token",
                }
            ),
            {"github": {"Authorization": "Bearer github-token"}},
        ),
        (
            Headers(
                {
                    "x-litellm-api-key": "test-key",
                    "x-mcp-github-x-api-key": "Basic base64-encoded-creds",
                }
            ),
            {"github": {"x-api-key": "Basic base64-encoded-creds"}},
        ),
    ],
)
def test_get_mcp_server_auth_headers_from_headers(headers, expected_result):
    """Test _get_mcp_server_auth_headers_from_headers method"""
    from starlette.datastructures import Headers

    headers = Headers(headers)
    result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
    assert result == expected_result


@pytest.mark.asyncio
async def test_get_team_object_permission_with_already_loaded_permission():
    """
    Test that _get_team_object_permission returns the already loaded object_permission
    from the team object without making an additional DB call.
    """
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable

    # Create mock object permission
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-123",
        mcp_servers=["server1", "server2"],
        mcp_access_groups=["group1"],
        vector_stores=["store1"],
    )
    
    # Create mock team object with object_permission already loaded
    mock_team_obj = LiteLLM_TeamTable(
        team_id="team-123",
        object_permission=mock_object_permission,
        object_permission_id="perm-123",
    )
    
    # Create mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-123",
    )
    
    # Mock get_team_object to return our team with loaded permission
    # Also need to mock prisma_client from proxy_server
    mock_prisma = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ):
        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object"
        ) as mock_get_team:
            with patch(
                "litellm.proxy.auth.auth_checks.get_object_permission"
            ) as mock_get_perm:
                mock_get_team.return_value = mock_team_obj
                
                # Call the method
                result = await MCPRequestHandler._get_team_object_permission(
                    mock_user_auth
                )
                
                # Assert we got the object permission
                assert result == mock_object_permission
                assert result.mcp_servers == ["server1", "server2"]
                
                # Verify get_team_object was called
                mock_get_team.assert_called_once()
                
                # Verify get_object_permission was NOT called (since it was already loaded)
                mock_get_perm.assert_not_called()


@pytest.mark.asyncio
async def test_get_team_object_permission_fetches_from_db_when_not_loaded():
    """
    Test that _get_team_object_permission fetches from DB when object_permission
    is not loaded but object_permission_id exists.
    """
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable

    # Create mock object permission (to be returned from DB)
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-456",
        mcp_servers=["server3", "server4"],
        mcp_access_groups=["group2"],
        vector_stores=["store2"],
    )
    
    # Create mock team object WITHOUT object_permission loaded (but has ID)
    mock_team_obj = LiteLLM_TeamTable(
        team_id="team-456",
        object_permission=None,
        object_permission_id="perm-456",
    )
    
    # Create mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-456",
    )
    
    # Mock the methods
    # Also need to mock prisma_client from proxy_server
    mock_prisma = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ):
        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object"
        ) as mock_get_team:
            with patch(
                "litellm.proxy.auth.auth_checks.get_object_permission"
            ) as mock_get_perm:
                mock_get_team.return_value = mock_team_obj
                mock_get_perm.return_value = mock_object_permission
                
                # Call the method
                result = await MCPRequestHandler._get_team_object_permission(
                    mock_user_auth
                )
                
                # Assert we got the object permission
                assert result == mock_object_permission
                assert result.mcp_servers == ["server3", "server4"]
                
                # Verify get_team_object was called
                mock_get_team.assert_called_once()
                
                # Verify get_object_permission WAS called (since it wasn't loaded)
                mock_get_perm.assert_called_once_with(
                    object_permission_id="perm-456",
                    prisma_client=mock.ANY,
                    user_api_key_cache=mock.ANY,
                    parent_otel_span=mock_user_auth.parent_otel_span,
                    proxy_logging_obj=mock.ANY,
                )


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_uses_helper():
    """
    Test that _get_allowed_mcp_servers_for_team properly uses _get_team_object_permission
    helper which handles both loaded and unloaded object_permission cases.
    """
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable

    # Create mock object permission with servers and access groups
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-789",
        mcp_servers=["direct-server1", "direct-server2"],
        mcp_access_groups=["dev-group"],
        vector_stores=[],
    )
    
    # Create mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-789",
    )
    
    # Mock the helper methods
    with patch.object(
        MCPRequestHandler, "_get_team_object_permission"
    ) as mock_get_team_perm:
        with patch.object(
            MCPRequestHandler, "_get_mcp_servers_from_access_groups"
        ) as mock_get_access_group_servers:
            # Configure mocks
            mock_get_team_perm.return_value = mock_object_permission
            mock_get_access_group_servers.return_value = ["group-server1", "group-server2"]
            
            # Call the method
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(
                mock_user_auth
            )
            
            # Assert the result contains both direct and access group servers
            assert set(result) == {
                "direct-server1",
                "direct-server2",
                "group-server1",
                "group-server2",
            }
            
            # Verify _get_team_object_permission was called (the helper we fixed)
            mock_get_team_perm.assert_called_once_with(mock_user_auth)
            
            # Verify access groups were resolved
            mock_get_access_group_servers.assert_called_once_with(["dev-group"])


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_with_no_object_permission():
    """
    Test that _get_allowed_mcp_servers_for_team returns empty list when
    team has no object_permission.
    """
    # Create mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-no-perm",
    )
    
    # Mock the helper to return None (no object permission)
    with patch.object(
        MCPRequestHandler, "_get_team_object_permission"
    ) as mock_get_team_perm:
        mock_get_team_perm.return_value = None
        
        # Call the method
        result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(
            mock_user_auth
        )
        
        # Assert empty list is returned
        assert result == []
        
        # Verify the helper was called
        mock_get_team_perm.assert_called_once_with(mock_user_auth)

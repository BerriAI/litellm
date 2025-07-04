import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._types import UserAPIKeyAuth, SpecialHeaders
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

            # Assert the result
            assert result == expected_result

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

            # Verify user_api_key_auth was called with correct parameters
            mock_auth.assert_called_once()
            call_args = mock_auth.call_args

            # Check that api_key parameter is correct
            assert call_args.kwargs["api_key"] == expected_api_key

            # Check that request parameter is a Request object
            request_param = call_args.kwargs["request"]
            assert isinstance(request_param, Request)

            # Verify the request has the correct scope
            assert request_param.scope == scope

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

            # Assert the results
            assert auth_result == mock_auth_result
            assert mcp_auth_header == expected_result["mcp_auth"]
            assert mcp_servers_result == expected_result["mcp_servers"]

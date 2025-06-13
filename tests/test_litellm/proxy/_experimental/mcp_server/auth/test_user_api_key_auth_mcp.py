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
    UserAPIKeyAuthMCP,
)
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
class TestUserAPIKeyAuthMCP:

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
            result = await UserAPIKeyAuthMCP._get_allowed_mcp_servers_for_key(
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
        "headers,expected_api_key",
        [
            # Test case 1: x-litellm-api-key header present
            (
                [(b"x-litellm-api-key", b"test-api-key-123")],
                "test-api-key-123",
            ),
            # Test case 2: Authorization header present (fallback)
            (
                [(b"authorization", b"Bearer test-auth-token")],
                "Bearer test-auth-token",
            ),
            # Test case 3: Both headers present (primary should win)
            (
                [
                    (b"x-litellm-api-key", b"primary-key"),
                    (b"authorization", b"Bearer fallback-token"),
                ],
                "primary-key",
            ),
            # Test case 4: Case insensitive headers
            (
                [(b"X-LITELLM-API-KEY", b"case-insensitive-key")],
                "case-insensitive-key",
            ),
            # Test case 5: No relevant headers
            (
                [(b"content-type", b"application/json")],
                "",
            ),
            # Test case 6: Empty headers
            ([], ""),
        ],
    )
    async def test_user_api_key_auth_mcp(self, headers, expected_api_key):
        """Test user_api_key_auth_mcp method with various header scenarios"""

        # Create ASGI scope with headers
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": headers,
        }

        # Mock the user_api_key_auth function
        mock_auth_result = UserAPIKeyAuth(
            api_key=expected_api_key,
            user_id="test-user-id",
            team_id="test-team-id",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth"
        ) as mock_user_api_key_auth:
            mock_user_api_key_auth.return_value = mock_auth_result

            # Call the method
            result = await UserAPIKeyAuthMCP.user_api_key_auth_mcp(scope)

            # Assert the result
            assert result == mock_auth_result

            # Verify user_api_key_auth was called with correct parameters
            mock_user_api_key_auth.assert_called_once()
            call_args = mock_user_api_key_auth.call_args

            # Check that api_key parameter is correct
            assert call_args.kwargs["api_key"] == expected_api_key

            # Check that request parameter is a Request object
            request_param = call_args.kwargs["request"]
            assert isinstance(request_param, Request)

            # Verify the request has the correct scope
            assert request_param.scope == scope

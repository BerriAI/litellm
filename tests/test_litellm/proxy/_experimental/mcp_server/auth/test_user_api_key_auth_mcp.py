import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

from starlette.datastructures import Headers

from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._types import (
    SpecialHeaders,
    SpecialMCPServerNames,
    UserAPIKeyAuth,
)


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
        with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key") as mock_key_servers:
            with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team") as mock_team_servers:
                # Set up return values
                mock_key_servers.return_value = key_servers
                mock_team_servers.return_value = team_servers

                # Call the method
                result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth=mock_user_auth)

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
        with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key") as mock_key_servers:
            with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team") as mock_team_servers:
                # Configure mocks to return the test data
                mock_key_servers.return_value = key_servers
                mock_team_servers.return_value = team_servers

                # Call the method
                result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)

                # Assert the result (order-independent comparison)
                assert sorted(result) == sorted(expected_servers)

                # Verify the mock functions were called correctly
                mock_key_servers.assert_called_once_with(user_api_key_auth)
                mock_team_servers.assert_called_once_with(user_api_key_auth)

    @pytest.mark.parametrize(
        "require_key_mcp_access_defined,expected",
        [
            # Default (flag off): a key with no MCP scope of its own inherits
            # the team's servers.
            (False, ["team_server1", "team_server2"]),
            # Flag on: the team is a ceiling, not a default — the key inherits
            # nothing and must grant servers explicitly.
            (True, []),
        ],
    )
    async def test_require_key_mcp_access_defined_gates_team_inheritance(
        self, require_key_mcp_access_defined, expected
    ):
        """The require_key_mcp_access_defined general setting flips an empty key
        from inheriting its team's MCP servers (default) to inheriting none."""
        auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="test-team")
        with (
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_key",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_team",
                new_callable=AsyncMock,
                return_value=["team_server1", "team_server2"],
            ),
            patch.object(
                MCPRequestHandler,
                "_get_key_access_group_mcp_server_extras",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "litellm.proxy.proxy_server.general_settings",
                {"require_key_mcp_access_defined": require_key_mcp_access_defined},
            ),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)

        assert sorted(result) == sorted(expected)

    @pytest.mark.parametrize(
        "key_servers,grants,expected,scenario",
        [
            # Explicit key subset is still honored under the flag (intersected
            # with the team ceiling) — the flag only removes empty-key inheritance.
            (["team_server1"], [], ["team_server1"], "explicit_subset_survives"),
            # An access-group grant is the escape hatch: it surfaces even though
            # the key inherits nothing from the team.
            ([], ["granted_server"], ["granted_server"], "access_group_grant_survives"),
        ],
    )
    async def test_require_key_mcp_access_defined_preserves_explicit_grants(
        self, key_servers, grants, expected, scenario
    ):
        """With require_key_mcp_access_defined on, a key still reaches servers it
        grants explicitly or via an access group — only blanket team inheritance
        is removed."""
        auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            access_group_ids=["grp"] if grants else [],
        )
        with (
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_key",
                new_callable=AsyncMock,
                return_value=key_servers,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_team",
                new_callable=AsyncMock,
                return_value=["team_server1", "team_server2"],
            ),
            patch.object(
                MCPRequestHandler,
                "_get_key_access_group_mcp_server_extras",
                new_callable=AsyncMock,
                return_value=grants,
            ),
            patch(
                "litellm.proxy.proxy_server.general_settings",
                {"require_key_mcp_access_defined": True},
            ),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)

        assert sorted(result) == sorted(expected)

    @pytest.mark.parametrize("team_servers", [[], ["team_server1", "team_server2"]])
    async def test_no_mcp_servers_sentinel_returns_empty(self, team_servers):
        """A key scoped to the no-mcp-servers sentinel resolves to zero servers,
        overriding team inheritance and never leaking the sentinel marker."""
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="test-team")
        key_object_permission = MagicMock()
        key_object_permission.mcp_servers = [SpecialMCPServerNames.no_mcp_servers.value]

        with (
            patch.object(
                MCPRequestHandler,
                "_get_key_object_permission",
                return_value=key_object_permission,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_team",
                new_callable=AsyncMock,
                return_value=team_servers,
            ),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)

        assert result == []

    async def test_get_allowed_mcp_servers_for_key_returns_sentinel_marker(self):
        """_get_allowed_mcp_servers_for_key surfaces the sentinel unexpanded so the
        caller can short-circuit, ignoring any other entries on the key."""
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = MagicMock()
        key_object_permission.mcp_servers = [
            SpecialMCPServerNames.no_mcp_servers.value,
            "some-other-server",
        ]

        with patch.object(
            MCPRequestHandler,
            "_get_key_object_permission",
            return_value=key_object_permission,
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

        assert result == [SpecialMCPServerNames.no_mcp_servers.value]

    async def test_permission_inheritance_edge_cases(self):
        """Test edge cases in permission inheritance"""

        # Test case: None values in database
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_objectpermissiontable.find_unique.return_value = None
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
        mock_prisma_client.db.litellm_objectpermissiontable.find_unique.side_effect = Exception("DB Error")

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)
            assert result == []  # Should handle exception gracefully

    @pytest.mark.parametrize(
        "key_servers,team_servers,grant_servers,expected,scenario",
        [
            # Key has no own scope, restrictive team ceiling {test}, server
            # granted only via key.access_group_ids → caller sees team's server
            # AND the grant (grant is added on top of the ceiling).
            (
                [],
                ["test"],
                ["context7"],
                ["context7", "test"],
                "grant_over_team_ceiling",
            ),
            # key {a} ∩ team {b} = {} ; the grant still surfaces, proving grants
            # are unioned with the ceiling, not intersected against it.
            (
                ["a"],
                ["b"],
                ["context7"],
                ["context7"],
                "grant_survives_empty_intersection",
            ),
            # No grant → ceiling behavior is unchanged (no additive leakage).
            (["x", "y"], ["x"], [], ["x"], "no_grant_keeps_intersection"),
        ],
    )
    async def test_access_group_grants_are_additive_over_ceiling(
        self, key_servers, team_servers, grant_servers, expected, scenario
    ):
        """Regression: key.access_group_ids grants are unioned on top of the
        key/team MCP ceiling, so a grant reaches the caller even when the team
        ceiling does not include it (and even when key ∩ team is empty)."""
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            access_group_ids=["grp-mcp"],
        )
        with (
            patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key") as mock_key,
            patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team") as mock_team,
            patch.object(MCPRequestHandler, "_get_key_access_group_mcp_server_extras") as mock_grants,
        ):
            mock_key.return_value = key_servers
            mock_team.return_value = team_servers
            mock_grants.return_value = grant_servers
            result = await MCPRequestHandler.get_allowed_mcp_servers(mock_user_auth)
        assert sorted(result) == sorted(expected)

    async def test_access_group_extras_returns_empty_when_no_auth(self):
        """No auth object → no additive grants."""
        result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(None)
        assert result == []

    async def test_access_group_extras_returns_empty_without_access_group_ids(self):
        """A key with no resolvable access groups yields no additive grants
        (the `if not raw_server_ids: return []` branch)."""
        auth = UserAPIKeyAuth(api_key="k", access_group_ids=[])
        with (
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                new=AsyncMock(return_value=[]),
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(auth)
        assert result == []
        # expand_permission_list must not be reached when there are no raw ids.
        mock_mgr.expand_permission_list.assert_not_called()

    async def test_access_group_extras_expands_resolved_server_ids(self):
        """Resolved access-group server ids/names are expanded to server ids."""
        auth = UserAPIKeyAuth(api_key="k", access_group_ids=["grp-mcp"])
        with (
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                new=AsyncMock(return_value=["alias-a", "srv-b"]),
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.expand_permission_list.return_value = ["srv-a", "srv-b"]
            result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(auth)
        assert sorted(result) == ["srv-a", "srv-b"]
        mock_mgr.expand_permission_list.assert_called_once_with(["alias-a", "srv-b"])

    async def test_access_group_extras_swallows_errors(self):
        """Resolution failures degrade to no grants rather than raising."""
        auth = UserAPIKeyAuth(api_key="k", access_group_ids=["grp-mcp"])
        with patch(
            "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
            new=AsyncMock(side_effect=Exception("db down")),
        ):
            result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(auth)
        assert result == []

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
                "test-auth-token",
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
                token=("test-token-sha256-empty-hash" if api_key else None),
                api_key=api_key,
                user_id="test-user-id" if api_key else None,
                team_id="test-team-id" if api_key else None,
                user_role=None,
                request_route=None,
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ):
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

    def test_duplicate_authorization_header_is_rejected(self):
        """A request carrying more than one Authorization header is malformed for bearer auth and,
        for the client-forwarded token modes, would make which upstream token is forwarded ambiguous.
        The ingress header converter must reject it with a 400 rather than silently keeping one."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/tp_server",
            "headers": [
                (b"authorization", b"Bearer upstream-token-a"),
                (b"authorization", b"Bearer upstream-token-b"),
                (b"content-type", b"application/json"),
            ],
        }
        with pytest.raises(HTTPException) as exc_info:
            MCPRequestHandler._safe_get_headers_from_scope(scope)
        assert exc_info.value.status_code == 400
        assert "Authorization" in str(exc_info.value.detail)

    def test_single_authorization_header_is_forwarded_verbatim(self):
        """The rejection must not disturb the normal single-Authorization case: the value passes
        through unchanged (guards against the duplicate check over-matching)."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/tp_server",
            "headers": [
                (b"authorization", b"Bearer upstream-token"),
                (b"content-type", b"application/json"),
            ],
        }
        headers = MCPRequestHandler._safe_get_headers_from_scope(scope)
        assert headers.get("authorization") == "Bearer upstream-token"


@pytest.mark.asyncio
class TestMCPOAuth2AuthFlow:
    """Test suite for OAuth2 authentication flow in MCP requests.

    Tests the fix for the 'Capabilities: none' bug where OAuth2 tokens
    from upstream MCP providers (e.g., Atlassian) were mistakenly validated
    as LiteLLM API keys, causing auth failures and empty tool listings.
    """

    async def test_oauth2_token_in_authorization_header_fallback(self):
        """
        When only the Authorization header is present with a non-LiteLLM OAuth2
        token AND the target server delegates auth to upstream, LiteLLM skips its
        own validation entirely (so the upstream token is never mistaken for a
        virtual key) and forwards the bearer upstream.
        """
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/atlassian_mcp",
            "headers": [
                (b"authorization", b"Bearer atlassian-oauth2-access-token-xyz"),
            ],
        }

        oauth2_server = MagicMock()
        oauth2_server.auth_type = MCPAuth.oauth2
        oauth2_server.delegate_auth_to_upstream = True
        oauth2_server.has_client_credentials = False

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = oauth2_server
            (
                auth_result,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            assert isinstance(auth_result, UserAPIKeyAuth)
            # The upstream token is never validated as a LiteLLM key ...
            mock_auth.assert_not_called()
            # ... and is preserved for upstream forwarding.
            assert oauth2_headers.get("Authorization") == "Bearer atlassian-oauth2-access-token-xyz"

    async def test_explicit_litellm_key_with_oauth2_authorization(self):
        """
        When both x-litellm-api-key AND Authorization header are present,
        LiteLLM key should be used for auth and Authorization preserved for OAuth2.
        """
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/atlassian_mcp",
            "headers": [
                (b"x-litellm-api-key", b"sk-litellm-valid-key"),
                (b"authorization", b"Bearer atlassian-oauth2-token"),
            ],
        }

        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(api_key=api_key, user_id="test-user")

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ) as mock_auth:
            (
                auth_result,
                mcp_auth_header,
                mcp_servers,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            # LiteLLM key should be used for auth
            mock_auth.assert_called_once()
            call_args = mock_auth.call_args
            assert call_args.kwargs["api_key"] == "sk-litellm-valid-key"

            # OAuth2 headers should still contain the Authorization token
            assert oauth2_headers.get("Authorization") == "Bearer atlassian-oauth2-token"

    async def test_litellm_key_in_authorization_backward_compat(self):
        """
        Backward compatibility: when only Authorization header is present
        with a valid LiteLLM key (not OAuth2), auth should succeed normally.
        """
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/some_server",
            "headers": [
                (b"authorization", b"Bearer sk-litellm-valid-key"),
            ],
        }

        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(api_key=api_key, user_id="test-user")

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth,
        ) as mock_auth:
            (
                auth_result,
                _,
                _,
                _,
                _,
                _,
            ) = await MCPRequestHandler.process_mcp_request(scope)

            # Should succeed with the LiteLLM key from Authorization header
            from litellm.proxy.utils import hash_token

            assert auth_result.api_key == hash_token("sk-litellm-valid-key")
            mock_auth.assert_called_once()

    async def test_non_auth_http_exception_still_raises(self):
        """
        If user_api_key_auth raises a non-401/403 HTTPException (e.g., 500),
        it should NOT be caught by the OAuth2 fallback.
        """
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/some_server",
            "headers": [
                (b"authorization", b"Bearer some-token"),
            ],
        }

        async def mock_user_api_key_auth_server_error(api_key, request):
            raise HTTPException(status_code=500, detail="Internal server error")

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth_server_error,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 500

    async def test_proxy_exception_non_delegate_oauth2_propagates(self):
        """
        Production raises ProxyException (not HTTPException) on auth failure. For
        a non-delegate oauth2 server the bearer is treated as a LiteLLM credential
        and a 401 must propagate as a real auth error, not be exchanged for an
        anonymous upstream-passthrough session.
        """
        from litellm.proxy._types import ProxyException
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/atlassian_mcp",
            "headers": [
                (b"authorization", b"Bearer atlassian-oauth2-access-token-xyz"),
            ],
        }

        async def mock_user_api_key_auth_proxy_exception(api_key, request):
            raise ProxyException(
                message="Authentication Error: Invalid API key",
                type="auth_error",
                param="api_key",
                code=401,
            )

        oauth2_server = MagicMock()
        oauth2_server.auth_type = MCPAuth.oauth2
        oauth2_server.delegate_auth_to_upstream = False
        oauth2_server.is_oauth_passthrough = False

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_proxy_exception,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = oauth2_server
            with pytest.raises(ProxyException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert str(exc_info.value.code) == "401"

    async def test_proxy_exception_non_auth_still_raises(self):
        """
        ProxyException with non-401/403 code should NOT be caught.
        """
        from litellm.proxy._types import ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/some_server",
            "headers": [
                (b"authorization", b"Bearer some-token"),
            ],
        }

        async def mock_user_api_key_auth_500(api_key, request):
            raise ProxyException(
                message="Internal error",
                type="server_error",
                param=None,
                code=500,
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth_500,
        ):
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(scope)


@pytest.mark.asyncio
class TestMCPPublicRouteGuard:
    """
    Regression tests for GHSA-7cwm-3279-qf3c / HW6xR21d:
    the public-route bypass at the top of process_mcp_request must match
    the exact `/.well-known/` path prefix, not a substring of the URL.
    """

    async def test_well_known_substring_in_query_does_not_bypass_auth(self):
        """
        URL with `.well-known` smuggled into the query string must still
        require valid LiteLLM auth.
        """
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/private_server",
            "query_string": b"redirect=.well-known/oauth-protected-resource",
            "headers": [(b"authorization", b"Bearer sk-bogus")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            # Explicit unresolvable target — proves auth still fails even
            # when the registry has no info to fall back to.
            mock_mgr.get_mcp_server_by_name.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_well_known_segment_in_middle_of_path_does_not_bypass_auth(self):
        """
        Path containing `.well-known` as a non-prefix component (e.g. a server
        name or sub-path) must still require auth.
        """
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/.well-known-fake/tools",
            "headers": [(b"authorization", b"Bearer sk-bogus")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_legitimate_well_known_path_still_bypasses_auth(self):
        """
        Real OAuth discovery routes registered under /.well-known/ must remain
        public so unauthenticated clients can fetch them per RFC 8414/9728.
        """
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/.well-known/oauth-protected-resource",
            "headers": [],
        }

        # No mock needed — public path should not call user_api_key_auth at all
        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
        ) as mock_auth:
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(scope)
            mock_auth.assert_not_called()
            assert isinstance(auth_result, UserAPIKeyAuth)


@pytest.mark.asyncio
class TestMCPPassthroughColdStartAdmission:
    @staticmethod
    def _make_passthrough_server():
        server = MagicMock()
        server.is_oauth_passthrough = True
        return server

    async def test_cold_start_ignores_header_without_path_target(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"x-mcp-servers", b"passthrough_server")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp._is_mcp_passthrough_cold_start"
            ) as mock_cold_start,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

            assert exc_info.value.status_code == 401
            # Cold-start admission must not fire for the aggregate ``/mcp``
            # route — only path-targeted routes are eligible for OAuth
            # discovery admission.
            mock_cold_start.assert_not_called()

    async def test_cold_start_rejects_server_specific_authorization_header(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [
                (
                    b"x-mcp-passthrough_server-authorization",
                    b"Bearer upstream-token",
                )
            ],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

            assert exc_info.value.status_code == 401

    async def test_cold_start_rejects_legacy_mcp_auth_header(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [(b"x-mcp-auth", b"Bearer upstream-token")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

            assert exc_info.value.status_code == 401

    async def test_cold_start_fails_closed_when_client_ip_hides_server(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.IPAddressUtils.get_mcp_client_ip",
                return_value="203.0.113.10",
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

            assert exc_info.value.status_code == 401
            mock_mgr.get_mcp_server_by_name.assert_any_call("passthrough_server", client_ip="203.0.113.10")

    async def test_cold_start_propagates_non_401_http_error(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_forbidden(api_key, request):
            raise HTTPException(status_code=403, detail="Forbidden")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_forbidden,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

            assert exc_info.value.status_code == 403

    async def test_cold_start_propagates_non_auth_proxy_exception(self):
        from litellm.proxy._types import ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_server_error(api_key, request):
            raise ProxyException(
                message="Internal error",
                type="server_error",
                param=None,
                code=500,
            )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_server_error,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(scope)

    async def test_cold_start_allows_401_for_path_passthrough_target(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(scope)

            assert isinstance(auth_result, UserAPIKeyAuth)
            mock_mgr.get_mcp_server_by_name.assert_any_call("passthrough_server", client_ip="")

    async def test_cold_start_allows_proxy_exception_401_for_path_target(self):
        from litellm.proxy._types import ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise ProxyException(
                message="Authentication Error",
                type="auth_error",
                param="api_key",
                code=401,
            )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = (
                TestMCPPassthroughColdStartAdmission._make_passthrough_server()
            )
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(scope)

            assert isinstance(auth_result, UserAPIKeyAuth)
            mock_mgr.get_mcp_server_by_name.assert_any_call("passthrough_server", client_ip="")


@pytest.mark.asyncio
class TestMCPOAuth2FallbackTargetGating:
    """
    Regression tests for GHSA-h8fm-g6wc-j228 / HW6xR21d:
    The OAuth2 passthrough fallback must only fire when the target MCP server
    is operator-configured for ``auth_type=oauth2``. A failed LiteLLM-auth
    against a non-OAuth2 server (api_key, bearer_token, basic, etc.) must
    propagate as a real auth error, not be exchanged for an anonymous session.
    """

    @staticmethod
    def _make_server(auth_type, is_oauth_passthrough=False):
        server = MagicMock()
        server.auth_type = auth_type
        # MagicMock would otherwise auto-create truthy stand-ins for any
        # attribute access (including ``is_oauth_passthrough``), which
        # would silently flip the passthrough fallback gate on. Pin the
        # boolean explicitly so non-passthrough fixtures stay non-passthrough.
        server.is_oauth_passthrough = is_oauth_passthrough
        return server

    async def test_fallback_blocked_when_target_is_not_oauth2(self):
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/api_key_server",
            "headers": [(b"authorization", b"Bearer anything-at-all")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPOAuth2FallbackTargetGating._make_server(
                MCPAuth.api_key
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_fallback_blocked_when_target_unresolvable(self):
        """
        If the target server cannot be resolved from path or x-mcp-servers,
        we cannot prove it is OAuth2-mode, so we must fail closed.
        """
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/never_registered_server",
            "headers": [(b"authorization", b"Bearer anything")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_non_delegate_oauth2_does_not_fall_back_to_anonymous(self):
        """
        An ``auth_type=oauth2`` server that has NOT opted into
        ``delegate_auth_to_upstream`` must not exchange a failed LiteLLM auth for
        an anonymous session: forwarding an arbitrary bearer upstream is only
        allowed once the operator explicitly delegates auth. A failed validation
        here is a genuine 401 and propagates (which is also what keeps the
        success-path trace free of a phantom 401, since no doomed validation runs
        for a delegated server).
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/atlassian_mcp",
            "headers": [
                (b"authorization", b"Bearer atlassian-oauth2-access-token-xyz"),
            ],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPOAuth2FallbackTargetGating._make_server(
                MCPAuth.oauth2
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_fallback_allowed_when_target_is_passthrough(self):
        """
        Cold-start return per RFC 9728 / MCP Authorization spec: client
        discovered the upstream IdP via the gateway's protected-resource
        metadata, completed OAuth, and is returning with
        ``Authorization: Bearer <upstream-token>``. The bearer is not a
        LiteLLM key but the target is a pass-through server, so admission
        falls back to anonymous and forwards the bearer upstream.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [(b"authorization", b"Bearer upstream-token-xyz")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPOAuth2FallbackTargetGating._make_server(
                auth_type=MCPAuth.none,
                is_oauth_passthrough=True,
            )
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(scope)
            assert isinstance(auth_result, UserAPIKeyAuth)
            assert auth_result.api_key is None

    async def test_fallback_blocked_when_client_ip_hides_oauth2_target(self):
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/hidden_oauth2_server",
            "headers": [(b"authorization", b"Bearer upstream-token")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.IPAddressUtils.get_mcp_client_ip",
                return_value="203.0.113.10",
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

            assert exc_info.value.status_code == 401
            # Lookup may run twice — once for the oauth2-target fallback gate
            # and once for the passthrough-target fallback gate. Both must
            # resolve to ``None`` (hidden by client IP) so neither bypass
            # opens. Use ``assert_any_call`` to assert the IP-scoped lookup
            # happened without locking the count.
            mock_mgr.get_mcp_server_by_name.assert_any_call("hidden_oauth2_server", client_ip="203.0.113.10")

    async def test_fallback_blocked_when_any_target_in_header_is_not_oauth2(self):
        """
        x-mcp-servers can list multiple targets. If ANY of them is non-OAuth2,
        the fallback must be blocked — otherwise an attacker can mix one
        OAuth2-mode server in to enable bypass against the others.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [
                (b"authorization", b"Bearer anything"),
                (b"x-mcp-servers", b"oauth2_server,api_key_server"),
            ],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        def mock_lookup(name, client_ip=None):
            if name == "oauth2_server":
                return TestMCPOAuth2FallbackTargetGating._make_server(MCPAuth.oauth2)
            return TestMCPOAuth2FallbackTargetGating._make_server(MCPAuth.api_key)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.side_effect = mock_lookup
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_proxy_exception_with_non_numeric_code_propagates(self):
        """
        ``ProxyException`` normalises ``code`` via ``str()`` in its __init__,
        so callers may produce ``"None"`` or any non-numeric string when no
        explicit code was supplied. The exception handler must not coerce
        with ``int(...)`` (which would raise ``ValueError`` and rewrite the
        auth error as an unhandled 500); it must simply re-raise.
        """
        from litellm.proxy._types import ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/atlassian_mcp",
            "headers": [(b"authorization", b"Bearer anything")],
        }

        async def mock_user_api_key_auth_no_code(api_key, request):
            raise ProxyException(
                message="Authentication Error",
                type="auth_error",
                param="api_key",
                code=None,
            )

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=mock_user_api_key_auth_no_code,
        ):
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(scope)


@pytest.mark.asyncio
class TestMCPDelegateAuthToUpstream:
    """
    Tests for the ``delegate_auth_to_upstream`` per-server flag.

    When set on an ``auth_type=oauth2`` MCP server, LiteLLM must skip its own
    API-key/SSO check entirely so the client completes PKCE directly with the
    upstream MCP server. The gate must fail closed for any non-oauth2 server,
    any mixed-target request, and any request where the target cannot be
    resolved.
    """

    @staticmethod
    def _make_server(auth_type, delegate_auth_to_upstream=False):
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        return MCPServer(
            server_id="test-server-id",
            name="test-server",
            transport="http",
            auth_type=auth_type,
            delegate_auth_to_upstream=delegate_auth_to_upstream,
        )

    def test_build_mcp_server_table_preserves_delegate_auth_to_upstream(self):
        """Registry → API list rows must expose delegate_auth_to_upstream for the UI."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        manager = MCPServerManager()
        delegated = MCPServer(
            server_id="delegated-1",
            name="delegated",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=True,
        )
        assert manager._build_mcp_server_table(delegated).delegate_auth_to_upstream is True

        not_delegated = delegated.model_copy(update={"delegate_auth_to_upstream": False})
        assert manager._build_mcp_server_table(not_delegated).delegate_auth_to_upstream is False

    def test_build_mcp_server_table_preserves_oauth_passthrough(self):
        """Registry → API list rows must expose oauth_passthrough for the UI.

        ``oauth_passthrough`` is the dedicated non-oauth2 pass-through opt-in,
        distinct from ``delegate_auth_to_upstream`` (oauth2-only). Both must
        round-trip independently so neither flag silently implies the other.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        manager = MCPServerManager()
        passthrough = MCPServer(
            server_id="passthrough-1",
            name="passthrough",
            transport="http",
            auth_type=MCPAuth.none,
            extra_headers=["Authorization"],
            oauth_passthrough=True,
            available_on_public_internet=True,
        )
        row = manager._build_mcp_server_table(passthrough)
        assert row.oauth_passthrough is True
        # The oauth2-only flag must remain independent and default off.
        assert row.delegate_auth_to_upstream is False

        not_passthrough = passthrough.model_copy(update={"oauth_passthrough": False})
        assert manager._build_mcp_server_table(not_passthrough).oauth_passthrough is False

    async def test_delegate_skips_litellm_auth_with_no_authorization(self):
        """
        oauth2 + delegate_auth_to_upstream=True, no Authorization header at
        all → anonymous UserAPIKeyAuth and ``user_api_key_auth`` is never
        called.
        """
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/delegated_oauth_server",
            "headers": [],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.oauth2,
                delegate_auth_to_upstream=True,
            )
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(scope)
            assert isinstance(auth_result, UserAPIKeyAuth)
            mock_auth.assert_not_called()

    async def test_delegate_with_upstream_token_in_authorization_skips_litellm_auth(
        self,
    ):
        """
        oauth2 + delegate_auth_to_upstream=True with an upstream OAuth token in
        ``Authorization``: the delegate gate fires before any LiteLLM validation,
        so ``user_api_key_auth`` is never called and the bearer is forwarded
        upstream untouched. Skipping the doomed validation is what keeps a tool
        call that actually succeeds from carrying a phantom 401 auth span.
        """
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/delegated_oauth_server",
            "headers": [(b"authorization", b"Bearer upstream-pkce-token")],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.oauth2,
                delegate_auth_to_upstream=True,
            )
            (
                auth_result,
                _,
                _,
                _,
                oauth2_headers,
                _,
            ) = await MCPRequestHandler.process_mcp_request(scope)
            assert isinstance(auth_result, UserAPIKeyAuth)
            assert oauth2_headers.get("Authorization") == "Bearer upstream-pkce-token"
            mock_auth.assert_not_called()

    async def test_delegate_off_still_requires_litellm_auth(self):
        """
        oauth2 server but delegate flag is OFF → existing behaviour: a missing
        / invalid LiteLLM key still 401s (no anonymous fast-path).
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/non_delegated_oauth_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.oauth2,
                delegate_auth_to_upstream=False,
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_delegate_ignored_for_non_oauth2_server(self):
        """
        Defense in depth: even if an operator turns on delegate_auth_to_upstream
        for a non-oauth2 server (api_key, bearer_token, etc.), the gate must
        not fire — only oauth2 servers may delegate.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/api_key_server",
            "headers": [],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.api_key,
                delegate_auth_to_upstream=True,
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_delegate_mixed_targets_fail_closed(self):
        """
        x-mcp-servers can list multiple targets. If ANY of them does not opt in
        to delegate_auth_to_upstream, the bypass must NOT fire — otherwise an
        attacker could mix one delegated server in to skip auth on the others.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [
                (b"x-mcp-servers", b"delegated_oauth,plain_oauth"),
            ],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        def mock_lookup(name, client_ip=None):
            if name == "delegated_oauth":
                return TestMCPDelegateAuthToUpstream._make_server(
                    auth_type=MCPAuth.oauth2,
                    delegate_auth_to_upstream=True,
                )
            return TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.oauth2,
                delegate_auth_to_upstream=False,
            )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.side_effect = mock_lookup
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_delegate_no_resolvable_target_fail_closed(self):
        """
        If the target server cannot be resolved at all (e.g. admin/REST path
        that isn't ``/mcp/{name}`` or ``/{name}/mcp``), we cannot prove the
        gate's preconditions, so we must fail closed and run normal auth.
        """
        from fastapi import HTTPException

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/admin/whatever",
            "headers": [],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_explicit_litellm_key_takes_precedence_over_delegate(self):
        """
        When ``x-litellm-api-key`` is present, normal auth runs even for a
        delegate server, so ``user_id`` is resolved and any stored upstream
        OAuth credentials can be looked up and forwarded. The bypass only
        fires when no LiteLLM key is supplied.
        """
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/delegated_oauth_server",
            "headers": [(b"x-litellm-api-key", b"Bearer sk-1234")],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
                return_value=UserAPIKeyAuth(user_id="real-user"),
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.oauth2,
                delegate_auth_to_upstream=True,
            )
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(scope)
            assert isinstance(auth_result, UserAPIKeyAuth)
            assert auth_result.user_id == "real-user"
            mock_auth.assert_called_once()

    async def test_authorization_bearer_on_delegate_server_treated_as_upstream(self):
        """
        On a delegate server the ``Authorization`` header is, by contract, an
        upstream token rather than a LiteLLM key — even when it is sk-shaped. It
        is forwarded upstream without LiteLLM validation, so ``user_api_key_auth``
        is not called and no LiteLLM identity is resolved. Callers who need
        LiteLLM identity / spend tracking on a delegate server must supply
        ``x-litellm-api-key`` (see
        test_explicit_litellm_key_takes_precedence_over_delegate).
        """
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/delegated_oauth_server",
            "headers": [(b"authorization", b"Bearer sk-1234")],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
                return_value=UserAPIKeyAuth(user_id="real-user"),
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.oauth2,
                delegate_auth_to_upstream=True,
            )
            (
                auth_result,
                _,
                _,
                _,
                oauth2_headers,
                _,
            ) = await MCPRequestHandler.process_mcp_request(scope)
            assert isinstance(auth_result, UserAPIKeyAuth)
            assert auth_result.user_id is None
            assert oauth2_headers.get("Authorization") == "Bearer sk-1234"
            mock_auth.assert_not_called()

    async def test_delegate_ignored_for_client_credentials_server(self):
        """
        oauth2 + delegate_auth_to_upstream=True but oauth2_flow=client_credentials
        → bypass must NOT fire; normal LiteLLM auth must be attempted.

        M2M servers fetch the upstream token automatically using stored
        credentials, so allowing anonymous bypass would let any external
        caller invoke tools as LiteLLM's service account.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/m2m_server",
            "headers": [],
        }

        m2m_server = MCPServer(
            server_id="m2m-server-id",
            name="m2m_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            oauth2_flow="client_credentials",
        )

        async def mock_auth_raises(*_args, **_kwargs):
            raise HTTPException(status_code=401, detail="No key provided")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_auth_raises,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = m2m_server
            # No delegate bypass → normal auth is attempted → 401 raised
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401
            mock_auth.assert_called_once()

    async def test_delegate_ignored_for_unstamped_m2m_shaped_server(self):
        """
        oauth2 + delegate + oauth2_flow=None but the M2M credential shape
        (client_id/secret + token_url, no authorization_url) → bypass must NOT
        fire. A legacy row that was never stamped still resolves to
        client_credentials by shape, and reading the bare column here would
        reopen the anonymous bypass to a server that runs upstream as LiteLLM's
        service account. Fails closed like the client_credentials case above.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/legacy_m2m_server",
            "headers": [],
        }

        legacy_m2m_server = MCPServer(
            server_id="legacy-m2m-id",
            name="legacy_m2m_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            oauth2_flow=None,
            client_id="cid",
            client_secret="csecret",
            token_url="https://idp.example.com/token",
        )
        assert legacy_m2m_server.has_client_credentials is False

        async def mock_auth_raises(*_args, **_kwargs):
            raise HTTPException(status_code=401, detail="No key provided")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_auth_raises,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = legacy_m2m_server
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401
            mock_auth.assert_called_once()

    async def test_delegate_bypass_for_pure_pkce_server(self):
        """
        oauth2 + delegate + oauth2_flow=None and NO stored client credentials
        (pure PKCE, the common delegate case) → bypass must still fire. The
        shape resolves to a non-M2M flow, so the security gate leaves it alone;
        the fail-closed rule targets the M2M shape specifically, not every
        unstamped row.
        """
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/pkce_server",
            "headers": [],
        }

        pkce_server = MCPServer(
            server_id="pkce-server-id",
            name="pkce_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            oauth2_flow=None,
        )

        async def mock_auth_raises(*_args, **_kwargs):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="No key provided")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_auth_raises,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = pkce_server
            auth, *_rest = await MCPRequestHandler.process_mcp_request(scope)
            mock_auth.assert_not_called()
            assert auth.api_key is None

    async def test_delegate_bypass_for_internal_server(self):
        """
        Delegate + oauth2 interactive servers bypass LiteLLM auth even when
        ``available_on_public_internet`` is False (internal MCPs).
        """
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/internal_server",
            "headers": [],
        }

        internal_server = MCPServer(
            server_id="internal-server-id",
            name="internal_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=False,
        )

        async def mock_auth_raises(*_args, **_kwargs):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="No key provided")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_auth_raises,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = internal_server
            auth, *_rest = await MCPRequestHandler.process_mcp_request(scope)
            mock_auth.assert_not_called()
            assert auth.api_key is None

    async def test_get_allowed_servers_excludes_client_credentials_delegate(self):
        """
        get_allowed_mcp_servers must not surface M2M (client_credentials) delegate
        servers to anonymous callers even if delegate_auth_to_upstream=True.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        manager = MCPServerManager()
        pkce_server = MCPServer(
            server_id="pkce-server",
            name="pkce_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=True,
        )
        m2m_server = MCPServer(
            server_id="m2m-server",
            name="m2m_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            oauth2_flow="client_credentials",
            available_on_public_internet=True,
        )
        manager.registry = {
            pkce_server.server_id: pkce_server,
            m2m_server.server_id: m2m_server,
        }

        with patch.object(
            MCPRequestHandler,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await manager.get_allowed_mcp_servers(None)

        assert "pkce-server" in result
        assert "m2m-server" not in result

    async def test_get_allowed_servers_excludes_unstamped_m2m_shape_delegate(self):
        """
        The anonymous allow-list must also exclude an M2M-shape delegate server whose
        oauth2_flow was never stamped (null column, verbatim-read as non-M2M). Reading
        the bare has_client_credentials here would surface it to anonymous callers; the
        resolved-flow check fails closed on the shape, matching the auth gate.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        manager = MCPServerManager()
        pkce_server = MCPServer(
            server_id="pkce-server",
            name="pkce_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=True,
        )
        unstamped_m2m = MCPServer(
            server_id="unstamped-m2m",
            name="unstamped_m2m",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            oauth2_flow=None,
            client_id="cid",
            client_secret="csecret",
            token_url="https://idp.example.com/token",
        )
        assert unstamped_m2m.has_client_credentials is False
        manager.registry = {
            pkce_server.server_id: pkce_server,
            unstamped_m2m.server_id: unstamped_m2m,
        }

        with patch.object(
            MCPRequestHandler,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await manager.get_allowed_mcp_servers(None)

        assert "pkce-server" in result
        assert "unstamped-m2m" not in result

    async def test_get_allowed_servers_includes_internal_delegate(self):
        """
        Internal-only (available_on_public_internet=False) delegate servers
        appear in the anonymous allow-list like public delegate servers.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        manager = MCPServerManager()
        public_server = MCPServer(
            server_id="public-server",
            name="public_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=True,
        )
        internal_server = MCPServer(
            server_id="internal-server",
            name="internal_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=False,
        )
        manager.registry = {
            public_server.server_id: public_server,
            internal_server.server_id: internal_server,
        }

        with patch.object(
            MCPRequestHandler,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await manager.get_allowed_mcp_servers(None)

        assert "public-server" in result
        assert "internal-server" in result

    async def test_true_passthrough_skips_litellm_auth_anonymously(self):
        """auth_type=true_passthrough performs no admission auth: the caller's Authorization is an
        upstream token forwarded unchanged and user_api_key_auth is never called."""
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/true_passthrough_server",
            "headers": [(b"authorization", b"Bearer upstream-token")],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = TestMCPDelegateAuthToUpstream._make_server(
                auth_type=MCPAuth.true_passthrough,
            )
            (
                auth_result,
                _,
                _,
                _,
                oauth2_headers,
                _,
            ) = await MCPRequestHandler.process_mcp_request(scope)
            assert isinstance(auth_result, UserAPIKeyAuth)
            assert auth_result.api_key is None
            assert oauth2_headers.get("Authorization") == "Bearer upstream-token"
            mock_auth.assert_not_called()

    async def test_true_passthrough_mixed_targets_fail_closed(self):
        """One true_passthrough target mixed with a non-passthrough target must NOT skip admission."""
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"x-mcp-servers", b"tp_server,plain_server")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        def mock_lookup(name, client_ip=None):
            if name == "tp_server":
                return TestMCPDelegateAuthToUpstream._make_server(
                    auth_type=MCPAuth.true_passthrough,
                )
            return TestMCPDelegateAuthToUpstream._make_server(auth_type=MCPAuth.api_key)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.side_effect = mock_lookup
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401

    async def test_get_allowed_servers_includes_true_passthrough(self):
        """Anonymous callers can reach true_passthrough servers; admission is delegated upstream."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        manager = MCPServerManager()
        tp_server = MCPServer(
            server_id="tp-server",
            name="tp_server",
            transport="http",
            auth_type=MCPAuth.true_passthrough,
            available_on_public_internet=True,
        )
        manager.registry = {tp_server.server_id: tp_server}

        with patch.object(
            MCPRequestHandler,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await manager.get_allowed_mcp_servers(None)

        assert "tp-server" in result

    def test_extract_target_server_names_matches_routing_parser(self):
        """
        Regression: _extract_target_server_names_from_path must match the
        downstream regex parser in server.py::_get_mcp_servers_in_path.

        Previously, a request to ``/mcp/<delegated>/garbage`` was parsed as
        targeting ``<delegated>`` by the auth gate (bypassing LiteLLM auth)
        while the routing layer parsed it as ``<delegated>/garbage`` — when
        that name did not resolve, the request fell back to the anonymous
        allow-list which can include ``allow_all_keys`` servers that normally
        require a LiteLLM key.
        """
        from litellm.proxy._experimental.mcp_server.server import (
            _get_mcp_servers_in_path,
        )

        cases = [
            # Single server, single segment.
            ("/mcp/foo", ["foo"]),
            # Server name with one embedded slash (two segments).
            ("/mcp/foo/bar", ["foo/bar"]),
            # Server name with embedded slash + extra path → name stays at two segments.
            ("/mcp/foo/bar/tools", ["foo/bar"]),
            # Comma-separated servers, no trailing path.
            ("/mcp/foo,bar", ["foo", "bar"]),
            # Comma-separated servers with trailing path.
            ("/mcp/foo,bar/tools", ["foo", "bar"]),
            # Alternative form ``/<server>/mcp`` is also parsed (both auth
            # parser and routing parser handle it for defense-in-depth — some
            # entry points may not be rewritten by ``dynamic_mcp_route``).
            ("/foo/mcp", ["foo"]),
            ("/foo/mcp/tools", ["foo"]),
            # Non-MCP paths → empty (fail closed).
            ("/.well-known/oauth-authorization-server", []),
            ("/v1/keys", []),
            ("/", []),
        ]
        for path_input, expected in cases:
            assert MCPRequestHandler._extract_target_server_names_from_path(path_input) == expected, (
                f"path={path_input!r} → expected {expected!r}"
            )
            assert (_get_mcp_servers_in_path(path_input) or []) == expected, (
                f"path={path_input!r} → routing expected {expected!r}"
            )

    async def test_delegate_does_not_bypass_on_extra_path_segment(self):
        """
        Regression: ``/mcp/<delegated>/<garbage>`` must NOT bypass auth.

        The bypass key check is now performed against the same parsed target
        as downstream routing — ``<delegated>/<garbage>`` — which will not
        resolve to a delegate-enabled server, so normal LiteLLM auth runs.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/delegated_server/extra",
            "headers": [],
        }

        delegate_server = TestMCPDelegateAuthToUpstream._make_server(
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
        )

        def lookup_by_name(name, **_kwargs):
            # Only the *exact* delegated name resolves. Anything else (e.g.
            # ``delegated_server/extra``) returns None so the bypass fails.
            # ``**_kwargs`` accepts the ``client_ip`` kwarg the cold-start
            # admission path now forwards (real signature:
            # ``get_mcp_server_by_name(name, client_ip=None)``).
            if name == "delegated_server":
                return delegate_server
            return None

        async def mock_auth_raises(*_args, **_kwargs):
            raise HTTPException(status_code=401, detail="No key provided")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_auth_raises,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.side_effect = lookup_by_name
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401
            # Auth was attempted (not bypassed) because the parsed target
            # name does not match any registered delegate server.
            mock_auth.assert_called_once()

    async def test_delegate_ignores_x_mcp_servers_header_for_mcp_paths(self):
        """
        Regression (header/path TOCTOU): For ``/mcp/...`` routes, downstream
        routing overrides ``x-mcp-servers`` with the path-derived names.
        The auth bypass must do the same — otherwise an attacker could send
        ``x-mcp-servers: <delegated>`` while the URL path targets a
        non-delegate server, flipping the auth gate on a server that should
        require a LiteLLM key.
        """
        from fastapi import HTTPException

        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/non_delegate_server",
            "headers": [(b"x-mcp-servers", b"delegated_server")],
        }

        delegate_server = MCPServer(
            server_id="delegate-id",
            name="delegated_server",
            transport="http",
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
            available_on_public_internet=True,
        )
        non_delegate = MCPServer(
            server_id="non-delegate-id",
            name="non_delegate_server",
            transport="http",
            auth_type=MCPAuth.api_key,
        )

        def lookup_by_name(name, **_kwargs):
            # ``**_kwargs`` accepts the ``client_ip`` kwarg the cold-start
            # admission path now forwards (real signature:
            # ``get_mcp_server_by_name(name, client_ip=None)``).
            return {
                "delegated_server": delegate_server,
                "non_delegate_server": non_delegate,
            }.get(name)

        async def mock_auth_raises(*_args, **_kwargs):
            raise HTTPException(status_code=401, detail="No key provided")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_auth_raises,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.side_effect = lookup_by_name
            # Bypass MUST NOT fire — path-derived target is the non-delegate
            # server. Normal auth runs and 401s.
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)
            assert exc_info.value.status_code == 401
            mock_auth.assert_called_once()

    async def test_resolve_target_server_names_prefers_path_over_header(self):
        """
        ``_resolve_target_server_names`` must:

        - For ``/mcp/<name>`` paths, return the path-derived list and ignore
          the header (mirrors downstream routing).
        - For non-MCP paths, fall back to the header (including the explicit
          empty-list case, which fails closed).
        """
        # Path matches /mcp/... — header is ignored.
        assert MCPRequestHandler._resolve_target_server_names(path="/mcp/foo", mcp_servers_header=["evil"]) == ["foo"]
        assert MCPRequestHandler._resolve_target_server_names(path="/mcp/foo,bar", mcp_servers_header=["evil"]) == [
            "foo",
            "bar",
        ]
        assert MCPRequestHandler._resolve_target_server_names(path="/foo/mcp", mcp_servers_header=["evil"]) == ["foo"]
        # Path does not match — header is trusted.
        assert MCPRequestHandler._resolve_target_server_names(
            path="/.well-known/oauth-authorization-server",
            mcp_servers_header=["foo"],
        ) == ["foo"]
        # Explicit empty list on a non-MCP path → empty (fail closed).
        assert (
            MCPRequestHandler._resolve_target_server_names(
                path="/.well-known/oauth-authorization-server",
                mcp_servers_header=[],
            )
            == []
        )
        # No header on a non-MCP path → empty.
        assert (
            MCPRequestHandler._resolve_target_server_names(
                path="/.well-known/oauth-authorization-server",
                mcp_servers_header=None,
            )
            == []
        )


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
    def test_get_mcp_client_side_auth_header_name(self, env_var, general_setting, expected_header_name):
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
    def test_get_mcp_auth_header_from_headers_with_custom_name(self, custom_header_name, headers, expected_auth_header):
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
            result = MCPRequestHandler._get_mcp_auth_header_from_headers(extracted_headers)

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
                    token="test-token-sha256-empty-hash",
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
        headers = Headers({"x-litellm-api-key": "test-key", "content-type": "application/json"})
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
        headers = Headers({"x-litellm-api-key": "test-key", "x-mcp-github-authorization": ""})
        result = MCPRequestHandler._get_mcp_server_auth_headers_from_headers(headers)
        assert result == {"github": {"Authorization": ""}}

        # Test case 9: Edge case - very long header value
        long_token = "Bearer " + "x" * 1000
        headers = Headers({"x-litellm-api-key": "test-key", "x-mcp-github-authorization": long_token})
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
                token="test-token-sha256-empty-hash",
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
            assert mcp_servers is None  # x-mcp-access-groups is not parsed as mcp_servers
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
                token="test-token-sha256-empty-hash",
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


def test_mcp_path_based_server_segregation(monkeypatch):
    # Import the MCP server FastAPI app and context getter
    from litellm.proxy._experimental.mcp_server.server import app, get_auth_context

    captured_mcp_servers = {}

    # Patch the session manager to send a dummy response and capture context
    async def dummy_handle_request(scope, receive, send):
        """Dummy handler for testing"""
        # Get auth context (includes client_ip as 7th value)
        (
            user_api_key_auth,
            mcp_auth_header,
            mcp_servers,
            mcp_server_auth_headers,
            oauth2_headers,
            raw_headers,
            client_ip,
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
        "litellm.proxy._experimental.mcp_server.server.session_manager_stateless",
        MagicMock(handle_request=dummy_handle_request),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.session_manager_stateful",
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
    response = client.get("/mcp/zapier,group1/tools", headers={"x-litellm-api-key": "test"})
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
        with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
            with patch("litellm.proxy.auth.auth_checks.get_object_permission") as mock_get_perm:
                mock_get_team.return_value = mock_team_obj

                # Call the method
                result = await MCPRequestHandler._get_team_object_permission(mock_user_auth)

                # Assert we got the object permission
                assert result == mock_object_permission
                assert result.mcp_servers == ["server1", "server2"]

                # Verify get_team_object was called
                mock_get_team.assert_called_once()

                # Verify get_object_permission was NOT called (since it was already loaded)
                mock_get_perm.assert_not_called()


@pytest.mark.asyncio
async def test_get_team_object_permission_with_core_auth_auto_loading():
    """
    Test that _get_team_object_permission returns the object_permission that was
    automatically loaded by get_team_object() in the core auth flow.

    Note: After migrating permission loading to core auth (get_team_object in auth_checks.py),
    the team object returned by get_team_object() should already have object_permission loaded
    when an object_permission_id exists.
    """
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable

    # Create mock object permission
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-456",
        mcp_servers=["server3", "server4"],
        mcp_access_groups=["group2"],
        vector_stores=["store2"],
    )

    # Create mock team object WITH object_permission already loaded
    # (This is what get_team_object() returns after the core auth migration)
    mock_team_obj = LiteLLM_TeamTable(
        team_id="team-456",
        object_permission=mock_object_permission,  # Already loaded by core auth
        object_permission_id="perm-456",
    )

    # Create mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-456",
    )

    # Mock the methods
    mock_prisma = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ):
        with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
            mock_get_team.return_value = mock_team_obj

            # Call the method
            result = await MCPRequestHandler._get_team_object_permission(mock_user_auth)

            # Assert we got the object permission (already loaded by core auth)
            assert result == mock_object_permission
            assert result.mcp_servers == ["server3", "server4"]

            # Verify get_team_object was called
            mock_get_team.assert_called_once()


@pytest.mark.asyncio
async def test_get_team_object_permission_ui_session_team_skips_db_lookup():
    """
    UI session tokens carry the virtual team_id "litellm-dashboard" (UI_TEAM_ID),
    which is never persisted. The lookup must short-circuit to None without
    calling get_team_object; otherwise every MCP tools listing from the
    dashboard logs a "Team doesn't exist in db" warning per server.
    """
    from litellm.proxy._types import UI_TEAM_ID

    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id=UI_TEAM_ID,
    )

    mock_prisma = MagicMock()
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
            result = await MCPRequestHandler._get_team_object_permission(mock_user_auth)

            assert result is None
            mock_get_team.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper_name,expected",
    [
        ("_get_allowed_mcp_servers_for_team", []),
        ("_get_mcp_access_groups_for_team", []),
    ],
)
async def test_team_mcp_helpers_ui_session_team_skip_db_lookup(helper_name, expected):
    """
    The server-permission and access-group helpers hit get_team_object with the
    session's team_id too; for the virtual UI team each used to 404 into its
    own swallowed warning per MCP listing. They must short-circuit without a
    DB lookup.
    """
    from litellm.proxy._types import UI_TEAM_ID

    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id=UI_TEAM_ID,
    )

    mock_prisma = MagicMock()
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team:
            helper = getattr(MCPRequestHandler, helper_name)
            result = await helper(mock_user_auth)

            assert result == expected
            mock_get_team.assert_not_called()


@pytest.mark.asyncio
async def test_get_allowed_tools_for_server_ui_session_team_keeps_key_restrictions():
    """
    Regression: the 404 raised by get_team_object for the virtual UI team used
    to escape into get_allowed_tools_for_server's blanket except, dropping
    key-level tool restrictions (fail-open) and logging a warning. With the
    short-circuit, key restrictions still apply for UI sessions.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import UI_TEAM_ID

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id=UI_TEAM_ID,
    )
    key_perm = MagicMock()
    key_perm.mcp_tool_permissions = {"server_1": ["tool_a"]}

    mock_prisma = MagicMock()
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            side_effect=HTTPException(
                status_code=404,
                detail={"error": "Team doesn't exist in db. Team=litellm-dashboard."},
            ),
        ):
            with patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_perm):
                result = await MCPRequestHandler.get_allowed_tools_for_server(
                    server_id="server_1",
                    user_api_key_auth=user_api_key_auth,
                )

                assert result == ["tool_a"]


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_uses_helper():
    """
    Test that _get_allowed_mcp_servers_for_team resolves both legacy
    object_permission fields (mcp_servers, mcp_access_groups) and the unified
    team.access_group_ids → access_mcp_server_ids path.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    # Register placeholder ids in the manager so expand_permission_list resolves them.
    for sid in ("direct-server1", "direct-server2"):
        global_mcp_server_manager.registry[sid] = MCPServer(
            server_id=sid,
            name=sid,
            server_name=sid,
            url=f"https://{sid}.example.com",
            transport=MCPTransport.http,
        )
    try:
        mock_object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-789",
            mcp_servers=["direct-server1", "direct-server2"],
            mcp_access_groups=["dev-group"],
            vector_stores=[],
        )
        mock_team = LiteLLM_TeamTable(
            team_id="team-789",
            access_group_ids=[],
            object_permission_id="perm-789",
        )
        mock_team.object_permission = mock_object_permission

        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="team-789",
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=mock_team,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_mcp_servers_from_access_groups",
                new_callable=AsyncMock,
                return_value=["group-server1", "group-server2"],
            ) as mock_get_access_group_servers,
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(mock_user_auth)

            assert set(result) == {
                "direct-server1",
                "direct-server2",
                "group-server1",
                "group-server2",
            }

            mock_get_access_group_servers.assert_called_once_with(["dev-group"])
    finally:
        for sid in ("direct-server1", "direct-server2"):
            global_mcp_server_manager.registry.pop(sid, None)


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_with_no_object_permission():
    """
    Test that _get_allowed_mcp_servers_for_team returns empty list when
    the team has no object_permission and no access_group_ids.
    """
    from litellm.proxy._types import LiteLLM_TeamTable

    mock_team = LiteLLM_TeamTable(
        team_id="team-no-perm",
        access_group_ids=[],
        object_permission_id=None,
    )

    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-no-perm",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team,
        ),
    ):
        result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(mock_user_auth)

        assert result == []


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_without_user_auth_returns_empty():
    """Ensure helper returns empty list when no user auth is provided."""

    result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(None)

    assert result == []


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_without_team_id_returns_empty():
    """Ensure helper returns empty list when user lacks a team_id."""

    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id=None,
    )

    result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(mock_user_auth)

    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_api_key_auth, prisma_client_value, scenario",
    [
        (None, object(), "no_user"),
        (
            UserAPIKeyAuth(api_key="test-key", user_id="test-user"),
            object(),
            "no_object_permission_id",
        ),
        (
            UserAPIKeyAuth(
                api_key="test-key",
                user_id="test-user",
                object_permission_id="perm-123",
            ),
            None,
            "no_prisma_client",
        ),
    ],
)
async def test_get_allowed_mcp_servers_for_key_guard_conditions(user_api_key_auth, prisma_client_value, scenario):
    """Ensure guard clauses return [] before hitting get_object_permission."""

    with patch(
        "litellm.proxy.auth.auth_checks.get_object_permission",
        new_callable=AsyncMock,
    ) as mock_get_perm:
        with patch("litellm.proxy.proxy_server.prisma_client", prisma_client_value):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

    assert result == []
    mock_get_perm.assert_not_called()


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_key_returns_empty_when_db_returns_none():
    """Ensure [] is returned when get_object_permission yields None."""

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        object_permission_id="perm-123",
    )

    mock_prisma = object()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.auth.auth_checks.get_object_permission",
            new_callable=AsyncMock,
        ) as mock_get_perm,
    ):
        mock_get_perm.return_value = None

        result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

    assert result == []
    mock_get_perm.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_key_prefers_in_memory_permission():
    """Ensure in-memory object_permission is used without hitting the DB."""

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    # Register "direct-server" in the manager so permission expansion resolves it.
    # Without this, expand_permission_list drops unknown ids as stale — which is
    # the correct production behavior but unrelated to what this test asserts.
    global_mcp_server_manager.registry["direct-server"] = MCPServer(
        server_id="direct-server",
        name="direct-server",
        server_name="direct-server",
        url="https://direct-server.example.com",
        transport=MCPTransport.http,
    )
    try:
        perms = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-in-memory",
            mcp_servers=["direct-server"],
            mcp_access_groups=["grp-alpha"],
        )
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            object_permission=perms,
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_object_permission",
            new_callable=AsyncMock,
        ) as mock_get_perm:
            with patch.object(MCPRequestHandler, "_get_mcp_servers_from_access_groups") as mock_access_groups:
                mock_access_groups.return_value = ["group-server"]

                result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

        assert set(result) == {"direct-server", "group-server"}
        mock_get_perm.assert_not_called()
        mock_access_groups.assert_called_once_with(["grp-alpha"])
    finally:
        global_mcp_server_manager.registry.pop("direct-server", None)


@pytest.mark.asyncio
class TestAgentMCPPermissions:
    """Test agent-level MCP server and tool permission intersection."""

    async def test_get_allowed_mcp_servers_agent_intersection(self):
        """Key/team allow [server_1, server_2]; agent allows [server_1]. Result = [server_1]."""
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            agent_id="agent-123",
        )
        with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key") as mock_key:
            with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team") as mock_team:
                with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_agent") as mock_agent:
                    mock_key.return_value = ["server_1", "server_2"]
                    mock_team.return_value = []
                    mock_agent.return_value = ["server_1"]
                    result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth=user_api_key_auth)
                    assert sorted(result) == ["server_1"]
                    mock_agent.assert_called_once_with(user_api_key_auth)

    async def test_get_allowed_mcp_servers_agent_no_restriction(self):
        """Agent with no object_permission returns []; no intersection applied (inherit key/team)."""
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="agent-456",
        )
        with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key") as mock_key:
            with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team") as mock_team:
                with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_agent") as mock_agent:
                    mock_key.return_value = ["server_1", "server_2"]
                    mock_team.return_value = []
                    mock_agent.return_value = []  # no agent-level restriction
                    result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth=user_api_key_auth)
                    assert sorted(result) == ["server_1", "server_2"]
                    mock_agent.assert_called_once_with(user_api_key_auth)

    async def test_get_allowed_mcp_servers_key_team_agent_intersection(self):
        """Key allows [1, 2], agent allows [2, 3]. Result = [2]."""
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="agent-789",
        )
        with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key") as mock_key:
            with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team") as mock_team:
                with patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_agent") as mock_agent:
                    mock_key.return_value = ["server_1", "server_2"]
                    mock_team.return_value = []
                    mock_agent.return_value = ["server_2", "server_3"]
                    result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth=user_api_key_auth)
                    assert sorted(result) == ["server_2"]

    async def test_get_allowed_tools_for_server_agent_intersection(self):
        """Key allows [tool_a, tool_b], agent allows [tool_a]. Result = [tool_a]."""
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="agent-tools",
        )
        key_perm = MagicMock()
        key_perm.mcp_tool_permissions = {"server_1": ["tool_a", "tool_b"]}
        team_perm = None
        with patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_perm):
            with patch.object(
                MCPRequestHandler,
                "_get_team_object_permission",
                new_callable=AsyncMock,
                return_value=team_perm,
            ):
                with patch.object(
                    MCPRequestHandler,
                    "_get_agent_tool_permissions_for_server",
                    new_callable=AsyncMock,
                    return_value=["tool_a"],
                ) as mock_agent_tools:
                    result = await MCPRequestHandler.get_allowed_tools_for_server(
                        server_id="server_1",
                        user_api_key_auth=user_api_key_auth,
                    )
                    assert result == ["tool_a"]
                    mock_agent_tools.assert_called_once()
                    call_kwargs = mock_agent_tools.call_args.kwargs
                    assert call_kwargs["server_id"] == "server_1"
                    assert call_kwargs["user_api_key_auth"] == user_api_key_auth

    async def test_get_allowed_tools_for_server_agent_no_restriction(self):
        """Agent has no tool permissions for server; key/team result is unchanged."""
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="agent-no-tools",
        )
        key_perm = MagicMock()
        key_perm.mcp_tool_permissions = {"server_1": ["tool_a", "tool_b"]}
        with patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_perm):
            with patch.object(
                MCPRequestHandler,
                "_get_team_object_permission",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with patch.object(
                    MCPRequestHandler,
                    "_get_agent_tool_permissions_for_server",
                    new_callable=AsyncMock,
                    return_value=None,
                ):
                    result = await MCPRequestHandler.get_allowed_tools_for_server(
                        server_id="server_1",
                        user_api_key_auth=user_api_key_auth,
                    )
                    assert sorted(result) == ["tool_a", "tool_b"]

    async def test_get_agent_object_permission_uses_shared_helper(self):
        """``_get_agent_object_permission`` must resolve the agent's
        ``object_permission_id`` and then defer to the shared
        ``get_object_permission`` helper so cache entries are shared with the
        org / team / key paths."""
        from litellm.caching.dual_cache import DualCache

        cache = DualCache()
        agent_row = MagicMock()
        agent_row.object_permission_id = "perm-xyz"
        prisma_client = MagicMock()
        prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=agent_row)
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="agent-shared",
        )
        expected_perm = MagicMock()

        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
            patch(
                "litellm.proxy.auth.auth_checks.get_object_permission",
                new_callable=AsyncMock,
                return_value=expected_perm,
            ) as mock_get_perm,
        ):
            result = await MCPRequestHandler._get_agent_object_permission(user_api_key_auth)
            assert result is expected_perm
            mock_get_perm.assert_awaited_once()
            assert mock_get_perm.await_args.kwargs["object_permission_id"] == "perm-xyz"

            # Second call: the agent_id -> object_permission_id mapping is
            # cached, so the agent row is not re-fetched.
            prisma_client.db.litellm_agentstable.find_unique.reset_mock()
            await MCPRequestHandler._get_agent_object_permission(user_api_key_auth)
            prisma_client.db.litellm_agentstable.find_unique.assert_not_called()

    async def test_get_agent_object_permission_caches_missing_permission(self):
        """When the agent has no ``object_permission_id`` the sentinel must be
        cached so subsequent requests do not hit the DB again."""
        from litellm.caching.dual_cache import DualCache

        cache = DualCache()
        agent_row = MagicMock()
        agent_row.object_permission_id = None
        prisma_client = MagicMock()
        prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=agent_row)
        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="agent-no-perm",
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
            patch(
                "litellm.proxy.auth.auth_checks.get_object_permission",
                new_callable=AsyncMock,
            ) as mock_get_perm,
        ):
            assert await MCPRequestHandler._get_agent_object_permission(user_api_key_auth) is None
            assert await MCPRequestHandler._get_agent_object_permission(user_api_key_auth) is None

            mock_get_perm.assert_not_awaited()
            prisma_client.db.litellm_agentstable.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_permission_servers_included_in_allowed_servers():
    """
    Servers listed only in mcp_tool_permissions (not in mcp_servers)
    should still be accessible.

    Regression test for https://github.com/BerriAI/litellm/issues/21954
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    # Register the server id so expand_permission_list resolves it rather than
    # dropping it as stale.
    global_mcp_server_manager.registry["server_id_123"] = MCPServer(
        server_id="server_id_123",
        name="server_id_123",
        server_name="server_id_123",
        url="https://server-id-123.example.com",
        transport=MCPTransport.http,
    )
    try:
        perm = MagicMock()
        perm.mcp_servers = []
        perm.mcp_access_groups = []
        perm.mcp_tool_permissions = {"server_id_123": ["tool_a", "tool_b"]}

        user_api_key_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
        )

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=perm),
            patch.object(
                MCPRequestHandler,
                "_get_mcp_servers_from_access_groups",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(
                user_api_key_auth=user_api_key_auth,
            )
            assert "server_id_123" in result
    finally:
        global_mcp_server_manager.registry.pop("server_id_123", None)


# ---------------------------------------------------------------------------
# Org-level MCP permission tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOrgMCPPermissions:
    """Tests for org-level MCP server permission enforcement."""

    def _make_auth(self, org_id=None, team_id=None) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id=team_id,
            org_id=org_id,
        )

    @pytest.mark.parametrize(
        "key_servers,team_servers,org_servers,expected,scenario",
        [
            (
                ["s1", "s2"],
                [],
                None,
                ["s1", "s2"],
                "no_org_id",
            ),
            (
                ["s1", "s2"],
                [],
                [],
                ["s1", "s2"],
                "org_empty_no_restriction",
            ),
            (
                [],
                [],
                ["org_s1", "org_s2"],
                ["org_s1", "org_s2"],
                "org_only_ceiling",
            ),
            (
                ["s1", "s2"],
                [],
                ["s1", "org_only"],
                ["s1"],
                "org_intersection",
            ),
            (
                ["s1", "s2"],
                [],
                ["org_s1"],
                [],
                "no_overlap_denied",
            ),
            (
                ["s1", "s2"],
                ["s1", "s2", "s3"],
                ["s1"],
                ["s1"],
                "team_then_org",
            ),
            (
                ["s1"],
                ["s2"],
                ["s1", "s2", "org_s1"],
                [],
                "key_team_conflict_not_expanded_by_org",
            ),
        ],
    )
    async def test_get_allowed_mcp_servers_with_org(
        self,
        key_servers,
        team_servers,
        org_servers,
        expected,
        scenario,
    ):
        org_id = "org-123" if org_servers is not None else None
        auth = self._make_auth(org_id=org_id)

        with (
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_key",
                new_callable=AsyncMock,
                return_value=key_servers,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_team",
                new_callable=AsyncMock,
                return_value=team_servers,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_org",
                new_callable=AsyncMock,
                return_value=org_servers if org_servers is not None else [],
            ),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
            assert sorted(result) == sorted(expected), f"scenario={scenario}"

    async def test_get_org_object_permission_no_org_id(self):
        auth = self._make_auth(org_id=None)
        result = await MCPRequestHandler._get_org_object_permission(auth)
        assert result is None

    async def test_get_org_object_permission_no_prisma(self):
        auth = self._make_auth(org_id="org-123")
        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_org_object_permission",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_org(auth)
            assert result == []

    async def test_get_allowed_mcp_servers_for_org_direct_servers(self):
        auth = self._make_auth(org_id="org-123")

        mock_perm = MagicMock()
        mock_perm.mcp_servers = ["org_server_1", "org_server_2"]
        mock_perm.mcp_access_groups = []
        mock_perm.mcp_tool_permissions = {}

        with (
            patch.object(
                MCPRequestHandler,
                "_get_org_object_permission",
                new_callable=AsyncMock,
                return_value=mock_perm,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_mcp_servers_from_access_groups",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_org(auth)
            assert sorted(result) == ["org_server_1", "org_server_2"]

    async def test_get_allowed_mcp_servers_for_org_access_groups(self):
        auth = self._make_auth(org_id="org-123")

        mock_perm = MagicMock()
        mock_perm.mcp_servers = []
        mock_perm.mcp_access_groups = ["group-a"]
        mock_perm.mcp_tool_permissions = {}

        with (
            patch.object(
                MCPRequestHandler,
                "_get_org_object_permission",
                new_callable=AsyncMock,
                return_value=mock_perm,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_mcp_servers_from_access_groups",
                new_callable=AsyncMock,
                return_value=["group_server_1"],
            ),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_org(auth)
            assert "group_server_1" in result

    async def test_get_allowed_mcp_servers_for_org_tool_permissions_only(self):
        auth = self._make_auth(org_id="org-123")

        mock_perm = MagicMock()
        mock_perm.mcp_servers = []
        mock_perm.mcp_access_groups = []
        mock_perm.mcp_tool_permissions = {"tool_only_server": ["tool_x"]}

        with (
            patch.object(
                MCPRequestHandler,
                "_get_org_object_permission",
                new_callable=AsyncMock,
                return_value=mock_perm,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_mcp_servers_from_access_groups",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_org(auth)
            assert "tool_only_server" in result

    async def test_get_allowed_mcp_servers_for_org_no_object_permission(self):
        auth = self._make_auth(org_id="org-123")

        with patch.object(
            MCPRequestHandler,
            "_get_org_object_permission",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_org(auth)
            assert result == []

    async def test_get_allowed_tools_for_server_org_ceiling(self):
        auth = self._make_auth(org_id="org-123")

        key_perm = MagicMock()
        key_perm.mcp_tool_permissions = {"server_1": ["tool_a", "tool_b", "tool_c"]}

        org_perm = MagicMock()
        org_perm.mcp_tool_permissions = {"server_1": ["tool_a", "tool_b"]}

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_perm),
            patch.object(
                MCPRequestHandler,
                "_get_team_object_permission",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_org_object_permission",
                new_callable=AsyncMock,
                return_value=org_perm,
            ),
        ):
            result = await MCPRequestHandler.get_allowed_tools_for_server(
                server_id="server_1",
                user_api_key_auth=auth,
            )
            assert sorted(result) == ["tool_a", "tool_b"]

    async def test_get_allowed_tools_for_server_org_no_restriction(self):
        auth = self._make_auth(org_id="org-123")

        key_perm = MagicMock()
        key_perm.mcp_tool_permissions = {"server_1": ["tool_a", "tool_b"]}

        org_perm = MagicMock()
        org_perm.mcp_tool_permissions = {}

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_perm),
            patch.object(
                MCPRequestHandler,
                "_get_team_object_permission",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                MCPRequestHandler,
                "_get_org_object_permission",
                new_callable=AsyncMock,
                return_value=org_perm,
            ),
        ):
            result = await MCPRequestHandler.get_allowed_tools_for_server(
                server_id="server_1",
                user_api_key_auth=auth,
            )
            assert sorted(result) == ["tool_a", "tool_b"]


# ---------------------------------------------------------------------------
# LIT-3189: key unified access_group_ids extend team MCP scope
# ---------------------------------------------------------------------------


def _patch_proxy_server_globals_for_mcp():
    """Non-None mocks so the helper's None-guard doesn't short-circuit."""
    return [
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
    ]


def _fake_mcp_access_group(
    access_group_id,
    access_mcp_server_ids=None,
    assigned_team_ids=None,
    assigned_key_ids=None,
):
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    return LiteLLM_AccessGroupTable(
        access_group_id=access_group_id,
        access_group_name=access_group_id,
        access_mcp_server_ids=access_mcp_server_ids or [],
        assigned_team_ids=assigned_team_ids or [],
        assigned_key_ids=assigned_key_ids or [],
    )


def _start_patches(patches):
    for p in patches:
        p.start()


def _stop_patches(patches):
    for p in patches:
        p.stop()


@pytest.mark.asyncio
async def test_mcp_key_access_group_extras_when_team_authorized():
    """Group's assigned_team_ids includes key's team and grants an MCP server → server returned."""
    valid_token = UserAPIKeyAuth(
        token="test-token",
        access_group_ids=["mcp-premium"],
        team_id="team-a",
    )
    fake_ag = _fake_mcp_access_group(
        access_group_id="mcp-premium",
        access_mcp_server_ids=["srv-stripe"],
        assigned_team_ids=["team-a"],
    )

    mock_mgr = MagicMock()
    mock_mgr.expand_permission_list.side_effect = lambda x: list(x)

    patches = _patch_proxy_server_globals_for_mcp() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            mock_mgr,
        ),
    ]
    _start_patches(patches)
    try:
        result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(valid_token)
        assert result == ["srv-stripe"]
    finally:
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_mcp_key_access_group_extras_when_key_directly_authorized():
    """Group's assigned_key_ids includes the key's token → server returned (per-key auth)."""
    valid_token = UserAPIKeyAuth(
        token="test-token-hashed",
        access_group_ids=["mcp-per-key"],
        team_id="team-a",
    )
    fake_ag = _fake_mcp_access_group(
        access_group_id="mcp-per-key",
        access_mcp_server_ids=["srv-stripe"],
        assigned_team_ids=[],
        assigned_key_ids=["test-token-hashed"],
    )

    mock_mgr = MagicMock()
    mock_mgr.expand_permission_list.side_effect = lambda x: list(x)

    patches = _patch_proxy_server_globals_for_mcp() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            mock_mgr,
        ),
    ]
    _start_patches(patches)
    try:
        result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(valid_token)
        assert result == ["srv-stripe"]
    finally:
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_mcp_key_access_group_extras_when_key_has_no_groups():
    """Empty access_group_ids → no extras, no DB read."""
    valid_token = UserAPIKeyAuth(
        token="test-token",
        access_group_ids=[],
        team_id="team-a",
    )
    result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(valid_token)
    assert result == []


@pytest.mark.asyncio
async def test_mcp_key_access_group_extras_when_group_has_no_servers():
    """Group authorizes the team but its access_mcp_server_ids is empty → no extras."""
    valid_token = UserAPIKeyAuth(
        token="test-token",
        access_group_ids=["mcp-empty"],
        team_id="team-a",
    )
    fake_ag = _fake_mcp_access_group(
        access_group_id="mcp-empty",
        access_mcp_server_ids=[],
        assigned_team_ids=["team-a"],
    )

    patches = _patch_proxy_server_globals_for_mcp() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    _start_patches(patches)
    try:
        result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(valid_token)
        assert result == []
    finally:
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_mcp_key_access_group_extras_granted_even_when_group_authorizes_neither():
    """Grants are ungated: attaching the group to the key is itself the grant, so its
    servers are contributed even when assigned_team_ids/assigned_key_ids exclude this
    caller. (A team member self-assigning a foreign group to reach past the team
    ceiling is a known, accepted-for-now tradeoff; restricting who may set
    key.access_group_ids is a separate concern.)"""
    valid_token = UserAPIKeyAuth(
        token="team-a-token",
        access_group_ids=["team-b-mcp-group"],
        team_id="team-a",
    )
    fake_ag = _fake_mcp_access_group(
        access_group_id="team-b-mcp-group",
        access_mcp_server_ids=["srv-finance-only"],
        assigned_team_ids=["team-b"],
        assigned_key_ids=["team-b-token"],
    )

    patches = _patch_proxy_server_globals_for_mcp() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    _start_patches(patches)
    try:
        result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(valid_token)
        assert result == ["srv-finance-only"]
    finally:
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_mcp_key_access_group_extras_when_get_access_object_raises():
    """Group lookup failure is treated as no authorization (does not crash)."""
    valid_token = UserAPIKeyAuth(
        token="test-token",
        access_group_ids=["missing-mcp-group"],
        team_id="team-a",
    )
    patches = _patch_proxy_server_globals_for_mcp() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            side_effect=Exception("not found"),
        ),
    ]
    _start_patches(patches)
    try:
        result = await MCPRequestHandler._get_key_access_group_mcp_server_extras(valid_token)
        assert result == []
    finally:
        _stop_patches(patches)


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_unions_key_access_group_extras():
    """End-to-end: team has [srv-team], key access group grants [srv-extra] → both in final list.

    Without this fix [srv-extra] would be intersected away because the team doesn't list it.
    """
    auth = UserAPIKeyAuth(
        token="test-token",
        api_key="test-key",
        team_id="team-a",
        access_group_ids=["mcp-extra-group"],
    )

    with (
        patch.object(
            MCPRequestHandler,
            "_get_allowed_mcp_servers_for_key",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch.object(
            MCPRequestHandler,
            "_get_allowed_mcp_servers_for_team",
            new_callable=AsyncMock,
            return_value=["srv-team"],
        ),
        patch.object(
            MCPRequestHandler,
            "_get_key_access_group_mcp_server_extras",
            new_callable=AsyncMock,
            return_value=["srv-extra"],
        ),
    ):
        result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert sorted(result) == ["srv-extra", "srv-team"]


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_no_union_when_no_authorized_extras():
    """End-to-end: no authorized extras → behavior identical to today (team ceiling enforced)."""
    auth = UserAPIKeyAuth(
        token="test-token",
        api_key="test-key",
        team_id="team-a",
        access_group_ids=["mcp-foreign-group"],
    )

    with (
        patch.object(
            MCPRequestHandler,
            "_get_allowed_mcp_servers_for_key",
            new_callable=AsyncMock,
            return_value=["srv-key-only"],
        ),
        patch.object(
            MCPRequestHandler,
            "_get_allowed_mcp_servers_for_team",
            new_callable=AsyncMock,
            return_value=["srv-team"],
        ),
        patch.object(
            MCPRequestHandler,
            "_get_key_access_group_mcp_server_extras",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        # key ∩ team = {} (no overlap), extras = [] → final = []
        result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert result == []


# ---------------------------------------------------------------------------
# Issue #27657: team unified access_group_ids resolve to MCP servers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_access_group_ids_resolve_to_mcp_servers():
    """A virtual key with empty access_group_ids inherits MCP servers from
    its team's access_group_ids (mirror of the model-side resolution).

    Reproduction of https://github.com/BerriAI/litellm/issues/27657:
    the runtime used to ignore team.access_group_ids when computing the
    MCP scope, so virtual keys saw empty server lists even when their
    team had an MCP-granting access group attached.
    """
    from litellm.proxy._types import LiteLLM_TeamTable

    mock_team = LiteLLM_TeamTable(
        team_id="team-a",
        access_group_ids=["mcp-premium"],
        object_permission_id=None,
    )

    auth = UserAPIKeyAuth(
        token="test-token-hash",
        api_key="sk-test",
        team_id="team-a",
        access_group_ids=[],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team,
        ),
        patch(
            "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=["srv-stripe"],
        ) as mock_resolver,
    ):
        result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)

    assert result == ["srv-stripe"]
    mock_resolver.assert_called_once()
    assert mock_resolver.call_args.kwargs["access_group_ids"] == ["mcp-premium"]


@pytest.mark.asyncio
async def test_team_access_group_ids_union_with_object_permission():
    """When both legacy object_permission and unified team.access_group_ids
    grant MCP servers, the final list is their union."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    for sid in ("srv-direct",):
        global_mcp_server_manager.registry[sid] = MCPServer(
            server_id=sid,
            name=sid,
            server_name=sid,
            url=f"https://{sid}.example.com",
            transport=MCPTransport.http,
        )
    try:
        mock_object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-1",
            mcp_servers=["srv-direct"],
            mcp_access_groups=[],
            vector_stores=[],
        )
        mock_team = LiteLLM_TeamTable(
            team_id="team-a",
            access_group_ids=["mcp-premium"],
            object_permission_id="perm-1",
        )
        mock_team.object_permission = mock_object_permission

        auth = UserAPIKeyAuth(
            token="test-token-hash",
            api_key="sk-test",
            team_id="team-a",
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=mock_team,
            ),
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                new_callable=AsyncMock,
                return_value=["srv-stripe"],
            ),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)

        assert set(result) == {"srv-direct", "srv-stripe"}
    finally:
        global_mcp_server_manager.registry.pop("srv-direct", None)


@pytest.mark.asyncio
async def test_team_access_group_ids_empty_returns_no_extras():
    """Empty team.access_group_ids → resolver called with [], short-circuits
    without DB access, no extras added."""
    from litellm.proxy._types import LiteLLM_TeamTable

    mock_team = LiteLLM_TeamTable(
        team_id="team-a",
        access_group_ids=[],
        object_permission_id=None,
    )

    auth = UserAPIKeyAuth(
        token="test-token-hash",
        api_key="sk-test",
        team_id="team-a",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team,
        ),
        patch(
            "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_resolver,
    ):
        result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)

    assert result == []
    mock_resolver.assert_called_once()
    assert mock_resolver.call_args.kwargs["access_group_ids"] == []


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_includes_team_access_group_extras_end_to_end():
    """End-to-end: virtual key has nothing of its own, team has an MCP
    access group → key sees the granted server through get_allowed_mcp_servers."""
    auth = UserAPIKeyAuth(
        token="test-token",
        api_key="sk-test",
        team_id="team-a",
        access_group_ids=[],
    )

    with (
        patch.object(
            MCPRequestHandler,
            "_get_allowed_mcp_servers_for_key",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch.object(
            MCPRequestHandler,
            "_get_allowed_mcp_servers_for_team",
            new_callable=AsyncMock,
            return_value=["srv-stripe"],
        ),
        patch.object(
            MCPRequestHandler,
            "_get_key_access_group_mcp_server_extras",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert result == ["srv-stripe"]


@pytest.mark.asyncio
async def test_allowed_mcp_servers_for_key_excludes_access_group_ids():
    """The key's own ceiling (which is intersected against the team) must NOT resolve
    access_group_ids — those are additive grants handled separately, so folding them
    in here is exactly the bug this fix removes. A key with only access_group_ids and
    no object_permission yields an empty ceiling, and the group resolver is never
    called from this path."""
    auth = UserAPIKeyAuth(
        token="test-token-hash",
        api_key="sk-test",
        access_group_ids=["mcp-premium"],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=["srv-stripe"],
        ) as mock_resolver,
    ):
        result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(auth)

    assert result == []
    mock_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_allowed_mcp_servers_for_key_uses_object_permission_not_access_groups():
    """The key's own ceiling is built from object_permission alone. Even when the key
    also carries access_group_ids that would resolve to other servers, those grants do
    NOT enter this (intersected) scope — only the object_permission server comes back.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    global_mcp_server_manager.registry["srv-direct"] = MCPServer(
        server_id="srv-direct",
        name="srv-direct",
        server_name="srv-direct",
        url="https://srv-direct.example.com",
        transport=MCPTransport.http,
    )
    try:
        perms = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-1",
            mcp_servers=["srv-direct"],
            mcp_access_groups=[],
            vector_stores=[],
        )
        auth = UserAPIKeyAuth(
            token="test-token-hash",
            api_key="sk-test",
            access_group_ids=["mcp-premium"],
            object_permission=perms,
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                new_callable=AsyncMock,
                return_value=["srv-stripe"],
            ) as mock_resolver,
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(auth)

        assert set(result) == {"srv-direct"}
        mock_resolver.assert_not_called()
    finally:
        global_mcp_server_manager.registry.pop("srv-direct", None)


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_surfaces_ungated_key_access_group_grant_end_to_end():
    """End-to-end: a teamless key has an MCP-granting access group on its
    access_group_ids. The grant is resolved ungated by the additive extras path and
    surfaces through get_allowed_mcp_servers, even though the key's own ceiling
    (object_permission) is empty."""
    auth = UserAPIKeyAuth(
        token="test-token",
        api_key="sk-test",
        access_group_ids=["mcp-group"],
    )

    patches = _patch_proxy_server_globals_for_mcp() + [
        patch(
            "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=["srv-deepwiki"],
        ),
    ]
    _start_patches(patches)
    try:
        extras = await MCPRequestHandler._get_key_access_group_mcp_server_extras(auth)
        assert extras == ["srv-deepwiki"]

        result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert result == ["srv-deepwiki"]
    finally:
        _stop_patches(patches)


def test_expand_permission_list_does_not_honor_all_proxy_sentinel():
    """The all-proxy sentinel is a team-only grant. The shared expand_permission_list
    also feeds the key/org/end_user/agent resolvers, so it must NOT expand the
    sentinel to the full registry; it passes through as an inert literal (denied
    downstream). Concrete ids still resolve normally. If the sentinel were expanded
    here, any stored key/org/end_user permission holding it would silently gain every
    server."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import SpecialMCPServerName
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    sentinel = SpecialMCPServerName.all_proxy_servers.value
    for sid in ("srv-x", "srv-y"):
        global_mcp_server_manager.registry[sid] = MCPServer(
            server_id=sid,
            name=sid,
            server_name=sid,
            url=f"https://{sid}.example.com",
            transport=MCPTransport.http,
        )
    try:
        result = global_mcp_server_manager.expand_permission_list([sentinel])
        assert set(result).isdisjoint({"srv-x", "srv-y"})
        assert result == [sentinel]
        assert global_mcp_server_manager.expand_permission_list(["srv-x"]) == ["srv-x"]
    finally:
        for sid in ("srv-x", "srv-y"):
            global_mcp_server_manager.registry.pop(sid, None)


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_for_team_expands_all_proxy_sentinel_dynamically():
    """The TEAM resolver expands the all-proxy sentinel to every registered server and
    picks up a server registered later, so a team scoped to all-proxy tracks the live
    registry without any change to its stored permission. Reverting the team-side
    expansion collapses this to the inert literal and the result no longer contains the
    real servers."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionTable,
        LiteLLM_TeamTable,
        SpecialMCPServerName,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    for sid in ("srv-x", "srv-y"):
        global_mcp_server_manager.registry[sid] = MCPServer(
            server_id=sid,
            name=sid,
            server_name=sid,
            url=f"https://{sid}.example.com",
            transport=MCPTransport.http,
        )
    try:
        team_perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="team-perm",
            mcp_servers=[SpecialMCPServerName.all_proxy_servers.value],
            mcp_access_groups=[],
            vector_stores=[],
        )
        team_obj = LiteLLM_TeamTable(
            team_id="team-1",
            access_group_ids=[],
            object_permission_id="team-perm",
        )
        team_obj.object_permission = team_perm
        auth = UserAPIKeyAuth(token="test-token", api_key="sk-test", team_id="team-1")

        patches = _patch_proxy_server_globals_for_mcp() + [
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=team_obj,
            ),
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ]
        _start_patches(patches)
        try:
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
            assert set(result) == {"srv-x", "srv-y"}

            global_mcp_server_manager.registry["srv-z"] = MCPServer(
                server_id="srv-z",
                name="srv-z",
                server_name="srv-z",
                url="https://srv-z.example.com",
                transport=MCPTransport.http,
            )
            result_after = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
            assert "srv-z" in result_after
        finally:
            _stop_patches(patches)
    finally:
        for sid in ("srv-x", "srv-y", "srv-z"):
            global_mcp_server_manager.registry.pop(sid, None)


@pytest.mark.asyncio
async def test_key_with_all_proxy_sentinel_does_not_grant_all_servers():
    """Security regression: the all-proxy sentinel is a team-only grant. A KEY whose
    stored object_permission holds the sentinel (via a stale write, a configured
    default, or a bug) must NOT be silently widened to every server at runtime. A
    teamless key with the sentinel resolves to no real server — never srv-secret or the
    full registry. On the pre-hardening code the key path expanded the sentinel and
    this key would reach srv-secret."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionTable,
        SpecialMCPServerName,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    for sid in ("srv-x", "srv-y", "srv-secret"):
        global_mcp_server_manager.registry[sid] = MCPServer(
            server_id=sid,
            name=sid,
            server_name=sid,
            url=f"https://{sid}.example.com",
            transport=MCPTransport.http,
        )
    try:
        key_perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="key-perm",
            mcp_servers=[SpecialMCPServerName.all_proxy_servers.value],
            mcp_access_groups=[],
            vector_stores=[],
        )
        auth = UserAPIKeyAuth(token="test-token", api_key="sk-test", object_permission=key_perm)

        patches = _patch_proxy_server_globals_for_mcp()
        _start_patches(patches)
        try:
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        finally:
            _stop_patches(patches)

        assert "srv-secret" not in result
        assert set(result).isdisjoint(global_mcp_server_manager.get_registry().keys())
    finally:
        for sid in ("srv-x", "srv-y", "srv-secret"):
            global_mcp_server_manager.registry.pop(sid, None)


@pytest.mark.asyncio
async def test_get_allowed_mcp_servers_team_all_proxy_key_scoped_to_one_end_to_end():
    """End-to-end: a team scoped to the all-proxy sentinel is a ceiling of every
    registered server, so a key scoped to a single server (srv-x) resolves to
    exactly that server (key ∩ all-servers == key). If the sentinel branch is
    reverted the team ceiling collapses to the literal marker, the intersection
    empties, and the result is [] instead of ["srv-x"]."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionTable,
        LiteLLM_TeamTable,
        SpecialMCPServerName,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    for sid in ("srv-x", "srv-y"):
        global_mcp_server_manager.registry[sid] = MCPServer(
            server_id=sid,
            name=sid,
            server_name=sid,
            url=f"https://{sid}.example.com",
            transport=MCPTransport.http,
        )
    try:
        key_perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="key-perm",
            mcp_servers=["srv-x"],
            mcp_access_groups=[],
            vector_stores=[],
        )
        team_perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="team-perm",
            mcp_servers=[SpecialMCPServerName.all_proxy_servers.value],
            mcp_access_groups=[],
            vector_stores=[],
        )
        team_obj = LiteLLM_TeamTable(
            team_id="team-1",
            access_group_ids=[],
            object_permission_id="team-perm",
        )
        team_obj.object_permission = team_perm

        auth = UserAPIKeyAuth(
            token="test-token",
            api_key="sk-test",
            team_id="team-1",
            object_permission=key_perm,
        )

        patches = _patch_proxy_server_globals_for_mcp() + [
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=team_obj,
            ),
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                MCPRequestHandler,
                "_get_mcp_servers_from_access_groups",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ]
        _start_patches(patches)
        try:
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        finally:
            _stop_patches(patches)

        assert result == ["srv-x"]
    finally:
        for sid in ("srv-x", "srv-y"):
            global_mcp_server_manager.registry.pop(sid, None)


@pytest.mark.asyncio
class TestMCPDcrBridgeDelegateAdmission:
    """Admission-side arm for a DCR-bridge ``oauth_delegate`` client that authenticates with
    a single envelope bearer (LIT-4338).

    The arm fires only for a single ``is_dcr_bridge`` ``is_oauth_delegate`` target carrying an
    envelope-shaped Authorization. It opens the litellm-signed envelope, admits under the
    recovered identity WITHOUT re-validating (the signature is the proof), and injects the inner
    upstream token under the server's per-server auth-header key so egress forwards it. Everything
    else must stay on its existing admission path.
    """

    _MASTER_KEY = "sk-bridge-master-key-for-envelope-derivation"

    @staticmethod
    def _bridge_delegate_server(server_name="bridge_delegate_server", dcr_bridge=True, alias=None):
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        return MCPServer(
            server_id="bridge-server-id",
            name=server_name or "bridge-fallback-name",
            server_name=server_name,
            alias=alias,
            transport="http",
            auth_type=MCPAuth.oauth_delegate,
            dcr_bridge=dcr_bridge,
        )

    @classmethod
    def _mint_bridge_envelope(
        cls,
        *,
        user_id="envelope-user-42",
        server_id="bridge-server-id",
        access_token="inner-upstream-access-token",
        token_type="Bearer",
        expires_in=1800,
        minted_at=None,
        master_key=None,
    ):
        from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
            envelope_keys_from_master_key,
        )
        from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
            EnvelopeIdentity,
            SealedEnvelope,
            UpstreamTokenGrant,
            mint_envelope,
        )
        from pydantic import SecretStr

        keys = envelope_keys_from_master_key(master_key or cls._MASTER_KEY)
        now = minted_at or datetime.now(timezone.utc)
        sealed = mint_envelope(
            identity=EnvelopeIdentity(user_id=user_id, server_id=server_id),
            grant=UpstreamTokenGrant(
                access_token=SecretStr(access_token),
                token_type=token_type,
                expires_in=expires_in,
            ),
            keys=keys,
            now=now,
        )
        assert isinstance(sealed, SealedEnvelope), sealed
        return sealed.token.get_secret_value()

    async def test_valid_envelope_admits_identity_and_injects_inner_token(self):
        """A valid envelope on a single dcr_bridge oauth_delegate server admits under the envelope's
        identity WITHOUT re-validating (user_api_key_auth is never called) and injects the inner
        upstream token, keyed by the server name, for egress forwarding."""
        envelope = self._mint_bridge_envelope(user_id="envelope-user-42")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            (
                auth_result,
                _mcp_auth_header,
                _mcp_servers,
                mcp_server_auth_headers,
                _oauth2_headers,
                _raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

        # Signature is the proof of prior authentication: identity admitted, no re-validation.
        assert auth_result.user_id == "envelope-user-42"
        mock_auth.assert_not_called()
        # Inner upstream token injected under the per-server key so egress forwards it.
        assert mcp_server_auth_headers == {
            "bridge_delegate_server": {"Authorization": "Bearer inner-upstream-access-token"}
        }

    async def test_alias_only_server_injects_under_alias_egress_can_resolve(self):
        """When server_name is None, the inner token must be keyed under the alias (which egress
        resolves), never under server.name (which egress never looks up), so the forwarded token is
        not silently dropped."""
        envelope = self._mint_bridge_envelope()
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server(
                server_name=None, alias="bridge_alias"
            )
            (_auth, _h, _s, mcp_server_auth_headers, _o, _r) = await MCPRequestHandler.process_mcp_request(scope)

        assert mcp_server_auth_headers == {"bridge_alias": {"Authorization": "Bearer inner-upstream-access-token"}}

    async def test_server_with_no_alias_or_server_name_is_not_admitted_via_bridge_arm(self):
        """A bridge server egress cannot route to (no alias and no server_name) must not take the
        envelope arm; it fails closed to normal oauth2 admission rather than admitting and dropping
        the inner token under an unresolvable key."""
        envelope = self._mint_bridge_envelope()
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=401, detail="Invalid key"),
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server(server_name=None, alias=None)
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        mock_auth.assert_called_once()

    async def test_expired_envelope_fails_closed_401(self):
        """An envelope whose exp is in the past must fail closed with a 401, never fall through to
        anonymous admission."""
        expired = self._mint_bridge_envelope(
            expires_in=60,
            minted_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {expired}".encode("latin-1"))],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        mock_auth.assert_not_called()

    async def test_envelope_minted_for_a_different_server_fails_closed_401(self):
        """An envelope sealed for another server_id must be rejected when presented to this server,
        so a captured or misrouted envelope cannot forward one server's upstream credential to
        another. The signature verifies, but the server binding does not."""
        wrong_server = self._mint_bridge_envelope(server_id="some-other-server-id")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {wrong_server}".encode("latin-1"))],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        mock_auth.assert_not_called()

    async def test_envelope_under_wrong_master_key_fails_closed_401(self):
        """An envelope-shaped bearer whose signature does not verify under the proxy's derived keys
        (e.g. minted against a different master_key, or tampered) must fail closed with a 401."""
        foreign = self._mint_bridge_envelope(master_key="a-different-master-key-entirely")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {foreign}".encode("latin-1"))],
        }

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        mock_auth.assert_not_called()

    async def test_non_envelope_bearer_on_bridge_server_falls_through_to_oauth2_arm(self):
        """A plain (non-envelope) bearer on the same bridge server must NOT be admitted by the
        envelope arm: it falls through to the oauth2 arm, which validates it as a LiteLLM key and
        401s here. Proves the arm is gated on envelope shape, not merely on the target being a
        bridge server."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", b"Bearer plain-upstream-bearer-not-an-envelope")],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        # The envelope arm was skipped, so the oauth2 arm ran and validated the bearer.
        mock_auth.assert_called_once()

    async def test_explicit_litellm_key_wins_over_envelope_arm(self):
        """An explicit x-litellm-api-key is always a LiteLLM credential and its arm precedes the
        envelope arm: user_api_key_auth validates the key and NO inner token is injected, even
        though the Authorization header carries a valid envelope."""
        envelope = self._mint_bridge_envelope()
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [
                (b"x-litellm-api-key", b"sk-explicit-litellm-key"),
                (b"authorization", f"Bearer {envelope}".encode("latin-1")),
            ],
        }

        async def mock_user_api_key_auth(api_key, request):
            return UserAPIKeyAuth(api_key=api_key, user_id="litellm-key-user")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            (
                auth_result,
                _mcp_auth_header,
                _mcp_servers,
                mcp_server_auth_headers,
                _oauth2_headers,
                _raw_headers,
            ) = await MCPRequestHandler.process_mcp_request(scope)

        mock_auth.assert_called_once()
        assert mock_auth.call_args.kwargs["api_key"] == "sk-explicit-litellm-key"
        # The explicit-key arm admitted; the envelope arm never ran, so no inner token is injected.
        assert auth_result.user_id == "litellm-key-user"
        assert mcp_server_auth_headers == {}

    async def test_non_bridge_oauth_delegate_server_does_not_take_envelope_arm(self):
        """An oauth_delegate server that is NOT a DCR bridge (``dcr_bridge`` unset) must not take the
        envelope arm even for an envelope-shaped bearer: is_dcr_bridge is False, so the gate returns
        None and admission falls through to the oauth2 arm (which 401s here)."""
        envelope = self._mint_bridge_envelope()
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/plain_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server(
                server_name="plain_delegate_server", dcr_bridge=False
            )
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        # Not admitted by the envelope arm — the oauth2 arm ran instead.
        mock_auth.assert_called_once()

    async def test_multi_target_including_bridge_server_does_not_take_envelope_arm(self):
        """A multi-target request that includes the bridge server must not take the envelope arm:
        the gate requires exactly one target, so it returns None and admission falls through."""
        from litellm.types.mcp import MCPAuth

        envelope = self._mint_bridge_envelope()
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [
                (b"authorization", f"Bearer {envelope}".encode("latin-1")),
                (b"x-mcp-servers", b"bridge_delegate_server,other_server"),
            ],
        }

        def mock_lookup(name, client_ip=None):
            if name == "bridge_delegate_server":
                return self._bridge_delegate_server()
            return TestMCPDelegateAuthToUpstream._make_server(auth_type=MCPAuth.api_key)

        async def mock_user_api_key_auth_fails(api_key, request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                side_effect=mock_user_api_key_auth_fails,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
        ):
            mock_mgr.get_mcp_server_by_name.side_effect = mock_lookup
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        mock_auth.assert_called_once()

    async def test_admit_helper_returns_new_headers_without_mutating_input(self):
        """Unit: ``_admit_dcr_bridge_delegate`` must return a NEW headers dict that preserves the
        caller's existing per-server entries and adds the injected inner token, never mutating the
        input dict."""
        envelope = self._mint_bridge_envelope(user_id="unit-user")
        existing = {"other_server": {"Authorization": "Bearer someone-elses-token"}}

        with patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY):
            auth_result, new_headers = MCPRequestHandler._admit_dcr_bridge_delegate(
                server=self._bridge_delegate_server(),
                authorization_value=f"Bearer {envelope}",
                mcp_server_auth_headers=existing,
            )

        assert auth_result.user_id == "unit-user"
        # Input untouched.
        assert existing == {"other_server": {"Authorization": "Bearer someone-elses-token"}}
        # New dict carries both the pre-existing entry and the injected inner token.
        assert new_headers is not existing
        assert new_headers == {
            "other_server": {"Authorization": "Bearer someone-elses-token"},
            "bridge_delegate_server": {"Authorization": "Bearer inner-upstream-access-token"},
        }

    async def test_admit_helper_raises_500_when_master_key_missing(self):
        """Unit: without a configured master_key the gateway cannot derive envelope keys, so
        admission raises a 500 rather than silently admitting."""
        envelope = self._mint_bridge_envelope()
        with patch("litellm.proxy.proxy_server.master_key", None):
            with pytest.raises(HTTPException) as exc_info:
                MCPRequestHandler._admit_dcr_bridge_delegate(
                    server=self._bridge_delegate_server(),
                    authorization_value=f"Bearer {envelope}",
                    mcp_server_auth_headers=None,
                )
        assert exc_info.value.status_code == 500

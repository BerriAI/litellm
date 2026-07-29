import contextlib
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
    _is_mcp_admitted_user_subject,
)
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    ProxyException,
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

    async def test_admitted_subject_not_zeroed_by_require_key_mcp_access_defined(self):
        """10x-flow regression: with require_key_mcp_access_defined ON (team = ceiling for keys), a
        keyless gateway/bridge-admitted subject whose ONLY access path is team membership must still
        inherit the team's servers. The flag zeros empty *virtual keys* that must declare their own
        access; a keyless admitted user has no key to declare it on, so it must not be zeroed."""
        auth = UserAPIKeyAuth(api_key=None, user_id="sso-user")
        auth.mcp_admitted_user_subject = True
        with (
            patch.object(
                MCPRequestHandler, "_get_allowed_mcp_servers_for_key", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_team",
                new_callable=AsyncMock,
                return_value=["team_server1", "team_server2"],
            ),
            patch.object(
                MCPRequestHandler, "_get_key_access_group_mcp_server_extras", new_callable=AsyncMock, return_value=[]
            ),
            patch("litellm.proxy.proxy_server.general_settings", {"require_key_mcp_access_defined": True}),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert sorted(result) == ["team_server1", "team_server2"]

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

    def _toolset_only_object_permission(self, toolset_ids):
        key_object_permission = MagicMock()
        key_object_permission.mcp_servers = []
        key_object_permission.mcp_access_groups = []
        key_object_permission.mcp_tool_permissions = None
        key_object_permission.mcp_toolsets = toolset_ids
        return key_object_permission

    def _mock_manager_with_toolsets(self, toolset_perms):
        mock_manager = MagicMock()
        mock_manager.expand_permission_list = MagicMock(side_effect=lambda servers: servers)
        mock_manager.expand_tool_permissions = MagicMock(side_effect=lambda perms: perms or {})
        mock_manager.resolve_toolset_tool_permissions = AsyncMock(return_value=toolset_perms)
        return mock_manager

    async def test_get_allowed_mcp_servers_for_key_includes_toolset_servers(self):
        """A key granted only mcp_toolsets must reach the toolset's servers on
        every path (list, call, REST); regression for the list-ok/call-403 bug"""
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = self._toolset_only_object_permission(["toolset-1"])
        mock_manager = self._mock_manager_with_toolsets({"server-a": ["lookup_status"]})

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
            patch.object(MCPRequestHandler, "_get_mcp_servers_from_access_groups", AsyncMock(return_value=[])),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

        assert result == ["server-a"]
        mock_manager.resolve_toolset_tool_permissions.assert_awaited_once_with(toolset_ids=["toolset-1"])

    async def test_get_allowed_mcp_servers_for_key_skips_toolset_resolution_when_none_granted(self):
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = self._toolset_only_object_permission([])
        key_object_permission.mcp_servers = ["server-direct"]
        mock_manager = self._mock_manager_with_toolsets({})

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
            patch.object(MCPRequestHandler, "_get_mcp_servers_from_access_groups", AsyncMock(return_value=[])),
        ):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_auth)

        assert result == ["server-direct"]
        mock_manager.resolve_toolset_tool_permissions.assert_not_awaited()

    async def test_get_allowed_mcp_servers_toolset_only_key_end_to_end_inheritance(self):
        """The full get_allowed_mcp_servers flow (key/team inheritance, no team
        restriction) surfaces toolset-granted servers for a toolset-only key"""
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = self._toolset_only_object_permission(["toolset-1"])
        mock_manager = self._mock_manager_with_toolsets({"server-a": ["lookup_status"]})

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
            patch.object(MCPRequestHandler, "_get_mcp_servers_from_access_groups", AsyncMock(return_value=[])),
            patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_team", AsyncMock(return_value=[])),
            patch.object(MCPRequestHandler, "_get_key_access_group_mcp_server_extras", AsyncMock(return_value=[])),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)

        assert result == ["server-a"]

    async def test_toolset_servers_stay_capped_by_team_ceiling(self):
        """Toolset grants expand the KEY's scope, which the team ceiling still
        intersects; a toolset must never grant a server the team does not allow.
        Pins that toolset expansion lives in the intersected key scope, not the
        additive access-group path"""
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="test-team")
        key_object_permission = self._toolset_only_object_permission(["toolset-1"])
        mock_manager = self._mock_manager_with_toolsets(
            {"server-in-team": ["lookup_status"], "server-outside-team": ["other_tool"]}
        )

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
            patch.object(MCPRequestHandler, "_get_mcp_servers_from_access_groups", AsyncMock(return_value=[])),
            patch.object(
                MCPRequestHandler,
                "_get_allowed_mcp_servers_for_team",
                AsyncMock(return_value=["server-in-team", "server-unrelated"]),
            ),
            patch.object(MCPRequestHandler, "_get_key_access_group_mcp_server_extras", AsyncMock(return_value=[])),
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(user_api_key_auth)

        assert result == ["server-in-team"]

    async def test_get_allowed_tools_for_server_unions_toolset_and_direct_tools(self):
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = self._toolset_only_object_permission(["toolset-1"])
        key_object_permission.mcp_tool_permissions = {"server-a": ["direct_tool"]}
        mock_manager = self._mock_manager_with_toolsets({"server-a": ["lookup_status"]})

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch.object(MCPRequestHandler, "_get_team_object_permission", AsyncMock(return_value=None)),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await MCPRequestHandler.get_allowed_tools_for_server(
                server_id="server-a",
                user_api_key_auth=user_api_key_auth,
            )

        assert result is not None
        assert set(result) == {"direct_tool", "lookup_status"}

    async def test_get_allowed_tools_for_server_toolset_only_key_restricts_to_toolset_tools(self):
        """A toolset grant must RESTRICT the server's tools, not fall through to
        the allow-all default; otherwise merging servers alone would over-grant
        every tool on a toolset-referenced server"""
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = self._toolset_only_object_permission(["toolset-1"])
        mock_manager = self._mock_manager_with_toolsets({"server-a": ["lookup_status"]})

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch.object(MCPRequestHandler, "_get_team_object_permission", AsyncMock(return_value=None)),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            allowed = await MCPRequestHandler.get_allowed_tools_for_server(
                server_id="server-a",
                user_api_key_auth=user_api_key_auth,
            )
            is_granted_tool_allowed = await MCPRequestHandler.is_tool_allowed_for_server(
                tool_name="lookup_status",
                server_id="server-a",
                user_api_key_auth=user_api_key_auth,
            )
            is_other_tool_allowed = await MCPRequestHandler.is_tool_allowed_for_server(
                tool_name="delete_everything",
                server_id="server-a",
                user_api_key_auth=user_api_key_auth,
            )

        assert allowed == ["lookup_status"]
        assert is_granted_tool_allowed is True
        assert is_other_tool_allowed is False

    async def test_get_allowed_tools_for_server_without_restrictions_stays_allow_all(self):
        user_api_key_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        key_object_permission = self._toolset_only_object_permission([])
        mock_manager = self._mock_manager_with_toolsets({})

        with (
            patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
            patch.object(MCPRequestHandler, "_get_team_object_permission", AsyncMock(return_value=None)),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                mock_manager,
            ),
        ):
            result = await MCPRequestHandler.get_allowed_tools_for_server(
                server_id="server-a",
                user_api_key_auth=user_api_key_auth,
            )

        assert result is None

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
    envelope-shaped Authorization. It opens the litellm-signed envelope, reloads the live key
    record the sealed ``key_hash`` references so the caller is admitted under the key's current
    authorization context (team/org/object-permission) and revocation state, and injects the inner
    upstream token under the server's per-server auth-header key so egress forwards it. A key that
    is missing, blocked, or expired fails closed with a 401. Everything else must stay on its
    existing admission path.
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

    _KEY_HASH = "hashed-litellm-key-abc123"

    @classmethod
    def _mint_bridge_envelope(
        cls,
        *,
        key_hash=None,
        user_id=None,
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
            SealedEnvelope,
            UpstreamTokenGrant,
            key_hash_identity,
            mint_envelope,
            user_identity,
        )
        from pydantic import SecretStr

        identity = (
            user_identity(server_id=server_id, user_id=user_id)
            if user_id is not None
            else key_hash_identity(server_id=server_id, key_hash=key_hash or cls._KEY_HASH)
        )
        keys = envelope_keys_from_master_key(master_key or cls._MASTER_KEY)
        now = minted_at or datetime.now(timezone.utc)
        sealed = mint_envelope(
            identity=identity,
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

    @staticmethod
    def _reloaded_key(**overrides):
        """A live key record as ``get_key_object`` would return it: carries real authorization
        context (key identity, team, org, and an object-permission restricting MCP servers) so a
        test can prove admission admits under THAT context rather than a blank identity."""
        defaults = dict(
            user_id="envelope-user-42",
            api_key=TestMCPDcrBridgeDelegateAdmission._KEY_HASH,
            team_id="team-restricted",
            org_id="org-restricted",
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="op-1", mcp_servers=["only-this-server"]
            ),
        )
        defaults.update(overrides)
        return UserAPIKeyAuth(**defaults)

    @staticmethod
    @contextlib.contextmanager
    def _patch_key_reload(*, return_value=None, side_effect=None, team_blocked=False, owner=None, project_object=None):
        """Patch the live-policy dependencies of the admission arm: the ``get_key_object`` reload,
        the ``prisma_client`` / ``user_api_key_cache`` globals, and optionally the live objects the
        policy gates re-check. ``team_blocked=True`` patches the centralized gate's
        ``get_team_object`` at the ``user_api_key_auth`` namespace it actually calls; ``owner``
        patches the SCIM gate's ``get_user_object`` (``auth_checks`` namespace); ``project_object``
        patches the centralized gate's ``get_project_object``. Unpatched lookups hit the MagicMock
        prisma and are swallowed (``_safe_fetch`` / the SCIM gate's fail-open), so their checks
        skip. Yields the ``get_key_object`` mock so callers can assert the sealed ``key_hash`` was
        the reload key."""
        get_key_object = AsyncMock(return_value=return_value, side_effect=side_effect)
        patchers = [
            patch("litellm.proxy.auth.auth_checks.get_key_object", get_key_object),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        ]
        if team_blocked:
            patchers.append(
                patch(
                    "litellm.proxy.auth.user_api_key_auth.get_team_object",
                    AsyncMock(return_value=MagicMock(blocked=True)),
                )
            )
        if owner is not None:
            patchers.append(patch("litellm.proxy.auth.auth_checks.get_user_object", AsyncMock(return_value=owner)))
        if project_object is not None:
            patchers.append(
                patch(
                    "litellm.proxy.auth.user_api_key_auth.get_project_object",
                    AsyncMock(return_value=project_object),
                )
            )
        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            yield get_key_object

    @staticmethod
    @contextlib.contextmanager
    def _patch_user_reload(*, return_value=None, side_effect=None):
        """Patch the user-subject reload path an interactively-minted envelope takes: the
        ``get_user_object`` lookup ``_reload_admitted_user`` runs (which also drives the SCIM gate),
        plus the ``prisma_client`` / ``user_api_key_cache`` globals. The centralized gate's own
        fetches fail-safe to None under the MagicMock prisma, so an unblocked user admits. Yields the
        ``get_user_object`` mock so a caller can assert the sealed user_id was the reload key."""
        get_user_object = AsyncMock(return_value=return_value, side_effect=side_effect)
        with (
            patch("litellm.proxy.auth.auth_checks.get_user_object", get_user_object),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        ):
            yield get_user_object

    @staticmethod
    def _wrapped_user_lookup_error(original: BaseException) -> ValueError:
        """Reproduce get_user_object's real exception contract (litellm/proxy/auth/auth_checks.py): it
        catches every DB failure in a broad ``except`` and re-raises a bare ``ValueError``, so the
        original error (a missing-user Exception or a real outage) survives only as ``__context__``.
        Injecting a raw ConnectionError/Exception instead would exercise a shape production never
        produces and let a chain-blind outage classifier pass. That wrapping fidelity is itself pinned by
        test_get_user_object_wraps_db_outage_as_valueerror_preserving_context in test_auth_checks."""
        try:
            raise original
        except BaseException:
            try:
                raise ValueError(f"User doesn't exist in db. Got error - {original}")
            except ValueError as wrapped:
                return wrapped

    @staticmethod
    def _mcp_request(path="/mcp/bridge_delegate_server"):
        """A minimal ``Request`` for direct ``_admit_dcr_bridge_delegate`` calls, mirroring how
        ``process_mcp_request`` builds one from the ASGI scope with a stubbed empty JSON body."""
        from starlette.requests import Request

        request = Request(scope={"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""})

        async def mock_body():
            return b"{}"

        request.body = mock_body
        return request

    async def test_valid_envelope_reloads_live_key_and_admits_its_authorization_context(self):
        """A valid envelope admits under the LIVE key record the sealed key_hash references, not a
        blank identity: the reload is keyed by that exact hash, and the admitted auth carries the
        key's current team/org/object-permission (the MCP tool/server restrictions the finding was
        about). The heavyweight ``user_api_key_auth`` pipeline is still never invoked. The inner
        upstream token is injected under the per-server key for egress."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
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
            self._patch_key_reload(return_value=self._reloaded_key()) as get_key_object,
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

        # The live key was reloaded by the exact hash the envelope sealed.
        assert get_key_object.await_args.kwargs["hashed_token"] == self._KEY_HASH
        # Admission carries the reloaded key's authorization context, not a blank UserAPIKeyAuth.
        assert auth_result.user_id == "envelope-user-42"
        assert auth_result.team_id == "team-restricted"
        assert auth_result.org_id == "org-restricted"
        assert auth_result.object_permission is not None
        assert auth_result.object_permission.mcp_servers == ["only-this-server"]
        # The full raw-key auth pipeline is still bypassed for the envelope arm.
        mock_auth.assert_not_called()
        # Inner upstream token injected under the per-server key so egress forwards it.
        assert mcp_server_auth_headers == {
            "bridge_delegate_server": {"Authorization": "Bearer inner-upstream-access-token"}
        }

    async def test_user_subject_envelope_admits_under_the_reloaded_user(self):
        """An interactively-minted (user_id) envelope admits under the reloaded USER, not a key: the
        reload is keyed by the sealed user_id, the admitted auth carries that user_id, the raw-key
        pipeline is never invoked, and the inner upstream token is injected for egress. This is the
        interactive-DCR admission the whole flow exists for."""
        envelope = self._mint_bridge_envelope(user_id="sso-user-7")
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
            self._patch_user_reload(
                return_value=MagicMock(
                    user_id="sso-user-7",
                    organization_id=None,
                    metadata={"scim_active": True},
                    user_role=None,
                    object_permission=None,
                    object_permission_id=None,
                )
            ) as get_user_object,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            (auth_result, _h, _s, mcp_server_auth_headers, _o, _r) = await MCPRequestHandler.process_mcp_request(scope)

        assert get_user_object.await_args.kwargs["user_id"] == "sso-user-7"
        assert auth_result.user_id == "sso-user-7"
        mock_auth.assert_not_called()
        assert mcp_server_auth_headers == {
            "bridge_delegate_server": {"Authorization": "Bearer inner-upstream-access-token"}
        }

    async def test_user_subject_envelope_carries_the_users_mcp_object_permission(self):
        """The admitted user's own MCP object permission rides on the returned auth so the shared
        get_allowed_mcp_servers grants the user their litellm-granted servers, rather than admitting a
        bare user with no MCP access. Regression for the signed-in SSO client getting zero tools because
        the reload dropped the user's object permission."""
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="op-user-7", mcp_servers=["bridge_delegate_server"]
        )
        envelope = self._mint_bridge_envelope(user_id="sso-user-7")
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
            self._patch_user_reload(
                return_value=MagicMock(
                    user_id="sso-user-7",
                    organization_id=None,
                    metadata={"scim_active": True},
                    user_role=None,
                    object_permission=object_permission,
                    object_permission_id="op-user-7",
                )
            ),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            (auth_result, _h, _s, _headers, _o, _r) = await MCPRequestHandler.process_mcp_request(scope)

        assert auth_result.object_permission is not None
        assert auth_result.object_permission.mcp_servers == ["bridge_delegate_server"]

    async def test_user_subject_envelope_missing_user_fails_closed_401(self):
        """A user_id envelope whose user has since been deleted must fail closed with a 401, not a 500.
        get_user_object catches the missing row and re-raises a bare ValueError (it does not return None
        on the production path), so the reload must fail closed rather than let it propagate as an opaque
        500, and must not mistake the wrapped ValueError for a DB outage. Regression for the missing-user
        path surfacing as a 500."""
        envelope = self._mint_bridge_envelope(user_id="ghost-user")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_user_reload(side_effect=self._wrapped_user_lookup_error(Exception())),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    async def test_user_subject_envelope_db_outage_is_retryable_503(self):
        """A transient database outage while reloading the envelope's user is a retryable 503, not an
        opaque 500, matching the key path's contract so an interactive DCR client retries instead of
        treating a live identity as invalid. get_user_object wraps the outage in a bare ValueError, so this
        exercises the chain-aware classifier; a raw ConnectionError would falsely pass even the old
        chain-blind check because it is an OSError. Regression for the user reload dropping the 503 arm."""
        envelope = self._mint_bridge_envelope(user_id="sso-user-7")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_user_reload(
                side_effect=self._wrapped_user_lookup_error(ConnectionError("auth database unreachable"))
            ),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 503

    async def test_user_subject_envelope_scim_deactivated_user_fails_closed_401(self):
        """SCIM-deactivating the envelope's user revokes it immediately: the reloaded user carries
        scim_active False, so admission 401s rather than letting an offboarded user keep tool access
        until the envelope expires."""
        envelope = self._mint_bridge_envelope(user_id="offboarded-user")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_user_reload(
                return_value=MagicMock(user_id="offboarded-user", organization_id=None, metadata={"scim_active": False})
            ),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    async def test_revoked_key_envelope_fails_closed_401(self):
        """An envelope whose key has since been deleted must fail closed: ``get_key_object`` raises
        for the missing row, so admission 401s instead of admitting the caller as an unrestricted
        identity. This is the core regression for the dropped-authorization-context finding."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        revoked = self._patch_key_reload(
            side_effect=ProxyException(
                message="Authentication Error, Invalid proxy server token passed.",
                type="token_not_found_in_db",
                param="key",
                code=401,
            )
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            revoked,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401
        mock_auth.assert_not_called()

    async def test_blocked_key_envelope_fails_closed_401(self):
        """A reloaded key that is blocked must fail closed with a 401, so revoking a key by blocking
        it takes effect immediately for any envelope still holding its hash."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key(blocked=True)),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    async def test_expired_key_record_fails_closed_401(self):
        """A reloaded key past its expiry must fail closed with a 401, distinct from an expired
        envelope: even a still-valid envelope cannot outlive the key it was minted under."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        expired_at = datetime.now(timezone.utc) - timedelta(hours=1)

        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key(expires=expired_at)),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    async def test_blocked_team_envelope_fails_closed_401(self):
        """Blocking the key's TEAM must revoke its envelopes immediately: the reloaded key is active
        but its team is blocked, so admission 401s. Without the live team re-check, a caller could
        keep executing tools after an admin blocked the team, until the envelope expired."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key(), team_blocked=True),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    async def test_scim_deactivated_owner_envelope_fails_closed_401(self):
        """SCIM-deactivating the key's OWNER must revoke the user's envelopes immediately: the
        standard pipeline rejects every key of a deactivated user inline in the builder, so the
        admission arm mirrors that gate. Without it, IdP offboarding would leave the offboarded
        user's already-minted envelopes executing tools until they expired."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(
                return_value=self._reloaded_key(),
                owner=MagicMock(metadata={"scim_active": False}),
            ),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    async def test_scim_active_owner_envelope_admits(self):
        """A SCIM-ACTIVE owner must still be admitted: the gate rejects only an explicit
        ``scim_active: False``, so SCIM-managed users whose accounts are in good standing keep
        working (and non-SCIM deployments, which never set the flag, are untouched)."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(
                return_value=self._reloaded_key(),
                owner=MagicMock(metadata={"scim_active": True}),
            ),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            (auth_result, _, _, _, _, _) = await MCPRequestHandler.process_mcp_request(scope)

        assert auth_result.user_id == "envelope-user-42"

    async def test_blocked_project_envelope_fails_closed_401(self):
        """Blocking the key's PROJECT must revoke its envelopes immediately: the admitted identity
        runs through the standard pipeline's centralized policy gate, which rejects a blocked
        project exactly as it would for the same key presented directly. This is the regression for
        the project half of the revocation finding; the gate also covers future policy dimensions
        without the admission arm mirroring them one by one."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }

        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(
                return_value=self._reloaded_key(project_id="project-restricted"),
                project_object=MagicMock(blocked=True),
            ),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 401

    _POLICY_GATE = "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp._run_centralized_common_checks"

    async def _enforce_with_gate_error(self, error):
        """Drive _enforce_admitted_live_policy with the centralized gate raising ``error`` and return
        the HTTPException the arm maps it to."""
        with patch(self._POLICY_GATE, new=AsyncMock(side_effect=error)):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler._enforce_admitted_live_policy(
                    admitted=UserAPIKeyAuth(user_id="envelope-user-42"),
                    request=self._mcp_request(),
                    route="/mcp/bridge_delegate_server",
                )
        return exc_info.value

    async def test_over_budget_admission_surfaces_429_not_401(self):
        """A validly-authenticated but over-budget identity surfaces the standard pipeline's 429, not
        a misleading 401. Flattening budget to 401 told the caller their credential was invalid, which
        on a DCR client reads as broken auth and triggers a re-authorize that cannot fix a budget
        problem. Regression for the status-flattening finding on the live-policy gate."""
        import litellm

        mapped = await self._enforce_with_gate_error(litellm.BudgetExceededError(current_cost=10.0, max_budget=1.0))
        assert mapped.status_code == 429

    async def test_db_outage_during_policy_surfaces_503_not_401(self):
        """A transient database outage during the live-policy gate surfaces a retryable 503, not a 401
        that masks the outage as an auth failure and tells a valid caller to re-authenticate."""
        mapped = await self._enforce_with_gate_error(ConnectionError("could not reach database server"))
        assert mapped.status_code == 503

    async def test_db_outage_during_key_reload_surfaces_503_not_500(self):
        """A DB outage while reloading the admitted key surfaces a retryable 503, not the opaque 500 a
        raw get_key_object transport error would otherwise propagate as, and not a 401 that masks the
        outage as an auth failure. Regression for the reload-path exception gap."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(side_effect=ConnectionError("could not reach database server")),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 503

    async def test_envelope_for_key_barred_from_mcp_routes_is_rejected_403(self):
        """A key whose allowed_routes exclude MCP must not reach tools via an envelope: the arm runs
        RouteChecks.should_call_route before admitting, exactly as the standard pipeline does between
        the builder and common_checks. A route-restricted key can mint an envelope at the token
        endpoint (not itself an MCP route) and would otherwise replay it against MCP, because the
        centralized checks treat MCP as an inference route and never re-check allowed_routes; the
        route gate rejects it with its own 403."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key(allowed_routes=["/chat/completions"])),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 403

    async def test_envelope_rejected_by_proxy_wide_pre_db_gates_403(self):
        """The envelope arm runs the same proxy-wide pre-DB gates user_api_key_auth applies before any
        key lookup (request size, body safety, IP allowlist, general_settings route allowlist). Here
        the proxy route allowlist forbids MCP, so the envelope is turned away with a 403 before the
        identity is even reloaded, closing the gap where an envelope bypassed the IP/route allowlists
        the normal MCP admission path enforces."""
        envelope = self._mint_bridge_envelope(key_hash=self._KEY_HASH)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/bridge_delegate_server",
            "headers": [(b"authorization", f"Bearer {envelope}".encode("latin-1"))],
        }
        with (
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch("litellm.proxy.proxy_server.general_settings", {"allowed_routes": ["/chat/completions"]}),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server()
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(scope)

        assert exc_info.value.status_code == 403

    async def test_blocked_state_bare_exception_stays_401(self):
        """A blocked team/project raises a bare Exception (no status) in common_checks, which the
        standard pipeline renders as 401; the arm keeps failing those closed as 401, never a 500."""
        mapped = await self._enforce_with_gate_error(Exception("Team=team-x is blocked."))
        assert mapped.status_code == 401

    async def test_subcheck_httpexception_status_preserved(self):
        """A sub-check that raises its own HTTPException (e.g. a 403 model-access denial) keeps that
        status through the arm rather than being flattened to 401."""
        mapped = await self._enforce_with_gate_error(HTTPException(status_code=403, detail="model not allowed"))
        assert mapped.status_code == 403

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
            self._patch_key_reload(return_value=self._reloaded_key()),
        ):
            mock_mgr.get_mcp_server_by_name.return_value = self._bridge_delegate_server(
                server_name=None, alias="bridge_alias"
            )
            (_auth, _h, _s, mcp_server_auth_headers, _o, _r) = await MCPRequestHandler.process_mcp_request(scope)

        assert mcp_server_auth_headers == {"bridge_alias": {"Authorization": "Bearer inner-upstream-access-token"}}

    async def test_sealed_token_wins_over_caller_forwarded_alias_header(self):
        """When a bridge server has both a server_name and a distinct alias, the sealed inner token
        must occupy the alias slot, the identifier egress resolves first. Otherwise a caller who
        forwards x-mcp-{alias}-authorization keeps that entry at the higher-priority slot and pairs
        the admitted identity with an attacker-chosen upstream credential."""
        envelope = self._mint_bridge_envelope()
        attacker_forwarded = {"bridge_alias": {"Authorization": "Bearer ATTACKER-UPSTREAM-TOKEN"}}

        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key()),
        ):
            _auth, new_headers = await MCPRequestHandler._admit_dcr_bridge_delegate(
                server=self._bridge_delegate_server(server_name="bridge_name", alias="bridge_alias"),
                authorization_value=f"Bearer {envelope}",
                mcp_server_auth_headers=attacker_forwarded,
                request=self._mcp_request(),
                route="/mcp/bridge_name",
            )

        # The sealed token owns the alias slot, overwriting the caller's value; the attacker token
        # survives nowhere egress would resolve.
        assert new_headers == {"bridge_alias": {"Authorization": "Bearer inner-upstream-access-token"}}

    @pytest.mark.parametrize(
        "server_name,alias",
        [
            (None, "bridge_alias"),
            ("bridge_delegate_server", None),
            ("bridge_name", "bridge_alias"),
        ],
        ids=["alias_only", "server_name_only", "both"],
    )
    async def test_injection_key_agrees_with_egress_lookup(self, server_name, alias):
        """Round-trip the injected headers through the REAL egress resolver for every admissible
        server shape: whatever identifier the admission arm keys the sealed token under,
        ``lookup_mcp_server_auth_in_headers`` called the way egress calls it (alias first, then
        server_name) must recover exactly that token. This pins the agreement between the two key
        hierarchies so neither side can drift and silently drop the forwarded token."""
        from litellm.proxy._experimental.mcp_server.utils import lookup_mcp_server_auth_in_headers

        envelope = self._mint_bridge_envelope()
        server = self._bridge_delegate_server(server_name=server_name, alias=alias)

        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key()),
        ):
            _auth, new_headers = await MCPRequestHandler._admit_dcr_bridge_delegate(
                server=server,
                authorization_value=f"Bearer {envelope}",
                mcp_server_auth_headers=None,
                request=self._mcp_request(),
                route="/mcp/bridge_delegate_server",
            )

        resolved = lookup_mcp_server_auth_in_headers(
            new_headers,
            alias=server.alias,
            server_name=server.server_name,
        )
        assert resolved == {"Authorization": "Bearer inner-upstream-access-token"}

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
        envelope = self._mint_bridge_envelope()
        existing = {"other_server": {"Authorization": "Bearer someone-elses-token"}}

        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            self._patch_key_reload(return_value=self._reloaded_key(user_id="unit-user")),
        ):
            auth_result, new_headers = await MCPRequestHandler._admit_dcr_bridge_delegate(
                server=self._bridge_delegate_server(),
                authorization_value=f"Bearer {envelope}",
                mcp_server_auth_headers=existing,
                request=self._mcp_request(),
                route="/mcp/bridge_delegate_server",
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
                await MCPRequestHandler._admit_dcr_bridge_delegate(
                    server=self._bridge_delegate_server(),
                    authorization_value=f"Bearer {envelope}",
                    mcp_server_auth_headers=None,
                    request=self._mcp_request(),
                    route="/mcp/bridge_delegate_server",
                )
        assert exc_info.value.status_code == 500

    async def test_admit_helper_raises_500_when_no_db_connection(self):
        """Unit: with a valid envelope but no database to reload the key from, admission raises a 500
        rather than admitting on unresolved authorization."""
        envelope = self._mint_bridge_envelope()
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch("litellm.proxy.proxy_server.prisma_client", None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler._admit_dcr_bridge_delegate(
                    server=self._bridge_delegate_server(),
                    authorization_value=f"Bearer {envelope}",
                    mcp_server_auth_headers=None,
                    request=self._mcp_request(),
                    route="/mcp/bridge_delegate_server",
                )
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
class TestAggregateGatewayDcrChallenge:
    """The mcp_gateway_dcr front door: a 401 on the aggregate /mcp scope must
    carry the RFC 9728 resource_metadata challenge pointing at the gateway's
    own protected-resource metadata, and must NOT fire for named-server
    targets, explicit litellm keys, or non-401 failures."""

    _AUTH_PATCH_TARGET = "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth"
    _EXPECTED_RESOURCE_METADATA = 'resource_metadata="http://testserver/.well-known/oauth-protected-resource/mcp"'

    def _scope(self, path="/mcp", extra_headers=()):
        return {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"host", b"testserver"), *extra_headers],
        }

    def _auth_401(self):
        async def _raise(api_key, request):
            raise ProxyException(
                message="Authentication Error: Invalid API key",
                type="auth_error",
                param="api_key",
                code=401,
            )

        return _raise

    async def test_challenge_on_anonymous_aggregate_mcp(self):
        """Anonymous request to the aggregate /mcp: 401 plus
        the bare bearer challenge (no error attribute, RFC 6750 section 3.1)."""
        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(self._scope())
        assert exc_info.value.status_code == 401
        www_authenticate = (exc_info.value.headers or {})["WWW-Authenticate"]
        assert www_authenticate == f"Bearer {self._EXPECTED_RESOURCE_METADATA}"

    async def test_challenge_invalid_token_on_failed_bearer(self):
        """A bearer that fails LiteLLM admission at aggregate scope (an expired
        gateway session, a revoked key) re-challenges with error=invalid_token
        so a spec client re-authorizes instead of retrying the dead token."""
        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(
                    self._scope(extra_headers=((b"authorization", b"Bearer expired-session-token"),))
                )
        assert exc_info.value.status_code == 401
        www_authenticate = (exc_info.value.headers or {})["WWW-Authenticate"]
        assert www_authenticate == f'Bearer error="invalid_token", {self._EXPECTED_RESOURCE_METADATA}'

    async def test_challenge_inserts_server_root_path(self):
        """With SERVER_ROOT_PATH set the resource_metadata URL must carry the same path-inserted
        root segment the aggregate PRM route is registered with (both derive it from
        well_known_root_suffix), so a DCR client behind a sub-path is pointed at a route that
        exists instead of a 404. Regression: the challenge used to hard-code /mcp and omit the
        root path the route inserts."""
        import os

        with (
            patch.dict(os.environ, {"SERVER_ROOT_PATH": "/litellm"}),
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(self._scope())
        www_authenticate = (exc_info.value.headers or {})["WWW-Authenticate"]
        assert (
            'resource_metadata="http://testserver/.well-known/oauth-protected-resource/litellm/mcp"' in www_authenticate
        )

    async def test_no_challenge_for_explicit_litellm_key(self):
        """An explicit x-litellm-api-key declares a litellm-key client; a typo
        there must surface the real auth error, never a DCR challenge that
        would send SDKs into a sign-in flow."""
        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(
                    self._scope(extra_headers=((b"x-litellm-api-key", b"sk-typo"),))
                )

    async def test_no_challenge_for_named_servers_header(self):
        """x-mcp-servers names explicit targets; the per-server challenge paths
        own those, so the aggregate challenge must not fire."""
        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(self._scope(extra_headers=((b"x-mcp-servers", b"github"),)))

    async def test_no_challenge_for_path_named_server(self):
        """/mcp/{server} targets one server; the aggregate challenge must not
        fire even when that server does not resolve."""
        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(self._scope(path="/mcp/github"))

    async def test_no_challenge_for_client_supplied_mcp_auth(self):
        """Per-server x-mcp-{alias}-authorization headers mean the caller is
        not a cold-start DCR client; keep the original error."""
        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=self._auth_401()),
        ):
            with pytest.raises(ProxyException):
                await MCPRequestHandler.process_mcp_request(
                    self._scope(extra_headers=((b"x-mcp-github-authorization", b"Bearer upstream"),))
                )

    async def test_no_challenge_for_non_401_failure(self):
        """Only genuine 401s convert to a challenge; a 500 stays a 500."""

        async def _raise_500(api_key, request):
            raise ProxyException(message="boom", type="server_error", param=None, code=500)

        with (
            patch(self._AUTH_PATCH_TARGET, side_effect=_raise_500),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await MCPRequestHandler.process_mcp_request(self._scope())
        assert str(exc_info.value.code) == "500"


@pytest.mark.asyncio
class TestGatewaySessionAdmission:
    """The aggregate /mcp session-bearer admission arm (mcp_gateway_dcr). A valid session
    token admits under the LIVE litellm user it references; an invalid/expired/refresh/foreign
    token fails closed with the aggregate invalid_token challenge; the arm fires ONLY at the
    aggregate scope, never for named servers or per-server flows."""

    _MASTER_KEY = "sk-gateway-session-admission-master-key"

    def _session_bearer(self, user_id="sso-user-42", client_id="llm_dcrc_abc"):
        from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
            session_keys_from_master_key,
        )
        from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
            SessionPrincipal,
            mint_session_token,
            mint_session_refresh_token,
        )

        keys = session_keys_from_master_key(self._MASTER_KEY)
        principal = SessionPrincipal(user_id=user_id, client_id=client_id)
        return mint_session_token, mint_session_refresh_token, principal, keys

    def _access_token(self, **kw):
        from datetime import datetime, timezone

        mint, _refresh, principal, keys = self._session_bearer(**kw)
        return mint(principal, keys, datetime(2030, 1, 1, tzinfo=timezone.utc)).token.get_secret_value()

    def _scope(self, bearer, path="/mcp", extra_headers=()):
        return {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"host", b"testserver"), (b"authorization", f"Bearer {bearer}".encode()), *extra_headers],
        }

    @staticmethod
    @contextlib.contextmanager
    def _patch_user_reload(*, user_id, active=True, organization_id=None, tpm_limit=None, rpm_limit=None):
        get_user_object = AsyncMock(
            return_value=MagicMock(
                user_id=user_id,
                organization_id=organization_id,
                metadata={"scim_active": active} if not active else {"scim_active": True},
                user_role=None,
                object_permission=None,
                object_permission_id=None,
                tpm_limit=tpm_limit,
                rpm_limit=rpm_limit,
            )
        )
        with (
            patch("litellm.proxy.auth.auth_checks.get_user_object", get_user_object),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        ):
            yield get_user_object

    async def test_session_admission_binds_org_id_so_the_org_ceiling_applies(self):
        """The admitted auth carries the user's org_id, so get_allowed_mcp_servers keeps the
        org-level MCP ceiling in force for a gateway session instead of skipping it."""
        token = self._access_token(user_id="org-user")
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ),
            self._patch_user_reload(user_id="org-user", organization_id="org-123"),
        ):
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(self._scope(token))
        assert auth_result.org_id == "org-123"

    async def test_session_admission_copies_user_rate_limits(self):
        """Security regression: the reconstructed auth must carry the live user's RPM/TPM, exactly as
        the standard user-subject path does. The parallel limiter reads these off the auth object and
        treats None as unlimited, so a keyless subject with them unset would invoke tools past their
        configured user rate limits."""
        token = self._access_token(user_id="rl-user")
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ),
            self._patch_user_reload(user_id="rl-user", tpm_limit=1000, rpm_limit=50),
        ):
            auth_result, *_rest = await MCPRequestHandler.process_mcp_request(self._scope(token))
        assert auth_result.user_tpm_limit == 1000
        assert auth_result.user_rpm_limit == 50

    async def test_valid_session_admits_under_live_user_at_aggregate_scope(self):
        token = self._access_token(user_id="sso-user-42")
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            self._patch_user_reload(user_id="sso-user-42") as get_user_object,
        ):
            auth_result, _h, _servers, mcp_server_auth_headers, _o, _r = await MCPRequestHandler.process_mcp_request(
                self._scope(token)
            )
        assert get_user_object.await_args.kwargs["user_id"] == "sso-user-42"
        assert auth_result.user_id == "sso-user-42"
        mock_auth.assert_not_called()
        # Identity-only admission injects no per-server upstream credential (unlike the
        # bridge envelope arm); the headers dict is whatever the request carried, here empty.
        assert not mcp_server_auth_headers

    @pytest.mark.parametrize(
        "scenario, expect_challenge",
        [("expired", True), ("tampered", False), ("refresh_at_tool_edge", False), ("foreign_key", False)],
    )
    async def test_bad_session_bearer_fails_closed(self, scenario, expect_challenge):
        # Every non-admissible session-shaped bearer fails closed with 401; a valid-but-unusable one
        # (expired) additionally carries the invalid_token challenge so the DCR client re-authorizes.
        from datetime import datetime, timezone

        if scenario == "expired":
            mint, _refresh, principal, keys = self._session_bearer()
            bearer = mint(principal, keys, datetime(2020, 1, 1, tzinfo=timezone.utc)).token.get_secret_value()
        elif scenario == "tampered":
            token = self._access_token()
            bearer = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
        elif scenario == "refresh_at_tool_edge":
            _mint, refresh, principal, keys = self._session_bearer()
            bearer = refresh(principal, keys, datetime(2030, 1, 1, tzinfo=timezone.utc)).token.get_secret_value()
        else:  # foreign_key: minted under the real master key, presented while the proxy uses another
            bearer = self._access_token()
        master_key = "sk-a-totally-different-master-key" if scenario == "foreign_key" else self._MASTER_KEY
        with patch("litellm.proxy.proxy_server.master_key", master_key):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(self._scope(bearer))
        assert exc_info.value.status_code == 401
        if expect_challenge:
            assert 'error="invalid_token"' in (exc_info.value.headers or {})["WWW-Authenticate"]

    async def test_deactivated_user_fails_with_invalid_token_challenge(self):
        """A cryptographically valid bearer whose referenced user is SCIM-deactivated must fail with
        the aggregate invalid_token challenge (WWW-Authenticate), matching the expired/tampered arms,
        so the DCR client re-authorizes instead of getting a bare 401 with no challenge."""
        token = self._access_token(user_id="offboarded-user")
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ),
            self._patch_user_reload(user_id="offboarded-user", active=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await MCPRequestHandler.process_mcp_request(self._scope(token))
        assert exc_info.value.status_code == 401
        assert 'error="invalid_token"' in (exc_info.value.headers or {})["WWW-Authenticate"]

    async def test_session_bearer_scrubbed_from_egress_header_contexts(self):
        """Security regression (credential leak): after a keyless session admission, the session
        bearer must be removed from BOTH returned egress header contexts (oauth2_headers and the raw
        headers) so no passthrough/OBO egress can forward it upstream for replay as this user."""
        token = self._access_token(user_id="sso-user-42")
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ),
            self._patch_user_reload(user_id="sso-user-42"),
        ):
            _auth, _h, _servers, _msah, oauth2_headers, raw_headers = await MCPRequestHandler.process_mcp_request(
                self._scope(token)
            )
        # the request carried "Authorization: Bearer <session>"; both egress contexts must be scrubbed
        assert oauth2_headers is None
        assert not any(k.lower() == "authorization" for k in (raw_headers or {}))

    async def test_arm_does_not_fire_for_named_server(self):
        """A session-shaped bearer aimed at a named server (path scope) does not enter the
        aggregate arm; it is treated as an ordinary bearer on that server."""
        token = self._access_token()
        with (
            patch("litellm.proxy.proxy_server.master_key", self._MASTER_KEY),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
                side_effect=ProxyException(message="bad key", type="auth_error", param="api_key", code=401),
            ) as mock_auth,
        ):
            with pytest.raises((HTTPException, ProxyException)):
                await MCPRequestHandler.process_mcp_request(self._scope(token, path="/mcp/github"))
        mock_auth.assert_called_once()


def _make_team(team_id, mcp_servers, *, org_id=None, tool_perms=None, members=("sso-user",)):
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable, Member

    return LiteLLM_TeamTable(
        team_id=team_id,
        organization_id=org_id,
        members_with_roles=[Member(user_id=u, role="user") for u in members],
        access_group_ids=[],
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id=f"op-{team_id}", mcp_servers=mcp_servers, mcp_tool_permissions=tool_perms
        ),
    )


def _make_admitted_subject(user_id, *, org_id=None, own_servers=None, own_tool_perms=None):
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable

    op = None
    if own_servers is not None or own_tool_perms is not None:
        op = LiteLLM_ObjectPermissionTable(
            object_permission_id=f"userop-{user_id}",
            mcp_servers=own_servers or [],
            mcp_tool_permissions=own_tool_perms,
        )
    auth = UserAPIKeyAuth(user_id=user_id, api_key=None, org_id=org_id, object_permission=op)
    auth.mcp_admitted_user_subject = True
    return auth


@pytest.mark.asyncio
class TestUserSubjectTeamUnion:
    """_get_allowed_mcp_servers_for_team unions across ALL a user's teams for a keyless
    user-subject caller (the gateway DCR session bearer and bridge user-envelope), while a
    key-based caller keeps its single-team behavior byte-identically."""

    @contextlib.contextmanager
    def _patch(self, *, teams_by_id, user_teams=None, orgs_by_id=None):
        async def _get_team_object(team_id, **kw):
            return teams_by_id.get(team_id)

        async def _get_user_object(user_id, **kw):
            return MagicMock(user_id=user_id, teams=user_teams or [])

        async def _get_org_object(org_id, **kw):
            return (orgs_by_id or {}).get(org_id)

        async def _spend_from_fallback(counter_key, fallback_spend, max_budget=None, **kw):
            # The budget owners read cross-pod spend Redis-first with the row's spend as fallback;
            # unit tests have no Redis, so the fallback IS the spend.
            return fallback_spend

        with (
            patch("litellm.proxy.auth.auth_checks.get_team_object", _get_team_object),
            patch("litellm.proxy.auth.auth_checks.get_user_object", _get_user_object),
            patch("litellm.proxy.auth.auth_checks.get_org_object", _get_org_object),
            patch("litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups", AsyncMock(return_value=[])),
            patch("litellm.proxy.proxy_server.get_current_spend", _spend_from_fallback),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        ):
            yield

    async def test_keyless_user_unions_servers_across_all_their_teams(self):
        teams = {"team-a": _make_team("team-a", ["srv1", "srv2"]), "team-b": _make_team("team-b", ["srv2", "srv3"])}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"]):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv1", "srv2", "srv3"}

    async def test_key_based_caller_uses_single_team_only(self):
        """A key-based caller (api_key set) with a team_id sees ONLY that team, even though the
        same user belongs to other teams: key auth must be byte-identical to before."""
        teams = {"team-a": _make_team("team-a", ["srv1"]), "team-b": _make_team("team-b", ["srv2", "srv3"])}
        auth = UserAPIKeyAuth(user_id="sso-user", api_key="sk-hash", team_id="team-a")
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"]):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
        assert set(result) == {"srv1"}

    async def test_keyless_user_with_explicit_team_id_uses_that_team_only(self):
        """A keyless caller that already pins a team_id (not the user-subject fan-out shape)
        resolves only that team; the union is strictly for the no-team-id user-subject case."""
        teams = {"team-a": _make_team("team-a", ["srv1"]), "team-b": _make_team("team-b", ["srv2"])}
        auth = UserAPIKeyAuth(user_id="sso-user", api_key=None, team_id="team-a")
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"]):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
        assert set(result) == {"srv1"}

    async def test_keyless_user_with_no_teams_gets_nothing_from_teams(self):
        auth = _make_admitted_subject("lonely-user")
        with self._patch(teams_by_id={}, user_teams=[]):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
        assert result == []

    async def test_ui_session_team_id_still_resolves_to_nothing(self):
        from litellm.proxy._types import UI_TEAM_ID

        auth = UserAPIKeyAuth(user_id="dash-user", api_key="sk-hash", team_id=UI_TEAM_ID)
        with self._patch(teams_by_id={}, user_teams=["team-a"]):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
        assert result == []

    async def test_team_ids_helper_gates_on_shape(self):
        from litellm.proxy._types import UI_TEAM_ID

        # key-based with team -> that team
        assert await MCPRequestHandler._team_ids_for_mcp_grant(
            UserAPIKeyAuth(api_key="sk", team_id="t1", user_id="u")
        ) == ["t1"]
        # An admitted subject never fans out HERE: it resolves one source per team first, and each of
        # those pins a team_id, so this helper only ever answers the single-team question. The fan-out
        # itself is _admitted_subject_sources' job, asserted below.
        with self._patch(teams_by_id={}, user_teams=["t2", "t3"]):
            assert await MCPRequestHandler._team_ids_for_mcp_grant(_make_admitted_subject("u")) == []
        # keyless, no user_id -> nothing
        assert await MCPRequestHandler._team_ids_for_mcp_grant(UserAPIKeyAuth(api_key=None)) == []
        # keyless with a user_id but NOT admission-marked (JWT auth) -> nothing (unchanged behavior)
        with self._patch(teams_by_id={}, user_teams=["t2", "t3"]):
            assert (
                await MCPRequestHandler._team_ids_for_mcp_grant(UserAPIKeyAuth(api_key=None, user_id="jwt-user")) == []
            )
        # UI sentinel -> nothing
        assert (
            await MCPRequestHandler._team_ids_for_mcp_grant(
                UserAPIKeyAuth(api_key="sk", team_id=UI_TEAM_ID, user_id="u")
            )
            == []
        )

    async def test_org_outage_is_not_treated_as_a_missing_org(self):
        """A CONFIRMED-absent org places no ceiling; a FAILED lookup must not be read as the same
        fact. get_org_object used to relabel every error as "doesn't exist", so a DB outage silently
        dropped a real org's ceiling for as long as it lasted. Absent -> the team's grant stands;
        outage -> the keyless source denies."""
        from litellm.proxy.auth.auth_checks import OrganizationNotFoundError

        teams = {"t1": _make_team("t1", ["srv1"])}
        teams["t1"].organization_id = "org-a"
        auth = _make_admitted_subject("sso-user")

        absent = AsyncMock(side_effect=OrganizationNotFoundError("Organization doesn't exist in db."))
        with self._patch(teams_by_id=teams, user_teams=["t1"]):
            with patch("litellm.proxy.auth.auth_checks.get_org_object", absent):
                reachable = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(reachable) == {"srv1"}, "a deleted org places no ceiling"

        outage = AsyncMock(side_effect=RuntimeError("connection reset by peer"))
        with self._patch(teams_by_id=teams, user_teams=["t1"]):
            with patch("litellm.proxy.auth.auth_checks.get_org_object", outage):
                reachable = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert reachable == [], "an unresolvable ceiling must deny a keyless source, not be skipped"

    async def test_org_ceiling_fault_fails_closed_for_admitted_but_open_for_keys(self):
        """An unresolvable org ceiling is NOT the same fact as "this org places no restriction".

        For a virtual key the ceiling is one of several bounds and a DB blip must not lock working
        keys out, so it stays fail-open. For a keyless admitted subject the per-source org ceiling is
        the ONLY org bound, so dropping it on a fault would widen a cross-org user to servers their
        team's org forbids. That is escalation, not an availability blip, so it fails closed."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        boom = AsyncMock(side_effect=RuntimeError("org lookup exploded"))
        # The subject must actually REACH something, or the assertion passes either way and pins
        # nothing (a fail-open mutant survived an earlier version of this test for exactly that).
        auth = _make_admitted_subject("sso-user")
        auth.org_id = "org-a"
        auth.object_permission = LiteLLM_ObjectPermissionTable(object_permission_id="op-u", mcp_servers=["srv1"])
        with self._patch(teams_by_id={}, user_teams=[]):
            assert set(await MCPRequestHandler.get_allowed_mcp_servers(auth)) == {"srv1"}  # control
            with patch.object(MCPRequestHandler, "_get_org_object_permission", boom):
                admitted = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert admitted == [], "admitted subject must fail CLOSED when its org ceiling cannot resolve"

        key_auth = UserAPIKeyAuth(user_id="u", api_key="sk-hash", team_id="t1", org_id="org-a")
        with self._patch(teams_by_id={"t1": _make_team("t1", ["srv1"])}, user_teams=[]):
            with patch.object(MCPRequestHandler, "_get_org_object_permission", boom):
                keyed = await MCPRequestHandler.get_allowed_mcp_servers(key_auth)
        assert set(keyed) == {"srv1"}, "key auth must keep its long-standing fail-open behavior"

    async def test_only_the_attributing_team_bucket_is_charged(self):
        """A team's mcp_rpm_limit bounds that team's SHARED bucket. Charging every granting team let
        one cross-team user drain several teams' buckets on a single call, blocking their other
        members for access those teams did not provide. Exactly one source is charged, and it is the
        SAME source billing picks — one owner for both, so they cannot disagree."""
        from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3

        t1 = _make_team("t1", ["srv1"])
        t1.metadata = {"mcp_rpm_limit": {"srv1": 5}}
        t2 = _make_team("t2", ["srv1"])
        t2.metadata = {"mcp_rpm_limit": {"srv1": 9}}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id={"t1": t1, "t2": t2}, user_teams=["t1", "t2"]):
            auth.mcp_source_team_rpm_limits = await MCPRequestHandler._admitted_subject_team_rpm_limits(auth)
            billed = await MCPRequestHandler.attributing_source_for_server(auth, "srv1")
        assert auth.mcp_source_team_rpm_limits == {"t1": {"srv1": 5}}, "t2's shared bucket is untouched"
        assert billed is not None and billed.team_id == "t1", "throttling and billing pick the same source"

        descriptors: list = []
        limiter = _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=MagicMock())
        limiter._add_mcp_per_team_rate_limit_descriptor(auth, "srv1", descriptors)
        charged = {d["value"]: d["rate_limit"]["requests_per_unit"] for d in descriptors}
        assert charged == {"t1:srv1": 5}, "only the attributing team's bucket is charged"

    async def test_direct_user_grant_charges_no_team_bucket(self):
        """When the user's OWN grant reaches the server, no team provided the access, so no team
        bucket may be charged — the user's own rpm/tpm is what bounds them. Mirrors billing, which
        bills the user and their own org for exactly this case."""
        t1 = _make_team("t1", ["srv1"])
        t1.metadata = {"mcp_rpm_limit": {"srv1": 5}}
        auth = _make_admitted_subject("sso-user")
        auth.object_permission = LiteLLM_ObjectPermissionTable(object_permission_id="op-u", mcp_servers=["srv1"])
        with self._patch(teams_by_id={"t1": t1}, user_teams=["t1"]):
            limits = await MCPRequestHandler._admitted_subject_team_rpm_limits(auth)
            billed = await MCPRequestHandler.attributing_source_for_server(auth, "srv1")
        assert limits is None, "a direct user grant must not charge any team's shared bucket"
        assert billed is None, "and billing agrees: the user is billed, not a team"

    def _manager_with(self, server_ids, allow_all=()):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.types.mcp import MCPTransport

        manager = MCPServerManager()
        for sid in server_ids:
            manager.registry[sid] = MCPServer(
                server_id=sid,
                name=sid,
                server_name=sid,
                url="https://example.com/mcp",
                transport=MCPTransport.http,
                allow_all_keys=sid in allow_all,
            )
        manager._get_active_submitted_mcp_server_ids_for_user = AsyncMock(return_value=[])
        return manager

    async def test_team_derived_call_bills_the_granting_team_and_its_org(self):
        """ACCOUNTING half of team budgets. Without attribution the admitted auth kept team_id=None,
        so spend skipped team updates (the team's budget never accumulated, so it could never begin
        to block) and charged the user's PRIMARY org rather than the org owning the granting team."""
        t_grant = _make_team("t-grant", ["srv1"])
        t_grant.organization_id = "org-team"
        auth = _make_admitted_subject("sso-user")
        auth.org_id = "org-user-primary"
        with self._patch(teams_by_id={"t-grant": t_grant}, user_teams=["t-grant"]):
            source = await MCPRequestHandler.attributing_source_for_server(auth, "srv1")
        assert source is not None and source.team_id == "t-grant"
        assert source.org_id == "org-team", "the granting team's org is charged, not the user's primary"
        assert auth.team_id is None and auth.org_id == "org-user-primary", "authz object untouched"

    async def test_billing_auth_carries_team_and_org_onto_the_spend_object(self):
        """Asserted on billing_auth_for_tool_call itself, not on the source it picks: the source
        already carries the team's org by construction, so asserting there leaves the copy step
        unpinned (a mutant dropping org_id survived exactly that). This is the object spend reads."""
        t_grant = _make_team("t-grant", ["srv1"])
        t_grant.organization_id = "org-team"
        auth = _make_admitted_subject("sso-user")
        auth.org_id = "org-user-primary"
        server = MagicMock(server_id="srv1")
        with self._patch(teams_by_id={"t-grant": t_grant}, user_teams=["t-grant"]):
            with patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager._get_mcp_server_from_tool_name",
                MagicMock(return_value=server),
            ):
                billed = await MCPRequestHandler.billing_auth_for_tool_call(auth, tool_name="t-grant/tool_a")
        assert (billed.team_id, billed.org_id) == ("t-grant", "org-team")
        assert (auth.team_id, auth.org_id) == (None, "org-user-primary"), "authz object must be untouched"

    async def test_own_grant_bills_the_user_not_a_team(self):
        """A server the user's OWN grant reaches is not reached "through a team", so it bills the
        user and their own org — attributing it to an unrelated team the user happens to belong to
        would charge that team for access it never provided."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        t_other = _make_team("t-other", ["srv1"])
        auth = _make_admitted_subject("sso-user")
        auth.object_permission = LiteLLM_ObjectPermissionTable(object_permission_id="op-u", mcp_servers=["srv1"])
        with self._patch(teams_by_id={"t-other": t_other}, user_teams=["t-other"]):
            assert await MCPRequestHandler.attributing_source_for_server(auth, "srv1") is None

    async def test_billing_attribution_is_deterministic_across_several_granting_teams(self):
        """When several teams grant the same server the pick must be stable and reproducible rather
        than dependent on dict/roster ordering, or the same call bills different teams run to run."""
        teams = {"t-b": _make_team("t-b", ["srv1"]), "t-a": _make_team("t-a", ["srv1"])}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["t-b", "t-a"]):
            first = await MCPRequestHandler.attributing_source_for_server(auth, "srv1")
        with self._patch(teams_by_id=teams, user_teams=["t-a", "t-b"]):
            second = await MCPRequestHandler.attributing_source_for_server(auth, "srv1")
        assert first is not None and first.team_id == "t-a"
        assert second is not None and second.team_id == "t-a", "roster order must not change who is billed"

    async def test_billing_auth_leaves_non_admitted_callers_untouched(self):
        """Key and JWT billing must be byte-identical: the attribution wrapper returns the very same
        object for anything that is not a keyless admitted subject."""
        key_auth = UserAPIKeyAuth(user_id="u", api_key="sk-hash", team_id="t1", org_id="org-a")
        assert await MCPRequestHandler.billing_auth_for_tool_call(key_auth, tool_name="srv1-tool") is key_auth

    async def test_admitted_tools_never_run_the_single_credential_prelude(self):
        """ORDERING is the invariant: the admitted branch is the FIRST statement of the tools
        resolver, exactly as in the servers resolver. A fault in a lookup the subject never uses
        (its own mcp_toolsets) must not reach it at all — when this branch sat after the prelude,
        such a fault hit the fail-closed handler and denied tools its teams did grant."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        teams = {"t1": _make_team("t1", ["srv1"], tool_perms={"srv1": ["read"]})}
        auth = _make_admitted_subject("sso-user")
        # The subject must carry a toolset, or the prelude never resolves one and the fault below is
        # unreachable — the branch could sit anywhere and the test would still pass (it did).
        auth.object_permission = LiteLLM_ObjectPermissionTable(object_permission_id="op-u", mcp_toolsets=["ts-1"])
        boom = AsyncMock(side_effect=RuntimeError("toolset resolution exploded"))
        with self._patch(teams_by_id=teams, user_teams=["t1"]):
            with patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.resolve_toolset_tool_permissions",
                boom,
            ):
                tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        # The fault DOES fire, correctly, inside the subject's own source (which carries its
        # toolsets) — that source contributes nothing. What must not happen is the top-level
        # prelude running it first and denying the team's grant through the fail-closed handler.
        assert tools == ["read"], "a fault in the subject's own toolsets must not deny its team's tools"

    async def test_admitted_own_byom_servers_stay_open(self):
        """BYOM suppression-by-explicit-scope is a rule about a CREDENTIAL carrying its own
        mcp_servers list. An admitted subject's object_permission is the user's own row, whose
        mcp_servers column is [] by DB default — applying the rule would hide almost every admitted
        user's OWN submitted servers. A key with an explicit scope still gets no BYOM widening."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        manager = self._manager_with(["srv-byom"])
        manager._get_active_submitted_mcp_server_ids_for_user = AsyncMock(return_value=["srv-byom"])
        db_default_perm = LiteLLM_ObjectPermissionTable(object_permission_id="op-u", mcp_servers=[])

        admitted = _make_admitted_subject("sso-user")
        admitted.object_permission = db_default_perm
        scoped_key = UserAPIKeyAuth(user_id="u", api_key="sk-hash", object_permission=db_default_perm)

        assert await manager.operator_open_server_ids(admitted) == {"srv-byom"}
        assert await manager.operator_open_server_ids(scoped_key) == set(), "explicit key scope still suppresses BYOM"

    async def test_admitted_admin_is_scoped_to_grants_not_full_registry(self):
        """The wrapper's admin short-circuit hands the FULL registry to any admin-role auth before
        the grant union or the per-team org ceilings run. A session bearer is a third-party client
        credential, not the dashboard: an admin signing in through the connect flow gets their
        grants like anyone else. A real admin key keeps the dashboard behavior unchanged."""
        from litellm.proxy._types import LitellmUserRoles

        manager = self._manager_with(["srv-granted", "srv-secret"])
        admitted = _make_admitted_subject("admin-user")
        admitted.user_role = LitellmUserRoles.PROXY_ADMIN
        with patch.object(MCPRequestHandler, "get_allowed_mcp_servers", AsyncMock(return_value=["srv-granted"])):
            admitted_view = set(await manager.get_allowed_mcp_servers(admitted))
            key_admin_view = set(
                await manager.get_allowed_mcp_servers(
                    UserAPIKeyAuth(user_id="admin-user", api_key="sk-hash", user_role=LitellmUserRoles.PROXY_ADMIN)
                )
            )
        assert admitted_view == {"srv-granted"}, "an admitted admin gets their grants, not the registry"
        assert key_admin_view == {"srv-granted", "srv-secret"}, "admin KEY behavior must be unchanged"

    async def test_admitted_opt_out_via_wrapper_keeps_team_servers(self):
        """The wrapper's no_mcp_servers early-return is a KEY rule (a scoped credential's opt-out is
        absolute). The admitted subject's opt-out silences only its own source, which the resolver
        enforces per source — the wrapper must defer to it, or the resolver-level rule is dead code
        on the production path."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, SpecialMCPServerNames

        manager = self._manager_with(["srv-team"])
        opt_out = LiteLLM_ObjectPermissionTable(
            object_permission_id="op-u", mcp_servers=[SpecialMCPServerNames.no_mcp_servers.value]
        )
        admitted = _make_admitted_subject("sso-user")
        admitted.object_permission = opt_out
        with patch.object(MCPRequestHandler, "get_allowed_mcp_servers", AsyncMock(return_value=["srv-team"])):
            admitted_view = set(await manager.get_allowed_mcp_servers(admitted))
            key_view = await manager.get_allowed_mcp_servers(
                UserAPIKeyAuth(user_id="u", api_key="sk-hash", object_permission=opt_out)
            )
        assert "srv-team" in admitted_view, "user opt-out must not zero team grants on the wrapper path"
        assert key_view == [], "a key's opt-out stays absolute"

    async def test_open_channel_confers_reachability_not_a_ceiling_waiver(self):
        """An open channel (allow_all_keys / own BYOM) makes a server REACHABLE. It is not a waiver
        of the ceilings that bound it: the user's own mcp_tool_permissions still apply, exactly as a
        virtual key's key_tools do on the same allow_all server. Returning None outright let a
        session holder invoke tools their own policy excludes."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        auth = _make_admitted_subject("sso-user")
        # The user is restricted to `read` on srv-open, and NO grant source names that server —
        # it is reachable only through the open channel, which is exactly the bypass path.
        auth.object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="op-u", mcp_servers=[], mcp_tool_permissions={"srv-open": ["read"]}
        )
        open_ids = AsyncMock(return_value={"srv-open"})
        with self._patch(teams_by_id={}, user_teams=[]):
            with patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.operator_open_server_ids",
                open_ids,
            ):
                tools = await MCPRequestHandler.get_allowed_tools_for_server("srv-open", auth)
        assert tools == ["read"], "the user's own tool policy must still bind on an open-channel server"

    async def test_open_channel_server_gets_default_open_tools_for_admitted(self):
        """A server reachable through an open channel (allow_all_keys / own BYOM) is granted by NO
        source, so the source union alone returns [] — listable but uninvokable. The tools axis asks
        the same open-channel owner the server union uses, so the server is default-open for tools
        exactly as a virtual key experiences it."""
        auth = _make_admitted_subject("sso-user")
        open_ids = AsyncMock(return_value={"srv-open"})
        with self._patch(teams_by_id={}, user_teams=[]):
            with patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.operator_open_server_ids",
                open_ids,
            ):
                open_tools = await MCPRequestHandler.get_allowed_tools_for_server("srv-open", auth)
                closed_tools = await MCPRequestHandler.get_allowed_tools_for_server("srv-ungranted", auth)
        assert open_tools is None, "open-channel server must be default-open for tools"
        assert closed_tools == [], "a server no source or channel grants stays deny-all"

    async def test_over_budget_team_grants_nothing_and_healthy_team_stands(self):
        """Budget ENFORCEMENT is the sibling of blocked: a team that has already exceeded its
        max_budget is rejected outright for a virtual key pinned to it (common_checks), so it must
        not keep granting servers, tools or throttle scope to a keyless union subject either.
        Enforced through the SAME owner the key path uses (_team_max_budget_check). Distinct from
        budget ATTRIBUTION of new spend, which stays with the user (documented deferral)."""
        t_over = _make_team("t-over", ["srv1"])
        t_over.max_budget = 10.0
        t_over.spend = 11.0
        t_ok = _make_team("t-ok", ["srv2"])
        t_ok.max_budget = 10.0
        t_ok.spend = 1.0
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id={"t-over": t_over, "t-ok": t_ok}, user_teams=["t-over", "t-ok"]):
            servers = set(await MCPRequestHandler.get_allowed_mcp_servers(auth))
            limits = await MCPRequestHandler._admitted_subject_team_rpm_limits(auth)
        assert servers == {"srv2"}, "an over-budget team must stop granting; the healthy team stands"
        assert limits is None, "an over-budget team is not a source, so it stamps no throttle either"

    async def test_team_in_over_budget_org_grants_nothing(self):
        """The org axis of the same rule, judged against the TEAM's own org (not the caller's
        primary): a team owned by an org over its budget grants nothing, exactly as a key in that
        org is rejected by _organization_max_budget_check."""
        t_in_broke_org = _make_team("t-b", ["srv1"])
        t_in_broke_org.organization_id = "org-broke"
        # object_permission_id=None: the org has NO MCP ceiling, so the source is denied by the
        # budget gate alone. A truthy auto-Mock id here made an earlier version of this test pass
        # through the org-CEILING fault path with the budget gate deleted — vacuous.
        org = MagicMock(object_permission_id=None, litellm_budget_table=MagicMock(max_budget=5.0), spend=9.0)
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id={"t-b": t_in_broke_org}, user_teams=["t-b"], orgs_by_id={"org-broke": org}):
            servers = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert servers == [], "a team in an over-budget org must not grant through the union"

    async def test_one_faulting_team_does_not_deny_the_other_sources(self):
        """The unit of fault isolation is the SOURCE. One team's row being momentarily unreadable
        contributes nothing for THAT team (access only narrows) while the user's own grants and every
        other resolvable team stand — it must not collapse the whole union to deny-all on either
        axis."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        t_ok = _make_team("t-ok", ["srv1"], tool_perms={"srv1": ["read"]})
        auth = _make_admitted_subject("sso-user")
        auth.object_permission = LiteLLM_ObjectPermissionTable(object_permission_id="op-u", mcp_servers=["srv-own"])
        teams = {"t-ok": t_ok}  # t-boom absent from the map -> our patched get_team_object RAISES for it

        async def _team_or_boom(team_id, **kw):
            if team_id not in teams:
                raise RuntimeError(f"transient DB blip loading {team_id}")
            return teams[team_id]

        with self._patch(teams_by_id=teams, user_teams=["t-boom", "t-ok"]):
            with patch("litellm.proxy.auth.auth_checks.get_team_object", _team_or_boom):
                servers = set(await MCPRequestHandler.get_allowed_mcp_servers(auth))
                tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        assert servers == {"srv-own", "srv1"}, "healthy sources must stand when one team faults"
        assert tools == ["read"], "the healthy team's tool grant must survive the other team's fault"

    async def test_key_org_tool_ceiling_fault_keeps_key_restrictions(self):
        """Virtual-key tools axis mirrors its servers axis on an unresolvable org ceiling: the org
        intersect is SKIPPED and the key's own tool restrictions stand. Letting the fault escape
        collapsed the whole resolution to None (allow-all), which is fail-open WIDER than before the
        fault — key restrictions must never be dropped by an org lookup blip."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        key_auth = UserAPIKeyAuth(user_id="u", api_key="sk-hash", org_id="org-a")
        key_auth.object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="op-k", mcp_servers=["srv1"], mcp_tool_permissions={"srv1": ["read"]}
        )
        boom = AsyncMock(side_effect=RuntimeError("org permission load exploded"))
        with self._patch(teams_by_id={}, user_teams=[]):
            with patch.object(MCPRequestHandler, "_get_org_object_permission", boom):
                tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", key_auth)
        assert tools == ["read"], "key tool restrictions must survive an unresolvable org ceiling"

    async def test_team_rpm_limit_binds_only_within_that_teams_grant_scope(self):
        """A limit rides the same scope as the access it bounds. A roster team is charged ONLY for
        servers its own grant reaches: not for a server the user reaches through a DIFFERENT team
        (else this user's calls drain a bucket shared by that team's keys for access the team never
        provided), not for map entries beyond its grant, and never when the team is blocked."""
        # t-granting grants srv1 and limits it; also names srv9 in its map, which it does NOT grant.
        t_granting = _make_team("t-granting", ["srv1"])
        t_granting.metadata = {"mcp_rpm_limit": {"srv1": 5, "srv9": 7}}
        # t-other grants only srv2 but retains limit metadata for srv1 -> must not be charged for it.
        t_other = _make_team("t-other", ["srv2"])
        t_other.metadata = {"mcp_rpm_limit": {"srv1": 3}}
        # t-blocked grants srv1 and limits it, but is blocked -> grants nothing, charges nothing.
        t_blocked = _make_team("t-blocked", ["srv1"])
        t_blocked.metadata = {"mcp_rpm_limit": {"srv1": 2}}
        t_blocked.blocked = True

        auth = _make_admitted_subject("sso-user")
        teams = {"t-granting": t_granting, "t-other": t_other, "t-blocked": t_blocked}
        with self._patch(teams_by_id=teams, user_teams=["t-granting", "t-other", "t-blocked"]):
            limits = await MCPRequestHandler._admitted_subject_team_rpm_limits(auth)

        assert limits == {"t-granting": {"srv1": 5}}, (
            "only the granting team's bucket, and only for the server it grants"
        )

    async def test_non_roster_team_rpm_limit_does_not_apply(self):
        """The roster gates grants and throttles through one owner, so a team the user was removed
        from neither grants servers nor gets charged for their calls."""
        stale = _make_team("t-stale", ["srv1"], members=("someone-else",))
        stale.metadata = {"mcp_rpm_limit": {"srv1": 1}}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id={"t-stale": stale}, user_teams=["t-stale"]):
            limits = await MCPRequestHandler._admitted_subject_team_rpm_limits(auth)
        assert limits is None

    async def test_org_list_caps_a_source_but_never_becomes_a_grant(self):
        """The admitted model is a union of GRANTS, so an org allowlist may only narrow what a source
        already grants. For a virtual key with no lower-level restriction the org list legitimately
        BECOMES the allowed set, and inheriting that arm would hand every admitted user with an
        org_id their whole org's server list with no direct or team grant behind it."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        auth = _make_admitted_subject("sso-user")
        auth.org_id = "org-a"  # org allows srv1+srv2; the user and their teams grant NOTHING
        org_perm = AsyncMock(
            return_value=LiteLLM_ObjectPermissionTable(object_permission_id="op-org-a", mcp_servers=["srv1", "srv2"])
        )
        with self._patch(teams_by_id={}, user_teams=[]):
            with patch.object(MCPRequestHandler, "_get_org_object_permission", org_perm):
                result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert result == [], "an org ceiling must not grant servers the user was never granted"

    async def test_tool_ceiling_fails_closed_when_a_SOURCE_faults(self):
        """Each source is resolved through an UNMARKED auth, so a fault under a source must still
        deny. Returning None there would win the union as allow-all and drop every team/org tool
        ceiling on a DB blip -- the marker alone only covers faults raised before the fan-out."""
        auth = _make_admitted_subject("sso-user")
        teams = {"t1": _make_team("t1", ["srv1"])}
        # Fault INSIDE the tool resolution only. Faulting something the server path also uses would
        # make the source grant nothing, so the union would return [] without the tool path ever
        # running -- the test would pass while pinning nothing (an earlier version did exactly that).
        boom = AsyncMock(side_effect=RuntimeError("org tool ceiling exploded"))
        with self._patch(teams_by_id=teams, user_teams=["t1"]):
            assert await MCPRequestHandler.get_allowed_mcp_servers(auth) == ["srv1"]  # control: granted
            with patch.object(MCPRequestHandler, "_apply_agent_and_org_tool_ceilings", boom):
                tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        assert tools == [], "a source-level fault must deny tools, never collapse to allow-all"

    async def test_own_opt_out_silences_only_that_source_not_the_teams(self):
        """no_mcp_servers on the USER's own grants opts that source out. It must not zero the teams:
        the sources are independent, so an opt-out on one silences one. (The same sentinel on a
        virtual KEY still overrides team inheritance -- that is the key ceiling model, unchanged.)"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, SpecialMCPServerNames

        auth = _make_admitted_subject("sso-user")
        auth.object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="op-user", mcp_servers=[SpecialMCPServerNames.no_mcp_servers.value]
        )
        with self._patch(teams_by_id={"t1": _make_team("t1", ["srv1"])}, user_teams=["t1"]):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv1"}, "the user's own opt-out must not zero their team's grants"

    async def test_sources_fan_out_per_team_and_drop_non_roster_teams(self):
        """The fan-out lives here now. One source per grant source: the user's own grants (no team_id,
        carrying their object_permission) plus each team they are a LIVE roster member of. A team that
        lingers in the user's cached `teams` array but no longer lists them in members_with_roles is
        dropped, which is what revokes access after a team_member_delete the user row hasn't caught up
        on. Each team source carries that team's own org, which is what makes the shared resolver apply
        the team's owning-org ceiling rather than the caller's home org."""
        teams = {
            "t-member": _make_team("t-member", ["srv1"], members=("sso-user",)),
            "t-stale": _make_team("t-stale", ["srv2"], members=("someone-else",)),
        }
        teams["t-member"].organization_id = "org-a"
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["t-member", "t-stale"]):
            sources = await MCPRequestHandler._admitted_subject_sources(auth)

        assert [(s.team_id, s.org_id) for s in sources] == [(None, None), ("t-member", "org-a")]
        # The user's own source carries their grants; a team source must NOT, or the team would be
        # widened by grants the team never made.
        assert sources[0].object_permission is auth.object_permission
        assert sources[1].object_permission is None
        # Every source is an ordinary caller, so it cannot re-enter the admitted fan-out.
        assert all(not s.mcp_admitted_user_subject for s in sources)
        # Nothing that meters or elevates the request may ride along onto a per-source clone.
        assert all(s.api_key is None and s.user_role is None for s in sources)

    async def test_jwt_keyless_user_without_team_claim_does_not_union(self):
        """Regression for the review finding: a JWT-authenticated caller is also keyless with a
        user_id and (with no team claim) no team_id, but it is NOT admission-marked, so it must
        keep its prior behavior of inheriting no team grants rather than silently gaining the
        union across every team the user belongs to."""
        teams = {"team-a": _make_team("team-a", ["srv1"]), "team-b": _make_team("team-b", ["srv2"])}
        jwt_auth = UserAPIKeyAuth(user_id="jwt-user", api_key=None)  # no admission marker
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"]):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(jwt_auth)
        assert result == []

    async def test_forged_metadata_marker_on_a_real_key_grants_no_union(self):
        """Security regression (forged admission marker): the admitted-subject marker is a
        server-only ``UserAPIKeyAuth`` field, NOT a metadata key, precisely because virtual-key
        metadata is caller-controlled at key creation. A user who sets
        ``mcp_admitted_user_subject: true`` in their own key's metadata (api_key present, no
        team_id) must NOT be treated as an admitted subject and must gain no cross-team union."""
        teams = {"team-a": _make_team("team-a", ["srv1"]), "team-b": _make_team("team-b", ["srv2"])}
        forged = UserAPIKeyAuth(
            user_id="attacker",
            api_key="sk-real-key",
            metadata={"mcp_admitted_user_subject": True},  # caller-forged marker in key metadata
        )
        assert _is_mcp_admitted_user_subject(forged) is False
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"]):
            assert await MCPRequestHandler._team_ids_for_mcp_grant(forged) == []
            assert await MCPRequestHandler._get_allowed_mcp_servers_for_team(forged) == []

    async def test_admitted_subject_team_tool_restriction_binds(self):
        """Security regression (team tool restrictions bypassed): a keyless admitted subject whose
        granting team restricts ``srv1`` to ``{tool_a}`` must NOT receive allow-all on srv1. The
        single-team-id tool lookup returns None (allow-all) for a keyless multi-team user, dropping
        the exclusion; the union across granting teams restores it."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable, Member

        team = LiteLLM_TeamTable(
            team_id="team-a",
            members_with_roles=[Member(user_id="sso-user", role="user")],
            access_group_ids=[],
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="op-team-a",
                mcp_servers=["srv1"],
                mcp_tool_permissions={"srv1": ["tool_a"]},
            ),
        )
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id={"team-a": team}, user_teams=["team-a"]):
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        assert tools == ["tool_a"]

    async def test_blocked_team_grants_no_servers_to_admitted_subject(self):
        """Security regression: a blocked team grants nothing. The central policy gate enforces this
        for a key pinned to a single team_id, but a keyless admitted subject unions across ALL its
        teams (no team_id), so a blocked team's MCP grants must be dropped at the per-team resolver."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable, Member

        blocked = LiteLLM_TeamTable(
            team_id="team-blocked",
            blocked=True,
            members_with_roles=[Member(user_id="sso-user", role="user")],
            access_group_ids=[],
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-blk", mcp_servers=["srv-secret"]),
        )
        teams = {"team-ok": _make_team("team-ok", ["srv-ok"]), "team-blocked": blocked}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-ok", "team-blocked"]):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv-ok"}

    async def test_admitted_subject_not_on_team_roster_gets_no_grant(self):
        """Security regression (membership containment): a keyless subject whose user_id is NOT on a
        team's roster inherits nothing from it, even when the team id lingers in the user's (stale or
        cached) teams array. The team roster is the source of truth, so a removed or foreign
        membership revokes access at the union rather than granting it."""
        teams = {"team-x": _make_team("team-x", ["srv-x"], members=("someone-else",))}
        auth = _make_admitted_subject("sso-user")  # in user.teams for team-x, but NOT on its roster
        with self._patch(teams_by_id=teams, user_teams=["team-x"]):
            result = await MCPRequestHandler._get_allowed_mcp_servers_for_team(auth)
        assert result == []

    async def test_tool_resolution_fails_closed_on_db_error(self):
        """Security regression: ANY error resolving the tool allowlist for a keyless admitted subject
        must DENY the server's tools ([]) rather than collapse to allow-all (None). Patches an await
        OUTSIDE the multi-team fan-out (the team-object lookup) to prove the whole function fails
        closed, not just the one helper — mirroring the fail-closed server path."""
        auth = _make_admitted_subject("sso-user")
        with patch.object(
            MCPRequestHandler,
            "_get_team_object_permission",
            new=AsyncMock(side_effect=RuntimeError("db blip")),
        ):
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        assert tools == []

    async def test_admission_marker_cannot_be_set_from_validated_input(self):
        """Defense-in-depth: the mcp_admitted_user_subject marker is server-only. Supplying it in any
        validated input (constructor kwargs OR model_validate, e.g. a future JWT/key claim splat) is
        stripped by the before-validator, so ONLY the admission path's post-construction assignment
        can set it."""
        via_kwarg = UserAPIKeyAuth(user_id="u", api_key=None, mcp_admitted_user_subject=True)
        via_validate = UserAPIKeyAuth.model_validate({"user_id": "u", "mcp_admitted_user_subject": True})
        assert via_kwarg.mcp_admitted_user_subject is False
        assert via_validate.mcp_admitted_user_subject is False
        assert _is_mcp_admitted_user_subject(via_kwarg) is False
        assert _is_mcp_admitted_user_subject(via_validate) is False


@pytest.mark.asyncio
class TestAdmittedSubjectPerTeamOrgCap:
    """A keyless admitted subject unions grants across teams that may span organizations. Each team's
    grant (servers AND tools) is capped by that team's OWN org, and the user's direct grants by the
    user's own org — never the caller's primary org applied over the whole cross-org union. Guards the
    Veria 'team grants bypass their owning policies' finding."""

    #: sentinel for org_perms: org has an object_permission_id but its load returns None (a swallowed
    #: DB error / dangling id), which _object_permission_for_org must treat as fail-closed.
    LOAD_FAILS = "__load_fails__"

    @contextlib.contextmanager
    def _patch(self, *, teams_by_id, user_teams, org_perms=None, registry=None):
        """org_perms: {org_id: LiteLLM_ObjectPermissionTable | None | LOAD_FAILS}.
        - table  → org exists, ceiling = that permission.
        - None   → org exists but carries no object_permission (no ceiling).
        - LOAD_FAILS → org exists with an object_permission_id, but the permission load returns None.
        - org_id ABSENT from the map → org row missing: get_org_object RAISES a bare Exception, exactly
          as production does (it does NOT return None or raise HTTPException)."""
        org_perms = org_perms or {}

        async def _get_team_object(team_id, **kw):
            return teams_by_id.get(team_id)

        async def _get_user_object(user_id, **kw):
            return MagicMock(user_id=user_id, teams=user_teams)

        async def _get_org_object(org_id, **kw):
            if org_id not in org_perms:
                from litellm.proxy.auth.auth_checks import OrganizationNotFoundError

                # matches production: a CONFIRMED-absent org raises this specific type, so callers
                # can tell it apart from an outage (a bare Exception now means "lookup failed").
                raise OrganizationNotFoundError(f"Organization doesn't exist. Org={org_id}.")
            op = org_perms[org_id]
            has_permission_id = op is not None  # a table OR LOAD_FAILS carries an id; None does not
            return MagicMock(
                organization_id=org_id,
                object_permission_id=(f"orgop-{org_id}" if has_permission_id else None),
                # Real typed values: the budget owners compare these, and a bare MagicMock attribute
                # would explode the comparison and silently drop the source (bare-Mock rule).
                litellm_budget_table=None,
                spend=0.0,
            )

        async def _get_object_permission(object_permission_id, **kw):
            for oid, op in org_perms.items():
                if op is not None and op != self.LOAD_FAILS and object_permission_id == f"orgop-{oid}":
                    return op
            return None  # LOAD_FAILS (or an unknown id) → None, simulating get_object_permission's swallow

        cms = [
            patch("litellm.proxy.auth.auth_checks.get_team_object", _get_team_object),
            patch("litellm.proxy.auth.auth_checks.get_user_object", _get_user_object),
            patch("litellm.proxy.auth.auth_checks.get_org_object", _get_org_object),
            patch("litellm.proxy.auth.auth_checks.get_object_permission", _get_object_permission),
            patch(
                "litellm.proxy.auth.auth_checks._get_mcp_server_ids_from_access_groups",
                AsyncMock(return_value=[]),
            ),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
            patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        ]
        if registry is not None:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            # registry may be a list of bare server_ids (MagicMock servers) OR a dict of
            # {server_id: server_obj} for tests that need real alias/name resolution (config servers).
            reg = registry if isinstance(registry, dict) else {s: MagicMock() for s in registry}
            cms.append(patch.object(global_mcp_server_manager, "get_registry", return_value=reg))
        with contextlib.ExitStack() as es:
            for cm in cms:
                es.enter_context(cm)
            yield

    # ---- server axis ----

    async def test_team_grant_capped_by_its_own_org(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        teams = {"team-a": _make_team("team-a", ["srv1", "srv2"], org_id="org-a")}
        org_perms = {"org-a": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-a", mcp_servers=["srv1"])}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms=org_perms):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv1"}  # srv2 capped out by org-a's ceiling

    async def test_cross_org_teams_each_capped_by_own_org(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        teams = {
            "team-a": _make_team("team-a", ["srv1", "srv2"], org_id="org-a"),
            "team-b": _make_team("team-b", ["srv3", "srv4"], org_id="org-b"),
        }
        org_perms = {
            "org-a": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-a", mcp_servers=["srv1"]),
            "org-b": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-b", mcp_servers=["srv3"]),
        }
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"], org_perms=org_perms):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv1", "srv3"}  # each team clipped by its OWN org, then unioned

    async def test_all_proxy_grant_capped_by_org(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, SpecialMCPServerName

        teams = {"team-a": _make_team("team-a", [SpecialMCPServerName.all_proxy_servers.value], org_id="org-a")}
        org_perms = {"org-a": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-a", mcp_servers=["srv1"])}
        auth = _make_admitted_subject("sso-user")
        with self._patch(
            teams_by_id=teams, user_teams=["team-a"], org_perms=org_perms, registry=["srv1", "srv2", "srv3"]
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        # all_proxy expands to the whole registry, then org-a caps to {srv1} — the cell the old partial
        # patch missed (it returned the full registry before capping).
        assert set(result) == {"srv1"}

    async def test_org_row_without_object_permission_does_not_cap(self):
        teams = {"team-a": _make_team("team-a", ["srv1", "srv2"], org_id="org-a")}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms={"org-a": None}):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv1", "srv2"}  # empty ceiling = no restriction

    async def test_direct_grants_unioned_with_team_and_capped_by_user_org(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        teams = {"team-a": _make_team("team-a", ["srv1"], org_id="org-a")}
        org_perms = {
            "org-a": None,  # the team's org imposes no ceiling
            "org-u": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-u", mcp_servers=["srvD", "srv1"]),
        }
        auth = _make_admitted_subject("sso-user", org_id="org-u", own_servers=["srvD", "srvX"])
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms=org_perms):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        # direct {srvD,srvX} ∩ user-org {srvD,srv1} = {srvD}; UNIONed with team {srv1} (not intersected).
        # srvX capped out by the user's org; team's srv1 NOT clipped by the user's primary org.
        assert set(result) == {"srvD", "srv1"}

    async def test_single_team_key_uses_primary_org_cap_not_per_team(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        # A KEY (not admitted): the per-team org cap must NOT fire; the top-level primary-org cap applies,
        # byte-identical to before. team-a (org-a) grants {srv1,srv2}; the key's primary org is org-k.
        teams = {"team-a": _make_team("team-a", ["srv1", "srv2"], org_id="org-a")}
        org_perms = {
            "org-a": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-a", mcp_servers=["srv2"]),
            "org-k": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-k", mcp_servers=["srv1"]),
        }
        key_auth = UserAPIKeyAuth(user_id="u", api_key="sk-hash", team_id="team-a", org_id="org-k")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms=org_perms):
            result = await MCPRequestHandler.get_allowed_mcp_servers(key_auth)
        # If the per-team (org-a) cap wrongly fired, team-a would clip to {srv2} then org-k → {} (empty).
        # Correct key behavior: no per-team cap; primary-org (org-k) cap → {srv1}.
        assert set(result) == {"srv1"}

    # ---- tool axis ----

    async def test_org_tool_ceiling_binds_when_team_places_no_tool_restriction(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        # team grants srv1 with NO tool restriction; org-a restricts srv1's tools to {tool_a}.
        teams = {"team-a": _make_team("team-a", ["srv1"], org_id="org-a")}
        org_perms = {
            "org-a": LiteLLM_ObjectPermissionTable(
                object_permission_id="orgop-org-a",
                mcp_servers=["srv1"],
                mcp_tool_permissions={"srv1": ["tool_a"]},
            )
        }
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms=org_perms):
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        # Without the per-team org tool ceiling this would be None (all tools) — org-a's tool ceiling
        # would be bypassed exactly like the server case.
        assert tools == ["tool_a"]

    async def test_tool_union_across_cross_org_teams(self):
        teams = {
            "team-a": _make_team("team-a", ["srv1"], org_id="org-a", tool_perms={"srv1": ["t1"]}),
            "team-b": _make_team("team-b", ["srv1"], org_id="org-b", tool_perms={"srv1": ["t2"]}),
        }
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a", "team-b"], org_perms={"org-a": None, "org-b": None}):
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        assert set(tools) == {"t1", "t2"}

    async def test_tool_deny_all_when_team_grant_and_org_tool_ceiling_disjoint(self):
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        teams = {"team-a": _make_team("team-a", ["srv1"], org_id="org-a", tool_perms={"srv1": ["t1"]})}
        org_perms = {
            "org-a": LiteLLM_ObjectPermissionTable(
                object_permission_id="orgop-org-a",
                mcp_servers=["srv1"],
                mcp_tool_permissions={"srv1": ["t2"]},
            )
        }
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms=org_perms):
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        # team {t1} ∩ org {t2} = {} → deny every tool ([]), NOT allow-all (None).
        assert tools == []

    # ---- error contract (adversarial-review findings) ----

    async def test_missing_org_row_is_treated_as_no_ceiling_not_lockout(self):
        """A team's organization_id may point to an org row that no longer exists (deleted / not yet
        synced). get_org_object RAISES a bare Exception for that; it must be treated as 'no ceiling' and
        must NOT lock the admitted subject out of the team's grants (parity with the key path, which
        tolerates a deleted org)."""
        teams = {"team-a": _make_team("team-a", ["srv1", "srv2"], org_id="org-gone")}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms={}):  # org-gone absent → raises
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert set(result) == {"srv1", "srv2"}

    async def test_org_permission_load_failure_fails_closed(self):
        """The org carries an object_permission_id but the permission load returns None (a swallowed DB
        error / dangling id). The ceiling cannot be verified, so the admitted subject must fail CLOSED
        for that team — NOT skip the ceiling, which would leak org-forbidden servers."""
        teams = {"team-a": _make_team("team-a", ["srv1", "srv2"], org_id="org-a")}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms={"org-a": self.LOAD_FAILS}):
            # Asserted through the PUBLIC resolver: the per-source org ceiling is applied there now,
            # so calling the single-team helper would return [] for an admitted subject either way
            # and pin nothing.
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        assert result == []  # fail closed, not {srv1, srv2}

    # ---- open bot-thread findings (2026-07-21 re-review) ----

    async def test_org_less_team_grant_capped_by_user_primary_org(self):
        """HIGH (cursor): a team with NO organization_id must still be bounded by the user's PRIMARY
        org — otherwise, since admitted subjects skip the top-level primary-org cap, an org-less team's
        grant would bypass every org ceiling and reach servers the user's home org forbids."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        teams = {"team-noorg": _make_team("team-noorg", ["srv1", "srv2"], org_id=None)}
        org_perms = {"org-U": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-U", mcp_servers=["srv1"])}
        auth = _make_admitted_subject("sso-user", org_id="org-U")
        with self._patch(teams_by_id=teams, user_teams=["team-noorg"], org_perms=org_perms):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        # org-less team falls back to the user's primary org (org-U → {srv1}); srv2 capped out.
        assert set(result) == {"srv1"}

    async def test_tool_empty_contributions_fails_closed(self):
        """MEDIUM (greptile/cursor): when no source in the tool-resolution view grants the server (a
        TOCTOU/cache-lag inconsistency on a server that passed the server gate), the admitted path must
        fail CLOSED (deny all tools = []), NOT allow-all (None)."""
        teams = {"team-a": _make_team("team-a", ["srv1"], org_id="org-a")}
        auth = _make_admitted_subject("sso-user")
        with self._patch(teams_by_id=teams, user_teams=["team-a"], org_perms={"org-a": None}):
            # 'srv-nobody' is granted by neither the team nor the user directly → empty contributions.
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv-nobody", auth)
        assert tools == []

    async def test_tool_no_db_honors_in_memory_direct_restriction(self):
        """MEDIUM (cursor): with no DB, the tool path must still honor the user's OWN in-memory
        object_permission tool restriction (resolvable without a DB) rather than blanket-allow (None)."""
        auth = _make_admitted_subject(
            "sso-user", own_servers=["srv1"], own_tool_perms={"srv1": ["t1"]}
        )  # no org_id, direct grant of srv1 restricted to {t1}
        with patch("litellm.proxy.proxy_server.prisma_client", None):
            tools = await MCPRequestHandler.get_allowed_tools_for_server("srv1", auth)
        assert tools == ["t1"]  # in-memory restriction honored, not widened to all tools

    # ---- config.yaml-defined servers (incl. OAuth) ----

    async def test_config_defined_oauth_server_by_alias_reached_and_org_capped(self):
        """A config.yaml-defined MCP OAuth server flows through the SAME resolution as a DB server:
        the team grant (and the org ceiling) reference it by ALIAS, expand_permission_list resolves it
        via the config+DB registry union to its server_id, and the per-team org cap applies identically.
        (The config server's OAuth *client* persistence is #33768 — an orthogonal egress concern; this
        pins the grant/reachability side of the 10x flow for config-defined servers.)"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        cfg_server = MagicMock()
        cfg_server.server_id = "cfg-oauth-1"
        cfg_server.alias = "linear_cfg"
        cfg_server.server_name = "linear_cfg"
        cfg_server.name = "linear_cfg"

        # team grants the config server BY ALIAS alongside a DB-style bare id; org-a's ceiling lists
        # ONLY the config server (also by alias).
        teams = {"team-a": _make_team("team-a", ["linear_cfg", "srv-db"], org_id="org-a")}
        org_perms = {
            "org-a": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-a", mcp_servers=["linear_cfg"])
        }
        auth = _make_admitted_subject("sso-user")
        with self._patch(
            teams_by_id=teams,
            user_teams=["team-a"],
            org_perms=org_perms,
            registry={"cfg-oauth-1": cfg_server},
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        # 'linear_cfg' alias resolves to the config server_id and survives org-a's ceiling; 'srv-db'
        # (not in org-a's allowlist) is capped out — same per-team org cap, config server included.
        assert set(result) == {"cfg-oauth-1"}

    async def test_config_oauth_server_alias_resolution_feeds_the_org_cap(self):
        """A config-defined OAuth server granted BY ALIAS whose OWN org forbids it is capped out — AND the
        cap is proven to run on RESOLVED server_ids, not raw strings. A control config server, granted by
        alias and allowed by the org via its RESOLVED id, must survive: that inclusion is impossible unless
        expand_permission_list resolved the grant alias to the id the ceiling lists, so a broken alias path
        yields {} and FAILS this test — whereas a bare `assert empty` would pass even if resolution never
        ran (the weakness Cursor flagged)."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        forbidden = MagicMock()  # granted by alias, but its org forbids it → must be capped out
        forbidden.server_id = "cfg-oauth-1"
        forbidden.alias = forbidden.server_name = forbidden.name = "linear_cfg"
        control = MagicMock()  # granted by alias, allowed by the org via its RESOLVED id → must survive
        control.server_id = "control-id"
        control.alias = control.server_name = control.name = "control_alias"

        teams = {"team-b": _make_team("team-b", ["linear_cfg", "control_alias"], org_id="org-b")}
        # org-b's ceiling allows ONLY the control server, referenced by its RESOLVED server_id.
        org_perms = {
            "org-b": LiteLLM_ObjectPermissionTable(object_permission_id="orgop-org-b", mcp_servers=["control-id"])
        }
        auth = _make_admitted_subject("sso-user")
        with self._patch(
            teams_by_id=teams,
            user_teams=["team-b"],
            org_perms=org_perms,
            registry={"cfg-oauth-1": forbidden, "control-id": control},
        ):
            result = await MCPRequestHandler.get_allowed_mcp_servers(auth)
        # control survives ('control_alias' resolved to 'control-id', matching the id-based ceiling); the
        # forbidden config server ('cfg-oauth-1') is capped out. A broken alias path → {} → fails here.
        assert set(result) == {"control-id"}


@pytest.mark.asyncio
class TestSessionBearerEgressScrub:
    """The gateway session bearer / bridge envelope is an admission credential, never an upstream token.
    The leak-defense scrub is anchored to the credential SHAPE, so a session-shaped Authorization is
    stripped from every egress context even when it reaches a non-aggregate scope that never set the
    admission marker (design-review finding: a session bearer misdirected to a per-server true_passthrough
    path would otherwise be forwarded upstream verbatim and replayed against the aggregate endpoint)."""

    async def test_session_bearer_misdirected_to_passthrough_is_scrubbed(self):
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [(b"authorization", b"Bearer llm_session_synthetic-shaped-token")],
        }
        ttp_server = MagicMock()
        ttp_server.auth_type = MCPAuth.true_passthrough

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = ttp_server
            (_auth, _mah, _srv, _sah, oauth2_headers, raw_headers) = await MCPRequestHandler.process_mcp_request(scope)

        mock_auth.assert_not_called()  # true_passthrough → LiteLLM auth skipped (anonymous arm, no marker)
        assert oauth2_headers is None  # session-shaped bearer scrubbed from oauth2 egress
        assert all(k.lower() != "authorization" for k in raw_headers)  # ...and from raw egress headers

    async def test_legitimate_upstream_token_is_not_scrubbed(self):
        """A genuine upstream/passthrough token is never session- or envelope-shaped, so the shape-anchored
        scrub must leave it intact for forwarding (guards against over-stripping)."""
        from litellm.types.mcp import MCPAuth

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/passthrough_server",
            "headers": [(b"authorization", b"Bearer real-upstream-opaque-token-xyz")],
        }
        ttp_server = MagicMock()
        ttp_server.auth_type = MCPAuth.true_passthrough

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
                new_callable=AsyncMock,
            ),
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mock_mgr,
        ):
            mock_mgr.get_mcp_server_by_name.return_value = ttp_server
            (_auth, _mah, _srv, _sah, oauth2_headers, _raw) = await MCPRequestHandler.process_mcp_request(scope)

        assert oauth2_headers.get("Authorization") == "Bearer real-upstream-opaque-token-xyz"

    async def test_scrub_removes_gateway_credential_from_every_egress_context(self):
        """The scrub is anchored to the credential SHAPE and covers ALL egress contexts, not just
        Authorization: a session bearer placed in x-mcp-auth OR a per-server x-mcp-{alias}-authorization
        header is stripped too (the High-severity gap: those were forwarded upstream before)."""
        sess = "Bearer llm_session_abc"
        oauth2, raw, mcp_auth, per_server = MCPRequestHandler._scrub_gateway_admission_credentials(
            admitted=False,
            oauth2_headers={"Authorization": sess},
            raw_headers={
                "authorization": sess,
                "x-mcp-auth": "llm_session_xyz",
                "x-mcp-github-authorization": "llm_session_ghi",
            },
            mcp_auth_header="llm_session_xyz",
            mcp_server_auth_headers={"github": {"Authorization": "llm_session_ghi"}},
        )
        assert oauth2 is None
        assert "authorization" not in {k.lower() for k in raw}
        assert all("llm_session_" not in v for v in raw.values())  # x-mcp-auth + per-server raw values gone
        assert mcp_auth is None  # deprecated x-mcp-auth value scrubbed
        assert per_server == {}  # per-server session bearer removed → now-empty server dict dropped

    async def test_scrub_keeps_real_upstream_tokens(self):
        """A legitimate upstream token is never session-/envelope-shaped, so every context is forwarded
        unchanged — guards against over-stripping a real credential the caller meant for the upstream."""
        oauth2, raw, mcp_auth, per_server = MCPRequestHandler._scrub_gateway_admission_credentials(
            admitted=False,
            oauth2_headers={"Authorization": "Bearer real-upstream-xyz"},
            raw_headers={"authorization": "Bearer real-upstream-xyz", "x-mcp-github-authorization": "Bearer gh_real"},
            mcp_auth_header="some-api-key-123",
            mcp_server_auth_headers={"github": {"Authorization": "Bearer gh_real"}},
        )
        assert oauth2 == {"Authorization": "Bearer real-upstream-xyz"}
        assert raw["authorization"] == "Bearer real-upstream-xyz"
        assert mcp_auth == "some-api-key-123"
        assert per_server == {"github": {"Authorization": "Bearer gh_real"}}

    async def test_scrub_admitted_drops_authorization_but_keeps_injected_upstream_token(self):
        """An admitted subject's top-level Authorization is dropped unconditionally, while the real
        upstream token the bridge arm INJECTS into a per-server header (not gateway-shaped) survives."""
        oauth2, raw, mcp_auth, per_server = MCPRequestHandler._scrub_gateway_admission_credentials(
            admitted=True,
            oauth2_headers={"Authorization": "Bearer llm_session_abc"},
            raw_headers={"authorization": "Bearer llm_session_abc"},
            mcp_auth_header=None,
            mcp_server_auth_headers={"github": {"Authorization": "Bearer gh_injected_upstream"}},
        )
        assert oauth2 is None
        assert "authorization" not in {k.lower() for k in raw}
        assert per_server == {"github": {"Authorization": "Bearer gh_injected_upstream"}}

import pytest

from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_team_mcp_manager,
)
from litellm.proxy.management_endpoints import (
    mcp_management_endpoints as mgmt_endpoints,
)


class TestIsUserTeamMcpManager:
    def test_mcp_server_manager_role_returns_true(self):
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
        )
        team = LiteLLM_TeamTable(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager")
            ],
        )
        assert _is_user_team_mcp_manager(user_auth, team) is True

    def test_regular_user_role_returns_false(self):
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
        )
        team = LiteLLM_TeamTable(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="user")],
        )
        assert _is_user_team_mcp_manager(user_auth, team) is False

    def test_admin_role_returns_false(self):
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
        )
        team = LiteLLM_TeamTable(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="admin")],
        )
        assert _is_user_team_mcp_manager(user_auth, team) is False

    def test_user_not_in_team_returns_false(self):
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user2",
            api_key="sk-test",
        )
        team = LiteLLM_TeamTable(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager")
            ],
        )
        assert _is_user_team_mcp_manager(user_auth, team) is False


from unittest.mock import AsyncMock, MagicMock, patch
from litellm.proxy._types import LiteLLM_TeamTableCachedObj


@pytest.mark.asyncio
@pytest.mark.skipif(
    not mgmt_endpoints.MCP_AVAILABLE, reason="MCP module not installed"
)
class TestAssertCanManageTeamMcpServer:
    async def test_mcp_manager_with_team_id_succeeds(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _assert_can_manage_team_mcp_server,
        )
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user1", api_key="sk-test",
        )
        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="mcp_server_manager")],
        )
        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_team_object",
            AsyncMock(return_value=mock_team),
        ):
            team_id, team_obj = await _assert_can_manage_team_mcp_server(
                user_api_key_dict=user_auth, team_id="team1"
            )
            assert team_id == "team1"
            assert team_obj == mock_team

    async def test_regular_user_gets_403(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _assert_can_manage_team_mcp_server,
        )
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user1", api_key="sk-test",
        )
        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="user")],
        )
        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_team_object",
            AsyncMock(return_value=mock_team),
        ):
            with pytest.raises(Exception) as exc_info:
                await _assert_can_manage_team_mcp_server(
                    user_api_key_dict=user_auth, team_id="team1"
                )
            assert exc_info.value.status_code == 403

    async def test_admin_gets_403(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _assert_can_manage_team_mcp_server,
        )
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user1", api_key="sk-test",
        )
        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="admin")],
        )
        with patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_team_object",
            AsyncMock(return_value=mock_team),
        ):
            with pytest.raises(Exception) as exc_info:
                await _assert_can_manage_team_mcp_server(
                    user_api_key_dict=user_auth, team_id="team1"
                )
            assert exc_info.value.status_code == 403

    async def test_mcp_manager_server_in_team_succeeds(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _assert_can_manage_team_mcp_server,
        )
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user1", api_key="sk-test", team_id="team1",
        )
        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="mcp_server_manager")],
        )
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_team_object",
                AsyncMock(return_value=mock_team),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_team_allowed_mcp_servers",
                AsyncMock(return_value={"server1", "server2"}),
            ),
        ):
            team_id, team_obj = await _assert_can_manage_team_mcp_server(
                user_api_key_dict=user_auth, server_id="server1"
            )
            assert team_id == "team1"
            assert team_obj == mock_team

    async def test_mcp_manager_server_not_in_team_gets_403(self):
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _assert_can_manage_team_mcp_server,
        )
        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER, user_id="user1", api_key="sk-test", team_id="team1",
        )
        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[Member(user_id="user1", role="mcp_server_manager")],
        )
        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_team_object",
                AsyncMock(return_value=mock_team),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._get_team_allowed_mcp_servers",
                AsyncMock(return_value={"server2", "server3"}),
            ),
        ):
            with pytest.raises(Exception) as exc_info:
                await _assert_can_manage_team_mcp_server(
                    user_api_key_dict=user_auth, server_id="server1"
                )
            assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.skipif(
    not mgmt_endpoints.MCP_AVAILABLE, reason="MCP module not installed"
)
class TestCreateMcpServerAsManager:
    async def test_create_auto_assigns_to_team(self):
        """MCP manager creating a server should auto-assign it to their team's ObjectPermissionTable."""
        from litellm.proxy._types import NewMCPServerRequest
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_mcp_server,
        )

        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
            team_id="team1",
        )

        payload = NewMCPServerRequest(
            server_name="test-server",
            url="https://example.com/mcp",
            team_id="team1",
        )

        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager"),
            ],
            object_permission_id="perm1",
        )
        mock_team.object_permission = MagicMock(mcp_servers=["existing_server"])

        created_server = MagicMock()
        created_server.server_id = "new_server_id"
        created_server.credentials = None

        mock_handle_update = AsyncMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._assert_can_manage_team_mcp_server",
                AsyncMock(return_value=("team1", mock_team)),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.create_mcp_server",
                AsyncMock(return_value=created_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                MagicMock(
                    add_server=AsyncMock(),
                    reload_servers_from_database=AsyncMock(),
                ),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.handle_update_object_permission_common",
                mock_handle_update,
            ),
        ):
            await add_mcp_server(payload=payload, user_api_key_dict=user_auth)

            # Verify handle_update_object_permission_common was called with merged server list
            mock_handle_update.assert_called_once()
            call_kwargs = mock_handle_update.call_args.kwargs
            mcp_servers = call_kwargs["data_json"]["object_permission"]["mcp_servers"]
            assert "existing_server" in mcp_servers
            assert "new_server_id" in mcp_servers
            assert call_kwargs["existing_object_permission_id"] == "perm1"

    async def test_create_links_new_permission_to_team_when_none_exists(self):
        """When team has no object_permission_id, create should link the new one."""
        from litellm.proxy._types import NewMCPServerRequest
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_mcp_server,
        )

        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
            team_id="team1",
        )

        payload = NewMCPServerRequest(
            server_name="test_server",
            url="https://example.com/mcp",
            team_id="team1",
        )

        # Team with NO object_permission_id
        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager"),
            ],
            object_permission_id=None,
        )
        mock_team.object_permission = None

        created_server = MagicMock()
        created_server.server_id = "new_server_id"
        created_server.credentials = None

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.update = AsyncMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_prisma,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._assert_can_manage_team_mcp_server",
                AsyncMock(return_value=("team1", mock_team)),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
                AsyncMock(return_value=None),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.create_mcp_server",
                AsyncMock(return_value=created_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                MagicMock(
                    add_server=AsyncMock(),
                    reload_servers_from_database=AsyncMock(),
                ),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.handle_update_object_permission_common",
                AsyncMock(return_value="new_perm_id"),
            ),
        ):
            await add_mcp_server(payload=payload, user_api_key_dict=user_auth)

            # Verify the team was updated with the new object_permission_id
            mock_prisma.db.litellm_teamtable.update.assert_called_once_with(
                where={"team_id": "team1"},
                data={"object_permission_id": "new_perm_id"},
            )

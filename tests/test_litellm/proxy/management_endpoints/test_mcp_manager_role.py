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
            result = await _assert_can_manage_team_mcp_server(
                user_api_key_dict=user_auth, team_id="team1"
            )
            assert result == "team1"

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
            result = await _assert_can_manage_team_mcp_server(
                user_api_key_dict=user_auth, server_id="server1"
            )
            assert result == "team1"

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

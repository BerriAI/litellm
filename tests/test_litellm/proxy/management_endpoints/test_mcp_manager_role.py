import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

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
from fastapi import HTTPException
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

        mock_auto_assign = AsyncMock()

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
                "litellm.proxy.management_endpoints.mcp_management_endpoints._auto_assign_mcp_server_to_team",
                mock_auto_assign,
            ),
        ):
            await add_mcp_server(payload=payload, user_api_key_dict=user_auth)

            # Verify _auto_assign_mcp_server_to_team was called with the right args
            mock_auto_assign.assert_called_once()
            call_kwargs = mock_auto_assign.call_args.kwargs
            assert call_kwargs["server_id"] == "new_server_id"
            assert call_kwargs["team_id"] == "team1"
            assert call_kwargs["team_obj"] == mock_team

    async def test_auto_assign_links_new_permission_to_team(self):
        """_auto_assign_mcp_server_to_team should create permission and link to team."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _auto_assign_mcp_server_to_team,
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

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_teamtable.update = AsyncMock()

        # handle_update_object_permission sets object_permission_id in data_json
        async def fake_handle(data_json, existing_team_row):
            data_json["object_permission_id"] = "new_perm_id"
            return data_json

        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.handle_update_object_permission",
            side_effect=fake_handle,
        ):
            await _auto_assign_mcp_server_to_team(
                server_id="new_server_id",
                team_id="team1",
                team_obj=mock_team,
                prisma_client=mock_prisma,
            )

            # Verify the team was updated with the new object_permission_id
            mock_prisma.db.litellm_teamtable.update.assert_called_once_with(
                where={"team_id": "team1"},
                data={"object_permission_id": "new_perm_id"},
            )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not mgmt_endpoints.MCP_AVAILABLE, reason="MCP module not installed"
)
class TestEditMcpServerAsManager:
    async def test_edit_succeeds_for_mcp_manager(self):
        """MCP manager should be able to edit a server assigned to their team."""
        from litellm.proxy._types import UpdateMCPServerRequest
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            edit_mcp_server,
        )

        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
            team_id="team1",
        )

        payload = UpdateMCPServerRequest(
            server_id="server1",
            description="Updated description",
        )

        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager"),
            ],
        )

        updated_server = MagicMock()
        updated_server.server_id = "server1"
        updated_server.credentials = None

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
                "litellm.proxy.management_endpoints.mcp_management_endpoints.update_mcp_server",
                AsyncMock(return_value=updated_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                MagicMock(
                    update_server=AsyncMock(),
                    reload_servers_from_database=AsyncMock(),
                ),
            ),
        ):
            result = await edit_mcp_server(payload=payload, user_api_key_dict=user_auth)
            # Should not raise — edit succeeded
            assert result is not None

    async def test_edit_fails_for_server_not_in_team(self):
        """MCP manager should get 403 when editing a server not in their team."""
        from litellm.proxy._types import UpdateMCPServerRequest
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            edit_mcp_server,
        )

        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
            team_id="team1",
        )

        payload = UpdateMCPServerRequest(
            server_id="server_not_in_team",
            description="Updated description",
        )

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
                AsyncMock(side_effect=HTTPException(status_code=403, detail="Not in team")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await edit_mcp_server(payload=payload, user_api_key_dict=user_auth)
            assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.skipif(
    not mgmt_endpoints.MCP_AVAILABLE, reason="MCP module not installed"
)
class TestDeleteMcpServerAsManager:
    async def test_delete_succeeds_and_cleans_up_team(self):
        """MCP manager deleting a server should also remove it from team permissions."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            remove_mcp_server,
        )

        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
            team_id="team1",
        )

        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager"),
            ],
        )

        deleted_server = MagicMock()
        deleted_server.server_id = "server1"

        mock_remove_from_team = AsyncMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._assert_can_manage_team_mcp_server",
                AsyncMock(return_value=("team1", mock_team)),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.delete_mcp_server",
                AsyncMock(return_value=deleted_server),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
                MagicMock(
                    remove_server=MagicMock(),
                    reload_servers_from_database=AsyncMock(),
                ),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._remove_mcp_server_from_team",
                mock_remove_from_team,
            ),
        ):
            response = await remove_mcp_server(
                server_id="server1", user_api_key_dict=user_auth
            )
            assert response.status_code == 202

            # Verify team cleanup was called
            mock_remove_from_team.assert_called_once()
            call_kwargs = mock_remove_from_team.call_args.kwargs
            assert call_kwargs["server_id"] == "server1"
            assert call_kwargs["team_obj"] == mock_team

    async def test_delete_fails_for_server_not_in_team(self):
        """MCP manager should get 403 when deleting a server not in their team."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            remove_mcp_server,
        )

        user_auth = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="user1",
            api_key="sk-test",
            team_id="team1",
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints._assert_can_manage_team_mcp_server",
                AsyncMock(side_effect=HTTPException(status_code=403, detail="Not in team")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await remove_mcp_server(
                    server_id="server_not_in_team", user_api_key_dict=user_auth
                )
            assert exc_info.value.status_code == 403

    async def test_remove_mcp_server_from_team_helper(self):
        """_remove_mcp_server_from_team should update the permission list without the deleted server."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            _remove_mcp_server_from_team,
        )

        mock_team = LiteLLM_TeamTableCachedObj(
            team_id="team1",
            members_with_roles=[
                Member(user_id="user1", role="mcp_server_manager"),
            ],
            object_permission_id="perm1",
        )
        mock_team.object_permission = MagicMock(
            mcp_servers=["server1", "server2", "server3"]
        )

        mock_handle_common = AsyncMock()

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.handle_update_object_permission_common",
                mock_handle_common,
            ),
        ):
            await _remove_mcp_server_from_team(
                server_id="server2",
                team_obj=mock_team,
            )

            mock_handle_common.assert_called_once()
            call_kwargs = mock_handle_common.call_args.kwargs
            updated_servers = call_kwargs["data_json"]["object_permission"]["mcp_servers"]
            assert "server2" not in updated_servers
            assert "server1" in updated_servers
            assert "server3" in updated_servers
            assert call_kwargs["existing_object_permission_id"] == "perm1"

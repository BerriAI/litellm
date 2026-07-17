from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
import pytest

from litellm.proxy._types import (
    JWTLiteLLMRoleMap,
    LiteLLM_JWTAuth,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler


@pytest.mark.asyncio
async def test_map_user_to_teams_user_already_in_team():
    """Test that no action is taken when user is already in team"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(
        team_id="test_team_1",
        members_with_roles=[Member(user_id="test_user_1", role="user")],
    )

    # Mock team_member_add to ensure it's not called
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
    ) as mock_add:
        await JWTAuthManager.map_user_to_teams(user_object=user, team_object=team)
        mock_add.assert_not_called()


@pytest.mark.asyncio
async def test_map_user_to_teams_add_new_user():
    """Test that new user is added to team"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(team_id="test_team_1", members_with_roles=[])

    # Mock team_member_add
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
    ) as mock_add:
        await JWTAuthManager.map_user_to_teams(user_object=user, team_object=team)
        mock_add.assert_called_once()
        # Verify the correct data was passed to team_member_add
        call_args = mock_add.call_args[1]["data"]
        assert call_args.member.user_id == "test_user_1"
        assert call_args.member.role == "user"
        assert call_args.team_id == "test_team_1"


@pytest.mark.asyncio
async def test_map_user_to_teams_handles_already_in_team_exception():
    """Test that team_member_already_in_team exception is handled gracefully"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(team_id="test_team_1", members_with_roles=[])

    # Create a ProxyException with team_member_already_in_team error type
    already_in_team_exception = ProxyException(
        message="User test_user_1 already in team",
        type=ProxyErrorTypes.team_member_already_in_team,
        param="user_id",
        code="400",
    )

    # Mock team_member_add to raise the exception
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        side_effect=already_in_team_exception,
    ) as mock_add:
        with patch("litellm.proxy.auth.handle_jwt.verbose_proxy_logger") as mock_logger:
            # This should not raise an exception
            result = await JWTAuthManager.map_user_to_teams(
                user_object=user, team_object=team
            )

            # Verify the method completed successfully
            assert result is None
            mock_add.assert_called_once()


@pytest.mark.asyncio
async def test_map_user_to_teams_reraises_other_proxy_exceptions():
    """Test that other ProxyException types are re-raised"""
    # Setup test data
    user = LiteLLM_UserTable(user_id="test_user_1")
    team = LiteLLM_TeamTable(team_id="test_team_1", members_with_roles=[])

    # Create a ProxyException with a different error type
    other_exception = ProxyException(
        message="Some other error",
        type=ProxyErrorTypes.internal_server_error,
        param="some_param",
        code="500",
    )

    # Mock team_member_add to raise the exception
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        side_effect=other_exception,
    ) as mock_add:
        # This should re-raise the exception
        with pytest.raises(ProxyException) as exc_info:
            await JWTAuthManager.map_user_to_teams(user_object=user, team_object=team)


@pytest.mark.asyncio
async def test_map_user_to_teams_null_inputs():
    """Test that method handles null inputs gracefully"""
    # Test with null user
    await JWTAuthManager.map_user_to_teams(
        user_object=None, team_object=LiteLLM_TeamTable(team_id="test_team_1")
    )

    # Test with null team
    await JWTAuthManager.map_user_to_teams(
        user_object=LiteLLM_UserTable(user_id="test_user_1"), team_object=None
    )

    # Test with both null
    await JWTAuthManager.map_user_to_teams(user_object=None, team_object=None)


@pytest.mark.asyncio
async def test_find_team_with_model_access_reports_passthrough_allowlist_denial():
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()
    team = LiteLLM_TeamTable(
        team_id="team-a",
        models=["gpt-4"],
        metadata={},
    )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            return_value=team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.can_team_access_model",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.allowed_routes_check",
            return_value=True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.is_auth_enforced_pass_through_route",
            return_value=True,
        ) as mock_is_auth_enforced_pass_through_route,
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.check_passthrough_route_access",
            return_value=False,
        ) as mock_passthrough_check,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.find_team_with_model_access(
                team_ids={"team-a"},
                requested_model="gpt-4",
                route="/my-pass-through",
                request_method="POST",
                jwt_handler=jwt_handler,
                prisma_client=None,
                user_api_key_cache=MagicMock(),
                parent_otel_span=None,
                proxy_logging_obj=MagicMock(),
            )

    assert exc_info.value.status_code == 403
    assert "allowed_passthrough_routes" in exc_info.value.detail
    assert "requested model" not in exc_info.value.detail
    mock_is_auth_enforced_pass_through_route.assert_called_once_with(
        route="/my-pass-through", method="POST"
    )

    user_api_key_dict = mock_passthrough_check.call_args.kwargs["user_api_key_dict"]
    assert user_api_key_dict.metadata == {}
    assert user_api_key_dict.team_metadata == {}


@pytest.mark.asyncio
async def test_find_team_with_model_access_uses_request_method_for_passthrough_auth():
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()
    team = LiteLLM_TeamTable(
        team_id="team-a",
        models=["gpt-4"],
        metadata={},
    )
    mock_registered_routes = {
        "test-uuid-1:exact:/custom:GET": {
            "endpoint_id": "test-uuid-1",
            "path": "/custom",
            "type": "exact",
            "methods": ["GET"],
            "auth": False,
        },
        "test-uuid-2:exact:/custom:POST": {
            "endpoint_id": "test-uuid-2",
            "path": "/custom",
            "type": "exact",
            "methods": ["POST"],
            "auth": True,
        },
    }

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            return_value=team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.allowed_routes_check",
            return_value=True,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.route_registry.registered_pass_through_routes",
            mock_registered_routes,
        ),
        patch(
            "litellm.proxy.utils.get_server_root_path",
            return_value="/",
        ),
    ):
        team_id, team_obj = await JWTAuthManager.find_team_with_model_access(
            team_ids={"team-a"},
            requested_model=None,
            route="/custom",
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
            request_method="GET",
        )
        assert team_id == "team-a"
        assert team_obj == team

        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.find_team_with_model_access(
                team_ids={"team-a"},
                requested_model=None,
                route="/custom",
                jwt_handler=jwt_handler,
                prisma_client=None,
                user_api_key_cache=MagicMock(),
                parent_otel_span=None,
                proxy_logging_obj=MagicMock(),
                request_method="POST",
            )

    assert exc_info.value.status_code == 403
    assert "allowed_passthrough_routes" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auth_builder_proxy_admin_user_role():
    """Test that is_proxy_admin is True when user_object.user_role is PROXY_ADMIN"""
    # Setup test data
    api_key = "test_jwt_token"
    request_data = {"model": "gpt-4"}
    general_settings = {"enforce_rbac": False}
    route = "/chat/completions"

    # Create user object with PROXY_ADMIN role
    user_object = LiteLLM_UserTable(
        user_id="test_user_1", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Create mock JWT handler
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    # Mock all the dependencies and method calls
    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(
            JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
        ) as mock_check_rbac,
        patch.object(jwt_handler, "get_rbac_role", return_value=None) as mock_get_rbac,
        patch.object(jwt_handler, "get_scopes", return_value=[]) as mock_get_scopes,
        patch.object(
            jwt_handler, "get_object_id", return_value=None
        ) as mock_get_object_id,
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=("test_user_1", "test@example.com", True),
        ) as mock_get_user_info,
        patch.object(jwt_handler, "get_org_id", return_value=None) as mock_get_org_id,
        patch.object(
            jwt_handler, "get_end_user_id", return_value=None
        ) as mock_get_end_user_id,
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_check_admin,
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team,
        patch.object(
            JWTAuthManager, "get_all_team_ids", return_value=set()
        ) as mock_get_all_team_ids,
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team_access,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ) as mock_get_objects,
        patch.object(
            JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
        ) as mock_map_user,
        patch.object(
            JWTAuthManager, "validate_object_id", return_value=True
        ) as mock_validate_object,
    ):
        # Set up the mock return values
        mock_auth_jwt.return_value = {"sub": "test_user_1", "scope": ""}

        # Call the auth_builder method
        result = await JWTAuthManager.auth_builder(
            api_key=api_key,
            jwt_handler=jwt_handler,
            request_data=request_data,
            general_settings=general_settings,
            route=route,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        # Verify that is_proxy_admin is True
        assert result["is_proxy_admin"] is True
        assert result["user_object"] == user_object
        assert result["user_id"] == "test_user_1"


@pytest.mark.asyncio
async def test_auth_builder_non_proxy_admin_user_role():
    """Test that is_proxy_admin is False when user_object.user_role is not PROXY_ADMIN"""
    # Setup test data
    api_key = "test_jwt_token"
    request_data = {"model": "gpt-4"}
    general_settings = {"enforce_rbac": False}
    route = "/chat/completions"

    # Create user object with regular USER role
    user_object = LiteLLM_UserTable(
        user_id="test_user_1", user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Create mock JWT handler
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    # Mock all the dependencies and method calls
    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(
            JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
        ) as mock_check_rbac,
        patch.object(jwt_handler, "get_rbac_role", return_value=None) as mock_get_rbac,
        patch.object(jwt_handler, "get_scopes", return_value=[]) as mock_get_scopes,
        patch.object(
            jwt_handler, "get_object_id", return_value=None
        ) as mock_get_object_id,
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=("test_user_1", "test@example.com", True),
        ) as mock_get_user_info,
        patch.object(jwt_handler, "get_org_id", return_value=None) as mock_get_org_id,
        patch.object(
            jwt_handler, "get_end_user_id", return_value=None
        ) as mock_get_end_user_id,
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_check_admin,
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team,
        patch.object(
            JWTAuthManager, "get_all_team_ids", return_value=set()
        ) as mock_get_all_team_ids,
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team_access,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ) as mock_get_objects,
        patch.object(
            JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
        ) as mock_map_user,
        patch.object(
            JWTAuthManager, "validate_object_id", return_value=True
        ) as mock_validate_object,
    ):
        # Set up the mock return values
        mock_auth_jwt.return_value = {"sub": "test_user_1", "scope": ""}

        # Call the auth_builder method
        result = await JWTAuthManager.auth_builder(
            api_key=api_key,
            jwt_handler=jwt_handler,
            request_data=request_data,
            general_settings=general_settings,
            route=route,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        # Verify that is_proxy_admin is False
        assert result["is_proxy_admin"] is False
        assert result["user_object"] == user_object
        assert result["user_id"] == "test_user_1"


@pytest.mark.asyncio
async def test_sync_user_role_and_teams():
    from unittest.mock import MagicMock

    # Create mock objects for required types
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=mock_user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            jwt_litellm_role_map=[
                JWTLiteLLMRoleMap(
                    jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN
                )
            ],
            roles_jwt_field="roles",
            team_ids_jwt_field="my_id_teams",
            sync_user_role_and_teams=True,
        ),
    )

    token = {"roles": ["ADMIN"], "my_id_teams": ["team1", "team2"]}

    user = LiteLLM_UserTable(
        user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER.value, teams=["team2"]
    )

    prisma = AsyncMock()
    prisma.db.litellm_usertable.update = AsyncMock()

    with patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        new_callable=AsyncMock,
    ) as mock_patch:
        await JWTAuthManager.sync_user_role_and_teams(jwt_handler, token, user, prisma)

    prisma.db.litellm_usertable.update.assert_called_once()
    mock_patch.assert_called_once()
    assert user.user_role == LitellmUserRoles.PROXY_ADMIN.value
    assert set(user.teams) == {"team1", "team2"}


@pytest.mark.asyncio
async def test_sync_user_role_and_teams_cache_invalidation_on_role_change():
    """Test that user cache is updated when role changes."""
    mock_cache = AsyncMock()

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=AsyncMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            jwt_litellm_role_map=[
                JWTLiteLLMRoleMap(
                    jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN
                )
            ],
            roles_jwt_field="roles",
            team_ids_jwt_field="my_id_teams",
            sync_user_role_and_teams=True,
        ),
    )

    token = {"roles": ["ADMIN"], "my_id_teams": ["team1"]}
    user = LiteLLM_UserTable(
        user_id="u1",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        teams=["team1"],  # teams already match — only role differs
    )

    prisma = AsyncMock()
    prisma.db.litellm_usertable.update = AsyncMock()

    await JWTAuthManager.sync_user_role_and_teams(
        jwt_handler, token, user, prisma, user_api_key_cache=mock_cache
    )

    mock_cache.async_set_cache.assert_called_once()
    call_kwargs = mock_cache.async_set_cache.call_args
    assert call_kwargs.kwargs["key"] == "u1"
    assert isinstance(call_kwargs.kwargs["value"], LiteLLM_UserTable)
    assert call_kwargs.kwargs["value"].user_role == LitellmUserRoles.PROXY_ADMIN.value
    assert call_kwargs.kwargs["model_type"] == LiteLLM_UserTable


@pytest.mark.asyncio
async def test_sync_user_role_and_teams_cache_invalidation_on_team_change():
    """Test that user cache is updated when team memberships change."""
    mock_cache = AsyncMock()

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=AsyncMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            jwt_litellm_role_map=[
                JWTLiteLLMRoleMap(
                    jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN
                )
            ],
            roles_jwt_field="roles",
            team_ids_jwt_field="my_id_teams",
            sync_user_role_and_teams=True,
        ),
    )

    token = {"roles": ["ADMIN"], "my_id_teams": ["team1", "team2"]}
    user = LiteLLM_UserTable(
        user_id="u1",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,  # role already matches
        teams=["team2"],  # teams differ
    )

    prisma = AsyncMock()
    prisma.db.litellm_usertable.update = AsyncMock()

    with patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        new_callable=AsyncMock,
    ):
        await JWTAuthManager.sync_user_role_and_teams(
            jwt_handler, token, user, prisma, user_api_key_cache=mock_cache
        )

    mock_cache.async_set_cache.assert_called_once()
    call_kwargs = mock_cache.async_set_cache.call_args
    assert call_kwargs.kwargs["key"] == "u1"
    assert isinstance(call_kwargs.kwargs["value"], LiteLLM_UserTable)
    assert set(call_kwargs.kwargs["value"].teams) == {"team1", "team2"}
    assert call_kwargs.kwargs["model_type"] == LiteLLM_UserTable


@pytest.mark.asyncio
async def test_sync_user_role_and_teams_no_cache_write_when_nothing_changes():
    """Test that cache is NOT written when role and teams already match."""
    mock_cache = AsyncMock()

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=AsyncMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            jwt_litellm_role_map=[
                JWTLiteLLMRoleMap(
                    jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN
                )
            ],
            roles_jwt_field="roles",
            team_ids_jwt_field="my_id_teams",
            sync_user_role_and_teams=True,
        ),
    )

    token = {"roles": ["ADMIN"], "my_id_teams": ["team1"]}
    user = LiteLLM_UserTable(
        user_id="u1",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        teams=["team1"],
    )

    prisma = AsyncMock()

    await JWTAuthManager.sync_user_role_and_teams(
        jwt_handler, token, user, prisma, user_api_key_cache=mock_cache
    )

    mock_cache.async_set_cache.assert_not_called()


def test_get_all_jwt_team_ids_unions_singular_and_plural():
    """get_all_jwt_team_ids must include the singular team_id_jwt_field claim
    in addition to the plural team_ids_jwt_field, deduplicated."""
    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=MagicMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_id_jwt_field="team_id",
            team_ids_jwt_field="teams",
        ),
    )

    # singular only — Okta/Auth0 default shape
    assert jwt_handler.get_all_jwt_team_ids({"team_id": "team-low"}) == ["team-low"]

    # plural only — pre-fix shape
    assert jwt_handler.get_all_jwt_team_ids({"teams": ["a", "b"]}) == ["a", "b"]

    # both populated, no overlap
    assert jwt_handler.get_all_jwt_team_ids(
        {"team_id": "primary", "teams": ["a", "b"]}
    ) == ["a", "b", "primary"]

    # both populated with overlap — singular dedup'd
    assert jwt_handler.get_all_jwt_team_ids({"team_id": "a", "teams": ["a", "b"]}) == [
        "a",
        "b",
    ]

    # singular field as multi-element list (some IdPs) — merge all, preserve plural-first order
    assert jwt_handler.get_all_jwt_team_ids(
        {"team_id": ["primary", "secondary"], "teams": ["a"]}
    ) == [
        "a",
        "primary",
        "secondary",
    ]

    # neither populated
    assert jwt_handler.get_all_jwt_team_ids({}) == []


def test_get_all_jwt_team_ids_does_not_use_team_id_default():
    """team_id_default is a JWT-bearer-flow auth-builder fallback, not a token
    claim. It must NOT leak into get_all_jwt_team_ids — otherwise SSO logins
    would silently start adding users to the default team for any tenant that
    has team_id_default configured."""
    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=MagicMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_id_jwt_field="team_id",
            team_ids_jwt_field="teams",
            team_id_default="default-team",
        ),
    )

    # team_id claim missing — must not fall back to default-team
    assert jwt_handler.get_all_jwt_team_ids({"teams": []}) == []
    assert jwt_handler.get_all_jwt_team_ids({}) == []

    # only the plural is populated — default still must not be added
    assert jwt_handler.get_all_jwt_team_ids({"teams": ["a"]}) == ["a"]

    # team_id_jwt_field unset entirely + only default configured: still no default
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=MagicMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="teams",
            team_id_default="default-team",
        ),
    )
    assert jwt_handler.get_all_jwt_team_ids({"teams": []}) == []


@pytest.mark.asyncio
async def test_map_jwt_role_to_litellm_role():
    """Test JWT role mapping to LiteLLM roles with various patterns"""
    from unittest.mock import MagicMock

    # Create mock objects for required types
    mock_user_api_key_cache = MagicMock()

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=mock_user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            jwt_litellm_role_map=[
                # Exact match
                JWTLiteLLMRoleMap(
                    jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN
                ),
                # Wildcard patterns
                JWTLiteLLMRoleMap(
                    jwt_role="user_*", litellm_role=LitellmUserRoles.INTERNAL_USER
                ),
                JWTLiteLLMRoleMap(
                    jwt_role="team_?", litellm_role=LitellmUserRoles.TEAM
                ),
                JWTLiteLLMRoleMap(
                    jwt_role="dev_[123]", litellm_role=LitellmUserRoles.INTERNAL_USER
                ),
            ],
            roles_jwt_field="roles",
        ),
    )

    # Test exact match
    token = {"roles": ["ADMIN"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.PROXY_ADMIN

    # Test wildcard match with *
    token = {"roles": ["user_manager"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.INTERNAL_USER

    token = {"roles": ["user_"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.INTERNAL_USER

    # Test wildcard match with ?
    token = {"roles": ["team_1"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.TEAM

    token = {"roles": ["team_a"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.TEAM

    # Test character class match
    token = {"roles": ["dev_1"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.INTERNAL_USER

    token = {"roles": ["dev_2"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.INTERNAL_USER

    # Test no match
    token = {"roles": ["unknown_role"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    # Test multiple roles - should return first mapping match
    token = {"roles": ["user_test", "ADMIN"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result == LitellmUserRoles.PROXY_ADMIN  # ADMIN matches first mapping

    # Test empty roles
    token = {"roles": []}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    # Test no roles field
    token = {}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    # Test no role mappings configured
    jwt_handler.litellm_jwtauth.jwt_litellm_role_map = None
    token = {"roles": ["ADMIN"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    # Test empty role mappings
    jwt_handler.litellm_jwtauth.jwt_litellm_role_map = []
    token = {"roles": ["ADMIN"]}
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    # Test patterns that don't match character classes
    jwt_handler.litellm_jwtauth.jwt_litellm_role_map = [
        JWTLiteLLMRoleMap(
            jwt_role="dev_[123]", litellm_role=LitellmUserRoles.INTERNAL_USER
        ),
    ]
    token = {"roles": ["dev_4"]}  # 4 is not in [123]
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    # Test ? pattern that requires exactly one character
    jwt_handler.litellm_jwtauth.jwt_litellm_role_map = [
        JWTLiteLLMRoleMap(jwt_role="team_?", litellm_role=LitellmUserRoles.TEAM),
    ]
    token = {"roles": ["team_12"]}  # More than one character after underscore
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None

    token = {"roles": ["team_"]}  # No character after underscore
    result = jwt_handler.map_jwt_role_to_litellm_role(token)
    assert result is None


@pytest.mark.asyncio
async def test_nested_jwt_field_access():
    """
    Test that all JWT fields support dot notation for nested access

    This test verifies that:
    1. All JWT field methods can access nested values using dot notation
    2. Backward compatibility is maintained for flat field names
    3. Missing nested paths return appropriate defaults
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    # Create JWT handler
    jwt_handler = JWTHandler()

    # Test token with nested claims
    nested_token = {
        "user": {"sub": "u123", "email": "user@example.com"},
        "resource_access": {"my-client": {"roles": ["admin", "user"]}},
        "groups": ["team1", "team2"],
        "organization": {"id": "org456"},
        "profile": {"object_id": "obj789"},
        "customer": {"end_user_id": "customer123"},
        "tenant": {"team_id": "team456"},
    }

    # Test flat token for backward compatibility
    flat_token = {
        "sub": "u123",
        "email": "user@example.com",
        "roles": ["admin", "user"],
        "groups": ["team1", "team2"],
        "org_id": "org456",
        "object_id": "obj789",
        "end_user_id": "customer123",
        "team_id": "team456",
    }

    # Test 1: user_id_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_id_jwt_field="user.sub")
    assert jwt_handler.get_user_id(nested_token, None) == "u123"

    # Test 1b: user_id_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_id_jwt_field="sub")
    assert jwt_handler.get_user_id(flat_token, None) == "u123"

    # Test 2: user_email_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_email_jwt_field="user.email")
    assert jwt_handler.get_user_email(nested_token, None) == "user@example.com"

    # Test 2b: user_email_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_email_jwt_field="email")
    assert jwt_handler.get_user_email(flat_token, None) == "user@example.com"

    # Test 3: team_ids_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_ids_jwt_field="groups")
    assert jwt_handler.get_team_ids_from_jwt(nested_token) == ["team1", "team2"]

    # Test 3b: team_ids_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_ids_jwt_field="groups")
    assert jwt_handler.get_team_ids_from_jwt(flat_token) == ["team1", "team2"]

    # Test 4: org_id_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_id_jwt_field="organization.id")
    assert jwt_handler.get_org_id(nested_token, None) == "org456"

    # Test 4b: org_id_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_id_jwt_field="org_id")
    assert jwt_handler.get_org_id(flat_token, None) == "org456"

    # Test 5: object_id_jwt_field with nested access (requires role_mappings)
    from litellm.proxy._types import LitellmUserRoles, RoleMapping

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        object_id_jwt_field="profile.object_id",
        role_mappings=[
            RoleMapping(role="admin", internal_role=LitellmUserRoles.INTERNAL_USER)
        ],
    )
    assert jwt_handler.get_object_id(nested_token, None) == "obj789"

    # Test 5b: object_id_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        object_id_jwt_field="object_id",
        role_mappings=[
            RoleMapping(role="admin", internal_role=LitellmUserRoles.INTERNAL_USER)
        ],
    )
    assert jwt_handler.get_object_id(flat_token, None) == "obj789"

    # Test 6: end_user_id_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        end_user_id_jwt_field="customer.end_user_id"
    )
    assert jwt_handler.get_end_user_id(nested_token, None) == "customer123"

    # Test 6b: end_user_id_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(end_user_id_jwt_field="end_user_id")
    assert jwt_handler.get_end_user_id(flat_token, None) == "customer123"

    # Test 7: team_id_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="tenant.team_id")
    assert jwt_handler.get_team_id(nested_token, None) == "team456"

    # Test 7b: team_id_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="team_id")
    assert jwt_handler.get_team_id(flat_token, None) == "team456"

    # Test 8: roles_jwt_field with deeply nested access (already supported, but testing)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        roles_jwt_field="resource_access.my-client.roles"
    )
    assert jwt_handler.get_jwt_role(nested_token, []) == ["admin", "user"]

    # Test 9: user_roles_jwt_field with nested access (already supported, but testing)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_roles_jwt_field="resource_access.my-client.roles",
        user_allowed_roles=["admin", "user"],
    )
    assert jwt_handler.get_user_roles(nested_token, []) == ["admin", "user"]


@pytest.mark.asyncio
async def test_nested_jwt_field_missing_paths():
    """
    Test handling of missing nested paths in JWT tokens

    This test verifies that:
    1. Missing nested paths return appropriate defaults
    2. Partial paths that exist but don't have the final key return defaults
    3. team_id_default fallback works with nested fields
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    # Create JWT handler
    jwt_handler = JWTHandler()

    # Test token with missing nested paths
    incomplete_token = {
        "user": {
            "name": "test user"
            # missing "sub" and "email"
        },
        "resource_access": {
            "other-client": {"roles": ["viewer"]}
            # missing "my-client"
        },
        # missing "organization", "profile", "customer", "tenant", "groups"
    }

    # Test 1: Missing user.sub should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_id_jwt_field="user.sub")
    assert jwt_handler.get_user_id(incomplete_token, "default_user") == "default_user"

    # Test 2: Missing user.email should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_email_jwt_field="user.email")
    assert (
        jwt_handler.get_user_email(incomplete_token, "default@example.com")
        == "default@example.com"
    )

    # Test 3: Missing groups should return empty list
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_ids_jwt_field="groups")
    assert jwt_handler.get_team_ids_from_jwt(incomplete_token) == []

    # Test 4: Missing organization.id should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_id_jwt_field="organization.id")
    assert jwt_handler.get_org_id(incomplete_token, "default_org") == "default_org"

    # Test 5: Missing profile.object_id should return default (requires role_mappings)
    from litellm.proxy._types import LitellmUserRoles, RoleMapping

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        object_id_jwt_field="profile.object_id",
        role_mappings=[
            RoleMapping(role="admin", internal_role=LitellmUserRoles.INTERNAL_USER)
        ],
    )
    assert jwt_handler.get_object_id(incomplete_token, "default_obj") == "default_obj"

    # Test 6: Missing customer.end_user_id should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        end_user_id_jwt_field="customer.end_user_id"
    )
    assert (
        jwt_handler.get_end_user_id(incomplete_token, "default_customer")
        == "default_customer"
    )

    # Test 7: Missing tenant.team_id should use team_id_default fallback
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_id_jwt_field="tenant.team_id", team_id_default="fallback_team"
    )
    assert jwt_handler.get_team_id(incomplete_token, "default_team") == "fallback_team"

    # Test 8: Missing resource_access.my-client.roles should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        roles_jwt_field="resource_access.my-client.roles"
    )
    assert jwt_handler.get_jwt_role(incomplete_token, ["default_role"]) == [
        "default_role"
    ]

    # Test 9: Missing nested user roles should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_roles_jwt_field="resource_access.my-client.roles",
        user_allowed_roles=["admin", "user"],
    )
    assert jwt_handler.get_user_roles(incomplete_token, ["default_user_role"]) == [
        "default_user_role"
    ]


@pytest.mark.asyncio
async def test_metadata_prefix_handling_in_nested_fields():
    """
    Test that metadata. prefix is properly handled in nested JWT field access

    The get_nested_value function should remove metadata. prefix before traversing
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    # Create JWT handler
    jwt_handler = JWTHandler()

    # Test token with proper structure for metadata prefix removal
    token = {
        "user": {
            "email": "user@example.com"  # This will be accessed when metadata.user.email is used
        },
        "sub": "u123",
    }

    # Test 1: metadata.user.email should access user.email after prefix removal
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_email_jwt_field="metadata.user.email"
    )
    # The get_nested_value function removes "metadata." prefix, so "metadata.user.email" becomes "user.email"
    assert jwt_handler.get_user_email(token, None) == "user@example.com"

    # Test 2: user.sub should work normally without metadata prefix
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_id_jwt_field="sub")
    assert jwt_handler.get_user_id(token, None) == "u123"


@pytest.mark.asyncio
async def test_find_team_with_model_access_model_group(monkeypatch):
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "gpt-4o-mini"},
                "model_info": {"access_groups": ["test-group"]},
            }
        ]
    )
    import sys
    import types

    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    team = LiteLLM_TeamTable(team_id="team-1", models=["test-group"])

    async def mock_get_team_object(*args, **kwargs):  # type: ignore
        return team

    monkeypatch.setattr(
        "litellm.proxy.auth.handle_jwt.get_team_object", mock_get_team_object
    )

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    team_id, team_obj = await JWTAuthManager.find_team_with_model_access(
        team_ids={"team-1"},
        requested_model="gpt-4o-mini",
        route="/chat/completions",
        jwt_handler=jwt_handler,
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
    )

    assert team_id == "team-1"
    assert team_obj.team_id == "team-1"


@pytest.mark.asyncio
async def test_auth_builder_returns_team_membership_object():
    """
    Test that auth_builder returns the team_membership_object when user is a member of a team.
    """
    # Setup test data
    api_key = "test_jwt_token"
    request_data = {"model": "gpt-4"}
    general_settings = {"enforce_rbac": False}
    route = "/chat/completions"
    _team_id = "test_team_1"
    _user_id = "test_user_1"

    # Create mock objects
    from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamMembership

    mock_team_membership = LiteLLM_TeamMembership(
        user_id=_user_id,
        team_id=_team_id,
        budget_id="budget_123",
        spend=10.5,
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget_123", rpm_limit=100, tpm_limit=5000
        ),
    )

    user_object = LiteLLM_UserTable(
        user_id=_user_id, user_role=LitellmUserRoles.INTERNAL_USER
    )

    team_object = LiteLLM_TeamTable(team_id=_team_id)

    # Create mock JWT handler
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    # Mock all the dependencies and method calls
    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(
            JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
        ) as mock_check_rbac,
        patch.object(jwt_handler, "get_rbac_role", return_value=None) as mock_get_rbac,
        patch.object(jwt_handler, "get_scopes", return_value=[]) as mock_get_scopes,
        patch.object(
            jwt_handler, "get_object_id", return_value=None
        ) as mock_get_object_id,
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(_user_id, "test@example.com", True),
        ) as mock_get_user_info,
        patch.object(jwt_handler, "get_org_id", return_value=None) as mock_get_org_id,
        patch.object(
            jwt_handler, "get_end_user_id", return_value=None
        ) as mock_get_end_user_id,
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_check_admin,
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(_team_id, team_object),
        ) as mock_find_team,
        patch.object(
            JWTAuthManager, "get_all_team_ids", return_value=set()
        ) as mock_get_all_team_ids,
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team_access,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(
                user_object,
                None,
                None,
                mock_team_membership,
                user_object.user_id,
            ),
        ) as mock_get_objects,
        patch.object(
            JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
        ) as mock_map_user,
        patch.object(
            JWTAuthManager, "validate_object_id", return_value=True
        ) as mock_validate_object,
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ) as mock_sync_user,
    ):
        # Set up the mock return values
        mock_auth_jwt.return_value = {"sub": _user_id, "scope": ""}

        # Call the auth_builder method
        result = await JWTAuthManager.auth_builder(
            api_key=api_key,
            jwt_handler=jwt_handler,
            request_data=request_data,
            general_settings=general_settings,
            route=route,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        # Verify that team_membership_object is returned
        assert result["team_membership"] is not None, (
            "team_membership should be present"
        )
        assert result["team_membership"] == mock_team_membership, (
            "team_membership should match the mock object"
        )
        assert result["team_membership"].user_id == _user_id, (
            "team_membership user_id should match"
        )
        assert result["team_membership"].team_id == _team_id, (
            "team_membership team_id should match"
        )
        assert result["team_membership"].budget_id == "budget_123", (
            "team_membership budget_id should match"
        )
        assert result["team_membership"].spend == 10.5, (
            "team_membership spend should match"
        )


@pytest.mark.asyncio
async def test_auth_builder_with_oidc_userinfo_enabled():
    """Test that auth_builder uses OIDC UserInfo endpoint when enabled"""
    from unittest.mock import MagicMock

    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    # Setup test data
    api_key = "test_access_token"
    request_data = {"model": "gpt-4"}
    general_settings = {"enforce_rbac": False}
    route = "/chat/completions"

    user_object = LiteLLM_UserTable(
        user_id="test_user_1", user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Create JWT handler with OIDC UserInfo enabled
    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            oidc_userinfo_enabled=True,
            oidc_userinfo_endpoint="https://example.com/oauth2/userinfo",
            user_id_jwt_field="sub",
            user_email_jwt_field="email",
        ),
    )

    # Mock OIDC UserInfo response
    userinfo_response = {
        "sub": "test_user_1",
        "email": "test@example.com",
        "scope": "",
    }

    # Mock all the dependencies
    with (
        patch.object(
            jwt_handler, "get_oidc_userinfo", new_callable=AsyncMock
        ) as mock_get_userinfo,
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(
            JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
        ) as mock_check_rbac,
        patch.object(jwt_handler, "get_rbac_role", return_value=None) as mock_get_rbac,
        patch.object(jwt_handler, "get_scopes", return_value=[]) as mock_get_scopes,
        patch.object(
            jwt_handler, "get_object_id", return_value=None
        ) as mock_get_object_id,
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=("test_user_1", "test@example.com", True),
        ) as mock_get_user_info,
        patch.object(jwt_handler, "get_org_id", return_value=None) as mock_get_org_id,
        patch.object(
            jwt_handler, "get_end_user_id", return_value=None
        ) as mock_get_end_user_id,
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_check_admin,
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team,
        patch.object(
            JWTAuthManager, "get_all_team_ids", return_value=set()
        ) as mock_get_all_team_ids,
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team_access,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ) as mock_get_objects,
        patch.object(
            JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
        ) as mock_map_user,
        patch.object(
            JWTAuthManager, "validate_object_id", return_value=True
        ) as mock_validate_object,
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ) as mock_sync_user,
    ):
        # Set up mock return values
        mock_get_userinfo.return_value = userinfo_response

        # Call auth_builder
        result = await JWTAuthManager.auth_builder(
            api_key=api_key,
            jwt_handler=jwt_handler,
            request_data=request_data,
            general_settings=general_settings,
            route=route,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Verify that get_oidc_userinfo was called instead of auth_jwt
        mock_get_userinfo.assert_called_once_with(token=api_key)
        mock_auth_jwt.assert_not_called()  # Should not be called when OIDC is enabled

        # Verify the result
        assert result["user_id"] == "test_user_1"
        assert result["user_object"] == user_object


@pytest.mark.asyncio
async def test_auth_builder_with_oidc_userinfo_disabled():
    """Test that auth_builder uses JWT validation when OIDC UserInfo is disabled"""
    from unittest.mock import MagicMock

    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    # Setup test data
    api_key = "test_jwt_token"
    request_data = {"model": "gpt-4"}
    general_settings = {"enforce_rbac": False}
    route = "/chat/completions"

    user_object = LiteLLM_UserTable(
        user_id="test_user_1", user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Create JWT handler with OIDC UserInfo disabled
    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            oidc_userinfo_enabled=False,  # Disabled
            user_id_jwt_field="sub",
        ),
    )

    # Mock JWT validation response
    jwt_response = {
        "sub": "test_user_1",
        "scope": "",
    }

    # Mock all the dependencies
    with (
        patch.object(
            jwt_handler, "get_oidc_userinfo", new_callable=AsyncMock
        ) as mock_get_userinfo,
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(
            JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
        ) as mock_check_rbac,
        patch.object(jwt_handler, "get_rbac_role", return_value=None) as mock_get_rbac,
        patch.object(jwt_handler, "get_scopes", return_value=[]) as mock_get_scopes,
        patch.object(
            jwt_handler, "get_object_id", return_value=None
        ) as mock_get_object_id,
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=("test_user_1", None, None),
        ) as mock_get_user_info,
        patch.object(jwt_handler, "get_org_id", return_value=None) as mock_get_org_id,
        patch.object(
            jwt_handler, "get_end_user_id", return_value=None
        ) as mock_get_end_user_id,
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_check_admin,
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team,
        patch.object(
            JWTAuthManager, "get_all_team_ids", return_value=set()
        ) as mock_get_all_team_ids,
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ) as mock_find_team_access,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ) as mock_get_objects,
        patch.object(
            JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
        ) as mock_map_user,
        patch.object(
            JWTAuthManager, "validate_object_id", return_value=True
        ) as mock_validate_object,
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ) as mock_sync_user,
    ):
        # Set up mock return values
        mock_auth_jwt.return_value = jwt_response

        # Call auth_builder
        result = await JWTAuthManager.auth_builder(
            api_key=api_key,
            jwt_handler=jwt_handler,
            request_data=request_data,
            general_settings=general_settings,
            route=route,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Verify that auth_jwt was called instead of get_oidc_userinfo
        mock_auth_jwt.assert_called_once_with(token=api_key)
        mock_get_userinfo.assert_not_called()  # Should not be called when OIDC is disabled

        # Verify the result
        assert result["user_id"] == "test_user_1"
        assert result["user_object"] == user_object


@pytest.mark.asyncio
async def test_auth_builder_oidc_enabled_falls_back_to_jwt_auth_for_jwt_tokens():
    """
    Regression test for the is_jwt routing fix.

    When oidc_userinfo_enabled=True but the supplied token is a well-formed
    JWT (three dot-separated parts), auth_builder must call auth_jwt and skip
    get_oidc_userinfo.  Sending a standard JWT to the OIDC UserInfo endpoint
    is incorrect — the endpoint expects an opaque access token.
    """
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    # Three-part token: recognised as a JWT by is_jwt()
    api_key = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0X3VzZXIifQ.some_signature"
    request_data = {"model": "gpt-4"}
    general_settings = {"enforce_rbac": False}
    route = "/chat/completions"

    user_object = LiteLLM_UserTable(
        user_id="test_user_1", user_role=LitellmUserRoles.INTERNAL_USER
    )

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            oidc_userinfo_enabled=True,
            oidc_userinfo_endpoint="https://example.com/oauth2/userinfo",
            user_id_jwt_field="sub",
        ),
    )

    jwt_response = {"sub": "test_user_1", "scope": ""}

    with (
        patch.object(
            jwt_handler, "get_oidc_userinfo", new_callable=AsyncMock
        ) as mock_get_userinfo,
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=("test_user_1", None, None),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
    ):
        mock_auth_jwt.return_value = jwt_response

        result = await JWTAuthManager.auth_builder(
            api_key=api_key,
            jwt_handler=jwt_handler,
            request_data=request_data,
            general_settings=general_settings,
            route=route,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Token is a JWT, so standard JWT auth must be used even when
        # oidc_userinfo_enabled is True.
        mock_auth_jwt.assert_called_once_with(token=api_key)
        mock_get_userinfo.assert_not_called()

        assert result["user_id"] == "test_user_1"
        assert result["user_object"] == user_object


def test_get_team_id_from_header():
    """Test get_team_id_from_header returns team when valid, None when missing, raises on invalid."""
    from fastapi import HTTPException

    # Valid team in allowed list
    result = JWTAuthManager.get_team_id_from_header(
        request_headers={"x-litellm-team-id": "team-1"},
        allowed_team_ids={"team-1", "team-2"},
    )
    assert result == "team-1"

    # No header returns None
    result = JWTAuthManager.get_team_id_from_header(
        request_headers={"authorization": "Bearer token"},
        allowed_team_ids={"team-1"},
    )
    assert result is None

    # Invalid team raises 403
    with pytest.raises(HTTPException) as exc_info:
        JWTAuthManager.get_team_id_from_header(
            request_headers={"x-litellm-team-id": "invalid-team"},
            allowed_team_ids={"team-1", "team-2"},
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_auth_builder_uses_team_from_header_e2e():
    """Test auth_builder e2e flow: selects team from x-litellm-team-id header."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="groups",
            user_id_jwt_field="sub",
        ),
    )

    team_object = LiteLLM_TeamTable(team_id="team-2")
    user_object = LiteLLM_UserTable(
        user_id="user-1", user_role=LitellmUserRoles.INTERNAL_USER
    )

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_team,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
    ):
        mock_auth_jwt.return_value = {
            "sub": "user-1",
            "scope": "",
            "groups": ["team-1", "team-2"],
        }
        mock_get_team.return_value = team_object

        result = await JWTAuthManager.auth_builder(
            api_key="jwt-token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            request_headers={"x-litellm-team-id": "team-2"},
        )

        assert result["team_id"] == "team-2"
        assert result["team_object"] == team_object


@pytest.mark.asyncio
async def test_auth_builder_header_team_denies_auth_passthrough_without_allowlist():
    """Header-selected JWT teams must enforce team allowed_passthrough_routes."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="groups",
            user_id_jwt_field="sub",
        ),
    )

    team_object = LiteLLM_TeamTable(team_id="team-2", metadata={})

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            return_value=team_object,
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
        ) as mock_get_objects,
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.is_auth_enforced_pass_through_route",
            return_value=True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.check_passthrough_route_access",
            return_value=False,
        ) as mock_passthrough_check,
    ):
        mock_auth_jwt.return_value = {
            "sub": "user-1",
            "scope": "",
            "groups": ["team-1", "team-2"],
        }

        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.auth_builder(
                api_key="jwt-token",
                jwt_handler=jwt_handler,
                request_data={"model": "gpt-4"},
                general_settings={},
                route="/my-pass-through",
                prisma_client=None,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
                request_headers={"x-litellm-team-id": "team-2"},
                request_method="POST",
            )

    assert exc_info.value.status_code == 403
    assert "allowed_passthrough_routes" in exc_info.value.detail
    mock_get_objects.assert_not_called()
    user_api_key_dict = mock_passthrough_check.call_args.kwargs["user_api_key_dict"]
    assert user_api_key_dict.team_metadata == {}


@pytest.mark.asyncio
async def test_auth_builder_specific_team_denies_auth_passthrough_without_allowlist():
    """JWT-field-selected teams must enforce team allowed_passthrough_routes."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_id_jwt_field="team_id",
            user_id_jwt_field="sub",
        ),
    )

    team_object = LiteLLM_TeamTable(team_id="team-1", metadata={})

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            return_value=team_object,
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
        ) as mock_get_objects,
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.is_auth_enforced_pass_through_route",
            return_value=True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.check_passthrough_route_access",
            return_value=False,
        ) as mock_passthrough_check,
    ):
        mock_auth_jwt.return_value = {
            "sub": "user-1",
            "scope": "",
            "team_id": "team-1",
        }

        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.auth_builder(
                api_key="jwt-token",
                jwt_handler=jwt_handler,
                request_data={"model": "gpt-4"},
                general_settings={},
                route="/my-pass-through",
                prisma_client=None,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
                request_method="POST",
            )

    assert exc_info.value.status_code == 403
    assert "allowed_passthrough_routes" in exc_info.value.detail
    mock_get_objects.assert_not_called()
    user_api_key_dict = mock_passthrough_check.call_args.kwargs["user_api_key_dict"]
    assert user_api_key_dict.team_metadata == {}


@pytest.mark.asyncio
async def test_auth_builder_rbac_team_loads_team_for_passthrough_allowlist():
    """RBAC role-claim teams (team_object unset) must load team metadata before gating."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    team_object = LiteLLM_TeamTable(
        team_id="team-rbac",
        metadata={"allowed_passthrough_routes": ["/my-pass-through"]},
    )

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(jwt_handler, "get_rbac_role", return_value=LitellmUserRoles.TEAM),
        patch.object(jwt_handler, "get_object_id", return_value="team-rbac"),
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            return_value=team_object,
        ) as mock_get_team,
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(None, None, None, None, None),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.is_auth_enforced_pass_through_route",
            return_value=True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.check_passthrough_route_access",
            return_value=True,
        ) as mock_passthrough_check,
    ):
        mock_auth_jwt.return_value = {"scope": ""}

        result = await JWTAuthManager.auth_builder(
            api_key="jwt-token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={},
            route="/my-pass-through",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            request_method="POST",
        )

    assert result["team_id"] == "team-rbac"
    mock_get_team.assert_awaited_once()
    assert mock_get_team.await_args.kwargs["team_id"] == "team-rbac"
    user_api_key_dict = mock_passthrough_check.call_args.kwargs["user_api_key_dict"]
    assert user_api_key_dict.team_metadata == {
        "allowed_passthrough_routes": ["/my-pass-through"]
    }


@pytest.mark.asyncio
async def test_auth_builder_rbac_team_denies_passthrough_without_allowlist():
    """RBAC role-claim teams without an allowlist are still denied for passthrough."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    team_object = LiteLLM_TeamTable(team_id="team-rbac", metadata={})

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(jwt_handler, "get_rbac_role", return_value=LitellmUserRoles.TEAM),
        patch.object(jwt_handler, "get_object_id", return_value="team-rbac"),
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            return_value=team_object,
        ) as mock_get_team,
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.is_auth_enforced_pass_through_route",
            return_value=True,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.check_passthrough_route_access",
            return_value=False,
        ),
    ):
        mock_auth_jwt.return_value = {"scope": ""}

        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.auth_builder(
                api_key="jwt-token",
                jwt_handler=jwt_handler,
                request_data={"model": "gpt-4"},
                general_settings={},
                route="/my-pass-through",
                prisma_client=None,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
                request_method="POST",
            )

    assert exc_info.value.status_code == 403
    assert "allowed_passthrough_routes" in exc_info.value.detail
    mock_get_team.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_builder_admin_on_llm_route_honors_team_header():
    """JWT proxy_admin + x-litellm-team-id on an LLM API route -> team context is
    attached to the admin result so team TPM/RPM limits and attribution apply."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="groups",
            user_id_jwt_field="sub",
            admin_allowed_routes=[
                "management_routes",
                "info_routes",
                "openai_routes",
            ],
        ),
    )

    team_object = LiteLLM_TeamTable(team_id="team-low", tpm_limit=100, rpm_limit=2)

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "is_admin", return_value=True),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_team,
    ):
        mock_auth_jwt.return_value = {
            "sub": "admin-user",
            "scope": "",
            "groups": [],
        }
        mock_get_team.return_value = team_object

        result = await JWTAuthManager.auth_builder(
            api_key="jwt-token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            request_headers={"x-litellm-team-id": "team-low"},
        )

        assert result["is_proxy_admin"] is True
        assert result["team_id"] == "team-low"
        assert result["team_object"] == team_object
        mock_get_team.assert_called_once()


@pytest.mark.asyncio
async def test_auth_builder_admin_on_mgmt_route_ignores_team_header():
    """JWT proxy_admin + x-litellm-team-id on an admin management route -> header
    is ignored; no team fetch. Preserves pre-existing bypass behavior and avoids
    phantom team creation when team_id_upsert is enabled."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="groups",
            user_id_jwt_field="sub",
            team_id_upsert=True,
        ),
    )

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "is_admin", return_value=True),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_team,
    ):
        mock_auth_jwt.return_value = {
            "sub": "admin-user",
            "scope": "",
            "groups": [],
        }

        result = await JWTAuthManager.auth_builder(
            api_key="jwt-token",
            jwt_handler=jwt_handler,
            request_data={},
            general_settings={},
            route="/user/info",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            request_headers={"x-litellm-team-id": "totally-made-up-team"},
        )

        assert result["is_proxy_admin"] is True
        assert result["team_id"] is None
        assert result["team_object"] is None
        mock_get_team.assert_not_called()


@pytest.mark.asyncio
async def test_auth_builder_admin_on_llm_route_without_header_unchanged():
    """JWT proxy_admin on an LLM API route without x-litellm-team-id -> no team
    context (team limits not applied, admin keeps unrestricted access)."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="groups",
            user_id_jwt_field="sub",
            admin_allowed_routes=[
                "management_routes",
                "info_routes",
                "openai_routes",
            ],
        ),
    )

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "is_admin", return_value=True),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_team,
    ):
        mock_auth_jwt.return_value = {
            "sub": "admin-user",
            "scope": "",
            "groups": [],
        }

        result = await JWTAuthManager.auth_builder(
            api_key="jwt-token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
            request_headers={},
        )

        assert result["is_proxy_admin"] is True
        assert result["team_id"] is None
        assert result["team_object"] is None
        mock_get_team.assert_not_called()


@pytest.mark.asyncio
async def test_get_team_alias_with_nested_fields():
    """
    Test get_team_alias() method with nested JWT fields
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    jwt_handler = JWTHandler()

    # Test token with nested team name
    nested_token = {
        "organization": {"team": {"name": "engineering-team"}},
        "team_name": "flat-team",
    }

    # Test nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_alias_jwt_field="organization.team.name"
    )
    assert jwt_handler.get_team_alias(nested_token, None) == "engineering-team"

    # Test flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_alias_jwt_field="team_name")
    assert jwt_handler.get_team_alias(nested_token, None) == "flat-team"

    # Test missing field returns default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_alias_jwt_field="nonexistent.field"
    )
    assert jwt_handler.get_team_alias(nested_token, "default-team") == "default-team"

    # Test with team_alias_jwt_field not configured
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()  # team_alias_jwt_field is None
    assert jwt_handler.get_team_alias(nested_token, "default") is None


@pytest.mark.asyncio
async def test_is_required_team_id_with_team_alias_field():
    """
    Test that is_required_team_id() returns True when team_alias_jwt_field is set
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    jwt_handler = JWTHandler()

    # Neither field set - should return False
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()
    assert jwt_handler.is_required_team_id() is False

    # Only team_id_jwt_field set - should return True
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="team_id")
    assert jwt_handler.is_required_team_id() is True

    # Only team_alias_jwt_field set - should return True
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_alias_jwt_field="team_name")
    assert jwt_handler.is_required_team_id() is True

    # Both fields set - should return True
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_id_jwt_field="team_id", team_alias_jwt_field="team_name"
    )
    assert jwt_handler.is_required_team_id() is True


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_with_team_alias():
    """
    Test that find_and_validate_specific_team_id resolves team by name when team_id is not found
    """
    from unittest.mock import MagicMock

    from litellm.caching import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_TeamTable
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(team_alias_jwt_field="team_alias"),
    )

    # Token with team name (no team_id)
    jwt_token = {"sub": "user-1", "team_alias": "my-team"}

    # Mock team object returned by get_team_object_by_alias
    team_object = LiteLLM_TeamTable(team_id="resolved-team-id", team_alias="my-team")

    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object_by_alias", new_callable=AsyncMock
    ) as mock_get_by_alias:
        mock_get_by_alias.return_value = team_object

        team_id, result_team = await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=jwt_handler,
            jwt_valid_token=jwt_token,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Should have resolved team_id from team name
        assert team_id == "resolved-team-id"
        assert result_team == team_object
        mock_get_by_alias.assert_called_once_with(
            team_alias="my-team",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_find_and_validate_team_id_takes_precedence_over_name():
    """
    Test that team_id_jwt_field takes precedence over team_alias_jwt_field
    """
    from unittest.mock import MagicMock

    from litellm.caching import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_TeamTable
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_id_jwt_field="team_id", team_alias_jwt_field="team_alias"
        ),
    )

    # Token with both team_id and team name
    jwt_token = {"sub": "user-1", "team_id": "direct-team-id", "team_alias": "my-team"}

    # Mock team object returned by get_team_object (by ID)
    team_object = LiteLLM_TeamTable(team_id="direct-team-id")

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_by_id,
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object_by_alias",
            new_callable=AsyncMock,
        ) as mock_get_by_alias,
    ):
        mock_get_by_id.return_value = team_object

        team_id, result_team = await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=jwt_handler,
            jwt_valid_token=jwt_token,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Should use team_id directly, not resolve by name
        assert team_id == "direct-team-id"
        assert result_team == team_object
        mock_get_by_id.assert_called_once()
        mock_get_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_find_and_validate_raises_when_required_team_not_found():
    """
    Test that an exception is raised when team is required but neither team_id nor team_name is found
    """
    from litellm.caching import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_alias_jwt_field="team_alias"  # Required, but not in token
        ),
    )

    # Token without team info
    jwt_token = {"sub": "user-1"}

    with pytest.raises(Exception) as exc_info:
        await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=jwt_handler,
            jwt_valid_token=jwt_token,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert "No team found in token" in str(exc_info.value)
    assert "team_alias field 'team_alias'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_org_alias_with_nested_fields():
    """
    Test get_org_alias() method with nested JWT fields
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    jwt_handler = JWTHandler()

    # Test token with nested org name
    nested_token = {
        "company": {"organization": {"name": "acme-corp"}},
        "org_name": "flat-org",
    }

    # Test nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        org_alias_jwt_field="company.organization.name"
    )
    assert jwt_handler.get_org_alias(nested_token, None) == "acme-corp"

    # Test flat access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_alias_jwt_field="org_name")
    assert jwt_handler.get_org_alias(nested_token, None) == "flat-org"

    # Test missing field returns default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        org_alias_jwt_field="nonexistent.field"
    )
    assert jwt_handler.get_org_alias(nested_token, "default-org") == "default-org"

    # Test with org_alias_jwt_field not configured
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()
    assert jwt_handler.get_org_alias(nested_token, "default") is None


@pytest.mark.asyncio
async def test_get_objects_resolves_org_by_name():
    """
    Test that get_objects resolves organization by name when org_id is not provided
    """
    from litellm.caching import DualCache
    from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_OrganizationTable
    from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)

    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(org_alias_jwt_field="org_alias"),
    )

    # Mock org object returned by get_org_object_by_alias
    org_object = LiteLLM_OrganizationTable(
        organization_id="resolved-org-id",
        organization_alias="my-org",
        budget_id="budget-1",
        created_by="admin",
        updated_by="admin",
        models=[],
    )

    with patch(
        "litellm.proxy.auth.handle_jwt.get_org_object_by_alias", new_callable=AsyncMock
    ) as mock_get_by_alias:
        mock_get_by_alias.return_value = org_object

        (
            result_user_obj,
            result_org_obj,
            result_end_user_obj,
            result_team_membership,
            _result_user_id,
        ) = await JWTAuthManager.get_objects(
            user_id=None,
            user_email=None,
            org_id=None,  # No org_id provided
            end_user_id=None,
            team_id=None,
            valid_user_email=None,
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
            route="/chat/completions",
            org_alias="my-org",
        )

        # Should resolve org by alias - org_id can be derived from org_object.organization_id
        assert result_org_obj == org_object
        assert result_org_obj.organization_id == "resolved-org-id"
        mock_get_by_alias.assert_called_once_with(
            org_alias="my-org",
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )


# ---------------------------------------------------------------------------
# Fix 1: OIDC discovery URL resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_jwks_url_passthrough_for_direct_jwks_url():
    """Non-discovery URLs are returned unchanged."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = JWTHandler()
    handler.update_environment(
        prisma_client=None,
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )
    url = "https://login.microsoftonline.com/common/discovery/keys"
    result = await handler._resolve_jwks_url(url)
    assert result == url


@pytest.mark.asyncio
async def test_resolve_jwks_url_resolves_oidc_discovery_document():
    """
    A .well-known/openid-configuration URL should be fetched and its
    jwks_uri returned.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = JWTHandler()
    cache = DualCache()
    handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    discovery_url = (
        "https://login.microsoftonline.com/tenant/.well-known/openid-configuration"
    )
    jwks_url = "https://login.microsoftonline.com/tenant/discovery/keys"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jwks_uri": jwks_url, "issuer": "https://..."}

    mock_get = AsyncMock(return_value=mock_response)
    handler.http_handler.get = mock_get

    result = await handler._resolve_jwks_url(discovery_url)

    assert result == jwks_url
    mock_get.assert_called_once_with(discovery_url)


@pytest.mark.asyncio
async def test_resolve_jwks_url_caches_resolved_jwks_uri():
    """Resolved jwks_uri is cached — second call does not hit the network."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = JWTHandler()
    cache = DualCache()
    handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    discovery_url = (
        "https://login.microsoftonline.com/tenant/.well-known/openid-configuration"
    )
    jwks_url = "https://login.microsoftonline.com/tenant/discovery/keys"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jwks_uri": jwks_url}

    mock_get = AsyncMock(return_value=mock_response)
    handler.http_handler.get = mock_get

    first = await handler._resolve_jwks_url(discovery_url)
    second = await handler._resolve_jwks_url(discovery_url)

    assert first == jwks_url
    assert second == jwks_url
    # Network should only be hit once
    assert mock_get.call_count == 1


@pytest.mark.asyncio
async def test_resolve_jwks_url_raises_if_no_jwks_uri_in_discovery_doc():
    """Raise a helpful error if the discovery document has no jwks_uri."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = JWTHandler()
    handler.update_environment(
        prisma_client=None,
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    discovery_url = "https://example.com/.well-known/openid-configuration"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"issuer": "https://example.com"}  # no jwks_uri

    handler.http_handler.get = AsyncMock(return_value=mock_response)

    with pytest.raises(Exception, match="jwks_uri"):
        await handler._resolve_jwks_url(discovery_url)


# ---------------------------------------------------------------------------
# Fix 2: handle array values in team_id_jwt_field (e.g. AAD "roles" claim)
# ---------------------------------------------------------------------------


def _make_jwt_handler(team_id_jwt_field: str) -> JWTHandler:
    from litellm.caching.dual_cache import DualCache

    handler = JWTHandler()
    handler.update_environment(
        prisma_client=None,
        user_api_key_cache=DualCache(),
        litellm_jwtauth=LiteLLM_JWTAuth(team_id_jwt_field=team_id_jwt_field),
    )
    return handler


def test_get_team_id_returns_first_element_when_roles_is_list():
    """
    AAD sends roles as a list.  get_team_id() must return the first string
    element rather than the raw list (which would later crash with
    'unhashable type: list').
    """
    handler = _make_jwt_handler("roles")
    token = {"oid": "user-oid", "roles": ["team1"]}
    result = handler.get_team_id(token=token, default_value=None)
    assert result == "team1"


def test_get_team_id_returns_first_element_from_multi_value_roles_list():
    """When roles has multiple entries, the first one is used."""
    handler = _make_jwt_handler("roles")
    token = {"roles": ["team2", "team1"]}
    result = handler.get_team_id(token=token, default_value=None)
    assert result == "team2"


def test_get_team_id_returns_default_when_roles_list_is_empty():
    """Empty list should fall back to default_value."""
    handler = _make_jwt_handler("roles")
    token = {"roles": []}
    result = handler.get_team_id(token=token, default_value="fallback")
    assert result == "fallback"


def test_get_team_id_still_works_with_string_value():
    """String values (non-array) continue to work as before."""
    handler = _make_jwt_handler("appid")
    token = {"appid": "my-team-id"}
    result = handler.get_team_id(token=token, default_value=None)
    assert result == "my-team-id"


def test_get_team_id_list_result_is_hashable():
    """
    The value returned by get_team_id() must be hashable so it can be
    added to a set (the operation that previously crashed).
    """
    handler = _make_jwt_handler("roles")
    token = {"roles": ["team1"]}
    result = handler.get_team_id(token=token, default_value=None)
    # This must not raise TypeError
    s: set = set()
    s.add(result)
    assert "team1" in s


# ---------------------------------------------------------------------------
# Fix 3: helpful error message for dot-notation array indexing (roles.0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_hints_bracket_notation():
    """
    When team_id_jwt_field is set to 'roles.0' (unsupported dot-notation for
    array indexing) and no team is found, the exception message should suggest
    using 'roles' instead (and explain LiteLLM auto-unwraps list values).
    """
    from unittest.mock import MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = _make_jwt_handler("roles.0")
    # token has roles as a list — dot-notation won't find anything
    token = {"roles": ["team1"]}

    with pytest.raises(Exception) as exc_info:
        await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=handler,
            jwt_valid_token=token,
            prisma_client=None,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    error_msg = str(exc_info.value)
    # Should mention the bad field name and suggest the fix
    assert "roles.0" in error_msg, f"Expected field name in: {error_msg}"
    assert "roles" in error_msg and "list" in error_msg, (
        f"Expected hint about using 'roles' instead: {error_msg}"
    )


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_hints_bracket_index_notation():
    """
    When team_id_jwt_field is set to 'roles[0]' (bracket indexing, also unsupported
    in get_nested_value) the error message should suggest using 'roles' instead.
    """
    from unittest.mock import MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = _make_jwt_handler("roles[0]")
    token = {"roles": ["team1"]}

    with pytest.raises(Exception) as exc_info:
        await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=handler,
            jwt_valid_token=token,
            prisma_client=None,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    error_msg = str(exc_info.value)
    assert "roles[0]" in error_msg, f"Expected field name in: {error_msg}"
    assert "roles" in error_msg and "list" in error_msg, (
        f"Expected hint about using 'roles' instead: {error_msg}"
    )


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_no_hint_for_valid_field():
    """
    When team_id_jwt_field is a normal field name (no dot-notation) the
    error message should not contain a spurious bracket-notation hint.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache

    handler = _make_jwt_handler("appid")
    token = {}  # no appid — triggers the "no team found" path

    with pytest.raises(Exception) as exc_info:
        await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=handler,
            jwt_valid_token=token,
            prisma_client=None,
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    error_msg = str(exc_info.value)
    assert "Hint" not in error_msg


# ---------------------------------------------------------------------------
# Single-team DB fallback when JWT does not resolve team_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "user_id",
        "user_teams",
        "get_team_object_return",
        "expected_team_id",
        "expect_get_team_called",
        "expect_get_membership_called",
    ),
    [
        pytest.param(
            "user_single_team_fb",
            ["team_only_fb"],
            "resolved_row",
            "team_only_fb",
            True,
            True,
            id="one_db_team_resolves_team_and_membership",
        ),
        pytest.param(
            "user_multi_team_fb",
            ["team_a", "team_b"],
            "unused",
            None,
            False,
            False,
            id="two_db_teams_ambiguous_no_fallback",
        ),
        pytest.param(
            "user_zero_teams_fb",
            [],
            "unused",
            None,
            False,
            False,
            id="zero_db_teams_no_fallback",
        ),
        pytest.param(
            "user_orphan_team_fb",
            ["team_missing_in_db"],
            "http_404",
            None,
            True,
            False,
            id="one_team_id_but_row_missing_in_db",
        ),
        pytest.param(
            "user_orphan_team_non404_fb",
            ["team_err"],
            "http_500",
            None,
            True,
            False,
            id="get_team_object_raises_non404_still_no_raise",
        ),
    ],
)
@pytest.mark.asyncio
async def test_auth_builder_single_team_db_fallback_when_jwt_has_no_team(
    user_id: str,
    user_teams: list,
    get_team_object_return: Optional[str],
    expected_team_id: Optional[str],
    expect_get_team_called: bool,
    expect_get_membership_called: bool,
) -> None:
    """
    JWT does not set team_id (mocks return no team from token/header/routing). Behavior:
    - exactly one team on user + get_team_object returns a row -> set team + membership
    - two+ teams, or zero teams -> no get_team_object / no membership
    - one team id but get_team_object raises (e.g. 404/500) -> skip fallback, no team, no error
    """
    if len(user_teams) == 1 and get_team_object_return == "resolved_row":
        only = user_teams[0]
        team_table = LiteLLM_TeamTable(team_id=only)
        membership = LiteLLM_TeamMembership(
            user_id=user_id, team_id=only, litellm_budget_table=None
        )
        get_team_return_value = team_table
        membership_return_value = membership
    else:
        team_table = None
        membership = None
        get_team_return_value = None
        membership_return_value = None
    # "http_404" / "http_500" use get_team_object.side_effect, not return_value

    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=user_teams,
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
        ) as mock_get_team,
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
        ) as mock_get_membership,
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        if get_team_object_return in ("http_404", "http_500"):
            from fastapi import HTTPException

            code = 404 if get_team_object_return == "http_404" else 500
            mock_get_team.side_effect = HTTPException(
                status_code=code,
                detail={
                    "error": f"Team doesn't exist in db. Team={user_teams[0]}. Create team via `/team/new` call."
                },
            )
        else:
            mock_get_team.return_value = get_team_return_value
        if membership_return_value is not None:
            mock_get_membership.return_value = membership_return_value

        result = await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        assert result["team_id"] == expected_team_id
        if expected_team_id is not None:
            assert result["team_object"] == team_table
            assert result["team_membership"] == membership
        else:
            assert result["team_object"] is None
            if not expect_get_membership_called:
                assert result["team_membership"] is None

        if expect_get_team_called:
            mock_get_team.assert_called()
        else:
            mock_get_team.assert_not_called()
        if expect_get_membership_called:
            mock_get_membership.assert_called_once()
        else:
            mock_get_membership.assert_not_called()


@pytest.mark.asyncio
async def test_auth_builder_single_team_fallback_membership_error_skips_no_raise():
    """
    get_team_object succeeds but get_team_membership raises — do not set team; no exception.
    """
    from fastapi import HTTPException

    user_id = "u_mem_fail"
    team_id_val = "team_mem_fail"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=[team_id_val],
    )
    team_table = LiteLLM_TeamTable(team_id=team_id_val)
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager,
            "find_and_validate_specific_team_id",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
        patch.object(
            JWTAuthManager,
            "find_team_with_model_access",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
        ) as mock_get_team,
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
        ) as mock_get_membership,
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        mock_get_team.return_value = team_table
        mock_get_membership.side_effect = HTTPException(
            status_code=500, detail="membership lookup failed"
        )

        result = await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        assert result["team_id"] is None
        assert result["team_object"] is None
        assert result["team_membership"] is None
        mock_get_team.assert_called()
        mock_get_membership.assert_called_once()


# ---------------------------------------------------------------------------
# JWTHandler._build_decode_kwargs — VERIA-27 (audience + issuer verification)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _reset_unscoped_warning_flag():
    """Reset the once-per-process warning sentinel so each test sees a fresh
    state."""
    JWTHandler._unscoped_jwt_warning_emitted = False
    yield
    JWTHandler._unscoped_jwt_warning_emitted = False


def test_build_decode_kwargs_no_env_disables_both_verifications(
    monkeypatch, _reset_unscoped_warning_flag
):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_ISSUER", raising=False)

    kwargs = JWTHandler._build_decode_kwargs()

    assert kwargs["audience"] is None
    assert kwargs["issuer"] is None
    assert kwargs["options"] == {"verify_aud": False, "verify_iss": False}


def test_build_decode_kwargs_audience_only_enables_aud_verification(
    monkeypatch, _reset_unscoped_warning_flag
):
    monkeypatch.setenv("JWT_AUDIENCE", "my-proxy")
    monkeypatch.delenv("JWT_ISSUER", raising=False)

    kwargs = JWTHandler._build_decode_kwargs()

    assert kwargs["audience"] == "my-proxy"
    assert kwargs["issuer"] is None
    # verify_aud not in options means PyJWT will verify audience
    assert kwargs["options"] == {"verify_iss": False}


def test_build_decode_kwargs_issuer_only_enables_iss_verification(
    monkeypatch, _reset_unscoped_warning_flag
):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.setenv("JWT_ISSUER", "https://idp.example.com/")

    kwargs = JWTHandler._build_decode_kwargs()

    assert kwargs["audience"] is None
    assert kwargs["issuer"] == "https://idp.example.com/"
    assert kwargs["options"] == {"verify_aud": False}


def test_build_decode_kwargs_both_set_enables_full_verification(
    monkeypatch, _reset_unscoped_warning_flag
):
    monkeypatch.setenv("JWT_AUDIENCE", "my-proxy")
    monkeypatch.setenv("JWT_ISSUER", "https://idp.example.com/")

    kwargs = JWTHandler._build_decode_kwargs()

    assert kwargs["audience"] == "my-proxy"
    assert kwargs["issuer"] == "https://idp.example.com/"
    # No verification opt-outs — PyJWT verifies both claims by default.
    assert kwargs["options"] is None


def test_build_decode_kwargs_warns_once_when_unscoped(
    monkeypatch, _reset_unscoped_warning_flag, caplog
):
    """The warning about unscoped JWT auth should fire on the first call but
    not on every subsequent decode."""
    import logging

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    caplog.set_level(logging.WARNING)

    JWTHandler._build_decode_kwargs()
    JWTHandler._build_decode_kwargs()
    JWTHandler._build_decode_kwargs()

    matching = [
        r
        for r in caplog.records
        if "JWT auth is enabled" in r.getMessage()
        and "neither JWT_AUDIENCE nor JWT_ISSUER" in r.getMessage()
    ]
    assert len(matching) == 1, (
        f"Expected exactly one warning across 3 calls, got {len(matching)}"
    )


def test_build_decode_kwargs_no_warning_when_scoped(
    monkeypatch, _reset_unscoped_warning_flag, caplog
):
    import logging

    monkeypatch.setenv("JWT_AUDIENCE", "my-proxy")
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    caplog.set_level(logging.WARNING)

    JWTHandler._build_decode_kwargs()

    matching = [
        r
        for r in caplog.records
        if "neither JWT_AUDIENCE nor JWT_ISSUER" in r.getMessage()
    ]
    assert matching == []


# ---------------------------------------------------------------------------
# Defer to single-team DB fallback (PR #26418) when JWT claims are present
# but do not resolve to a LiteLLM team.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_unresolved_claim_returns_none():
    """With `team_claim_fallback=True`: team_id claim is present in the JWT
    but the team is missing in the DB — return (None, None) so the
    auth_builder single-team fallback can run, instead of raising and
    failing auth."""
    from fastapi import HTTPException

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_id_jwt_field="team_id",
        team_claim_fallback=True,
    )
    token = {"sub": "user-1", "team_id": "claim-team-not-in-db"}

    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object",
        new_callable=AsyncMock,
    ) as mock_get_team:
        mock_get_team.side_effect = HTTPException(status_code=404, detail="missing")

        team_id, team_object = await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=jwt_handler,
            jwt_valid_token=token,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

    assert team_id is None
    assert team_object is None


@pytest.mark.asyncio
async def test_find_team_with_model_access_unresolved_group_claim_returns_none(
    monkeypatch,
):
    """With `team_claim_fallback=True`: group claim resolves to team_ids that
    don't exist in the DB — return (None, None) instead of raising 403, so
    the single-team fallback can run."""
    import sys
    import types

    from fastapi import HTTPException

    from litellm.router import Router

    router = Router(
        model_list=[
            {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}
        ]
    )
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    async def raise_404(*_args, **_kwargs):
        raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr("litellm.proxy.auth.handle_jwt.get_team_object", raise_404)

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_claim_fallback=True)

    team_id, team_object = await JWTAuthManager.find_team_with_model_access(
        team_ids={"idp-group-a", "idp-group-b"},
        requested_model="gpt-4o-mini",
        route="/chat/completions",
        jwt_handler=jwt_handler,
        prisma_client=None,
        user_api_key_cache=None,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )

    assert team_id is None
    assert team_object is None


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_non_http_exception_still_propagates():
    """Regression guard: only the 404 HTTPException raised by
    `get_team_object` ("team doesn't exist in db") is softened. Other
    errors — e.g. "No DB Connected" — must still propagate so operator-side
    problems are loud."""
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="team_id")
    token = {"sub": "user-1", "team_id": "some-claim-team"}

    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object",
        new_callable=AsyncMock,
    ) as mock_get_team:
        mock_get_team.side_effect = RuntimeError("simulated infrastructure error")

        with pytest.raises(RuntimeError, match="simulated infrastructure error"):
            await JWTAuthManager.find_and_validate_specific_team_id(
                jwt_handler=jwt_handler,
                jwt_valid_token=token,
                prisma_client=None,
                user_api_key_cache=None,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_non_404_http_exception_propagates():
    """Regression guard: only 404 HTTPException is softened. If
    `get_team_object` is ever updated to raise a different HTTP status code
    (e.g. 403 for a blocked team), that error must still propagate rather
    than silently fall through to the single-team DB fallback."""
    from fastapi import HTTPException

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="team_id")
    token = {"sub": "user-1", "team_id": "some-claim-team"}

    for status_code in (400, 403, 500):
        with patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
        ) as mock_get_team:
            mock_get_team.side_effect = HTTPException(
                status_code=status_code, detail="non-404 failure"
            )

            with pytest.raises(HTTPException) as exc_info:
                await JWTAuthManager.find_and_validate_specific_team_id(
                    jwt_handler=jwt_handler,
                    jwt_valid_token=token,
                    prisma_client=None,
                    user_api_key_cache=None,
                    parent_otel_span=None,
                    proxy_logging_obj=None,
                )
            assert exc_info.value.status_code == status_code


@pytest.mark.asyncio
async def test_find_team_with_model_access_enforce_team_based_access_still_raises():
    """Regression guard: when no group claims are present and
    `enforce_team_based_model_access` is on, the original 403 still fires —
    the new soft-fail only applies to the unresolved-claim path inside the
    loop, not to the no-team-claims-at-all path at the top."""
    from fastapi import HTTPException

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(enforce_team_based_model_access=True)

    with pytest.raises(HTTPException) as exc_info:
        await JWTAuthManager.find_team_with_model_access(
            team_ids=set(),
            requested_model="gpt-4o-mini",
            route="/chat/completions",
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

    assert exc_info.value.status_code == 403
    assert "enforce_team_based_model_access" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_find_team_with_model_access_resolved_team_without_model_still_raises_403(
    monkeypatch,
):
    """Regression guard: when the JWT group claim DOES resolve to a real
    LiteLLM team but that team does not grant the requested model, keep the
    original 403. Only the unresolved-claim case is softened."""
    import sys
    import types

    from fastapi import HTTPException

    from litellm.router import Router

    router = Router(
        model_list=[
            {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
        ]
    )
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    team = LiteLLM_TeamTable(team_id="real-team", models=["gpt-3.5-turbo"])

    async def mock_get_team_object(*_args, **_kwargs):
        return team

    monkeypatch.setattr(
        "litellm.proxy.auth.handle_jwt.get_team_object", mock_get_team_object
    )

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    with pytest.raises(HTTPException) as exc_info:
        await JWTAuthManager.find_team_with_model_access(
            team_ids={"real-team"},
            requested_model="gpt-4o-mini",
            route="/chat/completions",
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

    assert exc_info.value.status_code == 403
    assert "No team has access to the requested model" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_unresolved_claim_default_raises():
    """Default `team_claim_fallback=False`: unresolved team_id claim must
    still raise — preserves the strict claim-based authorization boundary
    when the operator has not opted in to the fallback."""
    from fastapi import HTTPException

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_jwt_field="team_id")
    token = {"sub": "user-1", "team_id": "claim-team-not-in-db"}

    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object",
        new_callable=AsyncMock,
    ) as mock_get_team:
        mock_get_team.side_effect = HTTPException(status_code=404, detail="missing")

        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.find_and_validate_specific_team_id(
                jwt_handler=jwt_handler,
                jwt_valid_token=token,
                prisma_client=None,
                user_api_key_cache=None,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_find_team_with_model_access_unresolved_group_claim_default_raises(
    monkeypatch,
):
    """Default `team_claim_fallback=False`: group claims that don't resolve
    to any LiteLLM team must still raise 403 — preserves the strict
    claim-based authorization boundary."""
    import sys
    import types

    from fastapi import HTTPException

    from litellm.router import Router

    router = Router(
        model_list=[
            {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}
        ]
    )
    proxy_server_module = types.ModuleType("proxy_server")
    proxy_server_module.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    async def raise_404(*_args, **_kwargs):
        raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr("litellm.proxy.auth.handle_jwt.get_team_object", raise_404)

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    with pytest.raises(HTTPException) as exc_info:
        await JWTAuthManager.find_team_with_model_access(
            team_ids={"idp-group-a", "idp-group-b"},
            requested_model="gpt-4o-mini",
            route="/chat/completions",
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

    assert exc_info.value.status_code == 403


# GH #26789: JWT claim user_id must rebind to legacy DB row after fuzzy match.


def test_canonical_user_id_rebinds_to_legacy_uuid():
    """JWT email resolves to a legacy UUID row -> use the UUID for attribution."""
    legacy_uuid = "bb8ab11f-09aa-47ae-b063-6e80506ac3bc"
    jwt_email = "matt@example.com"
    user_object = LiteLLM_UserTable(user_id=legacy_uuid, user_email=jwt_email)

    assert (
        JWTAuthManager._canonical_user_id_from_db(
            user_id=jwt_email, user_object=user_object
        )
        == legacy_uuid
    )


def test_canonical_user_id_no_change_when_ids_match():
    """Fresh upserted user (row.user_id == claim) -> claim returned unchanged."""
    same = "alice@example.com"
    user_object = LiteLLM_UserTable(user_id=same, user_email=same)

    assert (
        JWTAuthManager._canonical_user_id_from_db(user_id=same, user_object=user_object)
        == same
    )


def test_canonical_user_id_returns_claim_when_no_user_object():
    """No resolved row (e.g. upsert disabled / brand new) -> keep the claim."""
    assert (
        JWTAuthManager._canonical_user_id_from_db(
            user_id="newcomer@example.com", user_object=None
        )
        == "newcomer@example.com"
    )


def test_canonical_user_id_returns_none_when_claim_none_and_no_object():
    """Defensive: no claim and no row -> stays None, never invents an id."""
    assert (
        JWTAuthManager._canonical_user_id_from_db(user_id=None, user_object=None)
        is None
    )


def test_canonical_user_id_no_change_when_db_user_id_falsy():
    """Defensive: an empty user_object.user_id must not clobber the claim."""

    class _Stub:
        user_id = ""

    assert (
        JWTAuthManager._canonical_user_id_from_db(
            user_id="jwt@example.com", user_object=_Stub()
        )
        == "jwt@example.com"
    )


@pytest.mark.asyncio
async def test_auth_jwt_expired_token_raises_401_jwk_path():
    """An expired JWT (access token) decoded via the JWK/dict public-key path
    must raise a ProxyException carrying a 401 status code so the status is
    preserved end-to-end (client response + OTel traces).
    """
    import jwt as jwt_lib

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    with (
        patch.object(
            jwt_handler, "get_public_key", new_callable=AsyncMock
        ) as mock_get_public_key,
        patch(
            "litellm.proxy.auth.handle_jwt.jwt.get_unverified_header",
            return_value={"kid": "test-kid"},
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.PyJWK.from_dict",
            return_value=MagicMock(key="fake-key"),
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.jwt.decode",
            side_effect=jwt_lib.ExpiredSignatureError("Signature has expired"),
        ),
    ):
        mock_get_public_key.return_value = {"kty": "RSA", "kid": "test-kid"}

        with pytest.raises(ProxyException) as exc_info:
            await jwt_handler.auth_jwt(token="expired.jwt.token")

        assert exc_info.value.code == str(401)
        assert exc_info.value.type == ProxyErrorTypes.expired_key.value
        assert "Token Expired" in exc_info.value.message


@pytest.mark.asyncio
async def test_auth_jwt_expired_token_raises_401_pem_cert_path():
    """Same as above but for the PEM-certificate (string public-key) decode path."""
    import jwt as jwt_lib

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth()

    mock_cert = MagicMock()
    mock_cert.public_key.return_value.public_bytes.return_value = b"fake-key"

    with (
        patch.object(
            jwt_handler, "get_public_key", new_callable=AsyncMock
        ) as mock_get_public_key,
        patch(
            "litellm.proxy.auth.handle_jwt.jwt.get_unverified_header",
            return_value={"kid": "test-kid"},
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.x509.load_pem_x509_certificate",
            return_value=mock_cert,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.jwt.decode",
            side_effect=jwt_lib.ExpiredSignatureError("Signature has expired"),
        ),
    ):
        mock_get_public_key.return_value = (
            "-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----"
        )

        with pytest.raises(ProxyException) as exc_info:
            await jwt_handler.auth_jwt(token="expired.jwt.token")

        assert exc_info.value.code == str(401)
        assert exc_info.value.type == ProxyErrorTypes.expired_key.value
        assert "Token Expired" in exc_info.value.message


def _base64url_encode_int(value: int) -> str:
    import base64

    value_bytes = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(value_bytes).decode("utf-8").rstrip("=")


def _get_rsa_key_and_jwk(kid: str):
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "n": _base64url_encode_int(value=public_numbers.n),
        "e": _base64url_encode_int(value=public_numbers.e),
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
    }
    return private_key, jwk


def _encode_rsa_jwt(
    private_key,
    issuer: str,
    audience: str,
    kid: str,
    extra_claims: Optional[dict] = None,
) -> str:
    import time

    import jwt
    from cryptography.hazmat.primitives import serialization

    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    current_time = int(time.time())
    claims = {
        "sub": "test-subject",
        "iss": issuer,
        "aud": audience,
        "iat": current_time,
        "exp": current_time + 300,
    }
    if extra_claims:
        claims.update(extra_claims)

    return jwt.encode(
        claims,
        private_key_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )


def _get_jwt_handler_with_issuer_keys(issuers: list, keys_by_url: dict) -> JWTHandler:
    from litellm.caching.dual_cache import DualCache

    cache = DualCache()
    for jwks_url, keys in keys_by_url.items():
        cache.set_cache(
            key=f"litellm_jwt_auth_keys_{jwks_url}",
            value=keys,
        )

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(issuers=issuers),
    )
    return jwt_handler


@pytest.mark.asyncio
async def test_get_public_key_fetches_and_caches_jwks_response():
    from unittest.mock import AsyncMock, MagicMock

    from litellm.caching.dual_cache import DualCache

    jwt_handler = JWTHandler()
    cache = DualCache()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(public_key_ttl=123),
    )
    expected_key_id = "cached-key"
    _, jwk = _get_rsa_key_and_jwk(kid=expected_key_id)
    mock_response = MagicMock()
    mock_response.json.return_value = {"keys": [jwk]}
    jwt_handler.http_handler.get = AsyncMock(return_value=mock_response)

    public_key = await jwt_handler._get_public_key_from_jwks_url(
        jwks_url="https://issuer.example.com/keys",
        kid=expected_key_id,
    )

    assert public_key == jwk
    cached_keys = await cache.async_get_cache(
        key="litellm_jwt_auth_keys_https://issuer.example.com/keys"
    )
    assert cached_keys == [jwk]


@pytest.mark.asyncio
async def test_get_public_key_tries_next_jwks_url_when_kid_missing(monkeypatch):
    from litellm.caching.dual_cache import DualCache

    first_jwks_url = "https://first.example.com/keys"
    second_jwks_url = "https://second.example.com/keys"
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", f"{first_jwks_url}, {second_jwks_url},,")
    _, first_jwk = _get_rsa_key_and_jwk(kid="first-key")
    _, second_jwk = _get_rsa_key_and_jwk(kid="second-key")
    cache = DualCache()
    cache.set_cache(key=f"litellm_jwt_auth_keys_{first_jwks_url}", value=[first_jwk])
    cache.set_cache(key=f"litellm_jwt_auth_keys_{second_jwks_url}", value=[second_jwk])
    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(),
    )

    public_key = await jwt_handler.get_public_key(kid="second-key")

    assert public_key == second_jwk


def test_get_jwks_url_for_issuer_falls_back_to_discovery_document():
    jwt_handler = JWTHandler()
    issuer_config = LiteLLM_JWTAuth(
        issuers=[
            {
                "issuer": "https://issuer.example.com/tenant/",
                "disable_audience_validation": True,
            }
        ]
    ).issuers[0]

    jwks_url = jwt_handler._get_jwks_url_for_issuer(issuer_config=issuer_config)

    assert (
        jwks_url == "https://issuer.example.com/tenant/.well-known/openid-configuration"
    )


@pytest.mark.asyncio
async def test_get_objects_team_membership_uses_rebound_user_id():
    """team_membership lookup uses resolved DB user_id, not JWT email claim."""
    from litellm.caching.caching import DualCache

    legacy_uuid = "bb8ab11f-09aa-47ae-b063-6e80506ac3bc"
    jwt_email = "matt@example.com"
    team_id = "team-1"

    resolved_user = LiteLLM_UserTable(user_id=legacy_uuid, user_email=jwt_email)
    captured = {}

    async def fake_get_user_object(*args, **kwargs):
        return resolved_user

    async def fake_get_team_membership(user_id, team_id, *args, **kwargs):
        captured["user_id"] = user_id
        captured["team_id"] = team_id
        return None

    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_id_jwt_field="email", user_id_upsert=True
    )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_user_object",
            side_effect=fake_get_user_object,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            side_effect=fake_get_team_membership,
        ),
    ):
        (
            user_object,
            _org_object,
            _end_user_object,
            _team_membership_object,
            effective_user_id,
        ) = await JWTAuthManager.get_objects(
            user_id=jwt_email,
            user_email=jwt_email,
            org_id=None,
            end_user_id=None,
            team_id=team_id,
            valid_user_email=None,
            jwt_handler=jwt_handler,
            prisma_client=MagicMock(),
            user_api_key_cache=DualCache(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
            route="/chat/completions",
        )

    assert user_object is not None and user_object.user_id == legacy_uuid
    assert effective_user_id == legacy_uuid
    assert captured["user_id"] == legacy_uuid, (
        "team_membership lookup must use the resolved DB user_id, not the JWT "
        f"email claim (got {captured['user_id']!r})"
    )
    assert captured["team_id"] == team_id


@pytest.mark.asyncio
async def test_multi_issuer_jwt_validates_selected_issuer_and_maps_claims(
    monkeypatch,
):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer_one = "https://issuer-one.example.com"
    issuer_two = "https://issuer-two.example.com"
    issuer_one_jwks_url = f"{issuer_one}/keys"
    issuer_two_jwks_url = f"{issuer_two}/keys"
    shared_kid = "shared-kid"

    _, issuer_one_jwk = _get_rsa_key_and_jwk(kid=shared_kid)
    issuer_two_private_key, issuer_two_jwk = _get_rsa_key_and_jwk(kid=shared_kid)

    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer_one,
                "jwks_url": issuer_one_jwks_url,
                "audience": "audience-one",
                "user_id_jwt_field": "email",
                "user_email_jwt_field": "email",
            },
            {
                "issuer": issuer_two,
                "jwks_url": issuer_two_jwks_url,
                "audience": "audience-two",
                "user_id_jwt_field": "repository_owner",
                "team_id_jwt_field": "repository",
            },
        ],
        keys_by_url={
            issuer_one_jwks_url: [issuer_one_jwk],
            issuer_two_jwks_url: [issuer_two_jwk],
        },
    )

    token = _encode_rsa_jwt(
        private_key=issuer_two_private_key,
        issuer=issuer_two,
        audience="audience-two",
        kid=shared_kid,
        extra_claims={
            "repository_owner": "example-org",
            "repository": "example-org/litellm-fork",
        },
    )

    claims = await jwt_handler.auth_jwt(token=token)

    assert claims[JWTHandler.LITELLM_JWT_ISSUER_CLAIM] == issuer_two
    assert jwt_handler.get_user_id(token=claims, default_value=None) == "example-org"
    assert jwt_handler.get_team_id(token=claims, default_value=None) == (
        "example-org/litellm-fork"
    )


@pytest.mark.asyncio
async def test_auth_jwt_issuer_path_expired_token_raises_401(monkeypatch):
    """An expired JWT validated through the issuer-scoped path
    (_auth_jwt_with_issuer) must raise a ProxyException carrying a 401 so the
    status is preserved end-to-end, just like the non-issuer path.
    """
    import time

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"
    kid = "expired-kid"

    private_key, jwk = _get_rsa_key_and_jwk(kid=kid)

    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[{"issuer": issuer, "jwks_url": jwks_url, "audience": "my-audience"}],
        keys_by_url={jwks_url: [jwk]},
    )

    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer=issuer,
        audience="my-audience",
        kid=kid,
        extra_claims={"exp": int(time.time()) - 100},
    )

    with pytest.raises(ProxyException) as exc_info:
        await jwt_handler.auth_jwt(token=token)

    assert exc_info.value.code == str(401)
    assert exc_info.value.type == ProxyErrorTypes.expired_key.value
    assert "Token Expired" in exc_info.value.message


@pytest.mark.asyncio
async def test_multi_issuer_jwt_maps_kubernetes_namespace_claim(monkeypatch):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://oidc.eks.eu-west-1.amazonaws.com/id/test-cluster"
    jwks_url = f"{issuer}/keys"
    private_key, jwk = _get_rsa_key_and_jwk(kid="k8s-key")
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer,
                "jwks_url": jwks_url,
                "audience": None,
                "disable_audience_validation": True,
                "user_id_jwt_field": "kubernetes\\.io.namespace",
            }
        ],
        keys_by_url={jwks_url: [jwk]},
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer=issuer,
        audience="kubernetes.default.svc",
        kid="k8s-key",
        extra_claims={"kubernetes.io": {"namespace": "example-namespace"}},
    )

    claims = await jwt_handler.auth_jwt(token=token)

    assert (
        jwt_handler.get_user_id(token=claims, default_value=None) == "example-namespace"
    )


@pytest.mark.asyncio
async def test_multi_issuer_jwt_unknown_issuer_falls_back_to_global_jwks(monkeypatch):
    """Tokens whose ``iss`` is not in the configured issuers list fall through
    to the legacy ``JWT_PUBLIC_KEY_URL`` path so operators can add the new
    ``issuers`` list to a live deployment without breaking existing tokens
    minted by non-configured IdPs. With no global JWKS configured, the legacy
    path surfaces a ``Missing JWT Public Key URL from environment.`` error.
    """
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    configured_issuer = "https://issuer.example.com"
    private_key, jwk = _get_rsa_key_and_jwk(kid="issuer-key")
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": configured_issuer,
                "jwks_url": f"{configured_issuer}/keys",
                "audience": "expected-audience",
            }
        ],
        keys_by_url={f"{configured_issuer}/keys": [jwk]},
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer="https://unknown-issuer.example.com",
        audience="expected-audience",
        kid="issuer-key",
    )

    with pytest.raises(Exception) as exc:
        await jwt_handler.auth_jwt(token=token)

    assert "Missing JWT Public Key URL from environment." in str(exc.value)


@pytest.mark.asyncio
async def test_multi_issuer_jwt_rejects_wrong_audience(monkeypatch):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"
    private_key, jwk = _get_rsa_key_and_jwk(kid="issuer-key")
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer,
                "jwks_url": jwks_url,
                "audience": "expected-audience",
            }
        ],
        keys_by_url={jwks_url: [jwk]},
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer=issuer,
        audience="wrong-audience",
        kid="issuer-key",
    )

    with pytest.raises(Exception) as exc:
        await jwt_handler.auth_jwt(token=token)

    assert "Validation fails" in str(exc.value)


@pytest.mark.asyncio
async def test_multi_issuer_jwt_same_kid_does_not_cross_issuer_keys(monkeypatch):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer_one = "https://issuer-one.example.com"
    issuer_two = "https://issuer-two.example.com"
    issuer_one_jwks_url = f"{issuer_one}/keys"
    issuer_two_jwks_url = f"{issuer_two}/keys"
    shared_kid = "shared-kid"
    issuer_one_private_key, issuer_one_jwk = _get_rsa_key_and_jwk(kid=shared_kid)
    _, issuer_two_jwk = _get_rsa_key_and_jwk(kid=shared_kid)
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer_one,
                "jwks_url": issuer_one_jwks_url,
                "audience": "audience-one",
            },
            {
                "issuer": issuer_two,
                "jwks_url": issuer_two_jwks_url,
                "audience": "audience-two",
            },
        ],
        keys_by_url={
            issuer_one_jwks_url: [issuer_one_jwk],
            issuer_two_jwks_url: [issuer_two_jwk],
        },
    )
    token = _encode_rsa_jwt(
        private_key=issuer_one_private_key,
        issuer=issuer_two,
        audience="audience-two",
        kid=shared_kid,
    )

    with pytest.raises(Exception) as exc:
        await jwt_handler.auth_jwt(token=token)

    assert "Validation fails" in str(exc.value)


@pytest.mark.asyncio
async def test_multi_issuer_jwt_missing_mapped_claim_leaves_user_id_unset(
    monkeypatch,
):
    """Mapped issuer claims behave like the global ``litellm_jwtauth`` path —
    present claims override the normalised value, missing ones simply leave
    the corresponding LiteLLM-internal claim absent (rather than failing the
    JWT outright). This keeps multi-issuer auth tolerant of tokens that omit
    optional fields like email or org id.
    """
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"
    private_key, jwk = _get_rsa_key_and_jwk(kid="issuer-key")
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer,
                "jwks_url": jwks_url,
                "audience": "expected-audience",
                "user_id_jwt_field": "email",
            }
        ],
        keys_by_url={jwks_url: [jwk]},
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer=issuer,
        audience="expected-audience",
        kid="issuer-key",
    )

    claims = await jwt_handler.auth_jwt(token=token)

    assert claims[jwt_handler.LITELLM_JWT_ISSUER_CLAIM] == issuer
    assert jwt_handler.LITELLM_USER_ID_CLAIM not in claims


def test_multi_issuer_jwt_requires_audience_unless_explicitly_disabled(
    monkeypatch,
):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"

    with pytest.raises(Exception) as exc:
        LiteLLM_JWTAuth(
            issuers=[
                {
                    "issuer": issuer,
                    "jwks_url": jwks_url,
                }
            ]
        )

    assert "must configure audience" in str(exc.value)


def test_multi_issuer_jwt_rejects_audience_with_disable_audience_validation():
    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"

    with pytest.raises(Exception) as exc:
        LiteLLM_JWTAuth(
            issuers=[
                {
                    "issuer": issuer,
                    "jwks_url": jwks_url,
                    "audience": "some-audience",
                    "disable_audience_validation": True,
                }
            ]
        )

    assert "cannot set audience and disable_audience_validation=True together" in str(
        exc.value
    )


@pytest.mark.asyncio
async def test_global_jwt_ignores_user_supplied_internal_claims(monkeypatch):
    from litellm.caching.dual_cache import DualCache

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_ISSUER", raising=False)

    jwks_url = "https://global-issuer.example.com/keys"
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", jwks_url)

    private_key, jwk = _get_rsa_key_and_jwk(kid="global-key")
    cache = DualCache()
    cache.set_cache(key=f"litellm_jwt_auth_keys_{jwks_url}", value=[jwk])

    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            user_id_jwt_field="email",
            user_email_jwt_field="email",
            team_id_jwt_field="team.id",
            team_ids_jwt_field="teams",
            org_id_jwt_field="org.id",
            end_user_id_jwt_field="end_user.id",
        ),
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer="https://global-issuer.example.com",
        audience="some-other-client",
        kid="global-key",
        extra_claims={
            "email": "real-user@example.com",
            "team": {"id": "real-team"},
            "teams": ["real-team", "secondary-team"],
            "org": {"id": "real-org"},
            "end_user": {"id": "real-end-user"},
            JWTHandler.LITELLM_JWT_ISSUER_CLAIM: "https://issuer.example.com",
            JWTHandler.LITELLM_USER_ID_CLAIM: "victim-user",
            JWTHandler.LITELLM_USER_EMAIL_CLAIM: "victim@example.com",
            JWTHandler.LITELLM_TEAM_ID_CLAIM: "victim-team",
            JWTHandler.LITELLM_TEAM_IDS_CLAIM: ["victim-team"],
            JWTHandler.LITELLM_ORG_ID_CLAIM: "victim-org",
            JWTHandler.LITELLM_END_USER_ID_CLAIM: "victim-end-user",
        },
    )

    claims = await jwt_handler.auth_jwt(token=token)

    assert jwt_handler.get_user_id(token=claims, default_value=None) == (
        "real-user@example.com"
    )
    assert jwt_handler.get_user_email(token=claims, default_value=None) == (
        "real-user@example.com"
    )
    assert jwt_handler.get_team_id(token=claims, default_value=None) == "real-team"
    assert jwt_handler.get_team_ids_from_jwt(token=claims) == [
        "real-team",
        "secondary-team",
    ]
    assert jwt_handler.get_org_id(token=claims, default_value=None) == "real-org"
    assert jwt_handler.get_end_user_id(token=claims, default_value=None) == (
        "real-end-user"
    )


@pytest.mark.asyncio
async def test_multi_issuer_jwt_strips_unmapped_internal_claims(monkeypatch):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"
    private_key, jwk = _get_rsa_key_and_jwk(kid="issuer-key")
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer,
                "jwks_url": jwks_url,
                "audience": "expected-audience",
                "user_email_jwt_field": "email",
            }
        ],
        keys_by_url={jwks_url: [jwk]},
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer=issuer,
        audience="expected-audience",
        kid="issuer-key",
        extra_claims={
            "email": "real-user@example.com",
            JWTHandler.LITELLM_USER_ID_CLAIM: "victim-user",
            JWTHandler.LITELLM_TEAM_ID_CLAIM: "victim-team",
        },
    )

    claims = await jwt_handler.auth_jwt(token=token)

    assert JWTHandler.LITELLM_USER_ID_CLAIM not in claims
    assert JWTHandler.LITELLM_TEAM_ID_CLAIM not in claims
    assert jwt_handler.get_user_id(token=claims, default_value=None) is None
    assert jwt_handler.get_team_id(token=claims, default_value=None) is None
    assert jwt_handler.get_user_email(token=claims, default_value=None) == (
        "real-user@example.com"
    )


@pytest.mark.asyncio
async def test_multi_issuer_jwt_does_not_emit_unscoped_global_warning(
    monkeypatch, caplog
):
    import logging

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_URL", raising=False)
    JWTHandler._unscoped_jwt_warning_emitted = False

    issuer = "https://issuer.example.com"
    jwks_url = f"{issuer}/keys"
    private_key, jwk = _get_rsa_key_and_jwk(kid="issuer-key")
    jwt_handler = _get_jwt_handler_with_issuer_keys(
        issuers=[
            {
                "issuer": issuer,
                "jwks_url": jwks_url,
                "audience": "expected-audience",
            }
        ],
        keys_by_url={jwks_url: [jwk]},
    )
    token = _encode_rsa_jwt(
        private_key=private_key,
        issuer=issuer,
        audience="expected-audience",
        kid="issuer-key",
    )

    with caplog.at_level(logging.WARNING):
        await jwt_handler.auth_jwt(token=token)

    assert "Tokens minted by any application" not in caplog.text
    assert JWTHandler._unscoped_jwt_warning_emitted is False


def test_build_decode_kwargs_warns_for_unscoped_global_fallback_in_mixed_deployment(
    monkeypatch, _reset_unscoped_warning_flag, caplog
):
    """The unscoped-fallback warning must fire even when per-issuer configs
    are set. In mixed deployments, tokens whose ``iss`` does not match any
    configured issuer fall through to the global path; if env-var scoping is
    absent that fallback IS unscoped, and the operator needs to be told."""
    import logging

    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    caplog.set_level(logging.WARNING)

    JWTHandler._build_decode_kwargs()

    matching = [
        r
        for r in caplog.records
        if "neither JWT_AUDIENCE nor JWT_ISSUER" in r.getMessage()
    ]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# fallback_to_db_teams: resolve team from DB memberships when JWT has no team
# claims (config flag on LiteLLM_JWTAuth)
# ---------------------------------------------------------------------------


def test_get_team_id_from_header_defers_to_db_membership_only_without_jwt_claims():
    """With fallback_to_db_teams=True, an x-litellm-team-id header is accepted
    provisionally only when the JWT carries no team claims (allowed set empty).
    When the JWT does carry team claims, the header must still be validated
    against them, and the flag-off behavior must keep rejecting unknown teams."""
    deferred = JWTAuthManager.get_team_id_from_header(
        request_headers={"x-litellm-team-id": "team-from-db"},
        allowed_team_ids=set(),
        fallback_to_db_teams=True,
    )
    assert deferred == "team-from-db"

    with pytest.raises(HTTPException) as exc_info:
        JWTAuthManager.get_team_id_from_header(
            request_headers={"x-litellm-team-id": "team-x"},
            allowed_team_ids={"team-1", "team-2"},
            fallback_to_db_teams=True,
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException):
        JWTAuthManager.get_team_id_from_header(
            request_headers={"x-litellm-team-id": "team-from-db"},
            allowed_team_ids=set(),
            fallback_to_db_teams=False,
        )


@pytest.mark.asyncio
async def test_find_team_with_model_access_defers_no_team_403_under_db_fallback():
    """find_team_with_model_access raises the early "no teams in token" 403 when
    enforcement is on, but defers (returns no team) so auth_builder's DB fallback
    can run when fallback_to_db_teams is enabled."""
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        enforce_team_based_model_access=True,
        fallback_to_db_teams=False,
    )
    with pytest.raises(HTTPException) as exc_info:
        await JWTAuthManager.find_team_with_model_access(
            team_ids=set(),
            requested_model="gpt-4",
            route="/chat/completions",
            jwt_handler=jwt_handler,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )
    assert exc_info.value.status_code == 403

    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        enforce_team_based_model_access=True,
        fallback_to_db_teams=True,
    )
    team_id, team_object = await JWTAuthManager.find_team_with_model_access(
        team_ids=set(),
        requested_model="gpt-4",
        route="/chat/completions",
        jwt_handler=jwt_handler,
        prisma_client=None,
        user_api_key_cache=MagicMock(),
        parent_otel_span=None,
        proxy_logging_obj=MagicMock(),
    )
    assert team_id is None
    assert team_object is None


def _db_fallback_handler(litellm_jwtauth: Optional[LiteLLM_JWTAuth] = None) -> JWTHandler:
    handler = JWTHandler()
    handler.litellm_jwtauth = litellm_jwtauth or LiteLLM_JWTAuth()
    return handler


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_skips_unresolvable_membership():
    """An orphaned membership (team row missing/erroring) is skipped and the next
    resolvable DB team is selected instead of aborting the fallback."""
    user_object = LiteLLM_UserTable(
        user_id="u_skip",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["ghost_team", "real_team"],
    )
    resolved = LiteLLM_TeamTable(team_id="real_team")

    async def fake_get_team(team_id, **kwargs):
        if team_id == "ghost_team":
            raise HTTPException(status_code=404, detail="missing")
        return resolved

    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object",
        new_callable=AsyncMock,
        side_effect=fake_get_team,
    ):
        (
            team_id,
            team_object,
            _membership,
        ) = await JWTAuthManager._resolve_db_team_fallback(
            user_object=user_object,
            user_id=None,
            requested_model=None,
            route="/chat/completions",
            jwt_handler=_db_fallback_handler(),
            enforce_team_based_model_access=True,
            team_id_upsert=False,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert team_id == "real_team"
    assert team_object is resolved


@pytest.mark.parametrize(
    (
        "fallback_to_db_teams",
        "user_teams",
        "header_team_id",
        "expected_team_id",
        "expect_403",
    ),
    [
        pytest.param(
            True, ["team_solo"], None, "team_solo", False, id="flag_on_single_db_team"
        ),
        pytest.param(
            True,
            ["team_a", "team_b"],
            None,
            "team_a",
            False,
            id="flag_on_multi_db_team_picks_first",
        ),
        pytest.param(
            True,
            ["team_a", "team_b"],
            "team_b",
            "team_b",
            False,
            id="flag_on_header_team_in_membership",
        ),
        pytest.param(
            True,
            ["team_a", "team_b"],
            "team_x",
            None,
            True,
            id="flag_on_header_team_not_in_membership_403",
        ),
        pytest.param(True, [], None, None, True, id="flag_on_no_db_team_enforced_403"),
        pytest.param(
            False,
            ["team_a", "team_b"],
            None,
            None,
            False,
            id="flag_off_multi_db_team_no_fallback",
        ),
        pytest.param(
            False,
            ["team_solo"],
            None,
            "team_solo",
            False,
            id="flag_off_single_db_team_upstream_fallback",
        ),
    ],
)
@pytest.mark.asyncio
async def test_auth_builder_db_team_fallback_when_jwt_has_no_team(
    fallback_to_db_teams: bool,
    user_teams: list,
    header_team_id: Optional[str],
    expected_team_id: Optional[str],
    expect_403: bool,
) -> None:
    """End-to-end auth_builder behavior with no JWT team claims.

    fallback_to_db_teams=True attributes usage to the user's first resolvable DB
    team, honors a valid x-litellm-team-id header, and rejects a header team the
    user does not belong to. The default (flag off) preserves the upstream
    single-team fallback: a lone DB team is resolved, multiple are ambiguous.
    """
    user_id = "u_db_fallback"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=user_teams,
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        enforce_team_based_model_access=True,
        fallback_to_db_teams=fallback_to_db_teams,
    )

    request_headers = {"x-litellm-team-id": header_team_id} if header_team_id else None

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    async def call_auth_builder():
        with (
            patch.object(
                jwt_handler, "auth_jwt", new_callable=AsyncMock
            ) as mock_auth_jwt,
            patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
            patch.object(jwt_handler, "get_rbac_role", return_value=None),
            patch.object(jwt_handler, "get_scopes", return_value=[]),
            patch.object(jwt_handler, "get_object_id", return_value=None),
            patch.object(
                JWTAuthManager,
                "get_user_info",
                new_callable=AsyncMock,
                return_value=(user_id, "u@example.com", True),
            ),
            patch.object(jwt_handler, "get_org_id", return_value=None),
            patch.object(jwt_handler, "get_end_user_id", return_value=None),
            patch.object(
                JWTAuthManager,
                "check_admin_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                JWTAuthManager,
                "find_and_validate_specific_team_id",
                new_callable=AsyncMock,
                return_value=(None, None),
            ),
            patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
            patch.object(
                JWTAuthManager,
                "find_team_with_model_access",
                new_callable=AsyncMock,
                return_value=(None, None),
            ),
            patch.object(
                JWTAuthManager,
                "get_objects",
                new_callable=AsyncMock,
                return_value=(user_object, None, None, None, user_id),
            ),
            patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
            patch.object(JWTAuthManager, "validate_object_id", return_value=True),
            patch.object(
                JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
            ),
            patch(
                "litellm.proxy.auth.handle_jwt.get_team_object",
                new_callable=AsyncMock,
                side_effect=fake_get_team,
            ),
            patch(
                "litellm.proxy.auth.handle_jwt.get_team_membership",
                new_callable=AsyncMock,
                return_value=LiteLLM_TeamMembership(
                    user_id=user_id,
                    team_id=user_teams[0] if user_teams else "none",
                    litellm_budget_table=None,
                ),
            ),
        ):
            mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
            return await JWTAuthManager.auth_builder(
                api_key="test_jwt_token",
                jwt_handler=jwt_handler,
                request_data={"model": "gpt-4"},
                general_settings={"enforce_rbac": False},
                route="/chat/completions",
                prisma_client=None,
                user_api_key_cache=None,
                parent_otel_span=None,
                proxy_logging_obj=None,
                request_headers=request_headers,
            )

    if expect_403:
        with pytest.raises(HTTPException) as exc_info:
            await call_auth_builder()
        assert exc_info.value.status_code == 403
    else:
        result = await call_auth_builder()
        assert result["team_id"] == expected_team_id


@pytest.mark.parametrize(
    "fallback_to_db_teams, expect_teams_stripped",
    [
        pytest.param(True, False, id="fallback_on_preserves_db_teams"),
        pytest.param(False, True, id="fallback_off_strips_db_teams"),
    ],
)
@pytest.mark.asyncio
async def test_sync_user_role_and_teams_no_claim_team_preservation(
    fallback_to_db_teams: bool,
    expect_teams_stripped: bool,
) -> None:
    """A no-team-claim JWT must not permanently strip a user's DB team memberships
    when fallback_to_db_teams is enabled — otherwise the DB fallback that runs
    right after has nothing to resolve and every request silently wipes the user
    out of their teams. With the flag off, the legacy mirror-the-IdP behavior
    (remove teams absent from the token) is preserved."""
    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=AsyncMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="teams",
            sync_user_role_and_teams=True,
            fallback_to_db_teams=fallback_to_db_teams,
        ),
    )

    token = {"sub": "u1"}
    user = LiteLLM_UserTable(
        user_id="u1",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        teams=["team_a", "team_b"],
    )
    prisma = AsyncMock()

    with patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        new_callable=AsyncMock,
    ) as mock_patch:
        await JWTAuthManager.sync_user_role_and_teams(jwt_handler, token, user, prisma)

    if expect_teams_stripped:
        mock_patch.assert_awaited_once()
        assert set(mock_patch.call_args.kwargs["teams_ids_to_remove_user_from"]) == {
            "team_a",
            "team_b",
        }
        assert user.teams == []
    else:
        mock_patch.assert_not_called()
        assert user.teams == ["team_a", "team_b"]


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_skips_team_without_model_access():
    """The DB-team fallback must apply the same per-team model-access check as the
    claim-based path: a DB team that cannot access the requested model is skipped
    in favor of one that can, instead of selecting the first membership blindly."""
    user_object = LiteLLM_UserTable(
        user_id="u_model_access",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["restricted_team", "allowed_team"],
    )
    teams = {
        "restricted_team": LiteLLM_TeamTable(
            team_id="restricted_team", models=["claude-3"]
        ),
        "allowed_team": LiteLLM_TeamTable(team_id="allowed_team", models=["gpt-4"]),
    }

    async def fake_get_team(team_id, **kwargs):
        return teams[team_id]

    async def fake_can_access(model, team_object, llm_router, team_model_aliases=None):
        if model in (team_object.models or []):
            return True
        raise ProxyException(
            message="team not allowed to access model",
            type=ProxyErrorTypes.team_model_access_denied,
            param="model",
            code=403,
        )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.can_team_access_model",
            new_callable=AsyncMock,
            side_effect=fake_can_access,
        ),
    ):
        (
            team_id,
            team_object,
            _membership,
        ) = await JWTAuthManager._resolve_db_team_fallback(
            user_object=user_object,
            user_id=None,
            requested_model="gpt-4",
            route="/chat/completions",
            jwt_handler=_db_fallback_handler(),
            enforce_team_based_model_access=True,
            team_id_upsert=False,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert team_id == "allowed_team"
    assert team_object is teams["allowed_team"]


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_enforces_team_allowed_routes():
    """The DB-team fallback must apply the same team_allowed_routes gate as the
    claim-based path: a route the JWT config excludes for team-role callers must
    not become reachable by selecting a DB team, even when that team can access
    the requested model. Without the gate, a teamless JWT could reach the
    info/management routes an admin narrowed team_allowed_routes to exclude."""
    user_object = LiteLLM_UserTable(
        user_id="u_routes",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["team_a"],
    )
    team = LiteLLM_TeamTable(team_id="team_a", models=["gpt-4"])
    handler = _db_fallback_handler(LiteLLM_JWTAuth(team_allowed_routes=["openai_routes"]))

    async def fake_get_team(team_id, **kwargs):
        return team

    async def fake_can_access(model, team_object, llm_router, team_model_aliases=None):
        return True

    async def resolve(route):
        return await JWTAuthManager._resolve_db_team_fallback(
            user_object=user_object,
            user_id=None,
            requested_model="gpt-4",
            route=route,
            jwt_handler=handler,
            enforce_team_based_model_access=False,
            team_id_upsert=False,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.can_team_access_model",
            new_callable=AsyncMock,
            side_effect=fake_can_access,
        ),
    ):
        excluded_team_id, excluded_team_object, _ = await resolve("/key/info")
        allowed_team_id, allowed_team_object, _ = await resolve("/chat/completions")

    assert excluded_team_id is None
    assert excluded_team_object is None
    assert allowed_team_id == "team_a"
    assert allowed_team_object is team


def test_validate_header_team_in_db_membership_does_not_leak_team_ids():
    """The 403 raised for an x-litellm-team-id header outside the user's DB
    memberships must not enumerate the user's team IDs back to the caller; any
    valid-JWT caller could otherwise probe header values to discover team IDs."""
    user_object = LiteLLM_UserTable(
        user_id="u_leak",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["secret_team_alpha", "secret_team_beta"],
    )

    with pytest.raises(HTTPException) as exc_info:
        JWTAuthManager._validate_header_team_in_db_membership(
            team_id="outsider_team",
            user_object=user_object,
        )

    detail = exc_info.value.detail
    assert exc_info.value.status_code == 403
    assert "secret_team_alpha" not in detail
    assert "secret_team_beta" not in detail
    assert "outsider_team" in detail


async def _run_auth_builder_with_header_team(
    jwt_auth_config: LiteLLM_JWTAuth,
    token: dict,
    header_team_id: str,
    user_object: LiteLLM_UserTable,
    fake_get_team,
    allowed_team_ids: set,
):
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = jwt_auth_config
    with (
        patch.object(
            jwt_handler, "auth_jwt", new_callable=AsyncMock, return_value=token
        ),
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_object.user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager, "get_all_team_ids", return_value=allowed_team_ids
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_object.user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
    ):
        return await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
            request_headers={"x-litellm-team-id": header_team_id},
        )


async def _team_lookup_404(team_id, **kwargs):
    raise HTTPException(
        status_code=404,
        detail=f"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call.",
    )


@pytest.mark.asyncio
async def test_auth_builder_header_team_not_found_matches_non_membership_denial() -> (
    None
):
    """A provisional x-litellm-team-id naming a nonexistent team must produce
    the exact same 403 shape as one naming an existing team outside the
    caller's memberships. Letting get_team_object's 404 surface would give any
    valid-JWT caller an oracle to probe which team ids exist."""
    user_object = LiteLLM_UserTable(
        user_id="u_oracle",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["team_member"],
    )
    config = LiteLLM_JWTAuth(
        enforce_team_based_model_access=True,
        fallback_to_db_teams=True,
    )
    token = {"sub": "u_oracle", "scope": ""}

    async def team_exists(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    with pytest.raises(HTTPException) as missing_exc:
        await _run_auth_builder_with_header_team(
            config, token, "team_ghost", user_object, _team_lookup_404, set()
        )
    with pytest.raises(HTTPException) as outsider_exc:
        await _run_auth_builder_with_header_team(
            config, token, "team_other", user_object, team_exists, set()
        )

    assert missing_exc.value.status_code == 403
    assert outsider_exc.value.status_code == 403
    assert missing_exc.value.detail.replace(
        "team_ghost", "<team>"
    ) == outsider_exc.value.detail.replace("team_other", "<team>")
    assert "exist" not in missing_exc.value.detail


@pytest.mark.asyncio
async def test_auth_builder_claim_backed_header_team_lookup_error_propagates() -> None:
    """When the JWT carries team claims the header team is not provisional, so
    a failed team lookup keeps the upstream contract: get_team_object's 404
    surfaces unchanged instead of being rewritten into the membership 403."""
    user_object = LiteLLM_UserTable(
        user_id="u_claimed",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["team_member"],
    )
    config = LiteLLM_JWTAuth(
        enforce_team_based_model_access=True,
        fallback_to_db_teams=True,
        team_ids_jwt_field="team_ids",
    )
    token = {"sub": "u_claimed", "scope": "", "team_ids": ["team_claimed"]}

    with pytest.raises(HTTPException) as exc_info:
        await _run_auth_builder_with_header_team(
            config, token, "team_claimed", user_object, _team_lookup_404, {"team_claimed"}
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_loads_team_membership():
    """The DB-team fallback must load the resolved team's membership row (when a
    user_id is known) so per-team membership budget limits are enforced on the
    fallback path the same as on the claim-based path; returning a None membership
    would silently skip LiteLLM_TeamMembership budget checks for every request."""
    user_object = LiteLLM_UserTable(
        user_id="u_membership",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["team_with_budget"],
    )
    membership = LiteLLM_TeamMembership(
        user_id="u_membership",
        team_id="team_with_budget",
        budget_id="budget_xyz",
        litellm_budget_table=None,
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    async def fake_get_membership(user_id, team_id, **kwargs):
        assert user_id == "u_membership"
        assert team_id == "team_with_budget"
        return membership

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
            side_effect=fake_get_membership,
        ),
    ):
        (
            team_id,
            team_object,
            team_membership,
        ) = await JWTAuthManager._resolve_db_team_fallback(
            user_object=user_object,
            user_id="u_membership",
            requested_model=None,
            route="/chat/completions",
            jwt_handler=_db_fallback_handler(),
            enforce_team_based_model_access=True,
            team_id_upsert=False,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert team_id == "team_with_budget"
    assert team_object is not None
    assert team_membership is membership
    assert team_membership.budget_id == "budget_xyz"


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_survives_membership_lookup_error():
    """A transient membership-lookup failure must not deny an otherwise-authorized
    request. get_team_membership swallows DB errors internally and returns None, so
    the fallback must return the resolved team with a None membership (budget
    enforcement degrades gracefully) instead of treating it as a denial."""
    user_object = LiteLLM_UserTable(
        user_id="u_flaky",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["team_flaky"],
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    async def none_on_db_error_membership(user_id, team_id, **kwargs):
        return None

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
            side_effect=none_on_db_error_membership,
        ),
    ):
        (
            team_id,
            team_object,
            team_membership,
        ) = await JWTAuthManager._resolve_db_team_fallback(
            user_object=user_object,
            user_id="u_flaky",
            requested_model=None,
            route="/chat/completions",
            jwt_handler=_db_fallback_handler(),
            enforce_team_based_model_access=True,
            team_id_upsert=False,
            prisma_client=None,
            user_api_key_cache=MagicMock(),
            parent_otel_span=None,
            proxy_logging_obj=MagicMock(),
        )

    assert team_id == "team_flaky"
    assert team_object is not None
    assert team_membership is None


@pytest.mark.asyncio
async def test_auth_builder_db_fallback_does_not_validate_rbac_team_against_db_membership():
    """When fallback_to_db_teams is on and the JWT carries an RBAC team role but no
    group/team claims, team_id is set from the RBAC object_id (not the provisional
    x-litellm-team-id header). That RBAC-asserted team must not be re-validated
    against the user's DB memberships; only a team that actually came from the
    header is provisional. Without the team_id == header_team_id guard, every such
    RBAC request 403s when the RBAC team is not also a DB membership."""
    rbac_team = "rbac_asserted_team"
    user_id = "u_rbac"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["unrelated_db_team"],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        enforce_team_based_model_access=True,
        fallback_to_db_teams=True,
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=LitellmUserRoles.TEAM),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=rbac_team),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        result = await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
            request_headers=None,
        )

    assert result["team_id"] == rbac_team


@pytest.mark.asyncio
async def test_resolve_db_team_fallback_distinguishes_no_membership_vs_model_denied():
    """When enforce_team_based_model_access is on, a user with no DB memberships
    and a user with memberships that all fail the model-access check must surface
    different 403s; collapsing both into the no-membership message hides the real
    cause and diverges from find_team_with_model_access's claim-based message."""
    membership_user = LiteLLM_UserTable(
        user_id="u_no_model",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["only_team"],
    )
    no_membership_user = LiteLLM_UserTable(
        user_id="u_empty",
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=[],
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id, models=["other"])

    async def fake_can_access(model, team_object, llm_router, team_model_aliases=None):
        raise ProxyException(
            message="team not allowed to access model",
            type=ProxyErrorTypes.team_model_access_denied,
            param="model",
            code=403,
        )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.can_team_access_model",
            new_callable=AsyncMock,
            side_effect=fake_can_access,
        ),
    ):
        with pytest.raises(HTTPException) as model_denied:
            await JWTAuthManager._resolve_db_team_fallback(
                user_object=membership_user,
                user_id=None,
                requested_model="gpt-4",
                route="/chat/completions",
                jwt_handler=_db_fallback_handler(),
                enforce_team_based_model_access=True,
                team_id_upsert=False,
                prisma_client=None,
                user_api_key_cache=MagicMock(),
                parent_otel_span=None,
                proxy_logging_obj=MagicMock(),
            )

        with pytest.raises(HTTPException) as no_member:
            await JWTAuthManager._resolve_db_team_fallback(
                user_object=no_membership_user,
                user_id=None,
                requested_model="gpt-4",
                route="/chat/completions",
                jwt_handler=_db_fallback_handler(),
                enforce_team_based_model_access=True,
                team_id_upsert=False,
                prisma_client=None,
                user_api_key_cache=MagicMock(),
                parent_otel_span=None,
                proxy_logging_obj=MagicMock(),
            )

    assert model_denied.value.status_code == 403
    assert "requested model" in model_denied.value.detail
    assert "gpt-4" in model_denied.value.detail
    assert "only_team" not in model_denied.value.detail

    assert no_member.value.status_code == 403
    assert "not a member of any team" in no_member.value.detail


@pytest.mark.asyncio
async def test_auth_builder_db_fallback_runs_when_only_team_id_default_set():
    """team_id_default makes JWTHandler.get_team_id return a non-None team for a
    claimless token. The fallback gate must look at real JWT team claims (not the
    operator-configured default) so fallback_to_db_teams still attributes to the
    user's DB memberships instead of silently routing to the default team."""
    user_id = "u_default_token"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["db_team_for_user"],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        fallback_to_db_teams=True,
        team_id_default="config_default_team",
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        result = await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
            request_headers=None,
        )

    assert result["team_id"] == "db_team_for_user"


@pytest.mark.asyncio
async def test_auth_builder_alias_only_token_resolves_alias_not_db_fallback():
    """An alias-only JWT (team_alias_jwt_field set, no team-id claims) must resolve
    its alias via find_and_validate_specific_team_id, not fall into the DB-membership
    fallback. get_all_jwt_team_ids ignores aliases, so without the get_team_alias
    clause in the db_team_fallback gate the alias is silently dropped and the request
    is mis-attributed to the user's first DB team instead of the alias-named team."""
    user_id = "u_alias_only"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["db_membership_team"],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        fallback_to_db_teams=True,
        team_alias_jwt_field="team_name",
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    async def fake_get_team_by_alias(team_alias, **kwargs):
        return LiteLLM_TeamTable(team_id="alias_resolved_team", team_alias=team_alias)

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object_by_alias",
            new_callable=AsyncMock,
            side_effect=fake_get_team_by_alias,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "team_name": "resolvable_alias"}
        result = await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
            request_headers=None,
        )

    assert result["team_id"] == "alias_resolved_team"


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_alias_wins_over_team_id_default():
    """When the JWT carries only an alias claim (no team_id claim) and
    team_id_default is configured, alias resolution must win. get_team_id
    silently substitutes team_id_default for a missing claim, which would
    otherwise mask the alias-resolved team and mis-attribute spend/access
    to the configured default team."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_alias_jwt_field="team_alias",
            team_id_default="config_default_team",
        ),
    )

    jwt_token = {"sub": "user-1", "team_alias": "my-team"}
    alias_team = LiteLLM_TeamTable(
        team_id="alias_resolved_team", team_alias="my-team"
    )

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_by_id,
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object_by_alias",
            new_callable=AsyncMock,
        ) as mock_get_by_alias,
    ):
        mock_get_by_alias.return_value = alias_team

        team_id, team_obj = await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=jwt_handler,
            jwt_valid_token=jwt_token,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert team_id == "alias_resolved_team"
    assert team_obj == alias_team
    mock_get_by_id.assert_not_called()
    mock_get_by_alias.assert_called_once()


@pytest.mark.asyncio
async def test_find_and_validate_specific_team_id_team_id_default_used_without_alias():
    """When the token carries neither a team_id nor an alias claim and
    team_id_default is configured, the default still resolves the team. The
    alias-precedence fix must not regress this baseline fallback behavior."""
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    jwt_handler = JWTHandler()
    user_api_key_cache = DualCache()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_alias_jwt_field="team_alias",
            team_id_default="config_default_team",
        ),
    )

    jwt_token = {"sub": "user-1"}
    default_team = LiteLLM_TeamTable(team_id="config_default_team")

    with (
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock
        ) as mock_get_by_id,
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object_by_alias",
            new_callable=AsyncMock,
        ) as mock_get_by_alias,
    ):
        mock_get_by_id.return_value = default_team

        team_id, team_obj = await JWTAuthManager.find_and_validate_specific_team_id(
            jwt_handler=jwt_handler,
            jwt_valid_token=jwt_token,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert team_id == "config_default_team"
    assert team_obj == default_team
    mock_get_by_alias.assert_not_called()


@pytest.mark.asyncio
async def test_auth_builder_db_fallback_enforces_passthrough_route_access():
    """A team selected only via _resolve_db_team_fallback must still pass the
    auth-enforced passthrough route check; previously the earlier gate ran while
    team_id was None and the fallback-resolved team bypassed it entirely."""
    user_id = "u_passthrough"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=["team_no_passthrough"],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(fallback_to_db_teams=True)

    passthrough_route = "/vertex_ai/v1/projects/p/locations/us/publishers/google/models/gemini:generateContent"

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id, metadata={})

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_membership",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.RouteChecks.is_auth_enforced_pass_through_route",
            return_value=True,
        ),
        patch.object(
            JWTAuthManager,
            "_team_has_passthrough_route_access",
            return_value=False,
        ),
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.auth_builder(
                api_key="test_jwt_token",
                jwt_handler=jwt_handler,
                request_data={"model": "gemini"},
                general_settings={"enforce_rbac": False},
                route=passthrough_route,
                prisma_client=None,
                user_api_key_cache=None,
                parent_otel_span=None,
                proxy_logging_obj=None,
                request_headers=None,
                request_method="POST",
            )

    assert exc_info.value.status_code == 403
    assert "passthrough route" in exc_info.value.detail


@pytest.mark.asyncio
async def test_sync_user_role_and_teams_singular_claim_reconciles_memberships():
    """When fallback_to_db_teams is on but the JWT carries a singular team claim
    (Okta/Auth0 default for users with one primary team), sync must treat it as a
    real claim and reconcile DB memberships against it. Otherwise stale DB teams
    persist and a subsequent claimless JWT for the same user is silently attributed
    to a team the IdP never asserted on the singular-claim login."""
    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=AsyncMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_id_jwt_field="primary_team",
            sync_user_role_and_teams=True,
            fallback_to_db_teams=True,
        ),
    )

    token = {"sub": "u_singular", "primary_team": "team_primary"}
    user = LiteLLM_UserTable(
        user_id="u_singular",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        teams=["team_stale_a", "team_stale_b"],
    )

    with patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        new_callable=AsyncMock,
    ) as mock_patch:
        await JWTAuthManager.sync_user_role_and_teams(
            jwt_handler, token, user, AsyncMock()
        )

    mock_patch.assert_awaited_once()
    assert set(mock_patch.call_args.kwargs["teams_ids_to_remove_user_from"]) == {
        "team_stale_a",
        "team_stale_b",
    }
    assert set(mock_patch.call_args.kwargs["teams_ids_to_add_user_to"]) == {
        "team_primary"
    }
    assert user.teams == ["team_primary"]


@pytest.mark.asyncio
async def test_auth_builder_provisional_header_team_is_not_upserted():
    """A provisional x-litellm-team-id (accepted only because the JWT carries no
    team claims) must not be upserted even when team_id_upsert is enabled: it is
    validated against DB membership afterwards, so upserting first would let an
    attacker-supplied header create an orphaned team row. A genuine membership
    team already exists, so the resolved request still succeeds."""
    user_id = "u_no_upsert"
    header_team = "header_supplied_team"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=[header_team],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        fallback_to_db_teams=True,
        team_id_upsert=True,
    )

    upsert_by_team: dict[str, Optional[bool]] = {}

    async def spy_get_team(team_id, **kwargs):
        upsert_by_team[team_id] = kwargs.get("team_id_upsert")
        return LiteLLM_TeamTable(team_id=team_id)

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=None),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=spy_get_team,
        ),
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        result = await JWTAuthManager.auth_builder(
            api_key="test_jwt_token",
            jwt_handler=jwt_handler,
            request_data={"model": "gpt-4"},
            general_settings={"enforce_rbac": False},
            route="/chat/completions",
            prisma_client=None,
            user_api_key_cache=None,
            parent_otel_span=None,
            proxy_logging_obj=None,
            request_headers={"x-litellm-team-id": header_team},
        )

    assert result["team_id"] == header_team
    assert upsert_by_team[header_team] is False


@pytest.mark.asyncio
async def test_auth_builder_header_cannot_override_rbac_team_under_db_fallback():
    """An RBAC team-role JWT already pins team_id to the asserted team. With
    fallback_to_db_teams on, a caller must not be able to substitute that team
    by sending x-litellm-team-id for any other team they happen to belong to:
    the provisional-header path is only for tokens with no team identity at all,
    so an RBAC token plus a non-claim header team is rejected with 403."""
    user_id = "u_rbac_override"
    rbac_team = "rbac_pinned_team"
    other_team = "other_db_team"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=[other_team],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(fallback_to_db_teams=True)

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    with (
        patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt,
        patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
        patch.object(jwt_handler, "get_rbac_role", return_value=LitellmUserRoles.TEAM),
        patch.object(jwt_handler, "get_scopes", return_value=[]),
        patch.object(jwt_handler, "get_object_id", return_value=rbac_team),
        patch.object(
            JWTAuthManager,
            "get_user_info",
            new_callable=AsyncMock,
            return_value=(user_id, "u@example.com", True),
        ),
        patch.object(jwt_handler, "get_org_id", return_value=None),
        patch.object(jwt_handler, "get_end_user_id", return_value=None),
        patch.object(
            JWTAuthManager,
            "check_admin_access",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
        patch.object(
            JWTAuthManager,
            "get_objects",
            new_callable=AsyncMock,
            return_value=(user_object, None, None, None, user_id),
        ),
        patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
        patch.object(JWTAuthManager, "validate_object_id", return_value=True),
        patch.object(
            JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.auth.handle_jwt.get_team_object",
            new_callable=AsyncMock,
            side_effect=fake_get_team,
        ),
    ):
        mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
        with pytest.raises(HTTPException) as exc_info:
            await JWTAuthManager.auth_builder(
                api_key="test_jwt_token",
                jwt_handler=jwt_handler,
                request_data={"model": "gpt-4"},
                general_settings={"enforce_rbac": False},
                route="/chat/completions",
                prisma_client=None,
                user_api_key_cache=None,
                parent_otel_span=None,
                proxy_logging_obj=None,
                request_headers={"x-litellm-team-id": other_team},
            )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_auth_builder_header_team_enforces_team_allowed_routes_under_db_fallback():
    """A claimless JWT with x-litellm-team-id under fallback_to_db_teams must
    obey the same team_allowed_routes gate as the auto-pick fallback path.
    Otherwise the header bypasses the route gate the JWT config narrows for
    team-role callers, letting management/info routes be reached with a
    team_id the auto-pick path would silently refuse to set."""
    user_id = "u_header_routes"
    header_team = "header_supplied_team"
    user_object = LiteLLM_UserTable(
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        teams=[header_team],
    )
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        fallback_to_db_teams=True,
        team_allowed_routes=["openai_routes"],
    )

    async def fake_get_team(team_id, **kwargs):
        return LiteLLM_TeamTable(team_id=team_id)

    async def call(route: str):
        with (
            patch.object(
                jwt_handler, "auth_jwt", new_callable=AsyncMock
            ) as mock_auth_jwt,
            patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock),
            patch.object(jwt_handler, "get_rbac_role", return_value=None),
            patch.object(jwt_handler, "get_scopes", return_value=[]),
            patch.object(jwt_handler, "get_object_id", return_value=None),
            patch.object(
                JWTAuthManager,
                "get_user_info",
                new_callable=AsyncMock,
                return_value=(user_id, "u@example.com", True),
            ),
            patch.object(jwt_handler, "get_org_id", return_value=None),
            patch.object(jwt_handler, "get_end_user_id", return_value=None),
            patch.object(
                JWTAuthManager,
                "check_admin_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(JWTAuthManager, "get_all_team_ids", return_value=set()),
            patch.object(
                JWTAuthManager,
                "get_objects",
                new_callable=AsyncMock,
                return_value=(user_object, None, None, None, user_id),
            ),
            patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock),
            patch.object(JWTAuthManager, "validate_object_id", return_value=True),
            patch.object(
                JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
            ),
            patch(
                "litellm.proxy.auth.handle_jwt.get_team_object",
                new_callable=AsyncMock,
                side_effect=fake_get_team,
            ),
        ):
            mock_auth_jwt.return_value = {"sub": user_id, "scope": ""}
            return await JWTAuthManager.auth_builder(
                api_key="test_jwt_token",
                jwt_handler=jwt_handler,
                request_data={"model": "gpt-4"},
                general_settings={"enforce_rbac": False},
                route=route,
                prisma_client=None,
                user_api_key_cache=None,
                parent_otel_span=None,
                proxy_logging_obj=None,
                request_headers={"x-litellm-team-id": header_team},
            )

    with pytest.raises(HTTPException) as exc_info:
        await call("/key/info")
    assert exc_info.value.status_code == 403
    assert "not allowed to access route" in exc_info.value.detail
    assert "/key/info" in exc_info.value.detail

    result = await call("/chat/completions")
    assert result["team_id"] == header_team


@pytest.mark.asyncio
async def test_sync_user_role_and_teams_singular_claim_only_recognized_under_flag():
    """Reading the singular team claim during sync is scoped to fallback_to_db_teams.
    With the flag off, sync keeps the upstream plural-only reconciliation, so a
    singular-only token is treated as claimless and existing DB teams are removed
    exactly as before this PR; the new dual-claim behavior must not silently change
    membership reconciliation for deployments that never opted in."""
    jwt_handler = JWTHandler()
    jwt_handler.update_environment(
        prisma_client=None,
        user_api_key_cache=AsyncMock(),
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_id_jwt_field="primary_team",
            sync_user_role_and_teams=True,
            fallback_to_db_teams=False,
        ),
    )

    token = {"sub": "u_flag_off", "primary_team": "team_primary"}
    user = LiteLLM_UserTable(
        user_id="u_flag_off",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        teams=["team_existing"],
    )

    with patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        new_callable=AsyncMock,
    ) as mock_patch:
        await JWTAuthManager.sync_user_role_and_teams(
            jwt_handler, token, user, AsyncMock()
        )

    mock_patch.assert_awaited_once()
    assert set(mock_patch.call_args.kwargs["teams_ids_to_remove_user_from"]) == {
        "team_existing"
    }
    assert mock_patch.call_args.kwargs["teams_ids_to_add_user_to"] == []
    assert user.teams == []

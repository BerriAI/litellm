from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import (
    JWTLiteLLMRoleMap,
    LiteLLM_JWTAuth,
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
    with patch.object(
        jwt_handler, "auth_jwt", new_callable=AsyncMock
    ) as mock_auth_jwt, patch.object(
        JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
    ) as mock_check_rbac, patch.object(
        jwt_handler, "get_rbac_role", return_value=None
    ) as mock_get_rbac, patch.object(
        jwt_handler, "get_scopes", return_value=[]
    ) as mock_get_scopes, patch.object(
        jwt_handler, "get_object_id", return_value=None
    ) as mock_get_object_id, patch.object(
        JWTAuthManager,
        "get_user_info",
        new_callable=AsyncMock,
        return_value=("test_user_1", "test@example.com", True),
    ) as mock_get_user_info, patch.object(
        jwt_handler, "get_org_id", return_value=None
    ) as mock_get_org_id, patch.object(
        jwt_handler, "get_end_user_id", return_value=None
    ) as mock_get_end_user_id, patch.object(
        JWTAuthManager, "check_admin_access", new_callable=AsyncMock, return_value=None
    ) as mock_check_admin, patch.object(
        JWTAuthManager,
        "find_and_validate_specific_team_id",
        new_callable=AsyncMock,
        return_value=(None, None),
    ) as mock_find_team, patch.object(
        JWTAuthManager, "get_all_team_ids", return_value=set()
    ) as mock_get_all_team_ids, patch.object(
        JWTAuthManager,
        "find_team_with_model_access",
        new_callable=AsyncMock,
        return_value=(None, None),
    ) as mock_find_team_access, patch.object(
        JWTAuthManager,
        "get_objects",
        new_callable=AsyncMock,
        return_value=(user_object, None, None),
    ) as mock_get_objects, patch.object(
        JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
    ) as mock_map_user, patch.object(
        JWTAuthManager, "validate_object_id", return_value=True
    ) as mock_validate_object:
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
    with patch.object(
        jwt_handler, "auth_jwt", new_callable=AsyncMock
    ) as mock_auth_jwt, patch.object(
        JWTAuthManager, "check_rbac_role", new_callable=AsyncMock
    ) as mock_check_rbac, patch.object(
        jwt_handler, "get_rbac_role", return_value=None
    ) as mock_get_rbac, patch.object(
        jwt_handler, "get_scopes", return_value=[]
    ) as mock_get_scopes, patch.object(
        jwt_handler, "get_object_id", return_value=None
    ) as mock_get_object_id, patch.object(
        JWTAuthManager,
        "get_user_info",
        new_callable=AsyncMock,
        return_value=("test_user_1", "test@example.com", True),
    ) as mock_get_user_info, patch.object(
        jwt_handler, "get_org_id", return_value=None
    ) as mock_get_org_id, patch.object(
        jwt_handler, "get_end_user_id", return_value=None
    ) as mock_get_end_user_id, patch.object(
        JWTAuthManager, "check_admin_access", new_callable=AsyncMock, return_value=None
    ) as mock_check_admin, patch.object(
        JWTAuthManager,
        "find_and_validate_specific_team_id",
        new_callable=AsyncMock,
        return_value=(None, None),
    ) as mock_find_team, patch.object(
        JWTAuthManager, "get_all_team_ids", return_value=set()
    ) as mock_get_all_team_ids, patch.object(
        JWTAuthManager,
        "find_team_with_model_access",
        new_callable=AsyncMock,
        return_value=(None, None),
    ) as mock_find_team_access, patch.object(
        JWTAuthManager,
        "get_objects",
        new_callable=AsyncMock,
        return_value=(user_object, None, None),
    ) as mock_get_objects, patch.object(
        JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
    ) as mock_map_user, patch.object(
        JWTAuthManager, "validate_object_id", return_value=True
    ) as mock_validate_object:
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
                JWTLiteLLMRoleMap(jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN)
            ],
            roles_jwt_field="roles",
            team_ids_jwt_field="my_id_teams",
            sync_user_role_and_teams=True
        ),
    )

    token = {"roles": ["ADMIN"], "my_id_teams": ["team1", "team2"]}

    user = LiteLLM_UserTable(user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER.value, teams=["team2"])

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
                JWTLiteLLMRoleMap(jwt_role="ADMIN", litellm_role=LitellmUserRoles.PROXY_ADMIN),
                # Wildcard patterns
                JWTLiteLLMRoleMap(jwt_role="user_*", litellm_role=LitellmUserRoles.INTERNAL_USER),
                JWTLiteLLMRoleMap(jwt_role="team_?", litellm_role=LitellmUserRoles.TEAM),
                JWTLiteLLMRoleMap(jwt_role="dev_[123]", litellm_role=LitellmUserRoles.INTERNAL_USER),
            ],
            roles_jwt_field="roles"
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
        JWTLiteLLMRoleMap(jwt_role="dev_[123]", litellm_role=LitellmUserRoles.INTERNAL_USER),
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
    from litellm.proxy.auth.handle_jwt import JWTHandler
    from litellm.proxy._types import LiteLLM_JWTAuth
    
    # Create JWT handler
    jwt_handler = JWTHandler()
    
    # Test token with nested claims
    nested_token = {
        "user": {
            "sub": "u123",
            "email": "user@example.com"
        },
        "resource_access": {
            "my-client": {
                "roles": ["admin", "user"]
            }
        },
        "groups": ["team1", "team2"],
        "organization": {
            "id": "org456"
        },
        "profile": {
            "object_id": "obj789"
        },
        "customer": {
            "end_user_id": "customer123"
        },
        "tenant": {
            "team_id": "team456"
        }
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
        "team_id": "team456"
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
    from litellm.proxy._types import RoleMapping, LitellmUserRoles
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        object_id_jwt_field="profile.object_id",
        role_mappings=[RoleMapping(role="admin", internal_role=LitellmUserRoles.INTERNAL_USER)]
    )
    assert jwt_handler.get_object_id(nested_token, None) == "obj789"
    
    # Test 5b: object_id_jwt_field with flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        object_id_jwt_field="object_id",
        role_mappings=[RoleMapping(role="admin", internal_role=LitellmUserRoles.INTERNAL_USER)]
    )
    assert jwt_handler.get_object_id(flat_token, None) == "obj789"

    # Test 6: end_user_id_jwt_field with nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(end_user_id_jwt_field="customer.end_user_id")
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
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(roles_jwt_field="resource_access.my-client.roles")
    assert jwt_handler.get_jwt_role(nested_token, []) == ["admin", "user"]

    # Test 9: user_roles_jwt_field with nested access (already supported, but testing)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_roles_jwt_field="resource_access.my-client.roles",
        user_allowed_roles=["admin", "user"]
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
    from litellm.proxy.auth.handle_jwt import JWTHandler
    from litellm.proxy._types import LiteLLM_JWTAuth
    
    # Create JWT handler
    jwt_handler = JWTHandler()
    
    # Test token with missing nested paths
    incomplete_token = {
        "user": {
            "name": "test user"
            # missing "sub" and "email"
        },
        "resource_access": {
            "other-client": {
                "roles": ["viewer"]
            }
            # missing "my-client"
        }
        # missing "organization", "profile", "customer", "tenant", "groups"
    }

    # Test 1: Missing user.sub should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_id_jwt_field="user.sub")
    assert jwt_handler.get_user_id(incomplete_token, "default_user") == "default_user"

    # Test 2: Missing user.email should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_email_jwt_field="user.email")
    assert jwt_handler.get_user_email(incomplete_token, "default@example.com") == "default@example.com"

    # Test 3: Missing groups should return empty list
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_ids_jwt_field="groups")
    assert jwt_handler.get_team_ids_from_jwt(incomplete_token) == []

    # Test 4: Missing organization.id should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_id_jwt_field="organization.id")
    assert jwt_handler.get_org_id(incomplete_token, "default_org") == "default_org"

    # Test 5: Missing profile.object_id should return default (requires role_mappings)
    from litellm.proxy._types import RoleMapping, LitellmUserRoles
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        object_id_jwt_field="profile.object_id",
        role_mappings=[RoleMapping(role="admin", internal_role=LitellmUserRoles.INTERNAL_USER)]
    )
    assert jwt_handler.get_object_id(incomplete_token, "default_obj") == "default_obj"

    # Test 6: Missing customer.end_user_id should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(end_user_id_jwt_field="customer.end_user_id")
    assert jwt_handler.get_end_user_id(incomplete_token, "default_customer") == "default_customer"

    # Test 7: Missing tenant.team_id should use team_id_default fallback
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        team_id_jwt_field="tenant.team_id",
        team_id_default="fallback_team"
    )
    assert jwt_handler.get_team_id(incomplete_token, "default_team") == "fallback_team"

    # Test 8: Missing resource_access.my-client.roles should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(roles_jwt_field="resource_access.my-client.roles")
    assert jwt_handler.get_jwt_role(incomplete_token, ["default_role"]) == ["default_role"]

    # Test 9: Missing nested user roles should return default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        user_roles_jwt_field="resource_access.my-client.roles",
        user_allowed_roles=["admin", "user"]
    )
    assert jwt_handler.get_user_roles(incomplete_token, ["default_user_role"]) == ["default_user_role"]


@pytest.mark.asyncio  
async def test_metadata_prefix_handling_in_nested_fields():
    """
    Test that metadata. prefix is properly handled in nested JWT field access
    
    The get_nested_value function should remove metadata. prefix before traversing
    """
    from litellm.proxy.auth.handle_jwt import JWTHandler
    from litellm.proxy._types import LiteLLM_JWTAuth
    
    # Create JWT handler
    jwt_handler = JWTHandler()
    
    # Test token with proper structure for metadata prefix removal
    token = {
        "user": {
            "email": "user@example.com"  # This will be accessed when metadata.user.email is used
        },
        "sub": "u123"
    }

    # Test 1: metadata.user.email should access user.email after prefix removal
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_email_jwt_field="metadata.user.email")
    # The get_nested_value function removes "metadata." prefix, so "metadata.user.email" becomes "user.email"
    assert jwt_handler.get_user_email(token, None) == "user@example.com"

    # Test 2: user.sub should work normally without metadata prefix
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(user_id_jwt_field="sub")
    assert jwt_handler.get_user_id(token, None) == "u123"

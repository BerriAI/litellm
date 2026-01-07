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
        return_value=(user_object, None, None, None),
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
        return_value=(user_object, None, None, None),
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
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

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
    from litellm.proxy._types import LitellmUserRoles, RoleMapping
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
    from litellm.proxy._types import LitellmUserRoles, RoleMapping
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
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

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
            budget_id="budget_123",
            rpm_limit=100,
            tpm_limit=5000
        )
    )
    
    user_object = LiteLLM_UserTable(
        user_id=_user_id, 
        user_role=LitellmUserRoles.INTERNAL_USER
    )
    
    team_object = LiteLLM_TeamTable(team_id=_team_id)

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
        return_value=(_user_id, "test@example.com", True),
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
        return_value=(_team_id, team_object),
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
        return_value=(user_object, None, None, mock_team_membership),
    ) as mock_get_objects, patch.object(
        JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
    ) as mock_map_user, patch.object(
        JWTAuthManager, "validate_object_id", return_value=True
    ) as mock_validate_object, patch.object(
        JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
    ) as mock_sync_user:
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
        assert result["team_membership"] is not None, "team_membership should be present"
        assert result["team_membership"] == mock_team_membership, "team_membership should match the mock object"
        assert result["team_membership"].user_id == _user_id, "team_membership user_id should match"
        assert result["team_membership"].team_id == _team_id, "team_membership team_id should match"
        assert result["team_membership"].budget_id == "budget_123", "team_membership budget_id should match"
        assert result["team_membership"].spend == 10.5, "team_membership spend should match"


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
    with patch.object(
        jwt_handler, "get_oidc_userinfo", new_callable=AsyncMock
    ) as mock_get_userinfo, patch.object(
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
        return_value=(user_object, None, None, None),
    ) as mock_get_objects, patch.object(
        JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
    ) as mock_map_user, patch.object(
        JWTAuthManager, "validate_object_id", return_value=True
    ) as mock_validate_object, patch.object(
        JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
    ) as mock_sync_user:
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
    with patch.object(
        jwt_handler, "get_oidc_userinfo", new_callable=AsyncMock
    ) as mock_get_userinfo, patch.object(
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
        return_value=("test_user_1", None, None),
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
        return_value=(user_object, None, None, None),
    ) as mock_get_objects, patch.object(
        JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock
    ) as mock_map_user, patch.object(
        JWTAuthManager, "validate_object_id", return_value=True
    ) as mock_validate_object, patch.object(
        JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock
    ) as mock_sync_user:
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
    user_object = LiteLLM_UserTable(user_id="user-1", user_role=LitellmUserRoles.INTERNAL_USER)

    with patch.object(jwt_handler, "auth_jwt", new_callable=AsyncMock) as mock_auth_jwt, \
         patch.object(JWTAuthManager, "check_rbac_role", new_callable=AsyncMock), \
         patch.object(JWTAuthManager, "check_admin_access", new_callable=AsyncMock, return_value=None), \
         patch("litellm.proxy.auth.handle_jwt.get_team_object", new_callable=AsyncMock) as mock_get_team, \
         patch.object(JWTAuthManager, "get_objects", new_callable=AsyncMock, return_value=(user_object, None, None, None)), \
         patch.object(JWTAuthManager, "map_user_to_teams", new_callable=AsyncMock), \
         patch.object(JWTAuthManager, "sync_user_role_and_teams", new_callable=AsyncMock):

        mock_auth_jwt.return_value = {"sub": "user-1", "scope": "", "groups": ["team-1", "team-2"]}
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
async def test_get_team_alias_with_nested_fields():
    """
    Test get_team_alias() method with nested JWT fields
    """
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler

    jwt_handler = JWTHandler()
    
    # Test token with nested team name
    nested_token = {
        "organization": {
            "team": {
                "name": "engineering-team"
            }
        },
        "team_name": "flat-team"
    }
    
    # Test nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_alias_jwt_field="organization.team.name")
    assert jwt_handler.get_team_alias(nested_token, None) == "engineering-team"
    
    # Test flat access (backward compatibility)
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_alias_jwt_field="team_name")
    assert jwt_handler.get_team_alias(nested_token, None) == "flat-team"
    
    # Test missing field returns default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(team_alias_jwt_field="nonexistent.field")
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
        team_id_jwt_field="team_id",
        team_alias_jwt_field="team_name"
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
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_alias_jwt_field="team_alias"
        ),
    )
    
    # Token with team name (no team_id)
    jwt_token = {
        "sub": "user-1",
        "team_alias": "my-team"
    }
    
    # Mock team object returned by get_team_object_by_alias
    team_object = LiteLLM_TeamTable(team_id="resolved-team-id", team_alias="my-team")
    
    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object_by_alias",
        new_callable=AsyncMock
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
            team_id_jwt_field="team_id",
            team_alias_jwt_field="team_alias"
        ),
    )
    
    # Token with both team_id and team name
    jwt_token = {
        "sub": "user-1",
        "team_id": "direct-team-id",
        "team_alias": "my-team"
    }
    
    # Mock team object returned by get_team_object (by ID)
    team_object = LiteLLM_TeamTable(team_id="direct-team-id")
    
    with patch(
        "litellm.proxy.auth.handle_jwt.get_team_object",
        new_callable=AsyncMock
    ) as mock_get_by_id, patch(
        "litellm.proxy.auth.handle_jwt.get_team_object_by_alias",
        new_callable=AsyncMock
    ) as mock_get_by_alias:
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
    jwt_token = {
        "sub": "user-1"
    }
    
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
        "company": {
            "organization": {
                "name": "acme-corp"
            }
        },
        "org_name": "flat-org"
    }
    
    # Test nested access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_alias_jwt_field="company.organization.name")
    assert jwt_handler.get_org_alias(nested_token, None) == "acme-corp"
    
    # Test flat access
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_alias_jwt_field="org_name")
    assert jwt_handler.get_org_alias(nested_token, None) == "flat-org"
    
    # Test missing field returns default
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(org_alias_jwt_field="nonexistent.field")
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
        litellm_jwtauth=LiteLLM_JWTAuth(
            org_alias_jwt_field="org_alias"
        ),
    )
    
    # Mock org object returned by get_org_object_by_alias
    org_object = LiteLLM_OrganizationTable(
        organization_id="resolved-org-id",
        organization_alias="my-org",
        budget_id="budget-1",
        created_by="admin",
        updated_by="admin",
        models=[]
    )
    
    with patch(
        "litellm.proxy.auth.handle_jwt.get_org_object_by_alias",
        new_callable=AsyncMock
    ) as mock_get_by_alias:
        mock_get_by_alias.return_value = org_object
        
        (
            result_user_obj,
            result_org_obj,
            result_end_user_obj,
            result_team_membership,
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




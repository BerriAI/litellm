import logging
import time
from collections.abc import Mapping
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import HTTPException
from pytest_mock import MockerFixture

from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    NewUserRequest,
    NewUserResponse,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.proxy.management_endpoints.scim.scim_v2 import (
    SCIMRosterSyncError,
    UserProvisionerHelpers,
    _apply_group_patch_updates,
    _extract_group_member_ids,
    _extract_ids_from_path_filter,
    _handle_group_membership_changes,
    _handle_team_membership_changes,
    _parse_member_entries,
    _process_group_patch_operations,
    _recompute_scim_member_roles,
    _resolve_group_member_ids,
    create_group,
    create_user,
    delete_group,
    delete_user,
    get_groups,
    get_users,
    get_service_provider_config,
    patch_group,
    patch_team_membership,
    patch_user,
    update_group,
    update_user,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIM_ENTERPRISE_USER_SCHEMA,
    SCIM_MANAGED_TEAM_METADATA_KEY,
    SCIM_TEAM_DATA_METADATA_KEY,
    SCIMGroup,
    SCIMMember,
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMServiceProviderConfig,
    SCIMUser,
    SCIMUserEmail,
    SCIMUserGroup,
    SCIMUserName,
)


@pytest.mark.asyncio
async def test_create_user_existing_user_conflict(mocker):
    """If a user already exists, create_user should raise ScimUserAlreadyExists"""

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="existing-user",
        name=SCIMUserName(familyName="User", givenName="Existing"),
        emails=[SCIMUserEmail(value="existing@example.com")],
    )

    # Create a properly structured mock for the prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value={"user_id": "existing-user"})
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    # Mock the _get_prisma_client_or_raise_exception to return our mock
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    mocked_new_user = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_user(user=scim_user)

    # Check that it's an HTTPException with status 409
    assert exc_info.value.status_code == 409
    assert "existing-user" in str(exc_info.value.detail)
    mocked_new_user.assert_not_called()


@pytest.mark.asyncio
async def test_create_user_defaults_to_viewer(mocker, monkeypatch):
    """If no role provided, new user should default to viewer"""

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="new-user",
        name=SCIMUserName(familyName="User", givenName="New"),
        emails=[SCIMUserEmail(value="new@example.com")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="new-user")),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    called_args = new_user_mock.call_args.kwargs["data"]
    assert called_args.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_create_user_ingests_enterprise_extension(mocker, monkeypatch):
    """A SCIM create payload carrying the enterprise extension block should land
    in the created user's metadata under scim_enterprise"""

    scim_user = SCIMUser.model_validate(
        {
            "schemas": [
                "urn:ietf:params:scim:schemas:core:2.0:User",
                SCIM_ENTERPRISE_USER_SCHEMA,
            ],
            "userName": "ent-user",
            "name": {"familyName": "User", "givenName": "Ent"},
            "emails": [{"value": "ent@example.com"}],
            SCIM_ENTERPRISE_USER_SCHEMA: {
                "costCenter": "CC-42",
                "department": "Platform",
            },
        }
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="ent-user")),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    created_metadata = new_user_mock.call_args.kwargs["data"].metadata
    assert created_metadata["scim_enterprise"] == {
        "costCenter": "CC-42",
        "department": "Platform",
    }


@pytest.mark.asyncio
async def test_create_user_ingests_entitlements_and_roles(mocker, monkeypatch):
    """A SCIM create payload carrying entitlements and roles should land in the
    created user's metadata under scim_entitlements and scim_roles"""

    scim_user = SCIMUser.model_validate(
        {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "entitled-user",
            "name": {"familyName": "User", "givenName": "Entitled"},
            "emails": [{"value": "entitled@example.com"}],
            "entitlements": [
                {
                    "value": "jira-software",
                    "display": "Jira Software",
                    "type": "app",
                    "primary": True,
                },
                "bare-entitlement",
            ],
            "roles": [{"value": "engineering-admin", "type": "role"}],
        }
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="entitled-user")),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    created_metadata = new_user_mock.call_args.kwargs["data"].metadata
    assert created_metadata["scim_entitlements"] == [
        {
            "value": "jira-software",
            "display": "Jira Software",
            "type": "app",
            "primary": True,
        },
        {"value": "bare-entitlement"},
    ]
    assert created_metadata["scim_roles"] == [{"value": "engineering-admin", "type": "role"}]


@pytest.mark.asyncio
async def test_create_user_uses_default_internal_user_params_role(mocker, monkeypatch):
    """If role is set in default_internal_user_params, new user should use that role"""

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="new-user",
        name=SCIMUserName(familyName="User", givenName="New"),
        emails=[SCIMUserEmail(value="new@example.com")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    # Set default_internal_user_params with a specific role
    default_params = {
        "user_role": LitellmUserRoles.PROXY_ADMIN,
    }
    monkeypatch.setattr("litellm.default_internal_user_params", default_params, raising=False)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="new-user")),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    called_args = new_user_mock.call_args.kwargs["data"]
    assert called_args.user_role == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.asyncio
async def test_scim_create_user_respects_default_role_set_via_ui(mocker, monkeypatch):
    """
    Default user role set to 'Internal User' via UI,
    but SCIM-created users get 'Internal Viewer' instead.

    The UI saves the setting via _update_litellm_setting, which should update
    litellm.default_internal_user_params in memory. Then SCIM create_user
    should read that in-memory value and assign the correct role.

    This test simulates the full flow:
    1. Start with default_internal_user_params = None
    2. Call update_internal_user_settings (the UI endpoint) to set role to INTERNAL_USER
    3. Create a user via SCIM
    4. Assert the user gets INTERNAL_USER (not INTERNAL_USER_VIEW_ONLY)
    """
    from litellm.proxy._types import DefaultInternalUserParams
    from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
        _update_litellm_setting,
    )

    # Step 1: Start with no default params (fresh proxy state)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    # Step 2: Simulate the UI saving "Internal User (Create/Delete/View)" as default role
    # Mock the proxy_config and store_model_in_db that _update_litellm_setting needs
    mock_proxy_config = mocker.MagicMock()
    mock_proxy_config.get_config = AsyncMock(return_value={"litellm_settings": {}})
    mock_proxy_config.save_config = AsyncMock()

    mocker.patch(
        "litellm.proxy.proxy_server.proxy_config",
        mock_proxy_config,
    )
    mocker.patch(
        "litellm.proxy.proxy_server.store_model_in_db",
        True,
    )

    import litellm
    from litellm.proxy._types import UserAPIKeyAuth

    settings = DefaultInternalUserParams(
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    await _update_litellm_setting(
        settings=settings,
        settings_key="default_internal_user_params",
        success_message="ok",
        user_api_key_dict=UserAPIKeyAuth(user_id="test-admin"),
    )

    # Verify the in-memory variable was actually updated
    assert litellm.default_internal_user_params is not None, (
        "BUG: _update_litellm_setting did not update litellm.default_internal_user_params in memory. "
        "The local variable reassignment (in_memory_var = ...) doesn't propagate back."
    )
    assert litellm.default_internal_user_params.get("user_role") == LitellmUserRoles.INTERNAL_USER

    # Step 3: Create a user via SCIM
    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="idontexist@example.com",
        emails=[SCIMUserEmail(value="idontexist@example.com")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="idontexist@example.com")),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    # Step 4: Verify the user got INTERNAL_USER, not INTERNAL_USER_VIEW_ONLY
    called_args = new_user_mock.call_args.kwargs["data"]
    assert called_args.user_role == LitellmUserRoles.INTERNAL_USER, (
        f"BUG: SCIM created user with role {called_args.user_role} instead of "
        f"{LitellmUserRoles.INTERNAL_USER}. The default_internal_user_params "
        f"in-memory variable was not updated by _update_litellm_setting."
    )


@pytest.mark.asyncio
async def test_get_users_filters_username_by_exposed_scim_username_for_okta(mocker):
    """
    Okta deprovisioning first locates a user with `userName eq "<email>"`.
    LiteLLM exposes SCIM userName from user_email, so the lookup must match
    user_email even when the internal user_id is a UUID.
    """
    user = LiteLLM_UserTable(
        user_id="internal-user-id",
        user_email="okta.user@example.com",
        user_alias="Okta User",
        teams=[],
        metadata={},
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[user])
    mock_prisma_client.db.litellm_usertable.count = AsyncMock(return_value=1)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(
            return_value=SCIMUser(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
                id="internal-user-id",
                userName="okta.user@example.com",
                emails=[SCIMUserEmail(value="okta.user@example.com")],
            )
        ),
    )

    response = await get_users(
        startIndex=1,
        count=10,
        filter='userName eq "okta.user@example.com"',
    )

    expected_where = {
        "OR": [
            {"user_email": "okta.user@example.com"},
            {"user_id": "okta.user@example.com"},
        ]
    }
    mock_prisma_client.db.litellm_usertable.find_many.assert_awaited_once_with(
        where=expected_where,
        skip=0,
        take=10,
        order={"created_at": "desc"},
    )
    mock_prisma_client.db.litellm_usertable.count.assert_awaited_once_with(where=expected_where)
    assert response.totalResults == 1
    assert response.Resources[0].id == "internal-user-id"


@pytest.mark.asyncio
async def test_get_users_filters_email_value_by_user_email(mocker):
    """
    SCIM clients can locate users with `emails.value eq "<email>"`; keep that
    filter as a direct user_email lookup alongside the userName fallback query.
    """
    user = LiteLLM_UserTable(
        user_id="internal-user-id",
        user_email="scim.user@example.com",
        user_alias="SCIM User",
        teams=[],
        metadata={},
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[user])
    mock_prisma_client.db.litellm_usertable.count = AsyncMock(return_value=1)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(
            return_value=SCIMUser(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
                id="internal-user-id",
                userName="scim.user@example.com",
                emails=[SCIMUserEmail(value="scim.user@example.com")],
            )
        ),
    )

    response = await get_users(
        startIndex=1,
        count=10,
        filter='emails.value eq "scim.user@example.com"',
    )

    expected_where = {"user_email": "scim.user@example.com"}
    mock_prisma_client.db.litellm_usertable.find_many.assert_awaited_once_with(
        where=expected_where,
        skip=0,
        take=10,
        order={"created_at": "desc"},
    )
    mock_prisma_client.db.litellm_usertable.count.assert_awaited_once_with(where=expected_where)
    assert response.totalResults == 1
    assert response.Resources[0].id == "internal-user-id"


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_no_email(mocker):
    """Should return None when new_user_request has no email"""
    mock_prisma_client = mocker.MagicMock()

    new_user_request = NewUserRequest(
        user_id="test-user",
        user_email=None,  # No email provided
        user_alias="Test User",
        teams=[],
        metadata={},
        auto_create_key=False,
    )

    result = await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    assert result is None


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_no_existing_user(mocker):
    """Should return None when no existing user is found with the email"""
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    new_user_request = NewUserRequest(
        user_id="test-user",
        user_email="test@example.com",
        user_alias="Test User",
        teams=["team1"],
        metadata={"key": "value"},
        auto_create_key=False,
    )

    result = await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    assert result is None
    mock_prisma_client.db.litellm_usertable.find_first.assert_called_once_with(where={"user_email": "test@example.com"})


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_existing_user_updated(mocker):
    """Should keep the existing user_id, sync team roster, and return SCIMUser

    Regression: a SCIM userName differing from the matched row's user_id used to
    re-key the user row, orphaning virtual keys, team rosters, memberships and
    spend logs that still referenced the old id.
    """
    existing_user = mocker.MagicMock()
    existing_user.user_id = "old-user-id"
    existing_user.user_email = "test@example.com"
    existing_user.user_alias = "Old Name"
    existing_user.teams = ["old-team"]
    existing_user.metadata = {"old": "data"}

    updated_user = {
        "user_id": "old-user-id",
        "user_email": "test@example.com",
        "user_alias": "New Name",
        "teams": ["new-team"],
        "metadata": '{"new": "data"}',
    }

    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="old-user-id",
        userName="test@example.com",
        name=SCIMUserName(familyName="Name", givenName="New"),
        emails=[SCIMUserEmail(value="test@example.com")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)

    mock_transform = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=mock_scim_user),
    )
    mock_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="new-user-id",
        user_email="test@example.com",
        user_alias="New Name",
        teams=["new-team"],
        metadata={"new": "data"},
        auto_create_key=False,
    )

    result = await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    assert result == mock_scim_user

    mock_prisma_client.db.litellm_usertable.find_first.assert_called_once_with(where={"user_email": "test@example.com"})

    update_calls = mock_prisma_client.db.litellm_usertable.update.call_args_list
    assert len(update_calls) == 1
    assert update_calls[0].kwargs == {
        "where": {"user_id": "old-user-id"},
        "data": {
            "user_email": "test@example.com",
            "user_alias": "New Name",
            "teams": ["new-team"],
            "metadata": '{"new": "data"}',
        },
    }

    mock_membership.assert_awaited_once_with(
        user_id="old-user-id",
        existing_teams=["old-team"],
        new_teams=["new-team"],
    )

    mock_transform.assert_called_once_with(updated_user)


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_roster_changes_use_existing_user_id(mocker):
    """Roster add/remove must be issued for the matched row's user_id, not the SCIM userName.

    Regression: the rename made removals run against the new id, so a roster still
    holding the old id reported "User not found in team" and the stale entry survived.
    """
    existing_user = mocker.MagicMock()
    existing_user.user_id = "oidc-sub-123"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = ["old-team"]
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mock_team_member_add = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(),
    )
    mock_team_member_delete = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=None),
    )

    new_user_request = NewUserRequest(
        user_id="scim-username",
        user_email="member@example.com",
        user_alias="Member",
        teams=["new-team"],
        metadata={},
        auto_create_key=False,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    assert mock_team_member_add.await_args.kwargs["data"].member.user_id == "oidc-sub-123"
    assert mock_team_member_delete.await_args.kwargs["data"].user_id == "oidc-sub-123"
    assert mock_prisma_client.db.litellm_usertable.update.await_args.kwargs["where"] == {"user_id": "oidc-sub-123"}


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_syncs_roster_and_dedups_teams(mocker):
    """Existing-email upsert must add the user to the team roster via the shared
    team_member_add path and dedup the teams built from repeated SCIM groups.

    Regression: previously the user's ``teams`` array was raw-written (with
    duplicates) and the team roster (members_with_roles / LiteLLM_TeamMembership)
    was never touched, so the user appeared in the group on their profile but was
    absent from the team directly.
    """
    existing_user = mocker.MagicMock()
    existing_user.user_id = "same-id"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = []
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=None),
    )
    mock_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="same-id",
        user_email="member@example.com",
        user_alias="Member",
        teams=["team-a", "team-a", "team-b"],
        metadata={},
        auto_create_key=False,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    mock_membership.assert_awaited_once_with(
        user_id="same-id",
        existing_teams=[],
        new_teams=["team-a", "team-b"],
    )

    update_calls = mock_prisma_client.db.litellm_usertable.update.call_args_list
    assert len(update_calls) == 1
    assert update_calls[0].kwargs["where"] == {"user_id": "same-id"}
    assert update_calls[0].kwargs["data"]["teams"] == ["team-a", "team-b"]


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_without_teams_preserves_memberships(mocker):
    """Adoption via POST /Users without ``groups`` must keep the user's existing teams.

    Regression: Entra manages membership exclusively through /Groups and never sends
    ``groups`` on POST /Users, so the empty team list was treated as the desired
    state and the adopted user was removed from every team roster and had ``teams``
    overwritten with [].
    """
    existing_user = mocker.MagicMock()
    existing_user.user_id = "adopted-id"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = ["team-a", "team-b"]
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mocker.patch(  # test-quality-ok: roster helpers are module-level, not injectable into the helper
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=None),
    )
    mock_team_member_add = mocker.patch(  # test-quality-ok: roster helpers are module-level, not injectable into the helper
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(),
    )
    mock_team_member_delete = mocker.patch(  # test-quality-ok: roster helpers are module-level, not injectable into the helper
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="entra-object-id",
        user_email="member@example.com",
        user_alias="Member",
        teams=[],
        metadata={},
        auto_create_key=False,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    mock_team_member_add.assert_not_awaited()
    mock_team_member_delete.assert_not_awaited()

    update_calls = mock_prisma_client.db.litellm_usertable.update.call_args_list
    assert len(update_calls) == 1
    assert update_calls[0].kwargs["where"] == {"user_id": "adopted-id"}
    assert update_calls[0].kwargs["data"]["teams"] == ["team-a", "team-b"]


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_roster_add_failure_blocks_teams_write(mocker):
    """A genuine roster add failure must propagate and must not persist the teams array.

    Regression: the roster sync went through patch_team_membership which swallowed
    real team_member_add failures, so the endpoint reported success and wrote a
    teams array listing a team the roster never received. The strict path now
    surfaces the failure so user.teams and members_with_roles cannot diverge.
    """
    existing_user = mocker.MagicMock()
    existing_user.user_id = "uid"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = []
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mock_team_member_add = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "Team not found"})),
    )

    new_user_request = NewUserRequest(
        user_id="uid",
        user_email="member@example.com",
        user_alias="Member",
        teams=["missing-team"],
        metadata={},
        auto_create_key=False,
    )

    with pytest.raises(SCIMRosterSyncError) as exc_info:
        await UserProvisionerHelpers.handle_existing_user_by_email(
            prisma_client=mock_prisma_client, new_user_request=new_user_request
        )

    mock_team_member_add.assert_awaited_once()
    assert "add uid to missing-team" in str(exc_info.value)
    assert mock_prisma_client.db.litellm_usertable.update.await_count == 0


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_roster_add_already_member_is_noop(mocker):
    """Being already in the team is benign even under the strict path: the upsert
    succeeds and the deduped teams array is still persisted."""
    existing_user = mocker.MagicMock()
    existing_user.user_id = "uid"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = []
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mock_team_member_add = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(
            side_effect=ProxyException(
                message="already in team",
                type=ProxyErrorTypes.team_member_already_in_team.value,
                param=None,
                code=400,
            )
        ),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=None),
    )

    new_user_request = NewUserRequest(
        user_id="uid",
        user_email="member@example.com",
        user_alias="Member",
        teams=["team-x"],
        metadata={},
        auto_create_key=False,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    mock_team_member_add.assert_awaited_once()
    update_calls = mock_prisma_client.db.litellm_usertable.update.call_args_list
    assert len(update_calls) == 1
    assert update_calls[0].kwargs["data"]["teams"] == ["team-x"]


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_roster_remove_failure_blocks_teams_write(mocker):
    """A genuine roster removal failure must propagate and must not persist the teams array,
    symmetrically with add failures, so user.teams cannot drop a team the roster still holds."""
    existing_user = mocker.MagicMock()
    existing_user.user_id = "uid"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = ["old-team"]
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mock_team_member_delete = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(side_effect=HTTPException(status_code=500, detail={"error": "No db connected"})),
    )

    mocker.patch(  # test-quality-ok: roster helpers are module-level, not injectable into the helper
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="uid",
        user_email="member@example.com",
        user_alias="Member",
        teams=["replacement-team"],
        metadata={},
        auto_create_key=False,
    )

    with pytest.raises(SCIMRosterSyncError) as exc_info:
        await UserProvisionerHelpers.handle_existing_user_by_email(
            prisma_client=mock_prisma_client, new_user_request=new_user_request
        )

    mock_team_member_delete.assert_awaited_once()
    assert "remove uid from old-team" in str(exc_info.value)
    assert mock_prisma_client.db.litellm_usertable.update.await_count == 0


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_roster_remove_already_absent_is_noop(mocker):
    """A user already absent from the team is the idempotent removal no-op even under the
    strict path: the upsert succeeds and the deduped teams array is still persisted."""
    existing_user = mocker.MagicMock()
    existing_user.user_id = "uid"
    existing_user.user_email = "member@example.com"
    existing_user.user_alias = "Member"
    existing_user.teams = ["old-team"]
    existing_user.metadata = {}

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={})

    mock_team_member_delete = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(side_effect=HTTPException(status_code=400, detail={"error": "User not found in team"})),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=None),
    )

    mocker.patch(  # test-quality-ok: roster helpers are module-level, not injectable into the helper
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="uid",
        user_email="member@example.com",
        user_alias="Member",
        teams=["replacement-team"],
        metadata={},
        auto_create_key=False,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client, new_user_request=new_user_request
    )

    mock_team_member_delete.assert_awaited_once()
    update_calls = mock_prisma_client.db.litellm_usertable.update.call_args_list
    assert len(update_calls) == 1
    assert update_calls[0].kwargs["data"]["teams"] == ["replacement-team"]


@pytest.mark.asyncio
async def test_handle_team_membership_changes_no_changes(mocker):
    """Should not call patch_team_membership when existing teams equal new teams"""
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Same teams - no changes
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1", "team2"],
        new_teams=["team1", "team2"],
    )

    # Should not be called since no changes
    mock_patch_team_membership.assert_not_called()


@pytest.mark.asyncio
async def test_handle_team_membership_changes_add_teams(mocker):
    """Should call patch_team_membership with teams to add"""
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Adding teams
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1"],
        new_teams=["team1", "team2", "team3"],
    )

    # Verify the call was made once
    mock_patch_team_membership.assert_called_once()

    # Check the arguments more flexibly to handle order variations
    call_args = mock_patch_team_membership.call_args
    assert call_args[1]["user_id"] == "test-user"
    assert set(call_args[1]["teams_ids_to_add_user_to"]) == {"team2", "team3"}
    assert call_args[1]["teams_ids_to_remove_user_from"] == []


@pytest.mark.asyncio
async def test_handle_team_membership_changes_remove_teams(mocker):
    """Should call patch_team_membership with teams to remove"""
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Removing teams
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1", "team2", "team3"],
        new_teams=["team1"],
    )

    # Verify the call was made once
    mock_patch_team_membership.assert_called_once()

    # Check the arguments more flexibly to handle order variations
    call_args = mock_patch_team_membership.call_args
    assert call_args[1]["user_id"] == "test-user"
    assert call_args[1]["teams_ids_to_add_user_to"] == []
    assert set(call_args[1]["teams_ids_to_remove_user_from"]) == {"team2", "team3"}


@pytest.mark.asyncio
async def test_handle_team_membership_changes_add_and_remove(mocker):
    """Should call patch_team_membership with both teams to add and remove"""
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Both adding and removing teams
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1", "team2"],
        new_teams=["team2", "team3"],
    )

    # Verify the call was made once
    mock_patch_team_membership.assert_called_once()

    # Check the arguments - team1 should be removed, team3 should be added, team2 stays
    call_args = mock_patch_team_membership.call_args
    assert call_args[1]["user_id"] == "test-user"
    assert call_args[1]["teams_ids_to_add_user_to"] == ["team3"]
    assert call_args[1]["teams_ids_to_remove_user_from"] == ["team1"]


@pytest.mark.asyncio
async def test_update_user_success(mocker):
    """Should successfully update user with PUT request"""
    # Mock existing user
    existing_user = mocker.MagicMock()
    existing_user.teams = ["old-team"]

    # Mock updated user
    updated_user = {
        "user_id": "test-user",
        "user_email": "updated@example.com",
        "user_alias": "Updated User",
        "teams": ["new-team"],
        "metadata": '{"scim_metadata": {"givenName": "Updated", "familyName": "User"}}',
    }

    # Mock SCIM user for request
    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="test-user",
        name=SCIMUserName(familyName="User", givenName="Updated"),
        emails=[SCIMUserEmail(value="updated@example.com")],
        groups=[SCIMUserGroup(value="new-team")],
    )

    # Mock SCIM user for response
    response_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="test-user",
        userName="test-user",
        name=SCIMUserName(familyName="User", givenName="Updated"),
        emails=[SCIMUserEmail(value="updated@example.com")],
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=response_scim_user),
    )

    # Call update_user
    result = await update_user(user_id="test-user", user=scim_user)

    # Verify result
    assert result == response_scim_user

    # Verify database update was called with correct data
    mock_prisma_client.db.litellm_usertable.update.assert_called_once()
    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["where"] == {"user_id": "test-user"}
    assert call_args[1]["data"]["user_email"] == "updated@example.com"
    assert call_args[1]["data"]["teams"] == ["new-team"]


@pytest.mark.asyncio
async def test_update_user_not_found(mocker):
    """Should raise 404 when user doesn't exist"""
    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="nonexistent-user",
        name=SCIMUserName(familyName="User", givenName="Test"),
        emails=[SCIMUserEmail(value="test@example.com")],
    )

    # Mock dependencies to raise HTTPException for user not found
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "User not found"})),
    )

    # Should raise ProxyException (which wraps the HTTPException)
    with pytest.raises(ProxyException):
        await update_user(user_id="nonexistent-user", user=scim_user)


@pytest.mark.asyncio
async def test_patch_user_success(mocker):
    """Should successfully patch user with PATCH request"""
    # Mock existing user
    existing_user = mocker.MagicMock()
    existing_user.teams = ["team1"]
    existing_user.metadata = {}

    # Mock updated user
    updated_user = {
        "user_id": "test-user",
        "user_alias": "Patched User",
        "teams": ["team1", "team2"],
        "metadata": '{"scim_metadata": {}}',
    }

    # Mock patch operations
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[
            SCIMPatchOperation(op="replace", path="displayName", value="Patched User"),
            SCIMPatchOperation(op="add", path="groups", value=[{"value": "team2"}]),
        ],
    )

    # Mock response SCIM user
    response_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="test-user",
        userName="test-user",
        name=SCIMUserName(familyName="User", givenName="Patched"),
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=response_scim_user),
    )

    # Call patch_user
    result = await patch_user(user_id="test-user", patch_ops=patch_ops)

    # Verify result
    assert result == response_scim_user

    # Verify database update was called
    mock_prisma_client.db.litellm_usertable.update.assert_called_once()
    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["where"] == {"user_id": "test-user"}


@pytest.mark.asyncio
async def test_patch_user_not_found(mocker):
    """Should raise 404 when user doesn't exist for patch"""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path="displayName", value="New Name")],
    )

    # Mock dependencies to raise HTTPException for user not found
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "User not found"})),
    )

    # Should raise ProxyException (which wraps the HTTPException)
    with pytest.raises(ProxyException):
        await patch_user(user_id="nonexistent-user", patch_ops=patch_ops)


@pytest.mark.asyncio
async def test_get_service_provider_config(mocker):
    """Test the get_service_provider_config endpoint"""
    # Mock the Request object
    mock_request = mocker.MagicMock()
    mock_request.url = "https://example.com/scim/v2/ServiceProviderConfig"

    # Call the endpoint
    result = await get_service_provider_config(mock_request)

    # Verify it returns the correct response
    assert isinstance(result, SCIMServiceProviderConfig)
    assert result.schemas == ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"]
    assert result.patch.supported is True
    assert result.bulk.supported is False
    assert result.meta is not None
    assert result.meta["resourceType"] == "ServiceProviderConfig"


@pytest.mark.asyncio
async def test_update_group_metadata_serialization_issue(mocker):
    """
    Test that update_group properly serializes metadata to avoid Prisma DataError.

    This test reproduces the issue where metadata was passed as a dict instead of
    a JSON string, causing: "Invalid argument type. `metadata` should be of any
    of the following types: `JsonNullValueInput`, `Json`"
    """
    from litellm.proxy.management_endpoints.scim.scim_v2 import update_group
    from litellm.types.proxy.management_endpoints.scim_v2 import SCIMGroup, SCIMMember

    # Create test data
    group_id = "test-group-id"
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Test Group",
        members=[SCIMMember(value="user1", display="User One")],
    )

    # Mock existing team with metadata
    mock_existing_team = mocker.MagicMock()
    mock_existing_team.team_id = group_id
    mock_existing_team.team_alias = "Old Group Name"
    mock_existing_team.members = ["user1"]
    mock_existing_team.metadata = {"existing_key": "existing_value"}
    mock_existing_team.created_at = None
    mock_existing_team.updated_at = None

    # Mock updated team response
    mock_updated_team = mocker.MagicMock()
    mock_updated_team.team_id = group_id
    mock_updated_team.team_alias = "Test Group"
    mock_updated_team.members = ["user1"]
    mock_updated_team.created_at = None
    mock_updated_team.updated_at = None

    # Create a properly structured mock for the prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock team operations
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_updated_team)

    # Mock user operations
    mock_user = mocker.MagicMock()
    mock_user.user_id = "user1"
    mock_user.user_email = "user1@example.com"  # Add proper string value for user_email
    mock_user.teams = [group_id]
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

    # Mock the _get_prisma_client_or_raise_exception to return our mock
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Mock the transformation function
    mock_scim_group_response = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Test Group",
        members=[SCIMMember(value="user1", display="User One")],
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=mock_scim_group_response),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Call the function that had the bug
    await update_group(group_id=group_id, group=scim_group)

    # Verify the team update was called
    mock_prisma_client.db.litellm_teamtable.update.assert_called_once()

    # Get the call arguments to verify metadata serialization
    call_args = mock_prisma_client.db.litellm_teamtable.update.call_args
    update_data = call_args[1]["data"]

    # Verify that metadata is properly serialized as a string, not a dict
    # This is the critical check that would have caught the original bug
    assert "metadata" in update_data
    metadata = update_data["metadata"]

    # The fix should ensure metadata is serialized as a JSON string
    assert isinstance(metadata, str), f"metadata should be a JSON string, but got {type(metadata)}"

    # Verify we can parse it back to verify it contains the expected data
    import json

    parsed_metadata = json.loads(metadata)
    assert "existing_key" in parsed_metadata
    assert "scim_data" in parsed_metadata


@pytest.mark.asyncio
async def test_team_membership_management(mocker):
    """
    Test that team membership changes work correctly:
    - Adding members to team
    - Removing members from team
    - members_with_roles is used as source of truth
    """
    from litellm.proxy._types import Member
    from litellm.proxy.management_endpoints.scim.scim_v2 import (
        _get_team_member_user_ids_from_team,
        _handle_group_membership_changes,
    )

    # Mock team with members_with_roles as source of truth
    mock_team = mocker.MagicMock()
    mock_team.members_with_roles = [
        Member(user_id="user1", role="user"),
        Member(user_id="user2", role="user"),
    ]
    mock_team.members = ["user1", "user2", "user3"]  # This should be ignored

    # Test that members_with_roles is source of truth
    member_ids = await _get_team_member_user_ids_from_team(mock_team)
    assert set(member_ids) == {"user1", "user2"}
    assert "user3" not in member_ids  # Should not be included even though in members

    # Mock patch_team_membership function
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Test adding and removing members
    group_id = "test-group-id"
    current_members = {"user1", "user2"}
    final_members = {"user2", "user3", "user4"}  # Remove user1, add user3 and user4

    await _handle_group_membership_changes(
        group_id=group_id, current_members=current_members, final_members=final_members
    )

    # Verify patch_team_membership was called correctly
    assert mock_patch_team_membership.call_count == 3

    # Check calls for adding members
    add_calls = [
        call for call in mock_patch_team_membership.call_args_list if call[1]["teams_ids_to_add_user_to"] == [group_id]
    ]
    assert len(add_calls) == 2  # user3 and user4

    add_user_ids = {call[1]["user_id"] for call in add_calls}
    assert add_user_ids == {"user3", "user4"}

    # Check calls for removing members
    remove_calls = [
        call
        for call in mock_patch_team_membership.call_args_list
        if call[1]["teams_ids_to_remove_user_from"] == [group_id]
    ]
    assert len(remove_calls) == 1  # user1

    remove_user_ids = {call[1]["user_id"] for call in remove_calls}
    assert remove_user_ids == {"user1"}

    # Verify all calls have correct structure
    for call in mock_patch_team_membership.call_args_list:
        assert "user_id" in call[1]
        assert "teams_ids_to_add_user_to" in call[1]
        assert "teams_ids_to_remove_user_from" in call[1]
        # Each call should either add OR remove, not both
        add_teams = call[1]["teams_ids_to_add_user_to"]
        remove_teams = call[1]["teams_ids_to_remove_user_from"]
        assert (len(add_teams) > 0) != (len(remove_teams) > 0)  # XOR - one should be empty


@pytest.mark.asyncio
async def test_update_group_e2e(mocker):
    """
    End-to-end test for update_group endpoint:
    - Updates group metadata (displayName)
    - Handles complete member replacement (add/remove members)
    - Verifies members_with_roles is updated as source of truth
    - Tests the full flow from SCIM request to database updates
    """
    from litellm.proxy._types import LiteLLM_TeamTable, Member
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    # Setup test data
    group_id = "test-team-123"

    # Mock existing team in database
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Old Team Name",
        members=["user1", "user2"],  # This should be ignored
        members_with_roles=[
            Member(user_id="user1", role="user"),
            Member(user_id="user2", role="user"),
        ],
        metadata={"existing_key": "existing_value"},
    )

    # Mock updated SCIM group request
    scim_group_update = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Updated Team Name",
        members=[
            SCIMMember(value="user2", display="User Two"),  # Keep user2
            SCIMMember(value="user3", display="User Three"),  # Add user3
            SCIMMember(value="user4", display="User Four"),  # Add user4
        ],
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock database operations
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)

    # Mock the updated team that gets returned from database
    updated_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Updated Team Name",
        members=["user2", "user3", "user4"],
        members_with_roles=[
            Member(user_id="user2", role="user"),
            Member(user_id="user3", role="user"),
            Member(user_id="user4", role="user"),
        ],
        metadata={
            "existing_key": "existing_value",
            "scim_data": scim_group_update.model_dump(),
        },
    )
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=updated_team)

    # Mock user validation (all users exist)
    mock_user = mocker.MagicMock()
    mock_user.user_id = "test-user"
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    # Mock patch_team_membership to track membership changes
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )

    # Mock SCIM transformation
    expected_scim_response = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Updated Team Name",
        members=[
            SCIMMember(value="user2", display="user2"),
            SCIMMember(value="user3", display="user3"),
            SCIMMember(value="user4", display="user4"),
        ],
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(return_value=expected_scim_response),
    )

    # Execute the update_group function
    result = await update_group(group_id=group_id, group=scim_group_update)

    # Verify database update was called with correct data
    mock_prisma_client.db.litellm_teamtable.update.assert_called_once()
    update_call_args = mock_prisma_client.db.litellm_teamtable.update.call_args

    # Check the update parameters
    assert update_call_args[1]["where"]["team_id"] == group_id
    update_data = update_call_args[1]["data"]
    assert update_data["team_alias"] == "Updated Team Name"

    # Verify metadata includes both existing data and SCIM data
    metadata_str = update_data["metadata"]
    import json

    metadata = json.loads(metadata_str)
    assert metadata["existing_key"] == "existing_value"
    assert "scim_data" in metadata
    assert metadata["scim_data"]["displayName"] == "Updated Team Name"

    # Verify team membership changes were handled correctly
    assert mock_patch_team_membership.call_count == 3  # Remove user1, add user3, add user4

    # Check membership changes
    call_args_list = mock_patch_team_membership.call_args_list

    # Find remove operation (user1)
    remove_calls = [call for call in call_args_list if call[1]["teams_ids_to_remove_user_from"] == [group_id]]
    assert len(remove_calls) == 1
    assert remove_calls[0][1]["user_id"] == "user1"
    assert remove_calls[0][1]["teams_ids_to_add_user_to"] == []

    # Find add operations (user3, user4)
    add_calls = [call for call in call_args_list if call[1]["teams_ids_to_add_user_to"] == [group_id]]
    assert len(add_calls) == 2
    add_user_ids = {call[1]["user_id"] for call in add_calls}
    assert add_user_ids == {"user3", "user4"}

    # Verify all add calls have empty remove lists
    for call in add_calls:
        assert call[1]["teams_ids_to_remove_user_from"] == []

    # Verify the response
    assert result.id == group_id
    assert result.displayName == "Updated Team Name"
    assert len(result.members) == 3

    # Verify SCIM transformation was called with updated team
    ScimTransformations.transform_litellm_team_to_scim_group.assert_called_once_with(updated_team)


@pytest.mark.asyncio
async def test_create_group_with_nonexistent_users_rejects(mocker, monkeypatch):
    """
    Test that creating a group with non-existent users is rejected when scim_upsert_user is False.
    Per SCIM 2.0 protocol, users must exist before being added to groups.
    This prevents security issues where users not assigned to app get provisioned via group membership.
    """

    # Mock the feature flag to False (SCIM 2.0 strict mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": False}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    group_id = "test-group-123"
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Test Group",
        members=[
            SCIMMember(value="existing-user", display="Existing User"),  # This user exists
            SCIMMember(value="new-user-1", display="New User 1"),  # This user doesn't exist
            SCIMMember(value="new-user-2", display="New User 2"),  # This user doesn't exist
        ],
    )

    #########################################################
    # We expect the request to be rejected with 400 error
    #########################################################

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock team operations - team doesn't exist yet
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock user lookup - only existing-user exists
    def mock_user_lookup(where):
        user_id = where["user_id"]
        if user_id == "existing-user":
            mock_user = mocker.MagicMock()
            mock_user.user_id = user_id
            return mock_user
        return None  # new-user-1 and new-user-2 don't exist

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=mock_user_lookup)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    # Execute the create_group function - should raise ProxyException
    with pytest.raises(ProxyException) as exc_info:
        await create_group(group=scim_group)

    # Verify it's a 400 Bad Request
    assert int(exc_info.value.code) == 400
    assert "does not exist" in str(exc_info.value.message)
    assert "new-user-1" in str(exc_info.value.message) or "new-user-2" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_update_group_with_nonexistent_users_rejects(mocker, monkeypatch):
    """
    Test that updating a group with non-existent users is rejected when scim_upsert_user is False.
    Per SCIM 2.0 protocol, users must exist before being added to groups.
    """

    # Mock the feature flag to False (SCIM 2.0 strict mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": False}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    group_id = "existing-group-456"

    # Mock existing team
    mock_existing_team = mocker.MagicMock()
    mock_existing_team.team_id = group_id
    mock_existing_team.team_alias = "Old Group Name"
    mock_existing_team.members = ["old-user"]
    mock_existing_team.members_with_roles = [{"user_id": "old-user", "role": "user"}]
    mock_existing_team.metadata = {"existing": "data"}

    # SCIM group update request
    scim_group_update = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Updated Group Name",
        members=[
            SCIMMember(value="existing-user", display="Existing User"),  # This user exists
            SCIMMember(value="new-user-3", display="New User 3"),  # This user doesn't exist
            SCIMMember(value="new-user-4", display="New User 4"),  # This user doesn't exist
        ],
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock team operations
    def mock_team_lookup(where):
        return mock_existing_team if where["team_id"] == group_id else None

    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(side_effect=mock_team_lookup)

    # Mock updated team response
    mock_updated_team = mocker.MagicMock()
    mock_updated_team.team_id = group_id
    mock_updated_team.team_alias = "Updated Group Name"
    mock_updated_team.members = ["existing-user", "new-user-3", "new-user-4"]
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_updated_team)

    # Mock user lookup - only existing-user exists
    def mock_user_lookup(where):
        user_id = where["user_id"]
        if user_id == "existing-user":
            mock_user = mocker.MagicMock()
            mock_user.user_id = user_id
            return mock_user
        return None  # new-user-3 and new-user-4 don't exist

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=mock_user_lookup)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_team_exists",
        AsyncMock(return_value=mock_existing_team),
    )

    # Execute the update_group function - should raise ProxyException
    with pytest.raises(ProxyException) as exc_info:
        await update_group(group_id=group_id, group=scim_group_update)

    # Verify it's a 400 Bad Request
    assert int(exc_info.value.code) == 400
    assert "does not exist" in str(exc_info.value.message)
    assert "new-user-3" in str(exc_info.value.message) or "new-user-4" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_create_group_with_nonexistent_users_creates_when_flag_true(mocker, monkeypatch):
    """
    Test that creating a group with non-existent users creates them when scim_upsert_user is True.
    This preserves backward compatible behavior.
    """

    # Mock the feature flag to True (backward compatible mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": True}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    group_id = "test-group-123"
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Test Group",
        members=[
            SCIMMember(value="existing-user", display="Existing User"),  # This user exists
            SCIMMember(value="new-user-1", display="New User 1"),  # This user doesn't exist - should be created
            SCIMMember(value="new-user-2", display="New User 2"),  # This user doesn't exist - should be created
        ],
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock team operations - team doesn't exist yet
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock user lookup - only existing-user exists initially
    def mock_user_lookup(where):
        user_id = where["user_id"]
        if user_id == "existing-user":
            mock_user = mocker.MagicMock()
            mock_user.user_id = user_id
            return mock_user
        return None  # new-user-1 and new-user-2 don't exist

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=mock_user_lookup)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])

    # Mock user creation
    created_user_1 = NewUserResponse(user_id="new-user-1", key="test-key-1")
    created_user_2 = NewUserResponse(user_id="new-user-2", key="test-key-2")
    mock_create_user = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(side_effect=[created_user_1, created_user_2]),
    )

    # Mock new_team
    mock_team = mocker.MagicMock()
    mock_team.team_id = group_id
    mock_new_team = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_team",
        AsyncMock(return_value=mock_team),
    )

    # Mock transformation
    mock_scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Test Group",
        members=[],
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=mock_scim_group),
    )

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    # Execute the create_group function - should succeed
    await create_group(group=scim_group)

    # Verify users were created
    assert mock_create_user.call_count == 2
    assert mock_create_user.call_args_list[0].kwargs["user_id"] == "new-user-1"
    assert mock_create_user.call_args_list[1].kwargs["user_id"] == "new-user-2"

    # Verify team was created
    mock_new_team.assert_called_once()


@pytest.mark.asyncio
async def test_extract_group_member_ids_with_flag_true_creates_users(mocker, monkeypatch):
    """
    Test that _extract_group_member_ids creates users when scim_upsert_user is True.
    """

    # Mock the feature flag to True (backward compatible mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": True}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="test-group",
        displayName="Test Group",
        members=[
            SCIMMember(value="existing-user", display="Existing User"),  # This user exists
            SCIMMember(value="new-user-1", display="New User 1"),  # This user doesn't exist - should be created
        ],
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock user lookup - only existing-user exists initially
    def mock_user_lookup(where):
        user_id = where["user_id"]
        if user_id == "existing-user":
            mock_user = mocker.MagicMock()
            mock_user.user_id = user_id
            return mock_user
        return None  # new-user-1 doesn't exist

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=mock_user_lookup)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])

    # Mock user creation
    created_user = NewUserResponse(user_id="new-user-1", key="test-key-1")
    mock_create_user = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=created_user),
    )

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    # Execute the function
    result = await _extract_group_member_ids(scim_group)

    # Verify result
    assert "existing-user" in result.existing_member_ids
    assert "existing-user" in result.all_member_ids
    assert "new-user-1" in result.all_member_ids
    assert len(result.created_users) == 1

    # Verify user was created
    mock_create_user.assert_called_once_with(user_id="new-user-1", created_via="scim_group_membership")


@pytest.mark.asyncio
async def test_extract_group_member_ids_with_flag_false_rejects(mocker, monkeypatch):
    """
    Test that _extract_group_member_ids rejects non-existent users when scim_upsert_user is False.
    """

    # Mock the feature flag to False (SCIM 2.0 strict mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": False}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="test-group",
        displayName="Test Group",
        members=[
            SCIMMember(value="existing-user", display="Existing User"),  # This user exists
            SCIMMember(value="new-user-1", display="New User 1"),  # This user doesn't exist - should be rejected
        ],
    )

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock user lookup - only existing-user exists
    def mock_user_lookup(where):
        user_id = where["user_id"]
        if user_id == "existing-user":
            mock_user = mocker.MagicMock()
            mock_user.user_id = user_id
            return mock_user
        return None  # new-user-1 doesn't exist

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=mock_user_lookup)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])

    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    # Execute the function - should raise HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await _extract_group_member_ids(scim_group)

    # Verify it's a 400 Bad Request
    assert exc_info.value.status_code == 400
    assert "does not exist" in str(exc_info.value.detail)
    assert "new-user-1" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_process_group_patch_operations_with_flag_true_creates_users(mocker, monkeypatch):
    """
    Test that _process_group_patch_operations creates users when scim_upsert_user is True.
    """

    # Mock the feature flag to True (backward compatible mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": True}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "new-user-1"}])],
    )

    # Mock existing team
    mock_existing_team = mocker.MagicMock()
    mock_existing_team.members = []
    mock_existing_team.metadata = {}

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock user lookup - new-user-1 doesn't exist
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock user creation
    created_user = NewUserResponse(user_id="new-user-1", key="test-key-1")
    mock_create_user = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=created_user),
    )

    # Execute the function
    update_data, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=mock_existing_team,
        prisma_client=mock_prisma_client,
    )

    # Verify result
    assert "new-user-1" in final_members

    # Verify user was created
    mock_create_user.assert_called_once_with(user_id="new-user-1", created_via="scim_group_patch")


@pytest.mark.asyncio
async def test_process_group_patch_operations_with_flag_false_rejects(mocker, monkeypatch):
    """
    Test that _process_group_patch_operations rejects non-existent users when scim_upsert_user is False.
    """

    # Mock the feature flag to False (SCIM 2.0 strict mode)
    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": False}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    # Test data
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "new-user-1"}])],
    )

    # Mock existing team
    mock_existing_team = mocker.MagicMock()
    mock_existing_team.members = []
    mock_existing_team.metadata = {}

    # Mock prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()

    # Mock user lookup - new-user-1 doesn't exist
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Execute the function - should raise HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await _process_group_patch_operations(
            patch_ops=patch_ops,
            existing_team=mock_existing_team,
            prisma_client=mock_prisma_client,
        )

    # Verify it's a 400 Bad Request
    assert exc_info.value.status_code == 400
    assert "does not exist" in str(exc_info.value.detail)
    assert "new-user-1" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_user_grants_admin_when_in_scim_admin_group(mocker, monkeypatch):
    """When scim_admin_group is configured and a created user's groups include it,
    the user is provisioned as PROXY_ADMIN."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="new-admin",
        emails=[SCIMUserEmail(value="new-admin@example.com")],
        groups=[SCIMUserGroup(value="litellm-admins", display="LiteLLM Admins")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="new-admin")),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    called_args = new_user_mock.call_args.kwargs["data"]
    assert called_args.user_role == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.asyncio
async def test_create_user_keeps_default_when_not_in_scim_admin_group(mocker, monkeypatch):
    """When scim_admin_group is configured but the user's groups don't include it,
    the user keeps the non-admin default role."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="regular-user",
        emails=[SCIMUserEmail(value="regular@example.com")],
        groups=[SCIMUserGroup(value="engineering", display="Engineering")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="regular-user")),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await create_user(user=scim_user)

    called_args = new_user_mock.call_args.kwargs["data"]
    assert called_args.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_update_user_demotes_admin_when_removed_from_scim_admin_group(mocker, monkeypatch):
    """Core demotion test: a PUT whose new groups no longer include the configured
    admin group must re-evaluate the role and write the non-admin default, so an
    admin removed from the IdP group is demoted without re-login."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    existing_user = mocker.MagicMock()
    existing_user.teams = ["litellm-admins"]
    existing_user.metadata = {}

    updated_user = {
        "user_id": "demote-me",
        "user_email": "demote@example.com",
        "user_alias": None,
        "teams": ["engineering"],
        "metadata": "{}",
    }

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="demote-me",
        emails=[SCIMUserEmail(value="demote@example.com")],
        groups=[SCIMUserGroup(value="engineering", display="Engineering")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await update_user(user_id="demote-me", user=scim_user)

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_update_user_does_not_force_role_when_scim_admin_group_unset(mocker, monkeypatch):
    """When scim_admin_group is unset, PUT must not touch user_role (current
    behavior preserved)."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    existing_user = mocker.MagicMock()
    existing_user.teams = ["litellm-admins"]
    existing_user.metadata = {}

    updated_user = {
        "user_id": "no-touch",
        "user_email": "no-touch@example.com",
        "user_alias": None,
        "teams": ["litellm-admins"],
        "metadata": "{}",
    }

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="no-touch",
        emails=[SCIMUserEmail(value="no-touch@example.com")],
        groups=[SCIMUserGroup(value="litellm-admins", display="LiteLLM Admins")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await update_user(user_id="no-touch", user=scim_user)

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert "user_role" not in call_args[1]["data"]


@pytest.mark.asyncio
async def test_update_user_demotes_when_default_params_lack_user_role(mocker, monkeypatch):
    """Regression: default_internal_user_params set without a user_role key must
    still resolve to the non-admin default on demotion, not silently skip and
    leave the user PROXY_ADMIN."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", {"max_budget": 10}, raising=False)

    existing_user = mocker.MagicMock()
    existing_user.teams = ["litellm-admins"]
    existing_user.metadata = {}

    updated_user = {
        "user_id": "demote-me",
        "user_email": "demote@example.com",
        "user_alias": None,
        "teams": ["engineering"],
        "metadata": "{}",
    }

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="demote-me",
        emails=[SCIMUserEmail(value="demote@example.com")],
        groups=[SCIMUserGroup(value="engineering", display="Engineering")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )

    await update_user(user_id="demote-me", user=scim_user)

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_patch_user_demotes_admin_when_removed_from_scim_admin_group(mocker, monkeypatch):
    """PATCH that drops the admin team from the resulting team set must write the
    non-admin default, mirroring the PUT demotion path."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    existing_user = mocker.MagicMock()
    existing_user.teams = ["litellm-admins"]
    existing_user.metadata = {}

    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path="groups", value=[{"value": "engineering"}])],
    )

    updated_user = {
        "user_id": "demote-me",
        "user_alias": None,
        "teams": ["engineering"],
        "metadata": "{}",
    }

    engineering_team = mocker.MagicMock()
    engineering_team.team_alias = "Engineering"

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=engineering_team)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(
            return_value=SCIMUser(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
                userName="demote-me",
            )
        ),
    )

    await patch_user(user_id="demote-me", patch_ops=patch_ops)

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_patch_user_grants_admin_by_team_display_name(mocker, monkeypatch):
    """PATCH carries groups as team ids, so admin-group matching must fall back to
    each team's display name; an admin group configured as a human-readable alias
    grants PROXY_ADMIN even when the team id differs."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "LiteLLM Admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    existing_user = mocker.MagicMock()
    existing_user.teams = []
    existing_user.metadata = {}

    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path="groups", value=[{"value": "team-abc-123"}])],
    )

    updated_user = {
        "user_id": "promote-me",
        "user_alias": None,
        "teams": ["team-abc-123"],
        "metadata": "{}",
    }

    admin_team = mocker.MagicMock()
    admin_team.team_alias = "LiteLLM Admins"

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=admin_team)

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(
            return_value=SCIMUser(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
                userName="promote-me",
            )
        ),
    )

    await patch_user(user_id="promote-me", patch_ops=patch_ops)

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.PROXY_ADMIN


def _scim_admin_prisma(mocker, *, user_teams):
    """Prisma double whose user resolves to user_teams and whose teams expose an
    alias equal to their id, used by the role-recompute helper tests."""
    user = mocker.MagicMock()
    user.user_id = "member-1"
    user.teams = user_teams

    def _team_find_unique(where):
        team = mocker.MagicMock()
        team.team_alias = where["team_id"]
        return team

    prisma = mocker.MagicMock()
    prisma.db = mocker.MagicMock()
    prisma.db.litellm_usertable = mocker.MagicMock()
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user)
    prisma.db.litellm_usertable.find_many = AsyncMock(return_value=())
    prisma.db.litellm_usertable.update = AsyncMock(return_value=user)
    prisma.db.litellm_teamtable = mocker.MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(side_effect=_team_find_unique)
    return prisma


@pytest.mark.asyncio
async def test_recompute_scim_member_roles_demotes_when_not_in_admin_group(mocker, monkeypatch):
    """The shared recompute helper writes the non-admin default for a member whose
    resulting teams no longer include the configured admin group."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    prisma = _scim_admin_prisma(mocker, user_teams=["engineering"])

    await _recompute_scim_member_roles(prisma, ["member-1"])

    call_args = prisma.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_recompute_scim_member_roles_grants_when_in_admin_group(mocker, monkeypatch):
    """The shared recompute helper grants PROXY_ADMIN when a member's resulting
    teams include the configured admin group."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    prisma = _scim_admin_prisma(mocker, user_teams=["litellm-admins"])

    await _recompute_scim_member_roles(prisma, ["member-1"])

    call_args = prisma.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.asyncio
async def test_recompute_scim_member_roles_noop_when_admin_group_unset(mocker, monkeypatch):
    """With scim_admin_group unset the recompute helper must not touch any role,
    preserving current behavior for SCIM group writes."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    prisma = _scim_admin_prisma(mocker, user_teams=["litellm-admins"])

    await _recompute_scim_member_roles(prisma, ["member-1"])

    prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_group_recomputes_roles_for_changed_members(mocker):
    """PUT /Groups must recompute the global role for every member whose
    membership changed, so an admin dropped from the admin group is demoted."""
    from litellm.proxy._types import LiteLLM_TeamTable, Member
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "test-team-123"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Admins",
        members=["user1", "user2"],
        members_with_roles=[
            Member(user_id="user1", role="user"),
            Member(user_id="user2", role="user"),
        ],
        metadata={},
    )
    scim_group_update = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Admins",
        members=[SCIMMember(value="user2"), SCIMMember(value="user3")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group_update),
    )

    await update_group(group_id=group_id, group=scim_group_update)

    recompute_mock.assert_awaited_once()
    assert set(recompute_mock.call_args[0][1]) == {"user1", "user3"}


@pytest.mark.asyncio
async def test_patch_group_recomputes_roles_for_changed_members(mocker):
    """PATCH /Groups must recompute the global role for every member whose
    membership changed, mirroring the PUT path."""
    from litellm.proxy._types import LiteLLM_TeamTable, Member
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "test-team-123"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Admins",
        members=["user1", "user2"],
        members_with_roles=[
            Member(user_id="user1", role="user"),
            Member(user_id="user2", role="user"),
        ],
        metadata={},
    )
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "user1"}])],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(
            return_value=SCIMGroup(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
                id=group_id,
                displayName="Admins",
            )
        ),
    )

    await patch_group(group_id=group_id, patch_ops=patch_ops)

    recompute_mock.assert_awaited_once()
    assert set(recompute_mock.call_args[0][1]) == {"user1"}


@pytest.mark.asyncio
async def test_delete_group_recomputes_roles_for_members(mocker):
    """DELETE /Groups must recompute the global role for the team's members, so
    deleting the admin group demotes everyone who was only admin through it."""
    from litellm.proxy._types import Member

    existing_team = mocker.MagicMock()
    existing_team.members_with_roles = [
        Member(user_id="user1", role="user"),
        Member(user_id="user2", role="user"),
    ]

    member = mocker.MagicMock()
    member.teams = ["test-team-123"]

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_teamtable.delete = AsyncMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=member)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.update = AsyncMock()

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    await delete_group(group_id="test-team-123")

    recompute_mock.assert_awaited_once()
    assert list(recompute_mock.call_args[0][1]) == ["user1", "user2"]


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_applies_role_when_admin_group_set(mocker):
    """When admin_group is configured, re-upserting an existing email persists the
    resolved role so a now-non-admin user can't keep a stale PROXY_ADMIN."""
    existing_user = mocker.MagicMock()
    existing_user.user_id = "old-user-id"

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={"user_id": "new-user-id"})
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="new-user-id",
        user_email="test@example.com",
        teams=["engineering"],
        metadata={},
        auto_create_key=False,
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client,
        new_user_request=new_user_request,
        admin_group="litellm-admins",
    )

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_leaves_role_when_admin_group_unset(mocker):
    """With admin_group unset, the existing-email upsert must not write user_role,
    preserving current behavior when the feature is off."""
    existing_user = mocker.MagicMock()
    existing_user.user_id = "old-user-id"

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={"user_id": "new-user-id"})
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )

    new_user_request = NewUserRequest(
        user_id="new-user-id",
        user_email="test@example.com",
        teams=["engineering"],
        metadata={},
        auto_create_key=False,
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    await UserProvisionerHelpers.handle_existing_user_by_email(
        prisma_client=mock_prisma_client,
        new_user_request=new_user_request,
    )

    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert "user_role" not in call_args[1]["data"]


@pytest.mark.asyncio
async def test_create_user_existing_email_upsert_demotes_when_admin_group_set(mocker, monkeypatch):
    """End-to-end create wiring: a SCIM POST that upserts an existing email while
    the user is not in the admin group must write the non-admin default, not leave
    a stale PROXY_ADMIN."""
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_admin_group": "litellm-admins"}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr("litellm.default_internal_user_params", None, raising=False)

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="returning-user",
        emails=[SCIMUserEmail(value="returning@example.com")],
        groups=[SCIMUserGroup(value="engineering", display="Engineering")],
    )

    existing_user = mocker.MagicMock()
    existing_user.user_id = "returning-user"

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value={"user_id": "returning-user"})

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    new_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(return_value=NewUserRequest(user_id="returning-user")),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=scim_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock(),
    )

    await create_user(user=scim_user)

    new_user_mock.assert_not_called()
    call_args = mock_prisma_client.db.litellm_usertable.update.call_args
    assert call_args[1]["data"]["user_role"] == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_create_group_recomputes_roles_for_members(mocker):
    """POST /Groups must recompute the global role for the new team's members, so a
    team created with the admin-group display name elevates its members."""
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "admin-team-1"
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="LiteLLM Admins",
        members=[SCIMMember(value="user1"), SCIMMember(value="user2")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_team",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )

    await create_group(group=scim_group)

    recompute_mock.assert_awaited_once()
    assert set(recompute_mock.call_args[0][1]) == {"user1", "user2"}


@pytest.mark.asyncio
async def test_update_group_rename_recomputes_retained_members(mocker):
    """A PUT that renames the group (alias changes) but leaves membership unchanged
    must still recompute retained members, since a rename can flip whether the
    group matches scim_admin_group by display name."""
    from litellm.proxy._types import LiteLLM_TeamTable, Member
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "test-team-123"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="LiteLLM Admins",
        members=["user1"],
        members_with_roles=[Member(user_id="user1", role="user")],
        metadata={},
    )
    scim_group_update = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Engineering",
        members=[SCIMMember(value="user1")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group_update),
    )

    await update_group(group_id=group_id, group=scim_group_update)

    recompute_mock.assert_awaited_once()
    assert set(recompute_mock.call_args[0][1]) == {"user1"}


@pytest.mark.asyncio
async def test_patch_group_rename_recomputes_retained_members(mocker):
    """A PATCH that renames the group (displayName op) but leaves membership
    unchanged must still recompute retained members, mirroring the PUT path."""
    from litellm.proxy._types import LiteLLM_TeamTable, Member
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "test-team-123"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="LiteLLM Admins",
        members=["user1"],
        members_with_roles=[Member(user_id="user1", role="user")],
        metadata={},
    )
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path="displayName", value="Engineering")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(
            return_value=SCIMGroup(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
                id=group_id,
                displayName="Engineering",
            )
        ),
    )

    await patch_group(group_id=group_id, patch_ops=patch_ops)

    recompute_mock.assert_awaited_once()
    assert set(recompute_mock.call_args[0][1]) == {"user1"}


@pytest.mark.asyncio
async def test_process_group_patch_operations_add_retains_existing_members(mocker, monkeypatch):
    """A SCIM group ``add`` operation must not drop members already in the team.

    Team membership lives in members_with_roles; team creation leaves the legacy
    ``members`` column empty. Seeding the patch result from that empty column
    made an ``add`` recompute the member set from scratch and remove everyone
    already in the team. The result set must be seeded from members_with_roles so
    existing members survive an add of a new one.
    """

    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": True}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    existing_team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],  # legacy column intentionally empty, as real teams leave it
        members_with_roles=[Member(user_id="existing-user", role="user")],
    )
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "new-user"}])],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    # new-user already exists in the DB
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock(user_id="new-user"))
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=mock_prisma_client,
    )

    assert final_members == {"existing-user", "new-user"}


@pytest.mark.asyncio
async def test_process_group_patch_operations_remove_uses_members_with_roles(mocker, monkeypatch):
    """A ``remove`` op must diff against members_with_roles, so removing one
    member leaves the rest of the team intact rather than emptying it."""

    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": True}}

    from litellm.proxy.proxy_server import proxy_config

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)

    existing_team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],
        members_with_roles=[
            Member(user_id="keep-user", role="user"),
            Member(user_id="drop-user", role="user"),
        ],
    )
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "drop-user"}])],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock(user_id="drop-user"))
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=mock_prisma_client,
    )

    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_get_groups_reports_members_from_members_with_roles(mocker):
    """GET /Groups must report members from members_with_roles (the source of
    truth), not the legacy ``members`` column that team creation leaves empty.
    Reporting an empty member list makes the IdP repeatedly re-provision."""
    team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],  # legacy column empty
        members_with_roles=[Member(user_id="member-1", role="user")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[team])
    mock_prisma_client.db.litellm_teamtable.count = AsyncMock(return_value=1)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=mocker.MagicMock(user_id="member-1", user_email="member-1@example.com")
    )
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    response = await get_groups(startIndex=1, count=10, filter=None)

    assert [m.value for m in response.Resources[0].members] == ["member-1"]


@pytest.mark.asyncio
async def test_apply_group_patch_updates_does_not_write_legacy_members(mocker):
    """The group PATCH apply must not write the legacy ``members`` column.

    Membership is reconciled onto the source of truth (members_with_roles and
    each member's user.teams) separately; writing the legacy column here too
    would create a second, unread copy of membership that can drift from the
    source of truth, which is the inconsistency this PR removes.
    """
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    updated = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=updated)

    result = await _apply_group_patch_updates(
        group_id="team-1",
        update_data={"team_alias": "Renamed"},
        prisma_client=mock_prisma_client,
    )

    assert result is updated
    mock_prisma_client.db.litellm_teamtable.update.assert_awaited_once()
    written = mock_prisma_client.db.litellm_teamtable.update.call_args.kwargs["data"]
    assert "members" not in written
    assert written["team_alias"] == "Renamed"


def _mock_prisma_for_delete_user(mocker, team):
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.delete = AsyncMock()
    return mock_prisma_client


def _patch_delete_user_dependencies(mocker, mock_prisma_client, existing_user):
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._set_user_keys_blocked",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._delete_rows_referencing_user",
        AsyncMock(),
    )


@pytest.mark.asyncio
async def test_delete_user_prunes_members_with_roles(mocker):
    """Deleting a SCIM user must remove them from every team they belong to via
    team_member_delete, which prunes members_with_roles (the source of truth for
    SCIM group membership) so GET /Groups no longer returns a dangling reference
    to the now-deleted user."""
    user_id = "scim-del-user"

    existing_user = mocker.MagicMock()
    existing_user.teams = ["team-1"]

    team = LiteLLM_TeamTable(
        team_id="team-1",
        members=[user_id, "other-user"],
        members_with_roles=[Member(user_id=user_id, role="user"), Member(user_id="other-user", role="admin")],
    )

    mock_prisma_client = _mock_prisma_for_delete_user(mocker, team)
    _patch_delete_user_dependencies(mocker, mock_prisma_client, existing_user)
    team_member_delete_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(),
    )

    await delete_user(user_id=user_id)

    team_member_delete_mock.assert_awaited_once()
    call = team_member_delete_mock.call_args
    assert call.kwargs["data"].team_id == "team-1"
    assert call.kwargs["data"].user_id == user_id
    assert call.kwargs["user_api_key_dict"].user_role == LitellmUserRoles.PROXY_ADMIN
    mock_prisma_client.db.litellm_usertable.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_user_surfaces_prune_failure_and_keeps_user(mocker):
    """A genuine failure while pruning members_with_roles must surface: the
    endpoint fails loudly and the user row is NOT deleted, so we never report a
    successful delete while leaving a dangling member (SCIM DELETE is idempotent,
    so the IdP retries)."""
    user_id = "scim-del-user"

    existing_user = mocker.MagicMock()
    existing_user.teams = ["team-1"]

    team = LiteLLM_TeamTable(
        team_id="team-1",
        members=[user_id],
        members_with_roles=[Member(user_id=user_id, role="user")],
    )

    mock_prisma_client = _mock_prisma_for_delete_user(mocker, team)
    _patch_delete_user_dependencies(mocker, mock_prisma_client, existing_user)
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(side_effect=Exception("database connection lost")),
    )

    with pytest.raises(ProxyException):
        await delete_user(user_id=user_id)

    mock_prisma_client.db.litellm_usertable.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_user_skips_teams_where_not_a_member(mocker):
    """If the user is not in a team's members_with_roles, deletion must treat that
    team as a no-op (no team_member_delete call, no error) and still delete the
    user, so a stale legacy membership can't block the delete."""
    user_id = "scim-del-user"

    existing_user = mocker.MagicMock()
    existing_user.teams = ["team-1"]

    team = LiteLLM_TeamTable(
        team_id="team-1",
        members=[user_id],
        members_with_roles=[Member(user_id="someone-else", role="admin")],
    )

    mock_prisma_client = _mock_prisma_for_delete_user(mocker, team)
    _patch_delete_user_dependencies(mocker, mock_prisma_client, existing_user)
    team_member_delete_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(),
    )

    await delete_user(user_id=user_id)

    team_member_delete_mock.assert_not_awaited()
    mock_prisma_client.db.litellm_usertable.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_patch_group_add_applies_delta_and_keeps_concurrent_add(mocker):
    """A group PATCH op:add must be applied as a delta against the live roster,
    not as a snapshot-based absolute target.

    When a concurrent PATCH has already added a member between this request's
    initial read and its post-write refresh, that member shows up in the
    refreshed roster but not in this request's snapshot-derived target. Diffing
    the refreshed roster against the snapshot target would issue a spurious
    team_member_delete for the concurrently-added member. Applying only this
    request's intended delta on top of the refreshed roster must retain them.
    """
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "team-concurrent"

    snapshot_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[Member(user_id="zed", role="user")],
        metadata={"externalId": "grp-ext"},
    )
    refreshed_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[
            Member(user_id="zed", role="user"),
            Member(user_id="alice", role="user"),
        ],
        metadata={"externalId": "grp-ext"},
    )
    final_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[
            Member(user_id="zed", role="user"),
            Member(user_id="alice", role="user"),
            Member(user_id="bob", role="user"),
        ],
        metadata={"externalId": "grp-ext"},
    )

    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "bob"}])],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        side_effect=[snapshot_team, refreshed_team, final_team]
    )
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=final_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    patch_membership_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(
            return_value=SCIMGroup(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
                id=group_id,
                displayName="Group",
            )
        ),
    )

    await patch_group(group_id=group_id, patch_ops=patch_ops)

    calls = patch_membership_mock.call_args_list

    removed_user_ids = {
        call.kwargs["user_id"] for call in calls if call.kwargs.get("teams_ids_to_remove_user_from") == [group_id]
    }
    assert removed_user_ids == set()

    added_user_ids = {
        call.kwargs["user_id"] for call in calls if call.kwargs.get("teams_ids_to_add_user_to") == [group_id]
    }
    assert added_user_ids == {"bob"}


@pytest.mark.asyncio
async def test_patch_group_replace_stays_absolute_against_concurrent_roster(mocker):
    """A group PATCH ``replace`` op declares the roster is exactly the given set,
    so it must reconcile as a set-to-target, not as a delta.

    Unlike ``add``/``remove``, ``replace`` is absolute. A member that another
    request added concurrently is present in the refreshed roster but not in the
    replace target, and ``replace`` must drop it. Rebasing the replace onto the
    refreshed roster (the delta behavior correct only for add/remove) would
    wrongly retain that concurrently-added member.
    """
    from litellm.proxy.management_endpoints.scim.scim_transformations import (
        ScimTransformations,
    )

    group_id = "team-replace-concurrent"

    snapshot_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[Member(user_id="zed", role="user")],
        metadata={"externalId": "grp-ext"},
    )
    refreshed_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[
            Member(user_id="alice", role="user"),
            Member(user_id="bob", role="user"),
        ],
        metadata={"externalId": "grp-ext"},
    )
    final_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[Member(user_id="alice", role="user")],
        metadata={"externalId": "grp-ext"},
    )

    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path="members", value=[{"value": "alice"}])],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        side_effect=[snapshot_team, refreshed_team, final_team]
    )
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=final_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    patch_membership_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(
            return_value=SCIMGroup(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
                id=group_id,
                displayName="Group",
            )
        ),
    )

    await patch_group(group_id=group_id, patch_ops=patch_ops)

    calls = patch_membership_mock.call_args_list

    removed_user_ids = {
        call.kwargs["user_id"] for call in calls if call.kwargs.get("teams_ids_to_remove_user_from") == [group_id]
    }
    assert removed_user_ids == {"bob"}

    added_user_ids = {
        call.kwargs["user_id"] for call in calls if call.kwargs.get("teams_ids_to_add_user_to") == [group_id]
    }
    assert added_user_ids == set()


@pytest.mark.parametrize(
    "path, attribute, expected",
    [
        ('members[value eq "user-1"]', "members", ["user-1"]),
        ("members[value eq 'user-1']", "members", ["user-1"]),
        ('members[value EQ "user-1"]', "members", ["user-1"]),
        ('members[ value  eq  "user-1" ]', "members", ["user-1"]),
        ('groups[value eq "team-1"]', "groups", ["team-1"]),
        ('members[value eq "Mixed-CASE-Id"]', "members", ["Mixed-CASE-Id"]),
        ('members[value eq "a\\"b"]', "members", ['a"b']),
        ('members[value eq "a\\\\b"]', "members", ["a\\b"]),
        ("members[value eq 'a\\'b']", "members", ["a'b"]),
        ("members", "members", []),
        ('groups[value eq "team-1"]', "members", []),
        (None, "members", []),
        ('members[value eq ""]', "members", []),
        ("members[value eq user-1]", "members", []),
        ("members[value eq unintendeduser]", "members", []),
    ],
)
def test_extract_ids_from_path_filter(path, attribute, expected):
    assert _extract_ids_from_path_filter(path, attribute) == expected


def test_extract_ids_from_path_filter_unterminated_is_linear():
    """A pathological unterminated quoted filter must not trigger super-linear
    backtracking; it returns no id and completes near-instantly."""
    pathological = 'members[value eq "' + ("\\" * 200)

    start = time.perf_counter()
    result = _extract_ids_from_path_filter(pathological, "members")
    elapsed = time.perf_counter() - start

    assert result == []
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_process_group_patch_remove_filtered_path_without_value(mocker):
    """Okta sends group membership removals as a filtered path with no request
    body value; the member id must be parsed out of members[value eq "..."]"""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path='members[value eq "user-1"]')],
    )

    existing_team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
    )

    prisma_client = mocker.MagicMock()
    prisma_client.db = mocker.MagicMock()
    prisma_client.db.litellm_usertable = mocker.MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=LiteLLM_UserTable(user_id="user-1"))
    prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=prisma_client,
    )

    assert final_members == {"user-2"}


@pytest.mark.asyncio
async def test_process_group_patch_add_filtered_path_without_value(mocker):
    """A filtered add path with no body value adds the id parsed from the filter."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path='members[value eq "user-3"]')],
    )

    existing_team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],
        members_with_roles=[Member(user_id="user-1", role="user")],
    )

    prisma_client = mocker.MagicMock()
    prisma_client.db = mocker.MagicMock()
    prisma_client.db.litellm_usertable = mocker.MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=LiteLLM_UserTable(user_id="user-3"))
    prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=prisma_client,
    )

    assert final_members == {"user-1", "user-3"}


@pytest.mark.asyncio
async def test_process_group_patch_replace_empty_value_does_not_use_path_filter(mocker):
    """An explicit empty replace value must clear membership rather than pull an
    id from the filtered path, which would retain one member and drop the rest."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path='members[value eq "user-1"]', value=[])],
    )

    existing_team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],
        members_with_roles=[
            Member(user_id="user-1", role="user"),
            Member(user_id="user-2", role="user"),
        ],
    )

    prisma_client = mocker.MagicMock()
    prisma_client.db = mocker.MagicMock()
    prisma_client.db.litellm_usertable = mocker.MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=LiteLLM_UserTable(user_id="user-1"))
    prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=prisma_client,
    )

    assert final_members == set()


def _member_resolution_prisma(
    mocker: MockerFixture,
    *,
    users: set[str],
    teams: set[str],
    unmanaged_teams: frozenset[str] = frozenset(),
    email_to_user_id: Mapping[str, str] | None = None,
    email_to_user_ids: Mapping[str, tuple[str, ...]] | None = None,
    sso_user_id_to_user_id: Mapping[str, str] | None = None,
) -> MagicMock:
    """Prisma mock where only the given ids resolve to a user row / team row.

    ``teams`` are teams a SCIM group write created, so they carry provenance;
    ``unmanaged_teams`` resolve too but look like a team an admin created here.
    """

    def team_row(team_id: str) -> LiteLLM_TeamTable | None:
        if team_id in teams:
            return LiteLLM_TeamTable(team_id=team_id, metadata={SCIM_MANAGED_TEAM_METADATA_KEY: True})
        if team_id in unmanaged_teams:
            return LiteLLM_TeamTable(team_id=team_id, metadata={})
        return None

    def user_row(where: Mapping[str, str]) -> LiteLLM_UserTable | None:
        user_id: Final = where["user_id"]
        if user_id in users:
            return LiteLLM_UserTable(user_id=user_id)
        return None

    prisma_client = mocker.MagicMock()
    prisma_client.db = mocker.MagicMock()
    prisma_client.db.litellm_usertable = mocker.MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(side_effect=user_row)

    emails_to_ids: Final[Mapping[str, tuple[str, ...]]] = (
        dict(email_to_user_ids)
        if email_to_user_ids is not None
        else ({email: (user_id,) for email, user_id in email_to_user_id.items()} if email_to_user_id else {})
    )
    ssos_to_ids: Final[Mapping[str, str]] = dict(sso_user_id_to_user_id) if sso_user_id_to_user_id else {}

    def identity_rows(where: Mapping[str, object], take: int | None = None) -> tuple[LiteLLM_UserTable, ...]:
        """Stand-in for the cross-field lookup, honouring the comparison mode
        production actually asks for per field, so a field that stops folding case, or
        starts folding it, fails here instead of passing.

        A caller that must know which accounts match rather than merely how many
        passes take=None, so an unbounded read returns every match.
        """
        clauses: Final = where["OR"]
        assert isinstance(clauses, list)
        fields: Final = tuple(next(iter(clause)) for clause in clauses)
        assert fields == ("sso_user_id", "user_email"), fields

        def comparison(clause: Mapping[str, object]) -> tuple[str, bool]:
            """The needle and whether production asked for a case-insensitive compare,
            read per field so a field that stops folding case fails here."""
            criterion = next(iter(clause.values()))
            if isinstance(criterion, str):
                return criterion, False
            assert isinstance(criterion, dict), criterion
            return criterion["equals"], criterion.get("mode") == "insensitive"

        sso_needle, sso_insensitive = comparison(clauses[0])
        email_needle, email_insensitive = comparison(clauses[1])

        def same(stored: str, needle: str, insensitive: bool) -> bool:
            return stored.casefold() == needle.casefold() if insensitive else stored == needle

        matched: Final = tuple(
            chain(
                (
                    user_id
                    for sso_user_id, user_id in ssos_to_ids.items()
                    if same(sso_user_id, sso_needle, sso_insensitive)
                ),
                (
                    user_id
                    for email, user_ids in emails_to_ids.items()
                    if same(email, email_needle, email_insensitive)
                    for user_id in user_ids
                ),
            )
        )
        found: Final = tuple(dict.fromkeys(matched))
        return tuple(LiteLLM_UserTable(user_id=user_id) for user_id in (found[:take] if take else found))

    def team_lookup(where: Mapping[str, str]) -> LiteLLM_TeamTable | None:
        team_id: Final = where["team_id"]
        return team_row(team_id)

    prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    prisma_client.db.litellm_usertable.find_many = AsyncMock(side_effect=identity_rows)
    prisma_client.db.litellm_teamtable = mocker.MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(side_effect=team_lookup)
    return prisma_client


@pytest.fixture
def scim_upsert_user_enabled(monkeypatch):
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": True}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)


@pytest.fixture
def scim_upsert_user_disabled(monkeypatch):
    from litellm.proxy.proxy_server import proxy_config

    async def mock_get_config():
        return {"litellm_settings": {"scim_upsert_user": False}}

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)


@pytest.mark.asyncio
async def test_create_group_ignores_nested_group_members(mocker, scim_upsert_user_enabled):
    """Entra sends nested groups as members with ``type: "Group"``. Treating that
    GUID as a user id provisioned a phantom internal user per nested group."""
    nested_group_id = "8f1e9d70-0000-4a0e-9a1e-nested"
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="parent-group",
        displayName="Parent Group",
        members=[
            SCIMMember(value="real-user", display="Real User", type="User"),
            SCIMMember(value=nested_group_id, display="Nested Group", type="Group"),
        ],
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=_member_resolution_prisma(mocker, users={"real-user"}, teams=set())),
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )
    new_team_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_team",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    await create_group(group=scim_group)

    create_user_mock.assert_not_called()
    assert new_team_mock.call_args.kwargs["data"].members_with_roles == [Member(user_id="real-user", role="user")]


@pytest.mark.asyncio
async def test_update_group_ignores_nested_group_members(mocker, scim_upsert_user_enabled):
    """PUT /Groups must drop nested-group members too, so a full sync from the IdP
    neither provisions nor enrolls the nested group's GUID."""
    group_id = "parent-group"
    nested_group_id = "8f1e9d70-0000-4a0e-9a1e-nested"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
        metadata={},
    )
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Parent Group",
        members=[
            SCIMMember(value="real-user", display="Real User", type="User"),
            SCIMMember(value=nested_group_id, display="Nested Group", type="Group"),
        ],
    )

    prisma_client = _member_resolution_prisma(mocker, users={"real-user"}, teams={group_id})
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_team_exists",
        AsyncMock(return_value=existing_team),
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )
    patch_membership_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )

    await update_group(group_id=group_id, group=scim_group)

    create_user_mock.assert_not_called()
    enrolled = {call.kwargs["user_id"] for call in patch_membership_mock.call_args_list}
    assert enrolled == {"real-user"}


@pytest.mark.asyncio
async def test_process_group_patch_operations_ignores_nested_group_members(mocker, scim_upsert_user_enabled):
    """PATCH bodies bypass SCIMGroup parsing, so ``type`` must be read off the raw
    member dicts; otherwise a nested group is indistinguishable from a user id."""
    nested_group_id = "8f1e9d70-0000-4a0e-9a1e-nested"
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[
            SCIMPatchOperation(
                op="add",
                path="members",
                value=[
                    {"value": "real-user", "display": "Real User", "type": "User"},
                    {"value": nested_group_id, "display": "Nested Group", "type": "Group"},
                ],
            )
        ],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="incumbent", role="user")],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"real-user"}, teams=set()),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"incumbent", "real-user"}


@pytest.mark.asyncio
async def test_process_group_patch_operations_ignores_lowercase_group_type(mocker, scim_upsert_user_enabled):
    """The ``type`` comparison is case-insensitive; IdPs are not consistent about it."""
    nested_group_id = "8f1e9d70-0000-4a0e-9a1e-nested"
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": nested_group_id, "type": "group"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users=set(), teams=set()),
    )

    create_user_mock.assert_not_called()
    assert final_members == set()


@pytest.mark.asyncio
async def test_process_group_patch_operations_skips_member_matching_existing_team(mocker, scim_upsert_user_enabled):
    """Okta sends filtered paths and untyped ids, so a nested group arrives with no
    ``type`` at all; an id that names an existing team is still not a user."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "child-team"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="incumbent", role="user")],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users=set(), teams={"child-team", "parent-group"}),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"incumbent"}


@pytest.mark.asyncio
async def test_process_group_patch_operations_prefers_user_over_team_for_colliding_id(mocker, scim_upsert_user_enabled):
    """Nothing stops a user id from also being a team id, so the user lookup has to
    win; ordering the team check first would silently stop syncing that user."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "dual-id"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"dual-id"}, teams={"dual-id"}),
    )

    assert final_members == {"dual-id"}


@pytest.mark.asyncio
async def test_create_group_strict_mode_accepts_group_and_team_members(mocker, scim_upsert_user_disabled):
    """Strict mode (scim_upsert_user=False) rejects unknown *users*; a nested group
    is not a user, so it must be dropped rather than 400 the whole sync."""
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="parent-group",
        displayName="Parent Group",
        members=[
            SCIMMember(value="real-user", type="User"),
            SCIMMember(value="nested-group-guid", type="Group"),
            SCIMMember(value="child-team"),
        ],
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=_member_resolution_prisma(mocker, users={"real-user"}, teams={"child-team"})),
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )
    new_team_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_team",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    await create_group(group=scim_group)

    create_user_mock.assert_not_called()
    assert new_team_mock.call_args.kwargs["data"].members_with_roles == [Member(user_id="real-user", role="user")]


@pytest.mark.asyncio
async def test_create_group_strict_mode_still_rejects_unknown_user(mocker, scim_upsert_user_disabled):
    """The strict-mode 400 must name the unknown *user* and stay quiet about the
    nested group sharing the request."""
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="parent-group",
        displayName="Parent Group",
        members=[
            SCIMMember(value="nested-group-guid", type="Group"),
            SCIMMember(value="unknown-user"),
        ],
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=_member_resolution_prisma(mocker, users=set(), teams=set())),
    )

    with pytest.raises(ProxyException) as exc_info:
        await create_group(group=scim_group)

    assert int(exc_info.value.code) == 400
    assert "unknown-user" in str(exc_info.value.message)
    assert "nested-group-guid" not in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_process_group_patch_remove_unknown_member_does_not_create_user(mocker, scim_upsert_user_enabled):
    """A ``remove`` of an id we don't know is an idempotent no-op. Upserting the id
    first, only to drop it from the roster, made removals a phantom-user factory."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "long-gone"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="keep-user", role="user")],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"keep-user"}, teams=set()),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"keep-user"}


@pytest.mark.parametrize(
    "operation",
    [
        SCIMPatchOperation(op="remove", path='members[value eq "long-gone"]', value=None),
        SCIMPatchOperation(op="remove", path="members", value=[{"value": "long-gone"}]),
        SCIMPatchOperation(op="remove", path="members", value=[{"value": "long-gone", "type": "Group"}]),
    ],
    ids=["path-filter", "unknown-id", "nested-group"],
)
@pytest.mark.asyncio
async def test_process_group_patch_remove_unknown_member_does_not_reject_in_strict_mode(
    mocker, scim_upsert_user_disabled, operation
):
    """Strict mode must not 400 a removal: refusing to drop an id the IdP already
    forgot leaves the roster permanently out of sync."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[operation],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="keep-user", role="user")],
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"keep-user"}, teams=set()),
    )

    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_process_group_patch_remove_drops_member_without_user_row(mocker, scim_upsert_user_enabled):
    """Phantom members already on a roster (their user row is gone) must still be
    removable, so the removal id is honoured even though it resolves to nothing."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "phantom"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[
            Member(user_id="keep-user", role="user"),
            Member(user_id="phantom", role="user"),
        ],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"keep-user"}, teams=set()),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"keep-user"}


_NESTED_GROUP_ID = "8f1e9d70-0000-4a0e-9a1e-nested"


@pytest.mark.parametrize(
    "member_entry, user_rows, team_rows",
    [
        ({"value": _NESTED_GROUP_ID, "type": "Group"}, {"keep-user", _NESTED_GROUP_ID}, set()),
        ({"value": _NESTED_GROUP_ID, "type": "Group"}, {"keep-user"}, set()),
        ({"value": _NESTED_GROUP_ID, "type": "Group"}, {"keep-user"}, {_NESTED_GROUP_ID}),
        ({"value": _NESTED_GROUP_ID}, {"keep-user"}, {_NESTED_GROUP_ID}),
    ],
    ids=["phantom-user-row-exists", "user-row-already-deleted", "child-group-is-a-team", "untyped-team-id"],
)
@pytest.mark.asyncio
async def test_process_group_patch_remove_discards_non_user_member(
    mocker, scim_upsert_user_enabled, member_entry, user_rows, team_rows
):
    """Rosters written before nested groups were understood still carry those ids,
    and the IdP removes them exactly as it added them; a removal that resolved its
    ids first would classify them as non-users and leave them stuck on the team."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[member_entry])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[
            Member(user_id="keep-user", role="user"),
            Member(user_id=_NESTED_GROUP_ID, role="user"),
        ],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users=user_rows, teams=team_rows),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_process_group_patch_add_keeps_member_typed_user_that_collides_with_team_id(
    mocker, scim_upsert_user_enabled
):
    """The team lookup only exists to catch nested groups that arrive untyped. An id
    the IdP calls a User is a user, and IdP ids collide with team ids easily."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "123456", "type": "User"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="123456", key="new-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users=set(), teams={"123456", "parent-group"}),
    )

    assert create_user_mock.call_args.kwargs["user_id"] == "123456"
    assert final_members == {"123456"}


@pytest.mark.parametrize("member_type", ["Device", " group ", "Machine"])
@pytest.mark.asyncio
async def test_process_group_patch_operations_skips_non_user_member_types(
    mocker, scim_upsert_user_enabled, member_type
):
    """A team holds users, so a member that declares itself to be anything else is
    dropped; enumerating the types worth skipping would leave the next one to leak."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "not-a-user", "type": member_type}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users=set(), teams=set()),
    )

    create_user_mock.assert_not_called()
    assert final_members == set()


@pytest.mark.parametrize(
    "team_metadata, expect_provisioned",
    [
        ({SCIM_MANAGED_TEAM_METADATA_KEY: True}, False),
        ({SCIM_TEAM_DATA_METADATA_KEY: {"displayName": "Child.Apps"}}, False),
        ({}, True),
        (None, True),
        ({SCIM_MANAGED_TEAM_METADATA_KEY: False}, True),
        ({SCIM_TEAM_DATA_METADATA_KEY: None}, True),
    ],
    ids=[
        "scim-managed",
        "legacy-scim-data",
        "admin-created",
        "no-metadata",
        "marker-unset",
        "legacy-key-without-value",
    ],
)
@pytest.mark.asyncio
async def test_process_group_patch_team_match_needs_scim_provenance(
    mocker, scim_upsert_user_enabled, team_metadata, expect_provisioned
):
    """A bare member id that names a team is only evidence of a nested group when the
    identity provider is what wrote that team. Teams created here can share an id with
    a real user, and skipping those members stops provisioning them entirely."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "child-team"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
    )
    prisma_client = _member_resolution_prisma(mocker, users=set(), teams=set())
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=LiteLLM_TeamTable(team_id="child-team", metadata=team_metadata)
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="child-team", key="new-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=prisma_client,
    )

    assert create_user_mock.called is expect_provisioned
    assert final_members == ({"child-team"} if expect_provisioned else set())


@pytest.mark.asyncio
async def test_create_group_strict_mode_rejects_id_matching_admin_created_team(mocker, scim_upsert_user_disabled):
    """Strict mode drops nested groups but reports unknown users. A team an admin
    created here says nothing about the member, so the member is an unknown user."""
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="parent-group",
        displayName="Parent Group",
        members=[SCIMMember(value="admin-team")],
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(
            return_value=_member_resolution_prisma(
                mocker, users=set(), teams=set(), unmanaged_teams=frozenset({"admin-team"})
            )
        ),
    )

    with pytest.raises(ProxyException) as exc_info:
        await create_group(group=scim_group)

    assert int(exc_info.value.code) == 400
    assert "admin-team" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_create_group_stamps_scim_provenance(mocker, scim_upsert_user_enabled):
    """The provenance the classifier reads only exists if the group writes stamp it;
    a SCIM-created team that carries no mark looks admin-created forever after."""
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="child-group",
        displayName="Child.Apps",
        members=[],
    )

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=_member_resolution_prisma(mocker, users=set(), teams=set())),
    )
    new_team_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_team",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    await create_group(group=scim_group)

    assert new_team_mock.call_args.kwargs["data"].metadata == {SCIM_MANAGED_TEAM_METADATA_KEY: True}


@pytest.mark.asyncio
@pytest.mark.parametrize("as_pydantic", [False, True])
async def test_create_group_applies_default_team_params(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    scim_upsert_user_enabled: None,
    as_pydantic: bool,
):
    """SCIM-created teams must honor litellm_settings.default_team_params, including
    models, the same way SSO auto-created teams do."""
    import litellm
    from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams

    default_params = {
        "models": ["no-default-models"],
        "max_budget": 25.0,
        "budget_duration": "30d",
        "tpm_limit": 100,
        "rpm_limit": 10,
    }
    monkeypatch.setattr(
        litellm,
        "default_team_params",
        DefaultTeamSSOParams(**default_params) if as_pydantic else default_params,
    )

    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="defaults-group",
        displayName="Defaults.Apps",
        members=[],
    )

    mocker.patch(  # test-quality-ok: endpoint collaborators are module-level, not injectable into create_group
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=_member_resolution_prisma(mocker, users=set(), teams=set())),
    )
    new_team_mock = mocker.patch(  # test-quality-ok: endpoint collaborators are module-level, not injectable into create_group
        "litellm.proxy.management_endpoints.scim.scim_v2.new_team",
        AsyncMock(return_value=mocker.MagicMock()),
    )
    mocker.patch(  # test-quality-ok: endpoint collaborators are module-level, not injectable into create_group
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )
    mocker.patch(  # test-quality-ok: endpoint collaborators are module-level, not injectable into create_group
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    await create_group(group=scim_group)

    team_request = new_team_mock.call_args.kwargs["data"]
    assert team_request.models == ["no-default-models"]
    assert team_request.max_budget == 25.0
    assert team_request.budget_duration == "30d"
    assert team_request.tpm_limit == 100
    assert team_request.rpm_limit == 10
    assert team_request.team_id == "defaults-group"
    assert team_request.team_alias == "Defaults.Apps"
    assert team_request.metadata == {SCIM_MANAGED_TEAM_METADATA_KEY: True}


@pytest.mark.asyncio
async def test_update_group_stamps_scim_provenance(mocker, scim_upsert_user_enabled):
    """A PUT full sync adopts a team the identity provider now owns, and the stamp has
    to land alongside the existing metadata rather than replacing it."""
    import json

    group_id = "child-group"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Child.Apps",
        members=[],
        members_with_roles=[],
        metadata={"existing_key": "kept"},
    )
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Child.Apps",
        members=[],
    )

    prisma_client = _member_resolution_prisma(mocker, users=set(), teams=set())
    prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_team_exists",
        AsyncMock(return_value=existing_team),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=scim_group),
    )

    await update_group(group_id=group_id, group=scim_group)

    written = json.loads(prisma_client.db.litellm_teamtable.update.call_args.kwargs["data"]["metadata"])
    assert written[SCIM_MANAGED_TEAM_METADATA_KEY] is True
    assert written["existing_key"] == "kept"
    assert SCIM_TEAM_DATA_METADATA_KEY in written


@pytest.mark.asyncio
async def test_process_group_patch_stamps_scim_provenance(mocker, scim_upsert_user_enabled):
    """PATCH is how Okta adopts a group, so a membership-only patch has to stamp the
    team too; otherwise the group it manages never gains provenance."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "real-user"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
        metadata={"existing_key": "kept"},
    )

    update_data, _, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"real-user"}, teams=set()),
    )

    assert update_data["metadata"][SCIM_MANAGED_TEAM_METADATA_KEY] is True
    assert update_data["metadata"]["existing_key"] == "kept"


@pytest.mark.parametrize("member_type", ["direct", "Device"])
@pytest.mark.asyncio
async def test_process_group_patch_keeps_existing_user_with_unrecognized_type(
    mocker, scim_upsert_user_enabled, member_type
):
    """Clients do stamp non-canonical types on real members (RFC 7643 defines
    ``direct`` for ``User.groups``). Dropping a member whose id is a live user would
    revoke that user's team access on the next full sync, so the type is only
    grounds for skipping once the user lookup has missed."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="add", path="members", value=[{"value": "real-user", "type": member_type}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"real-user"}, teams=set()),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"real-user"}


@pytest.mark.parametrize(
    "second_creation",
    [None, NewUserResponse(user_id="dup-user", key="second-key")],
    ids=["second-creation-fails", "both-creations-succeed"],
)
@pytest.mark.asyncio
async def test_resolve_group_member_ids_dedupes_repeated_member(mocker, scim_upsert_user_enabled, second_creation):
    """An id the request lists twice is one member. Admitting it twice writes a
    duplicate members_with_roles row, and the second creation of the same id fails
    against the real unique constraint even when the first one succeeded."""
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(side_effect=[NewUserResponse(user_id="dup-user", key="first-key"), second_creation]),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value="dup-user"), SCIMMember(value="dup-user")],
        created_via="scim_group_membership",
        prisma_client=_member_resolution_prisma(mocker, users=set(), teams=set()),
    )

    assert result.all_member_ids == ["dup-user"]


def _identity_lookup(value: str) -> object:
    """The single cross-field lookup the classifier is expected to issue."""
    return call(
        where={"OR": [{"sso_user_id": value}, {"user_email": {"equals": value, "mode": "insensitive"}}]},
        take=2,
    )


@pytest.mark.asyncio
async def test_resolve_group_member_ids_matches_sso_user_id(mocker, scim_upsert_user_enabled):
    """An OIDC subject in a group payload must resolve to the existing user's
    internal id instead of provisioning a placeholder."""
    prisma_client = _member_resolution_prisma(
        mocker,
        users=set(),
        teams=set(),
        sso_user_id_to_user_id={"member-sub": "sso-user"},
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value="member-sub")],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )

    create_user_mock.assert_not_called()
    assert result.existing_member_ids == ["sso-user"]
    assert result.created_users == []
    assert result.all_member_ids == ["sso-user"]
    assert prisma_client.db.litellm_usertable.find_many.await_args_list == [_identity_lookup("member-sub")]


@pytest.mark.asyncio
async def test_resolve_group_member_ids_matches_user_email(mocker, scim_upsert_user_enabled):
    """A group member email must resolve to the existing user's internal id
    when the identity provider sends email rather than the user id."""
    prisma_client = _member_resolution_prisma(
        mocker,
        users=set(),
        teams=set(),
        email_to_user_id={"member@example.com": "email-user"},
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value="member@example.com")],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )

    create_user_mock.assert_not_called()
    assert result.existing_member_ids == ["email-user"]
    assert result.created_users == []
    assert result.all_member_ids == ["email-user"]
    assert prisma_client.db.litellm_usertable.find_many.await_args_list == [_identity_lookup("member@example.com")]


@pytest.mark.parametrize(
    "pushed",
    ["MEMBER@EXAMPLE.COM", "Member@Example.com", " member@example.com "],
    ids=["upper", "mixed", "padded"],
)
@pytest.mark.asyncio
async def test_resolve_group_member_ids_matches_user_email_as_the_write_path_would(
    mocker, scim_upsert_user_enabled, pushed
):
    """The member value must be compared the way the layer that would reject a
    placeholder compares it.

    ``new_user`` refuses a duplicate email case-insensitively and after stripping, so
    a lookup that is stricter than that resolves nothing, creates a placeholder, and
    is refused by that same layer, which surfaces as a 500 on the whole group push.
    """
    prisma_client = _member_resolution_prisma(
        mocker,
        users=set(),
        teams=set(),
        email_to_user_id={"member@example.com": "email-user"},
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=None),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value=pushed)],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )

    create_user_mock.assert_not_called()
    assert result.all_member_ids == ["email-user"]


@pytest.mark.parametrize(
    "population",
    [
        {"email_to_user_ids": {"duplicate@example.com": ("email-user-a", "email-user-b")}},
        {"email_to_user_ids": {"duplicate@example.com": ("email-user-a",), "DUPLICATE@EXAMPLE.COM": ("email-user-b",)}},
        {
            "sso_user_id_to_user_id": {"duplicate@example.com": "sso-user"},
            "email_to_user_id": {"duplicate@example.com": "email-user"},
        },
    ],
    ids=["same-email-twice", "emails-differing-only-in-case", "one-account-by-sso-another-by-email"],
)
@pytest.mark.asyncio
async def test_resolve_group_member_ids_rejects_a_value_naming_two_accounts(
    mocker, scim_upsert_user_enabled, caplog, population
):
    """A value that names two accounts names a real person we cannot identify, so
    the write is refused rather than attributed to one of them.

    Every shape of collision is refused, not just two rows holding the same email
    verbatim: rows whose emails differ only in case are one row to the layer that
    rejects duplicates, and a value that is one account's SSO identity and another's
    email would otherwise be handed to whichever field happened to be searched first.

    It must not fall through to placeholder creation. That path can only fail: the
    placeholder carries ``user_email`` set to the member value, which the duplicate
    email check rejects, and the recovery lookup that follows searches by ``user_id``
    and so misses the very rows that caused the collision. The operator's data problem
    then surfaces as an HTTP 500 the identity provider retries forever.
    """
    prisma_client = _member_resolution_prisma(mocker, users=set(), teams=set(), **population)
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=None),
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_group_member_ids(
                members=[SCIMMember(value="duplicate@example.com")],
                created_via="scim_group_membership",
                prisma_client=prisma_client,
            )

    assert exc_info.value.status_code == 400
    assert "duplicate@example.com" in str(exc_info.value.detail)
    assert "more than one" in str(exc_info.value.detail)
    create_user_mock.assert_not_called()
    assert any(
        record.levelno >= logging.WARNING
        and "duplicate@example.com" in record.getMessage()
        and "more than one account" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_resolve_group_member_ids_does_not_fold_case_on_the_sso_identity(mocker, scim_upsert_user_enabled):
    """An email and an SSO identity are not comparable the same way.

    OIDC defines ``sub`` as case-sensitive and nothing folds its case on the way in,
    so two subjects differing only in case are two people. Folding it would hand the
    group to an account the provider never named, which is the mis-grant the email
    comparison is deliberately loose enough to avoid and this one is not.
    """
    prisma_client = _member_resolution_prisma(
        mocker,
        users=set(),
        teams=set(),
        sso_user_id_to_user_id={"AbC-subject": "other-user"},
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="abc-subject", key="placeholder-key")),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value="abc-subject")],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )

    assert result.existing_member_ids == []
    assert result.all_member_ids == ["abc-subject"]
    create_user_mock.assert_awaited_once_with(user_id="abc-subject", created_via="scim_group_membership")


@pytest.mark.asyncio
async def test_resolve_group_member_ids_ambiguous_email_outranks_upsert_rejection(mocker, scim_upsert_user_disabled):
    """Ambiguity does not depend on scim_upsert_user, so the operator gets the
    actionable message on either setting rather than being told to create a user that
    already exists twice."""
    prisma_client = _member_resolution_prisma(
        mocker,
        users=set(),
        teams=set(),
        email_to_user_ids={"duplicate@example.com": ("email-user-a", "email-user-b")},
    )

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_group_member_ids(
            members=[SCIMMember(value="duplicate@example.com")],
            created_via="scim_group_membership",
            prisma_client=prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert "more than one" in str(exc_info.value.detail)
    assert "does not exist" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_group_rejects_ambiguous_member_email(mocker, scim_upsert_user_enabled):
    """The refusal reaches the endpoint, so the identity provider sees a 400 on the
    group write rather than a 500 it will retry."""
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="ambiguous-group",
        displayName="Ambiguous Group",
        members=[SCIMMember(value="duplicate@example.com")],
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(
            return_value=_member_resolution_prisma(
                mocker,
                users=set(),
                teams=set(),
                email_to_user_ids={"duplicate@example.com": ("email-user-a", "email-user-b")},
            )
        ),
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=None),
    )

    with pytest.raises(ProxyException) as exc_info:
        await create_group(group=scim_group)

    assert int(exc_info.value.code) == 400
    assert "duplicate@example.com" in str(exc_info.value.message)
    create_user_mock.assert_not_called()


@pytest.mark.parametrize(
    "removed_by",
    ["member@example.com", "member-sub"],
    ids=["by-email", "by-sso-subject"],
)
@pytest.mark.asyncio
async def test_process_group_patch_remove_by_the_id_the_directory_added_with(
    mocker, scim_upsert_user_enabled, removed_by
):
    """A directory removes people by the same id it added them with.

    Resolving on add and not on remove would let someone keep a team after the
    directory took them out of the group: the roster holds the canonical user id, so
    subtracting the email or the subject the request names would match nothing.
    """
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": removed_by}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="real-user", role="user"), Member(user_id="keep-user", role="user")],
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="phantom-user", key="phantom-key")),
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(
            mocker,
            users={"real-user", "keep-user"},
            teams=set(),
            email_to_user_id={"member@example.com": "real-user"},
            sso_user_id_to_user_id={"member-sub": "real-user"},
        ),
    )

    create_user_mock.assert_not_called()
    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_process_group_patch_remove_still_drops_a_placeholder_by_its_literal_id(
    mocker, scim_upsert_user_enabled
):
    """An earlier release put unmatched ids on the roster verbatim, so a remove has to
    keep clearing the id as written even once it also resolves."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "legacy@example.com"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="legacy@example.com", role="user"), Member(user_id="keep-user", role="user")],
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(mocker, users={"keep-user"}, teams=set()),
    )

    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_process_group_patch_remove_when_the_id_turned_ambiguous_after_admission(
    mocker, scim_upsert_user_enabled
):
    """Ambiguity is a property of the table as it stands, not of the value.

    Someone admitted while their email was theirs alone must stay removable after a
    second account takes that email. Resolving the removal against the whole table
    would find two accounts, decline to pick, drop nobody, and still answer 200,
    leaving the person the directory just removed holding the team.
    """
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "shared@example.com"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="admitted-user", role="user"), Member(user_id="keep-user", role="user")],
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        # the newcomer took the address but never joined the group
        prisma_client=_member_resolution_prisma(
            mocker,
            users={"admitted-user", "keep-user"},
            teams=set(),
            email_to_user_ids={"shared@example.com": ("admitted-user", "newcomer")},
        ),
    )

    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_process_group_patch_remove_refuses_a_value_naming_one_member_by_id_and_another_by_email(
    mocker, scim_upsert_user_enabled
):
    """One value must never revoke two people.

    A SCIM-provisioned account is keyed by its userName, so a canonical user id that
    looks like an email is ordinary rather than exotic, and a second account can hold
    that address as its email. Counting the id as written and the resolved accounts
    separately makes each look singular, and the removal then takes both.
    """
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "shared@example.com"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[
            Member(user_id="shared@example.com", role="user"),
            Member(user_id="other-account", role="user"),
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        await _process_group_patch_operations(
            patch_ops=patch_ops,
            existing_team=existing_team,
            prisma_client=_member_resolution_prisma(
                mocker,
                users={"shared@example.com", "other-account"},
                teams=set(),
                email_to_user_id={"shared@example.com": "other-account"},
            ),
        )

    assert exc_info.value.status_code == 400
    assert "shared@example.com" in str(exc_info.value.detail)
    assert "more than one member of this group" in str(exc_info.value.detail)


@pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
@pytest.mark.asyncio
async def test_process_group_patch_remove_finds_the_member_past_the_bounded_read(
    mocker, scim_upsert_user_enabled, position
):
    """A removal has to know *which* accounts a value names, not merely whether it
    names several, so it reads them all.

    An add stops after two matches, which is all it needs to decide the value is
    ambiguous. Reusing that bounded read here would silently drop the member whenever
    the one on the roster sorted past the cap, which no fixture smaller than the cap
    can show. The member is placed at each position so the test cannot pass by luck
    of ordering.
    """
    strangers = ["stranger-one", "stranger-two"]
    sharers = tuple(strangers[:position] + ["admitted-user"] + strangers[position:])
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "shared@example.com"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="admitted-user", role="user"), Member(user_id="keep-user", role="user")],
    )

    _, final_members, _ = await _process_group_patch_operations(
        patch_ops=patch_ops,
        existing_team=existing_team,
        prisma_client=_member_resolution_prisma(
            mocker,
            users={"admitted-user", "keep-user"},
            teams=set(),
            email_to_user_ids={"shared@example.com": sharers},
        ),
    )

    assert final_members == {"keep-user"}


@pytest.mark.asyncio
async def test_process_group_patch_remove_refuses_when_two_members_share_the_id(mocker, scim_upsert_user_enabled):
    """When both accounts a value names are on the roster the removal is genuinely
    undecidable, so it fails rather than reporting a removal it did not perform or
    revoking a membership the directory did not name."""
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="members", value=[{"value": "shared@example.com"}])],
    )
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="member-a", role="user"), Member(user_id="member-b", role="user")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await _process_group_patch_operations(
            patch_ops=patch_ops,
            existing_team=existing_team,
            prisma_client=_member_resolution_prisma(
                mocker,
                users={"member-a", "member-b"},
                teams=set(),
                email_to_user_ids={"shared@example.com": ("member-a", "member-b")},
            ),
        )

    assert exc_info.value.status_code == 400
    assert "shared@example.com" in str(exc_info.value.detail)
    assert "more than one member of this group" in str(exc_info.value.detail)



@pytest.mark.asyncio
async def test_resolve_group_member_ids_exact_user_id_wins_when_it_names_nobody_else(
    mocker, scim_upsert_user_enabled
):
    """The canonical user id stays authoritative, including when the same account also
    holds that value as its email, which is how a SCIM-provisioned account is keyed."""
    prisma_client = _member_resolution_prisma(
        mocker,
        users={"member-id"},
        teams=set(),
        email_to_user_id={"member-id": "member-id"},
    )
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value="member-id")],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )

    create_user_mock.assert_not_called()
    assert result.existing_member_ids == ["member-id"]
    assert result.all_member_ids == ["member-id"]


@pytest.mark.parametrize(
    "population",
    [
        {"sso_user_id_to_user_id": {"member-id": "someone-else"}},
        {"email_to_user_id": {"member-id": "someone-else"}},
    ],
    ids=["another-account-by-sso", "another-account-by-email"],
)
@pytest.mark.asyncio
async def test_resolve_group_member_ids_refuses_a_user_id_that_names_another_account(
    mocker, scim_upsert_user_enabled, caplog, population
):
    """An exact user id is checked for collisions like every other match.

    Taking it on sight would hand the group to whichever account happened to be keyed
    by the value. The placeholders this bug provisioned are exactly that shape, since
    they are keyed by the very id the provider keeps pushing, so on a tenant that
    already has them the real account can never win. Refusing names the problem
    instead of silently landing on the placeholder again.
    """
    prisma_client = _member_resolution_prisma(mocker, users={"member-id"}, teams=set(), **population)
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=None),
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_group_member_ids(
                members=[SCIMMember(value="member-id")],
                created_via="scim_group_membership",
                prisma_client=prisma_client,
            )

    assert exc_info.value.status_code == 400
    assert "member-id" in str(exc_info.value.detail)
    create_user_mock.assert_not_called()
    assert any(
        record.levelno >= logging.WARNING and "someone-else" in record.getMessage() for record in caplog.records
    )



@pytest.mark.asyncio
async def test_resolve_group_member_ids_warns_before_creating_unmatched_placeholder(
    mocker, scim_upsert_user_enabled, caplog
):
    """An unmatched member still follows upsert behavior, but operators receive
    a warning before the placeholder can leave an SSO user teamless."""
    prisma_client = _member_resolution_prisma(mocker, users=set(), teams=set())
    create_user_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=NewUserResponse(user_id="placeholder", key="placeholder-key")),
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        result = await _resolve_group_member_ids(
            members=[SCIMMember(value="unmatched-id")],
            created_via="scim_group_membership",
            prisma_client=prisma_client,
        )

    create_user_mock.assert_awaited_once_with(user_id="unmatched-id", created_via="scim_group_membership")
    assert result.existing_member_ids == []
    assert result.created_users == [NewUserResponse(user_id="placeholder", key="placeholder-key")]
    assert result.all_member_ids == ["unmatched-id"]
    assert any(
        record.levelno >= logging.WARNING
        and "unmatched-id" in record.getMessage()
        and "matched no user by user_id, sso_user_id or user_email" in record.getMessage()
        and "real account stays teamless" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "operation",
    [
        SCIMPatchOperation(op="add", path="members", value=[{"value": "   "}]),
        SCIMPatchOperation(op="remove", path="members", value=[{"value": "   "}]),
        SCIMPatchOperation(op="remove", path='members[value eq "   "]', value=None),
    ],
    ids=["add", "remove", "remove-path-filter"],
)
@pytest.mark.asyncio
async def test_process_group_patch_rejects_blank_member_id(mocker, scim_upsert_user_enabled, operation):
    """A blank id names nobody. The removal path stopped resolving its members, so it
    has to keep rejecting one on its own."""
    patch_ops = SCIMPatchOp(schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"], Operations=[operation])
    existing_team = LiteLLM_TeamTable(
        team_id="parent-group",
        team_alias="Parent Group",
        members=[],
        members_with_roles=[Member(user_id="keep-user", role="user")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await _process_group_patch_operations(
            patch_ops=patch_ops,
            existing_team=existing_team,
            prisma_client=_member_resolution_prisma(mocker, users={"keep-user"}, teams=set()),
        )

    assert exc_info.value.status_code == 400


def test_scim_member_round_trips_type():
    """``type`` has to survive parsing; dropping it is what made a nested group
    look like a user id."""
    assert SCIMMember.model_validate({"value": "x", "type": "Group"}).type == "Group"
    assert SCIMMember(value="x").type is None


@pytest.mark.parametrize("junk_type", [123, True, {}, [], 1.5])
def test_scim_member_treats_non_string_type_as_absent(junk_type):
    """Before ``type`` was a field, junk in it was parsed away; typing the field must
    not start rejecting those requests, and both parsers have to agree it is typeless."""
    assert SCIMMember.model_validate({"value": "x", "type": junk_type}).type is None
    assert _parse_member_entries([{"value": "x", "type": junk_type}])[0].type is None


@pytest.mark.asyncio
async def test_get_groups_members_are_typed_as_users(mocker):
    """Group members we report back are always users, and saying so keeps the
    response from emitting a null ``type``."""
    team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        members=[],
        members_with_roles=[Member(user_id="member-1", role="user")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[team])
    mock_prisma_client.db.litellm_teamtable.count = AsyncMock(return_value=1)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=mocker.MagicMock(user_id="member-1", user_email="member-1@example.com")
    )
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )

    response = await get_groups(startIndex=1, count=10, filter=None)

    assert [m.type for m in response.Resources[0].members] == ["User"]


@pytest.mark.asyncio
async def test_update_user_roster_add_failure_propagates_and_skips_teams_write(mocker):
    """PUT /Users must surface a genuine roster add failure instead of returning 200.

    Regression: the failure was swallowed, the IdP recorded the push as successful
    and never retried, and the user row was still written with a teams array the
    team roster never received.
    """
    existing_user = mocker.MagicMock()
    existing_user.teams = ["old-team"]

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="test-user",
        name=SCIMUserName(familyName="User", givenName="Updated"),
        emails=[SCIMUserEmail(value="updated@example.com")],
        groups=[SCIMUserGroup(value="new-team")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock()

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "Team not found"})),
    )
    delete_mock = mocker.patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete", AsyncMock())

    with pytest.raises(ProxyException) as exc_info:
        await update_user(user_id="test-user", user=scim_user)

    delete_mock.assert_awaited_once()
    assert exc_info.value.code == "404"
    assert "add test-user to new-team" in exc_info.value.message
    mock_prisma_client.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_patch_user_roster_remove_failure_propagates_and_skips_teams_write(mocker):
    """PATCH /Users must surface a genuine roster remove failure instead of returning 200."""
    existing_user = mocker.MagicMock()
    existing_user.teams = ["team1", "team2"]
    existing_user.metadata = {}

    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="remove", path="groups", value=[{"value": "team2"}])],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.update = AsyncMock()

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(side_effect=HTTPException(status_code=500, detail={"error": "db unavailable"})),
    )

    with pytest.raises(ProxyException):
        await patch_user(user_id="test-user", patch_ops=patch_ops)

    mock_prisma_client.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_member", ["user0", "user1", "user2", "user3"])
async def test_handle_group_membership_changes_attempts_every_member_and_names_failures(mocker, failing_member):
    """One failing member must not strand the rest of the roster unattempted.

    Regression: reconciliation stopped at the first failure, so a group push carrying
    several membership changes left the later ones neither written nor reported, and the
    IdP got one opaque error. Every member is attempted now and only the writes that
    actually failed are named, so the next push closes exactly that gap.
    """

    async def add_member(**kwargs):
        if kwargs["data"].member.user_id == failing_member:
            raise HTTPException(status_code=500, detail={"error": "db unavailable"})

    async def remove_member(**kwargs):
        if kwargs["data"].user_id == failing_member:
            raise HTTPException(status_code=500, detail={"error": "db unavailable"})

    add_mock = AsyncMock(side_effect=add_member)
    delete_mock = AsyncMock(side_effect=remove_member)
    mocker.patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_add", add_mock)
    mocker.patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete", delete_mock)

    with pytest.raises(SCIMRosterSyncError) as exc_info:
        await _handle_group_membership_changes(
            group_id="group-1",
            current_members={"user0"},
            final_members={"user1", "user2", "user3"},
        )

    assert [call.kwargs["data"].member.user_id for call in add_mock.call_args_list] == ["user1", "user2", "user3"]
    assert [call.kwargs["data"].user_id for call in delete_mock.call_args_list] == ["user0"]

    message = str(exc_info.value)
    assert "1 of 4 team membership writes" in message
    failed_write = "remove user0 from group-1" if failing_member == "user0" else f"add {failing_member} to group-1"
    assert failed_write in message
    all_writes = {
        "remove user0 from group-1",
        "add user1 to group-1",
        "add user2 to group-1",
        "add user3 to group-1",
    }
    assert not [write for write in all_writes - {failed_write} if write in message]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_status, second_status, expected_status",
    [(404, 404, 404), (404, 500, 500), (500, 500, 500)],
)
async def test_roster_sync_error_status_follows_unanimous_failures(
    mocker, first_status, second_status, expected_status
):
    """Aggregating several failures must not flatten a unanimous 4xx into a 500.

    A push naming a team that does not exist is not retryable, so the IdP has to keep
    seeing the 404. Only a batch whose failures disagree falls back to 500.
    """

    async def add_member(**kwargs):
        status = first_status if kwargs["data"].team_id == "team-a" else second_status
        raise HTTPException(status_code=status, detail={"error": "nope"})

    mocker.patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_add", AsyncMock(side_effect=add_member))

    with pytest.raises(SCIMRosterSyncError) as exc_info:
        await patch_team_membership(
            user_id="user1",
            teams_ids_to_add_user_to=["team-a", "team-b"],
            teams_ids_to_remove_user_from=[],
            raise_on_error=True,
        )

    assert exc_info.value.status_code == expected_status
    assert "2 of 2 team membership writes" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_team", ["team-a", "team-b", "team-c"])
async def test_patch_team_membership_attempts_every_team_before_reporting(mocker, failing_team):
    """A failing team must not strand the same user's remaining adds and removes.

    Regression: the add loop bailed on the first failure, which skipped both the later
    adds and every removal, so a multi-team SCIM push reconciled only a prefix of the
    requested changes while reporting one failure.
    """

    async def add_member(**kwargs):
        if kwargs["data"].team_id == failing_team:
            raise HTTPException(status_code=500, detail={"error": "db unavailable"})

    add_mock = AsyncMock(side_effect=add_member)
    delete_mock = AsyncMock()
    mocker.patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_add", add_mock)
    mocker.patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete", delete_mock)

    with pytest.raises(SCIMRosterSyncError) as exc_info:
        await patch_team_membership(
            user_id="user1",
            teams_ids_to_add_user_to=["team-a", "team-b", "team-c"],
            teams_ids_to_remove_user_from=["team-d"],
            raise_on_error=True,
        )

    assert [call.kwargs["data"].team_id for call in add_mock.call_args_list] == ["team-a", "team-b", "team-c"]
    assert [call.kwargs["data"].team_id for call in delete_mock.call_args_list] == ["team-d"]

    message = str(exc_info.value)
    assert "1 of 4 team membership writes" in message
    assert f"add user1 to {failing_team}" in message
    assert not [team for team in {"team-a", "team-b", "team-c"} - {failing_team} if f"add user1 to {team}" in message]


@pytest.mark.asyncio
async def test_update_group_roster_failure_propagates(mocker):
    """PUT /Groups must fail loudly when a member roster write fails, instead of
    reporting a successful membership sync to the IdP."""
    group_id = "test-team-123"
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Engineering",
        members_with_roles=[Member(user_id="user1", role="user")],
        metadata={},
    )
    scim_group = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Engineering",
        members=[SCIMMember(value="user1"), SCIMMember(value="user2")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=existing_team)
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mocker.MagicMock())
    mock_prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())

    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(side_effect=HTTPException(status_code=500, detail={"error": "db unavailable"})),
    )
    recompute_mock = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    with pytest.raises(ProxyException) as exc_info:
        await update_group(group_id=group_id, group=scim_group)

    assert "add user2 to test-team-123" in exc_info.value.message
    recompute_mock.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_group_member_ids_raises_when_creation_fails(mocker, scim_upsert_user_enabled):
    """A member whose user row can neither be found nor created must fail the
    request. Regression: the resolver silently dropped that member and the group
    write reported success, so the IdP recorded the user as provisioned while the
    team roster was missing them."""
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=None),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_group_member_ids(
            members=[SCIMMember(value="member-1")],
            created_via="scim_group_membership",
            prisma_client=_member_resolution_prisma(mocker, users=set(), teams=set()),
        )

    assert exc_info.value.status_code == 500
    assert "member-1" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_resolve_group_member_ids_admits_member_created_concurrently(mocker, scim_upsert_user_enabled):
    """When creation fails because a concurrent request already created the user,
    the member is still admitted: the id resolves to a real user row, so failing
    or dropping it would be wrong either way."""
    prisma_client = _member_resolution_prisma(mocker, users=set(), teams=set())
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        side_effect=[None, LiteLLM_UserTable(user_id="raced-user")]
    )
    prisma_client.db.litellm_usertable.find_many = AsyncMock(return_value=())
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._create_user_if_not_exists",
        AsyncMock(return_value=None),
    )

    result = await _resolve_group_member_ids(
        members=[SCIMMember(value="raced-user")],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )

    assert result.all_member_ids == ["raced-user"]
    assert len(result.created_users) == 0


@pytest.mark.asyncio
async def test_handle_group_membership_changes_already_in_team_is_noop(mocker):
    """The strict path must keep treating an already-enrolled member as a no-op
    and continue with the remaining members instead of failing the sync."""
    mock_team_member_add = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(
            side_effect=ProxyException(
                message="already in team",
                type=ProxyErrorTypes.team_member_already_in_team.value,
                param=None,
                code=400,
            )
        ),
    )

    await _handle_group_membership_changes(
        group_id="group-1", current_members=set(), final_members={"user-1", "user-2"}
    )

    assert mock_team_member_add.await_count == 2


@pytest.mark.asyncio
async def test_patch_group_404s_when_team_deleted_mid_request(mocker):
    """A group deleted between the existence check and the write must 404.

    Prisma returns None from both the update and the refresh reads once the row is
    gone, and patch_group used to dereference that None while building the response.
    """
    group_id = "team-gone"

    snapshot_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Group",
        members_with_roles=[Member(user_id="zed", role="user")],
        metadata={"externalId": "grp-ext"},
    )

    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[SCIMPatchOperation(op="replace", path="displayName", value="Renamed")],
    )

    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable = mocker.MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(side_effect=[snapshot_team, None, None])
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=None)

    mocker.patch(  # test-quality-ok: stubs the collaborator so the test pins the endpoint's own error contract
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    mocker.patch(  # test-quality-ok: stubs the collaborator so the test pins the endpoint's own error contract
        "litellm.proxy.management_endpoints.scim.scim_v2._recompute_scim_member_roles",
        AsyncMock(),
    )

    with pytest.raises(ProxyException) as exc_info:
        await patch_group(group_id=group_id, patch_ops=patch_ops)

    assert exc_info.value.code == "404"
    assert f"Group not found with ID: {group_id}" in exc_info.value.message

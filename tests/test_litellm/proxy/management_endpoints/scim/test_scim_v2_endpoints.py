from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, NewUserRequest, ProxyException
from litellm.proxy.management_endpoints.scim.scim_v2 import (
    UserProvisionerHelpers,
    _handle_team_membership_changes,
    create_user,
    get_service_provider_config,
    patch_user,
    update_user,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMFeature,
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
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    monkeypatch.setattr(
        "litellm.default_internal_user_params", None, raising=False
    )

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
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    # Set default_internal_user_params with a specific role
    default_params = {
        "user_role": LitellmUserRoles.PROXY_ADMIN,
    }
    monkeypatch.setattr(
        "litellm.default_internal_user_params", default_params, raising=False
    )

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
        prisma_client=mock_prisma_client,
        new_user_request=new_user_request
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
        prisma_client=mock_prisma_client,
        new_user_request=new_user_request
    )
    
    assert result is None
    mock_prisma_client.db.litellm_usertable.find_first.assert_called_once_with(
        where={"user_email": "test@example.com"}
    )


@pytest.mark.asyncio
async def test_handle_existing_user_by_email_existing_user_updated(mocker):
    """Should update existing user and return SCIMUser when user with email exists"""
    # Mock existing user - create a proper mock object with attributes
    existing_user = mocker.MagicMock()
    existing_user.user_id = "old-user-id"
    existing_user.user_email = "test@example.com"
    existing_user.user_alias = "Old Name"
    existing_user.teams = ["old-team"]
    existing_user.metadata = {"old": "data"}
    
    # Mock updated user
    updated_user = {
        "user_id": "new-user-id",
        "user_email": "test@example.com", 
        "user_alias": "New Name",
        "teams": ["new-team"],
        "metadata": '{"new": "data"}'
    }
    
    # Mock SCIM user to be returned
    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="new-user-id",
        userName="new-user-id",
        name=SCIMUserName(familyName="Name", givenName="New"),
        emails=[SCIMUserEmail(value="test@example.com")],
    )
    
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=existing_user)
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=updated_user)
    
    # Mock the transformation function
    mock_transform = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=mock_scim_user)
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
        prisma_client=mock_prisma_client,
        new_user_request=new_user_request
    )
    
    # Verify the result
    assert result == mock_scim_user
    
    # Verify database operations
    mock_prisma_client.db.litellm_usertable.find_first.assert_called_once_with(
        where={"user_email": "test@example.com"}
    )
    
    mock_prisma_client.db.litellm_usertable.update.assert_called_once_with(
        where={"user_id": "old-user-id"},
        data={
            "user_id": "new-user-id",
            "user_email": "test@example.com", 
            "user_alias": "New Name",
            "teams": ["new-team"],
            "metadata": '{"new": "data"}',
        },
    )
    
    # Verify transformation was called
    mock_transform.assert_called_once_with(updated_user)


@pytest.mark.asyncio
async def test_handle_team_membership_changes_no_changes(mocker):
    """Should not call patch_team_membership when existing teams equal new teams"""
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock()
    )
    
    # Same teams - no changes
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1", "team2"],
        new_teams=["team1", "team2"]
    )
    
    # Should not be called since no changes
    mock_patch_team_membership.assert_not_called()


@pytest.mark.asyncio
async def test_handle_team_membership_changes_add_teams(mocker):
    """Should call patch_team_membership with teams to add"""
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock()
    )
    
    # Adding teams
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1"],
        new_teams=["team1", "team2", "team3"]
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
        AsyncMock()
    )
    
    # Removing teams
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1", "team2", "team3"],
        new_teams=["team1"]
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
        AsyncMock()
    )
    
    # Both adding and removing teams
    await _handle_team_membership_changes(
        user_id="test-user",
        existing_teams=["team1", "team2"],
        new_teams=["team2", "team3"]
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
        "metadata": '{"scim_metadata": {"givenName": "Updated", "familyName": "User"}}'
    }
    
    # Mock SCIM user for request
    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="test-user",
        name=SCIMUserName(familyName="User", givenName="Updated"),
        emails=[SCIMUserEmail(value="updated@example.com")],
        groups=[SCIMUserGroup(value="new-team")]
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
        AsyncMock(return_value=mock_prisma_client)
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user)
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock()
    )
    mock_transform = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=response_scim_user)
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
        AsyncMock(return_value=mocker.MagicMock())
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "User not found"}))
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
        "metadata": '{"scim_metadata": {}}'
    }
    
    # Mock patch operations
    patch_ops = SCIMPatchOp(
        schemas=["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
        Operations=[
            SCIMPatchOperation(op="replace", path="displayName", value="Patched User"),
            SCIMPatchOperation(op="add", path="groups", value=[{"value": "team2"}])
        ]
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
        AsyncMock(return_value=mock_prisma_client)
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=existing_user)
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._handle_team_membership_changes",
        AsyncMock()
    )
    mock_transform = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
        AsyncMock(return_value=response_scim_user)
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
        Operations=[
            SCIMPatchOperation(op="replace", path="displayName", value="New Name")
        ]
    )
    
    # Mock dependencies to raise HTTPException for user not found
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mocker.MagicMock())
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "User not found"}))
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

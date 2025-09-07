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
    update_group,
    update_user,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMFeature,
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
        members=[SCIMMember(value="user1", display="User One")]
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
    mock_prisma_client.db.litellm_usertable.update = AsyncMock(return_value=mock_user)
    
    # Mock the _get_prisma_client_or_raise_exception to return our mock
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    
    # Mock the transformation function
    mock_scim_group_response = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Test Group",
        members=[SCIMMember(value="user1", display="User One")]
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_team_to_scim_group",
        AsyncMock(return_value=mock_scim_group_response),
    )
    
    # Call the function that had the bug
    result = await update_group(group_id=group_id, group=scim_group)
    
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
        patch_team_membership,
    )

    # Mock team with members_with_roles as source of truth
    mock_team = mocker.MagicMock()
    mock_team.members_with_roles = [
        Member(user_id="user1", role="user"),
        Member(user_id="user2", role="user")
    ]
    mock_team.members = ["user1", "user2", "user3"]  # This should be ignored
    
    # Test that members_with_roles is source of truth
    member_ids = await _get_team_member_user_ids_from_team(mock_team)
    assert set(member_ids) == {"user1", "user2"}
    assert "user3" not in member_ids  # Should not be included even though in members
    
    # Mock patch_team_membership function
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock()
    )
    
    # Test adding and removing members
    group_id = "test-group-id"
    current_members = {"user1", "user2"}
    final_members = {"user2", "user3", "user4"}  # Remove user1, add user3 and user4
    
    await _handle_group_membership_changes(
        group_id=group_id,
        current_members=current_members,
        final_members=final_members
    )
    
    # Verify patch_team_membership was called correctly
    assert mock_patch_team_membership.call_count == 3
    
    # Check calls for adding members
    add_calls = [call for call in mock_patch_team_membership.call_args_list 
                if call[1]["teams_ids_to_add_user_to"] == [group_id]]
    assert len(add_calls) == 2  # user3 and user4
    
    add_user_ids = {call[1]["user_id"] for call in add_calls}
    assert add_user_ids == {"user3", "user4"}
    
    # Check calls for removing members  
    remove_calls = [call for call in mock_patch_team_membership.call_args_list
                   if call[1]["teams_ids_to_remove_user_from"] == [group_id]]
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
    from litellm.proxy.utils import safe_dumps

    # Setup test data
    group_id = "test-team-123"
    
    # Mock existing team in database
    existing_team = LiteLLM_TeamTable(
        team_id=group_id,
        team_alias="Old Team Name",
        members=["user1", "user2"],  # This should be ignored
        members_with_roles=[
            Member(user_id="user1", role="user"),
            Member(user_id="user2", role="user")
        ],
        metadata={"existing_key": "existing_value"}
    )
    
    # Mock updated SCIM group request
    scim_group_update = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Updated Team Name",
        members=[
            SCIMMember(value="user2", display="User Two"),  # Keep user2
            SCIMMember(value="user3", display="User Three"),  # Add user3
            SCIMMember(value="user4", display="User Four")   # Add user4
        ]
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
            Member(user_id="user4", role="user")
        ],
        metadata={
            "existing_key": "existing_value",
            "scim_data": scim_group_update.model_dump()
        }
    )
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=updated_team)
    
    # Mock user validation (all users exist)
    mock_user = mocker.MagicMock()
    mock_user.user_id = "test-user"
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    
    # Mock dependencies
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client)
    )
    
    # Mock patch_team_membership to track membership changes
    mock_patch_team_membership = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.patch_team_membership",
        AsyncMock()
    )
    
    # Mock SCIM transformation
    expected_scim_response = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id=group_id,
        displayName="Updated Team Name",
        members=[
            SCIMMember(value="user2", display="user2"),
            SCIMMember(value="user3", display="user3"), 
            SCIMMember(value="user4", display="user4")
        ]
    )
    mocker.patch.object(
        ScimTransformations,
        "transform_litellm_team_to_scim_group",
        AsyncMock(return_value=expected_scim_response)
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
    remove_calls = [call for call in call_args_list 
                   if call[1]["teams_ids_to_remove_user_from"] == [group_id]]
    assert len(remove_calls) == 1
    assert remove_calls[0][1]["user_id"] == "user1"
    assert remove_calls[0][1]["teams_ids_to_add_user_to"] == []
    
    # Find add operations (user3, user4)
    add_calls = [call for call in call_args_list 
                if call[1]["teams_ids_to_add_user_to"] == [group_id]]
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
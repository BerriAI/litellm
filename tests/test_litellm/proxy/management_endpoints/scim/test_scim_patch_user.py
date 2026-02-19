from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim.scim_v2 import patch_user
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUser,
    SCIMUserEmail,
    SCIMUserName,
)


@pytest.mark.asyncio
async def test_patch_user_updates_fields():
    mock_user = LiteLLM_UserTable(
        user_id="user-1",
        user_email="test@example.com",
        user_alias="Old",
        teams=[],
        metadata={},
    )

    # Create a proper copy to track updates  
    updated_user = LiteLLM_UserTable(
        user_id="user-1",
        user_email="test@example.com",
        user_alias="New Name",
        teams=[],
        metadata={"scim_active": False, "scim_metadata": {}},
    )

    async def mock_update(*, where, data):
        # Return the updated user object
        return updated_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)
    mock_db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock the transformation function to return a proper SCIMUser
    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="user-1",
        userName="user-1",
        displayName="New Name",
        name=SCIMUserName(familyName="Name", givenName="New"),
        emails=[SCIMUserEmail(value="test@example.com")],
        active=False,
    )

    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="replace", path="displayName", value="New Name"),
            SCIMPatchOperation(op="replace", path="active", value="False"),
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_client), \
         patch("litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user", 
               AsyncMock(return_value=mock_scim_user)):
        result = await patch_user(user_id="user-1", patch_ops=patch_ops)

    mock_db.litellm_usertable.update.assert_called_once()
    assert result.displayName == "New Name"
    assert result.active is False


@pytest.mark.asyncio
async def test_patch_user_manages_group_memberships():
    mock_user = LiteLLM_UserTable(
        user_id="user-2",
        user_email="test@example.com",
        user_alias="Old",
        teams=["old-team"],
        metadata={},
    )

    # Create updated user with final teams
    updated_user = LiteLLM_UserTable(
        user_id="user-2",
        user_email="test@example.com",
        user_alias="Old",
        teams=["new-team"],
        metadata={"scim_metadata": {}},
    )

    async def mock_update(*, where, data):
        # Return the updated user
        return updated_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)
    mock_db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Mock the transformation function to return a proper SCIMUser
    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="user-2",
        userName="user-2",
        displayName="Old",
        name=SCIMUserName(familyName="Family", givenName="Old"),
        emails=[SCIMUserEmail(value="test@example.com")],
        active=True,
    )

    async def mock_add(data, user_api_key_dict):
        # Mock team member add
        pass

    async def mock_delete(data, user_api_key_dict):
        # Mock team member delete
        pass

    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="add", path="groups", value=[{"value": "new-team"}]),
            SCIMPatchOperation(op="remove", path="groups", value=[{"value": "old-team"}]),
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_client), \
         patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
               AsyncMock(side_effect=mock_add)) as mock_add_fn, \
         patch("litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
               AsyncMock(side_effect=mock_delete)) as mock_del_fn, \
         patch("litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
               AsyncMock(return_value=mock_scim_user)):
        result = await patch_user(user_id="user-2", patch_ops=patch_ops)

    assert mock_add_fn.called
    assert mock_del_fn.called
    # Check that the database update was called with the correct teams
    call_args = mock_db.litellm_usertable.update.call_args
    assert "new-team" in call_args[1]["data"]["teams"]
    assert result == mock_scim_user


@pytest.mark.asyncio
async def test_patch_user_deprovision_without_path():
    """
    Test SCIM deprovisioning when operation has no path field.
    Some SCIM providers send: {"op": "replace", "value": {"active": false}}
    """
    mock_user = LiteLLM_UserTable(
        user_id="user-3",
        user_email="test@example.com",
        user_alias="Test User",
        teams=[],
        metadata={"scim_active": True, "scim_metadata": {"givenName": "Test", "familyName": "User"}},
    )

    updated_user = LiteLLM_UserTable(
        user_id="user-3",
        user_email="test@example.com",
        user_alias="Test User",
        teams=[],
        metadata={"scim_active": False, "scim_metadata": {"givenName": "Test", "familyName": "User"}},
    )

    async def mock_update(*, where, data):
        return updated_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)

    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="user-3",
        userName="user-3",
        displayName="Test User",
        name=SCIMUserName(familyName="User", givenName="Test"),
        emails=[SCIMUserEmail(value="test@example.com")],
        active=False,
    )

    # SCIM operation without path field
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="replace", value={"active": False}),
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_client), \
         patch("litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
               AsyncMock(return_value=mock_scim_user)):
        result = await patch_user(user_id="user-3", patch_ops=patch_ops)

    # Verify metadata was updated correctly
    call_args = mock_db.litellm_usertable.update.call_args
    metadata = call_args[1]["data"]["metadata"]
    
    # Parse JSON string back to dict if needed
    if isinstance(metadata, str):
        import json
        metadata = json.loads(metadata)
    
    assert metadata["scim_active"] is False
    assert "" not in metadata  # Ensure no empty string key
    assert result.active is False


@pytest.mark.asyncio
async def test_patch_user_multiple_fields_without_path():
    """
    Test SCIM operations without path containing multiple fields.
    """
    mock_user = LiteLLM_UserTable(
        user_id="user-4",
        user_email="old@example.com",
        user_alias="Old Name",
        teams=[],
        metadata={"scim_active": True, "scim_metadata": {"givenName": "Old", "familyName": "Name"}},
    )

    updated_user = LiteLLM_UserTable(
        user_id="user-4",
        user_email="old@example.com",
        user_alias="New Display Name",
        teams=[],
        metadata={
            "scim_active": False,
            "scim_metadata": {"givenName": "New", "familyName": "User"},
        },
    )

    async def mock_update(*, where, data):
        return updated_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)

    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="user-4",
        userName="user-4",
        displayName="New Display Name",
        name=SCIMUserName(familyName="User", givenName="New"),
        emails=[SCIMUserEmail(value="old@example.com")],
        active=False,
    )

    # SCIM operation without path but with multiple fields
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                value={
                    "active": False,
                    "displayName": "New Display Name",
                    "name": {"givenName": "New", "familyName": "User"},
                },
            ),
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_client), \
         patch("litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
               AsyncMock(return_value=mock_scim_user)):
        result = await patch_user(user_id="user-4", patch_ops=patch_ops)

    # Verify all fields were updated correctly
    call_args = mock_db.litellm_usertable.update.call_args
    update_data = call_args[1]["data"]
    metadata = update_data["metadata"]
    
    # Parse JSON string back to dict if needed
    if isinstance(metadata, str):
        import json
        metadata = json.loads(metadata)
    
    assert metadata["scim_active"] is False
    assert metadata["scim_metadata"]["givenName"] == "New"
    assert metadata["scim_metadata"]["familyName"] == "User"
    assert update_data["user_alias"] == "New Display Name"
    assert "" not in metadata  # Ensure no empty string key
    assert result.active is False




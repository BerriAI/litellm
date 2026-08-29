from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim.scim_v2 import _apply_patch_ops, patch_user
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
    # active=False triggers cascading key-block. No keys here, so return [].
    mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db.litellm_verificationtoken.update_many = AsyncMock(return_value=None)

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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
    ):
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
            SCIMPatchOperation(
                op="remove", path="groups", value=[{"value": "old-team"}]
            ),
        ]
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
            AsyncMock(side_effect=mock_add),
        ) as mock_add_fn,
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
            AsyncMock(side_effect=mock_delete),
        ) as mock_del_fn,
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
    ):
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
        metadata={
            "scim_active": True,
            "scim_metadata": {"givenName": "Test", "familyName": "User"},
        },
    )

    updated_user = LiteLLM_UserTable(
        user_id="user-3",
        user_email="test@example.com",
        user_alias="Test User",
        teams=[],
        metadata={
            "scim_active": False,
            "scim_metadata": {"givenName": "Test", "familyName": "User"},
        },
    )

    async def mock_update(*, where, data):
        return updated_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)
    mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db.litellm_verificationtoken.update_many = AsyncMock(return_value=None)

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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
    ):
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
        metadata={
            "scim_active": True,
            "scim_metadata": {"givenName": "Old", "familyName": "Name"},
        },
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
    mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db.litellm_verificationtoken.update_many = AsyncMock(return_value=None)

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

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
    ):
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


def _user_with_metadata(metadata):
    return LiteLLM_UserTable(
        user_id="user-mva",
        user_email="mva@example.com",
        user_alias=None,
        teams=[],
        metadata=metadata,
    )


def test_apply_patch_ops_replace_entitlements_writes_canonical_key():
    """A PATCH on path=entitlements must persist under scim_entitlements, not
    fall through to the generic handler's raw path key"""
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                path="entitlements",
                value=[{"value": "jira-software", "display": "Jira Software"}],
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({}), patch_ops=patch_ops
    )

    metadata = update_data["metadata"]
    assert metadata["scim_entitlements"] == [
        {"value": "jira-software", "display": "Jira Software"}
    ]
    assert "entitlements" not in metadata


def test_apply_patch_ops_add_roles_appends_to_existing():
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="add", path="roles", value=[{"value": "admin"}])
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({"scim_roles": [{"value": "viewer"}]}),
        patch_ops=patch_ops,
    )

    assert update_data["metadata"]["scim_roles"] == [
        {"value": "viewer"},
        {"value": "admin"},
    ]


def test_apply_patch_ops_remove_entitlements_clears_canonical_key():
    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="remove", path="entitlements")]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {"scim_entitlements": [{"value": "jira-software"}]}
        ),
        patch_ops=patch_ops,
    )

    assert "scim_entitlements" not in update_data["metadata"]


def test_apply_patch_ops_pathless_value_dict_handles_roles():
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                value={"roles": [{"value": "engineering-admin", "primary": True}]},
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({}), patch_ops=patch_ops
    )

    assert update_data["metadata"]["scim_roles"] == [
        {"value": "engineering-admin", "primary": True}
    ]


def test_apply_patch_ops_invalid_entitlements_value_raises_400():
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace", path="entitlements", value=[{"display": "no value"}]
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        _apply_patch_ops(existing_user=_user_with_metadata({}), patch_ops=patch_ops)

    assert exc_info.value.status_code == 400


def test_apply_patch_ops_add_without_value_raises_400_naming_value_member():
    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="add", path="entitlements")]
    )

    with pytest.raises(HTTPException) as exc_info:
        _apply_patch_ops(existing_user=_user_with_metadata({}), patch_ops=patch_ops)

    assert exc_info.value.status_code == 400
    assert "value" in str(exc_info.value.detail)


def test_apply_patch_ops_filtered_path_raises_400_instead_of_junk_metadata():
    """A filtered path must fail loudly rather than fall through to the generic
    handler, which would write a junk metadata key while reporting success"""
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="remove", path='roles[value eq "engineering-admin"]'
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        _apply_patch_ops(
            existing_user=_user_with_metadata(
                {"scim_roles": [{"value": "engineering-admin"}]}
            ),
            patch_ops=patch_ops,
        )

    assert exc_info.value.status_code == 400


def test_apply_patch_ops_remove_group_filtered_path_without_value():
    """Okta removes a user from a team with groups[value eq "..."] and no body
    value; the team id must be parsed from the filter so the remove takes effect"""
    user = LiteLLM_UserTable(
        user_id="user-fp",
        user_email="fp@example.com",
        teams=["team-1", "team-2"],
        metadata={},
    )
    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="remove", path='groups[value eq "team-1"]')]
    )

    _, final_team_set = _apply_patch_ops(existing_user=user, patch_ops=patch_ops)

    assert final_team_set == {"team-2"}


def test_apply_patch_ops_add_group_filtered_path_without_value():
    """A filtered add path with no body value adds the team id from the filter."""
    user = LiteLLM_UserTable(
        user_id="user-fp",
        user_email="fp@example.com",
        teams=["team-1"],
        metadata={},
    )
    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="add", path="groups[value eq 'team-3']")]
    )

    _, final_team_set = _apply_patch_ops(existing_user=user, patch_ops=patch_ops)

    assert final_team_set == {"team-1", "team-3"}


def test_apply_patch_ops_replace_groups_empty_value_does_not_use_path_filter():
    """A filtered replace with an explicit empty value must not resurrect the
    filter id; the team set is replaced with the empty value as given."""
    user = LiteLLM_UserTable(
        user_id="user-fp",
        user_email="fp@example.com",
        teams=["team-1", "team-2"],
        metadata={},
    )
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="replace", path='groups[value eq "team-1"]', value=[])
        ]
    )

    _, final_team_set = _apply_patch_ops(existing_user=user, patch_ops=patch_ops)

    assert final_team_set == set()

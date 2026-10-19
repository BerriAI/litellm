from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim.scim_v2 import _apply_patch_ops, patch_user
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIM_ENTERPRISE_METADATA_KEY,
    SCIM_ENTERPRISE_USER_SCHEMA,
    SCIMEnterpriseUser,
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
                op="replace", path="entitlements", value=[42]
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        _apply_patch_ops(existing_user=_user_with_metadata({}), patch_ops=patch_ops)

    assert exc_info.value.status_code == 400


def test_apply_patch_ops_replace_entitlements_without_value_member_is_stored_as_sent():
    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace", path="entitlements", value=[{"groups": ["S0506MKA55L"]}]
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({}), patch_ops=patch_ops
    )

    assert update_data["metadata"]["scim_entitlements"] == [{"groups": ["S0506MKA55L"]}]


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


@pytest.mark.asyncio
async def test_patch_user_pathless_enterprise_extension_merges_existing_metadata():
    mock_user: Final = _user_with_metadata({SCIM_ENTERPRISE_METADATA_KEY: {"employeeNumber": "42"}})
    mock_client: Final = MagicMock()
    mock_db: Final = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)

    async def mock_update(*, where: Mapping[str, object], data: Mapping[str, object]) -> LiteLLM_UserTable:
        assert where == {"user_id": "user-mva"}
        metadata_json: Final = data["metadata"]
        assert isinstance(metadata_json, str)
        metadata: Final = TypeAdapter(dict[str, object]).validate_json(metadata_json)
        return _user_with_metadata(metadata)

    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)
    mock_db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                value={
                    SCIM_ENTERPRISE_USER_SCHEMA: {
                        "department": "Eng",
                        "costCenter": "CC-10",
                    }
                },
            )
        ]
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
    ):
        result: Final = await patch_user(user_id="user-mva", patch_ops=patch_ops)

    update_data: Final = mock_db.litellm_usertable.update.call_args.kwargs["data"]
    metadata_json: Final = update_data["metadata"]
    assert isinstance(metadata_json, str)
    metadata: Final = TypeAdapter(dict[str, object]).validate_json(metadata_json)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "department": "Eng",
        "costCenter": "CC-10",
    }
    assert not any(isinstance(key, str) and key.casefold().startswith("urn:") for key in metadata)
    assert result.enterprise_user == SCIMEnterpriseUser(
        employeeNumber="42",
        department="Eng",
        costCenter="CC-10",
    )


def test_apply_patch_ops_replace_enterprise_urn_path_from_idp():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                path="urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:department",
                value="Eng",
            )
        ]
    )

    update_data, _ = _apply_patch_ops(existing_user=_user_with_metadata({}), patch_ops=patch_ops)

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {"department": "Eng"}
    assert not any(isinstance(key, str) and key.casefold().startswith("urn:") for key in metadata)


def test_apply_patch_ops_pathless_enterprise_manager_sub_attribute_merges():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                value={
                    f"{SCIM_ENTERPRISE_USER_SCHEMA.upper()}:MANAGER.VALUE": "manager-1",
                },
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {SCIM_ENTERPRISE_METADATA_KEY: {"manager": {"displayName": "Engineering Manager"}}}
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "manager": {"value": "manager-1", "displayName": "Engineering Manager"}
    }


def test_apply_patch_ops_replace_enterprise_manager_normalizes_case_insensitive_keys():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager",
                value={"VALUE": "mgr-2"},
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "employeeNumber": "42",
                    "manager": {"value": "mgr-1", "displayName": "Boss"},
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "manager": {"value": "mgr-2", "displayName": "Boss"},
    }


@pytest.mark.parametrize(
    ("path", "value"),
    (
        pytest.param(
            f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager",
            {"displayName": "Q"},
            id="pathed",
        ),
        pytest.param(
            None,
            {SCIM_ENTERPRISE_USER_SCHEMA: {"manager": {"displayName": "Q"}}},
            id="pathless",
        ),
        pytest.param(
            None,
            {SCIM_ENTERPRISE_USER_SCHEMA: {"MANAGER": {"displayName": "Q"}}},
            id="pathless-uppercase-key",
        ),
    ),
)
def test_apply_patch_ops_replace_manager_object_preserves_unspecified_sub_attributes(
    path: str | None,
    value: object,
) -> None:
    patch_ops: Final = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="replace", path=path, value=value)]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "department": "A",
                    "manager": {"value": "m", "displayName": "B"},
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "department": "A",
        "manager": {"value": "m", "displayName": "Q"},
    }


@pytest.mark.parametrize(
    ("manager_path", "manager_value"),
    (
        pytest.param(
            f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager",
            {"$ref": "new"},
            id="pathed",
        ),
        pytest.param(
            None,
            {SCIM_ENTERPRISE_USER_SCHEMA: {"manager": {"$ref": "new"}}},
            id="pathless",
        ),
    ),
)
def test_apply_patch_ops_batched_manager_ref_update_replaces_previous_link(
    manager_path: str | None,
    manager_value: object,
) -> None:
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:department",
                value="B",
            ),
            SCIMPatchOperation(
                op="replace",
                path=manager_path,
                value=manager_value,
            ),
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "department": "A",
                    "manager": {"value": "m", "$ref": "old"},
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "department": "B",
        "manager": {"value": "m", "$ref": "new"},
    }


def test_apply_patch_ops_add_enterprise_manager_merges_existing_sub_attributes():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="add",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager",
                value={"displayName": "New Boss"},
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "employeeNumber": "42",
                    "manager": {
                        "value": "mgr-1",
                        "$ref": "https://idp.example/Users/mgr-1",
                    },
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "manager": {
            "value": "mgr-1",
            "$ref": "https://idp.example/Users/mgr-1",
            "displayName": "New Boss",
        },
    }


def test_apply_patch_ops_pathless_add_enterprise_manager_merges_existing_sub_attributes():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="add",
                value={
                    SCIM_ENTERPRISE_USER_SCHEMA: {
                        "manager": {"displayName": "New Boss"},
                    }
                },
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "employeeNumber": "42",
                    "manager": {
                        "value": "mgr-1",
                        "$ref": "https://idp.example/Users/mgr-1",
                    },
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "manager": {
            "value": "mgr-1",
            "$ref": "https://idp.example/Users/mgr-1",
            "displayName": "New Boss",
        },
    }


def test_apply_patch_ops_remove_enterprise_manager_sub_attribute_preserves_other_fields():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="remove",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager.value",
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "employeeNumber": "42",
                    "manager": {"value": "mgr-1", "displayName": "Boss"},
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "manager": {"displayName": "Boss"},
    }


def test_apply_patch_ops_remove_enterprise_attribute_preserves_other_attributes():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="remove",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:costCenter",
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({SCIM_ENTERPRISE_METADATA_KEY: {"costCenter": "CC-10", "department": "Eng"}}),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {"department": "Eng"}


def test_apply_patch_ops_remove_last_enterprise_attribute_drops_metadata_key():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="remove",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:employeeNumber",
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({SCIM_ENTERPRISE_METADATA_KEY: {"employeeNumber": "42"}}),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert SCIM_ENTERPRISE_METADATA_KEY not in metadata


def test_apply_patch_ops_unknown_enterprise_attribute_preserves_existing_and_applies_other_ops() -> None:
    existing_enterprise: Final = {
        "employeeNumber": "42",
        "manager": {"value": "mgr-1", "displayName": "Boss"},
    }
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:shoeSize",
                value="10",
            ),
            SCIMPatchOperation(op="replace", path="displayName", value="Updated User"),
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({SCIM_ENTERPRISE_METADATA_KEY: existing_enterprise}),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == existing_enterprise
    assert not any(isinstance(key, str) and key.casefold().startswith("urn:") for key in metadata)
    assert update_data["user_alias"] == "Updated User"


def test_apply_patch_ops_display_name_only_preserves_enterprise_value_unchanged() -> None:
    existing_enterprise: Final = {"Department": "Eng", "manager": "mgr-1"}
    patch_ops: Final = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="replace", path="displayName", value="Ada")]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({SCIM_ENTERPRISE_METADATA_KEY: existing_enterprise}),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {"Department": "Eng", "manager": "mgr-1"}


@pytest.mark.parametrize("path", ("manager.foo", "manager.a.b", "department.x"))
def test_apply_patch_ops_unknown_enterprise_nested_attribute_preserves_existing(path: str) -> None:
    existing_enterprise: Final = {
        "department": "Engineering",
        "manager": {"value": "mgr-1", "displayName": "Boss"},
    }
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:{path}",
                value="ignored",
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata({SCIM_ENTERPRISE_METADATA_KEY: existing_enterprise}),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == existing_enterprise


@pytest.mark.parametrize("manager_id", ("mgr-id", ""))
def test_apply_patch_ops_add_enterprise_manager_string_replaces_existing_manager(manager_id: str) -> None:
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="add",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager",
                value=manager_id,
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "employeeNumber": "42",
                    "manager": {"value": "old", "displayName": "Old Boss"},
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "manager": {"value": manager_id},
    }


def test_apply_patch_ops_pathless_add_enterprise_manager_string_replaces_existing_manager() -> None:
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="add",
                value={SCIM_ENTERPRISE_USER_SCHEMA: {"manager": "mgr-id"}},
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {
                SCIM_ENTERPRISE_METADATA_KEY: {
                    "employeeNumber": "42",
                    "manager": {"value": "old", "displayName": "Old Boss"},
                }
            }
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert metadata[SCIM_ENTERPRISE_METADATA_KEY] == {
        "employeeNumber": "42",
        "manager": {"value": "mgr-id"},
    }


def test_apply_patch_ops_remove_enterprise_manager_drops_manager() -> None:
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="remove",
                path=f"{SCIM_ENTERPRISE_USER_SCHEMA}:manager",
            )
        ]
    )

    update_data, _ = _apply_patch_ops(
        existing_user=_user_with_metadata(
            {SCIM_ENTERPRISE_METADATA_KEY: {"manager": {"value": "mgr-1", "displayName": "Boss"}}}
        ),
        patch_ops=patch_ops,
    )

    metadata: Final = update_data["metadata"]
    assert isinstance(metadata, dict)
    assert SCIM_ENTERPRISE_METADATA_KEY not in metadata


def test_apply_patch_ops_pathless_enterprise_value_must_be_a_dict():
    patch_ops: Final = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(
                op="replace",
                value={SCIM_ENTERPRISE_USER_SCHEMA: "not-a-dict"},
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        _apply_patch_ops(existing_user=_user_with_metadata({}), patch_ops=patch_ops)

    assert exc_info.value.status_code == 400

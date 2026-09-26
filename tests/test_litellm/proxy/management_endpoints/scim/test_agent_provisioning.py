from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy.management_endpoints.scim.agent_provisioning import (
    SCIMProvisioningFailure,
    apply_user_patch,
    group_members_after_patch,
)
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import SCIM_AGENT_USER_SCHEMA
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMGroup, SCIMMember, SCIMPatchOp, SCIMUser

PARENT: Final = "11111111-1111-4111-8111-111111111111"

SUBJECT: Final = "22222222-2222-4222-8222-222222222222"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ABCDEF00-1234-4234-9234-123456789ABC", "abcdef00-1234-4234-9234-123456789abc"),
        ("opaque-directory-id", "opaque-directory-id"),
    ],
)
def test_directory_ids_normalize_uuids_without_rewriting_opaque_ids(value: str, expected: str) -> None:
    from litellm.types.proxy.management_endpoints.scim_agent_provisioning import canonical_directory_id

    assert canonical_directory_id(value) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False, None])
async def test_source_token_lookup_uses_current_writer_state(enabled: bool | None) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.scim.agent_provisioning import source_for_auth

    source: Final = SimpleNamespace(source_id="source", enabled=enabled) if enabled is not None else None
    client: Final = MagicMock()
    client.writer_db.litellm_scimsource.find_unique = AsyncMock(return_value=source)
    auth: Final = UserAPIKeyAuth(token="source-token-hash")
    if enabled is False:
        with pytest.raises(HTTPException) as failure:
            await source_for_auth(auth, client)
        assert failure.value.status_code == 403
    else:
        assert await source_for_auth(auth, client) is source
    client.writer_db.litellm_scimsource.find_unique.assert_awaited_once_with(where={"key_hash": "source-token-hash"})
    client.db.litellm_scimsource.find_unique.assert_not_called()
    assert await source_for_auth(None, client) is None
    assert await source_for_auth(UserAPIKeyAuth(), client) is None


@pytest.mark.asyncio
async def test_directory_documents_use_authoritative_activity_and_membership() -> None:
    from litellm.proxy.management_endpoints.scim.agent_provisioning import (
        group_document,
        remove_group_member,
        user_document,
    )

    row: Final = SimpleNamespace(id="row", active=False, document={"userName": "subject", "active": True})
    assert user_document(row).active is False
    assert user_document(row).id == "row"
    group: Final = SimpleNamespace(
        id="group",
        member_ids=["keep", "remove"],
        document={"displayName": "Directory", "members": [{"value": "stale"}]},
    )
    assert [member.value for member in group_document(group).members] == ["keep", "remove"]
    client: Final = MagicMock()
    client.litellm_scimresource.update = AsyncMock()
    await remove_group_member(client, group, "remove")
    client.litellm_scimresource.update.assert_awaited_once_with(where={"id": "group"}, data={"member_ids": ["keep"]})
    assert group.member_ids == ["keep", "remove"]


def agent_user() -> SCIMUser:
    return SCIMUser.model_validate(
        {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User", SCIM_AGENT_USER_SCHEMA],
            "id": "stable-scim-id",
            "externalId": SUBJECT,
            "userName": "agent@example.com",
            "active": True,
            SCIM_AGENT_USER_SCHEMA: {"identityParentId": PARENT},
        }
    )


@pytest.mark.parametrize("wire_value, expected", [("False", False), ("True", True), (False, False), (True, True)])
def test_entra_boolean_patch_preserves_identity_and_returns_json_boolean(wire_value: object, expected: bool) -> None:
    original: Final = agent_user()
    result: Final = apply_user_patch(
        original, SCIMPatchOp(Operations=[{"op": "Replace", "path": "active", "value": wire_value}])
    )
    assert isinstance(result, SCIMUser)
    assert result.active is expected
    assert result.id == original.id
    assert result.externalId == SUBJECT
    assert result.agent_user == original.agent_user
    assert result.model_dump(by_alias=True)["active"] is expected
    assert original.active is True


@pytest.mark.parametrize(
    "path, value",
    [
        ("externalId", PARENT),
        (SCIM_AGENT_USER_SCHEMA + ":identityParentId", SUBJECT),
        (None, {SCIM_AGENT_USER_SCHEMA: {"identityParentId": SUBJECT}}),
        ("active", "garbage"),
    ],
)
def test_identity_rebinding_and_invalid_active_patch_are_rejected(path: str | None, value: object) -> None:
    result: Final = apply_user_patch(
        agent_user(), SCIMPatchOp(Operations=[{"op": "replace", "path": path, "value": value}])
    )
    assert isinstance(result, SCIMProvisioningFailure)
    assert result.status == 400


def test_rename_does_not_reenable_a_disabled_subject() -> None:
    original: Final = agent_user().model_copy(update={"active": False})
    result: Final = apply_user_patch(
        original, SCIMPatchOp(Operations=[{"op": "replace", "value": {"displayName": "Renamed"}}])
    )
    assert isinstance(result, SCIMUser)
    assert result.displayName == "Renamed"
    assert result.active is False
    assert result.id == original.id


def test_agent_schema_without_parent_is_not_a_human() -> None:
    with pytest.raises(ValidationError, match="identityParentId"):
        SCIMUser.model_validate({"schemas": [SCIM_AGENT_USER_SCHEMA], "userName": "agent@example.com"})


def test_group_removal_preserves_other_members_and_repeat_removal_is_idempotent() -> None:
    group: Final = SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        displayName="Mixed",
        members=[SCIMMember(value="agent"), SCIMMember(value="human")],
    )
    patch: Final = SCIMPatchOp(Operations=[{"op": "Remove", "path": 'members[value eq "agent"]'}])
    result: Final = group_members_after_patch(group, patch)
    assert isinstance(result, SCIMGroup)
    assert result.members == [SCIMMember(value="human")]
    assert group_members_after_patch(result, patch) == result
    assert group.members == [SCIMMember(value="agent"), SCIMMember(value="human")]


def test_group_replace_removes_omitted_members_and_empty_replace_removes_all() -> None:
    group: Final = SCIMGroup(
        schemas=[], displayName="Mixed", members=[SCIMMember(value="agent"), SCIMMember(value="human")]
    )
    result: Final = group_members_after_patch(
        group, SCIMPatchOp(Operations=[{"op": "replace", "path": "members", "value": [{"value": "human"}]}])
    )
    assert isinstance(result, SCIMGroup)
    assert result.members == [SCIMMember(value="human")]
    empty: Final = group_members_after_patch(
        result, SCIMPatchOp(Operations=[{"op": "replace", "path": "members", "value": []}])
    )
    assert isinstance(empty, SCIMGroup)
    assert empty.members == []


@pytest.mark.parametrize(
    "path,value",
    [
        (SCIM_AGENT_USER_SCHEMA + ":identityParentId", PARENT),
        (None, {SCIM_AGENT_USER_SCHEMA: {"identityParentId": PARENT}}),
        ("schemas", [SCIM_AGENT_USER_SCHEMA]),
    ],
)
def test_human_patch_cannot_smuggle_an_agent_identity(path: str | None, value: object) -> None:
    from litellm.proxy.management_endpoints.scim.agent_provisioning import patch_changes_identity

    patch: Final = SCIMPatchOp(Operations=[{"op": "add", "path": path, "value": value}])
    assert patch_changes_identity(patch)


@pytest.mark.parametrize("identity_marker", [True, False])
def test_deep_patch_checks_identity_without_exhausting_the_call_stack(identity_marker: bool) -> None:
    from functools import reduce

    from litellm.proxy.management_endpoints.scim.agent_provisioning import patch_changes_identity

    leaf: Final = {"identityParentId": PARENT} if identity_marker else {"displayName": "Renamed"}
    nested: Final = reduce(lambda value, _: {"nested": [value]}, range(1200), leaf)
    patch: Final = SCIMPatchOp(Operations=[{"op": "replace", "value": nested}])
    assert patch_changes_identity(patch) is identity_marker


def test_patch_error_does_not_apply_later_operations() -> None:
    patch: Final = SCIMPatchOp(
        Operations=[
            {"op": "replace", "path": "externalId", "value": "foreign-subject"},
            {"op": "replace", "path": "displayName", "value": "renamed"},
        ]
    )
    result: Final = apply_user_patch(agent_user(), patch)
    assert isinstance(result, SCIMProvisioningFailure)
    assert result.status == 400


@pytest.mark.parametrize(
    "operation,expected",
    [
        ({"op": "add", "path": "members", "value": [{"value": "human"}, {"value": "second"}]}, ["human", "second"]),
        ({"op": "remove", "path": "members", "value": [{"value": "human"}]}, []),
        ({"op": "remove", "path": "members"}, []),
        ({"op": "replace", "path": "displayName", "value": "Renamed"}, ["human"]),
    ],
)
def test_group_patch_add_remove_and_rename_keep_membership_consistent(
    operation: dict[str, object], expected: list[str]
) -> None:
    group: Final = SCIMGroup(schemas=[], displayName="Original", members=[SCIMMember(value="human")])
    result: Final = group_members_after_patch(group, SCIMPatchOp.model_validate({"Operations": [operation]}))
    assert isinstance(result, SCIMGroup)
    assert [member.value for member in result.members or []] == expected
    assert result.displayName == ("Renamed" if operation["path"] == "displayName" else "Original")


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "path": "externalId", "value": "foreign"},
        {"op": "add", "path": "members", "value": [{"display": "missing-id"}]},
    ],
)
def test_group_patch_failure_cannot_apply_subsequent_membership_changes(operation: dict[str, object]) -> None:
    group: Final = SCIMGroup(schemas=[], displayName="Original", members=[SCIMMember(value="human")])
    result: Final = group_members_after_patch(
        group, SCIMPatchOp.model_validate({"Operations": [operation, {"op": "remove", "path": "members"}]})
    )
    assert isinstance(result, SCIMProvisioningFailure)
    assert result.status == 400
    assert group.members == [SCIMMember(value="human")]


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "remove"},
        {"op": "replace", "value": "invalid"},
        {"op": "remove", "path": "userName"},
    ],
)
def test_invalid_agent_profile_patch_is_rejected(operation: dict[str, object]) -> None:
    result: Final = apply_user_patch(agent_user(), SCIMPatchOp.model_validate({"Operations": [operation]}))
    assert isinstance(result, SCIMProvisioningFailure)
    assert result.status == 400


@pytest.mark.parametrize(
    "path,value",
    [("name.givenName", "New"), ("name.familyName", "Family"), ('emails[type eq "work"].value', "new@example.com")],
)
def test_native_profile_accepts_standard_entra_subattribute_updates(path: str, value: str) -> None:
    user: Final = SCIMUser.model_validate(
        {
            **agent_user().model_dump(by_alias=True),
            "name": {"givenName": "Old", "familyName": "Original"},
            "emails": [{"type": "work", "value": "old@example.com"}, {"type": "home", "value": "home@example.com"}],
        }
    )
    result: Final = apply_user_patch(user, SCIMPatchOp(Operations=[{"op": "replace", "path": path, "value": value}]))
    assert isinstance(result, SCIMUser)
    assert result.externalId == user.externalId
    assert result.agent_user == user.agent_user
    if path.startswith("name."):
        assert getattr(result.name, path.split(".")[1]) == value
        assert result.emails == user.emails
    else:
        assert result.emails[0].value == value
        assert result.emails[1].value == "home@example.com"
        assert result.name == user.name

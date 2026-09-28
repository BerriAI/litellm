import asyncio
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.scim.agent_provisioning import (
    SCIMProvisioningFailure,
    apply_user_patch,
    group_members_after_patch,
)
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import SCIM_AGENT_USER_SCHEMA
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMGroup, SCIMMember, SCIMPatchOp, SCIMUser

PARENT: Final = "11111111-1111-4111-8111-111111111111"

SUBJECT: Final = "22222222-2222-4222-8222-222222222222"




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

    row: Final = SimpleNamespace(
        id="row", active=False, document={"schemas": [], "userName": "subject", "active": True}
    )
    assert user_document(row).active is False
    assert user_document(row).id == "row"
    group: Final = SimpleNamespace(
        id="group",
        member_ids=["keep", "remove"],
        document={"schemas": [], "displayName": "Directory", "members": [{"value": "stale"}]},
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


def provisioning_fixture():
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from prisma import Prisma
    from prisma.models import LiteLLM_SCIMResource, LiteLLM_SCIMSource

    from litellm.proxy.management_endpoints.scim.agent_provisioning import AgentProvisioningService
    from litellm.proxy.utils import PrismaClient

    now: Final = datetime.now(timezone.utc)
    source: Final = LiteLLM_SCIMSource(
        source_id="source",
        display_name="Source",
        tenant_id=PARENT,
        key_hash="hash",
        enabled=True,
        group_mappings="[]",
        created_at=now,
        updated_at=now,
    )
    row: Final = LiteLLM_SCIMResource(
        id="stable-scim-id",
        source_id="source",
        kind="Users",
        external_id=SUBJECT,
        user_name="agent@example.com",
        display_name="Agent",
        document=agent_user().model_dump_json(by_alias=True),
        active=True,
        deleted=False,
        local_id="registered-agent",
        member_ids=[],
        created_at=now,
        updated_at=now,
    )
    client: Final = MagicMock(spec=PrismaClient)
    tx: Final = MagicMock(spec=Prisma)
    client.tx.return_value.__aenter__.return_value = tx
    client.writer_db = tx
    tx.execute_raw = AsyncMock(return_value=1)
    tx.litellm_scimsource.find_unique = AsyncMock(return_value=source)
    tx.litellm_scimresource.find_unique = AsyncMock(return_value=row)
    tx.litellm_scimresource.update_many = AsyncMock(return_value=1)
    tx.litellm_scimresource.update = AsyncMock(return_value=row)
    tx.litellm_scimresource.find_many = AsyncMock(return_value=[])
    tx.litellm_scimresource.count = AsyncMock(return_value=0)
    tx.litellm_agentstable.find_unique = AsyncMock(return_value={"agent_id": row.local_id})
    return AgentProvisioningService(client, source), tx, row


@pytest.mark.asyncio
async def test_profile_put_cannot_reenable_native_user_when_active_is_omitted() -> None:
    service, tx, row = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(update={"active": False})
    profile: Final = agent_user().model_dump(by_alias=True)
    incoming: Final = SCIMUser.model_validate({key: value for key, value in profile.items() if key != "active"})
    result: Final = await service.update_user(row.id, incoming)
    assert result.active is False
    assert result.id == row.id
    assert tx.litellm_scimresource.update_many.call_args.kwargs["data"]["active"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("username", [None, ""])
async def test_native_profile_update_rejects_blank_username_without_writing(username: str | None) -> None:
    service, tx, row = provisioning_fixture()
    request: Final = SCIMUser.model_validate({**agent_user().model_dump(by_alias=True), "userName": username})
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, request)
    assert failure.value.status_code == 400
    assert failure.value.detail == "userName is required"
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_registration_is_not_recreated_by_directory_replay() -> None:
    from fastapi import HTTPException

    service, tx, row = provisioning_fixture()
    tx.litellm_agentstable.find_unique.return_value = None
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, agent_user())
    assert failure.value.status_code == 409
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_source_is_rechecked_after_waiting_for_its_lock() -> None:
    from fastapi import HTTPException

    service, tx, row = provisioning_fixture()
    tx.litellm_scimsource.find_unique.return_value = service.source.model_copy(update={"enabled": False})
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, agent_user())
    assert failure.value.status_code == 403
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_native_profile_update_returns_conflict() -> None:
    from fastapi import HTTPException

    service, tx, row = provisioning_fixture()
    tx.litellm_scimresource.update_many.return_value = 0
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, agent_user())
    assert failure.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["Users", "Groups"])
async def test_scoped_delete_propagates_human_and_team_deprovisioning(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    document: Final = SCIMUser(schemas=[], userName="human@example.com").model_dump(mode="json")
    row: Final = native.model_copy(update={"kind": kind, "document": document})
    tx.litellm_scimresource.find_unique.return_value = row
    delete_user: Final = AsyncMock()
    delete_group: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "delete_user", delete_user)
    monkeypatch.setattr(scim_v2, "delete_group", delete_group)
    await service.delete(kind, row.id)
    if kind == "Users":
        delete_user.assert_awaited_once_with(user_id=row.local_id)
        delete_group.assert_not_awaited()
    else:
        delete_group.assert_awaited_once_with(group_id=row.local_id)
        delete_user.assert_not_awaited()
    tx.litellm_scimresource.update.assert_awaited_once_with(
        where={"id": row.id},
        data={"active": False, "deleted": True, "member_ids": []},
    )


@pytest.mark.asyncio
async def test_failed_human_deletion_remains_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    row: Final = native.model_copy(update={"document": SCIMUser(schemas=[], userName="human@example.com").model_dump()})
    tx.litellm_scimresource.find_unique.return_value = row
    monkeypatch.setattr(scim_v2, "delete_user", AsyncMock(side_effect=HTTPException(503, "unavailable")))
    with pytest.raises(HTTPException) as failure:
        await service.delete("Users", row.id)
    assert failure.value.status_code == 503
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_delete_is_idempotent_and_does_not_delete_a_human(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, row = provisioning_fixture()
    delete_user: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "delete_user", delete_user)
    await service.delete("Users", row.id)
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(update={"deleted": True, "active": False})
    await service.delete("Users", row.id)
    delete_user.assert_not_awaited()
    tx.litellm_scimresource.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_delete_removes_group_links_before_tombstoning_the_subject() -> None:
    from unittest.mock import call

    service, tx, row = provisioning_fixture()
    group: Final = row.model_copy(update={"id": "mixed-group", "kind": "Groups", "member_ids": [row.id, "human"]})
    tx.litellm_scimresource.find_many.return_value = [group]
    await service.delete("Users", row.id)
    tx.litellm_scimresource.update.assert_has_awaits(
        [
            call(where={"id": group.id}, data={"member_ids": ["human"]}),
            call(where={"id": row.id}, data={"active": False, "deleted": True, "member_ids": []}),
        ]
    )
    tx.litellm_scimresource.find_many.assert_awaited_once_with(
        where={"source_id": "source", "kind": "Groups", "member_ids": {"has": row.id}}
    )


@pytest.mark.asyncio
async def test_new_native_user_is_created_disabled_with_exact_subject_parent_and_source() -> None:
    service, tx, _ = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = None
    tx.litellm_scimresource.create = AsyncMock()
    tx.litellm_agentidentity.find_unique = AsyncMock(return_value=None)
    tx.litellm_agentstable.create = AsyncMock()
    tx.litellm_verifiedsubject.create = AsyncMock()
    result: Final = await service.create_user(agent_user())
    resource: Final = tx.litellm_scimresource.create.call_args.kwargs["data"]
    agent: Final = tx.litellm_agentstable.create.call_args.kwargs["data"]
    subject: Final = tx.litellm_verifiedsubject.create.call_args.kwargs["data"]
    assert result.id == resource["id"] == subject["scim_resource_id"]
    assert agent["agent_id"] == resource["local_id"] == subject["agent_id"]
    assert agent["enabled"] is False
    assert agent["execution_mode"] == "autonomous"
    assert agent["identity"]["create"]["provisioning_source_id"] == service.source.source_id
    assert agent["identity"]["create"]["client_id"] == subject["parent_client_id"] == PARENT
    assert subject["oid"] == resource["external_id"] == SUBJECT
    assert subject["kind"] == "agent_user"
    assert "user_id" not in subject


@pytest.mark.asyncio
async def test_native_create_replay_uses_existing_registration() -> None:
    service, tx, row = provisioning_fixture()
    tx.litellm_agentstable.create = AsyncMock()
    result: Final = await service.create_user(agent_user())
    assert result.id == row.id
    tx.litellm_agentstable.create.assert_not_awaited()
    assert tx.litellm_scimresource.update_many.call_args.kwargs["where"]["id"] == row.id


@pytest.mark.asyncio
async def test_native_create_cannot_adopt_a_preexisting_parent() -> None:
    service, tx, _ = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = None
    tx.litellm_agentidentity.find_unique = AsyncMock(return_value={"agent_id": "other"})
    tx.litellm_scimresource.create = AsyncMock()
    with pytest.raises(HTTPException) as failure:
        await service.create_user(agent_user())
    assert failure.value.status_code == 409
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"externalId": "invalid"}, {"externalId": None}, {"userName": None}])
async def test_native_create_rejects_missing_or_malformed_identity(changed: dict[str, object]) -> None:
    service, tx, _ = provisioning_fixture()
    tx.litellm_scimresource.create = AsyncMock()
    with pytest.raises(HTTPException) as failure:
        await service.create_user(agent_user().model_copy(update=changed))
    assert failure.value.status_code == 400
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["missing", "foreign", "deleted", "wrong-kind"])
async def test_resource_reads_are_source_and_kind_scoped(state: str) -> None:
    service, tx, row = provisioning_fixture()
    changes: Final = {
        "foreign": {"source_id": "foreign"},
        "deleted": {"deleted": True},
        "wrong-kind": {"kind": "Groups"},
    }
    tx.litellm_scimresource.find_unique.return_value = (
        None if state == "missing" else row.model_copy(update=changes[state])
    )
    with pytest.raises(HTTPException) as failure:
        await service.get("Users", row.id)
    assert failure.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attribute,field",
    [("userName", "user_name"), ("externalId", "external_id"), ("displayName", "display_name"), ("id", "id")],
)
async def test_scim_filter_keeps_source_boundary_and_total_count(attribute: str, field: str) -> None:
    service, tx, row = provisioning_fixture()
    tx.litellm_scimresource.find_many.return_value = [row]
    tx.litellm_scimresource.count = AsyncMock(return_value=3)
    result: Final = await service.list("Users", 2, 500, f'{attribute} eq "value"')
    query: Final = tx.litellm_scimresource.find_many.call_args.kwargs
    assert query["where"] == {"source_id": "source", "kind": "Users", "deleted": False, field: "value"}
    assert query["skip"] == 1 and query["take"] == 100
    assert result.totalResults == 3
    assert result.itemsPerPage == 1
    assert result.Resources[0].id == row.id


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_value", ['active eq "true"', 'userName sw "a"'])
async def test_unsupported_filter_is_rejected_before_database_query(filter_value: str) -> None:
    service, tx, _ = provisioning_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.list("Users", 1, 10, filter_value)
    assert failure.value.status_code == 400
    tx.litellm_scimresource.find_many.assert_not_awaited()


def group_rows(native):
    from litellm.types.proxy.management_endpoints.scim_v2 import SCIMGroup, SCIMUser

    human: Final = native.model_copy(
        update={
            "id": "human-scim",
            "local_id": "local-human",
            "external_id": "external-human",
            "document": SCIMUser(schemas=[], userName="human@example.com").model_dump(),
        }
    )
    document: Final = SCIMGroup(schemas=[], externalId="directory-group", displayName="Mixed")
    group: Final = native.model_copy(
        update={
            "id": "group-scim",
            "kind": "Groups",
            "local_id": None,
            "external_id": "directory-group",
            "document": document.model_dump(),
            "member_ids": [native.id, human.id],
        }
    )
    return human, group, document


@pytest.mark.asyncio
async def test_group_validation_reads_all_member_batches() -> None:
    from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

    service, tx, _ = provisioning_fixture()
    members: Final = tuple(f"member-{index}" for index in range(IN_LIST_CHUNK_SIZE + 1))
    tx.litellm_scimresource.count.side_effect = [IN_LIST_CHUNK_SIZE, 1]
    await service._validate_members(members)
    assert tx.litellm_scimresource.count.await_count == 2
    tx.litellm_scimresource.count.side_effect = [IN_LIST_CHUNK_SIZE, 0]
    with pytest.raises(HTTPException) as failure:
        await service._validate_members(members)
    assert failure.value.status_code == 400


@pytest.mark.asyncio
async def test_group_sync_includes_humans_from_later_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.management_endpoints.scim import scim_v2
    from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

    service, tx, native = provisioning_fixture()
    human, group, document = group_rows(native)
    members: Final = [f"member-{index}" for index in range(IN_LIST_CHUNK_SIZE)] + [human.id]
    native_rows: Final = [native.model_copy(update={"id": member}) for member in members[:-1]]
    tx.litellm_scimresource.find_many.side_effect = [native_rows, [human]]
    tx.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    create_team: Final = AsyncMock(return_value=scim_v2.ProvisionedGroupWrite(team_id="local-team", created=None, removals=(), additions=()))
    monkeypatch.setattr(scim_v2, "write_provisioned_group", create_team)
    await service._sync_human_members(tx, group.model_copy(update={"member_ids": members}), None)
    create_team.assert_awaited_once()
    assert create_team.call_args.args[2].members == [SCIMMember(value=human.local_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_group_member_validation_uses_the_source_transaction(operation: str) -> None:
    service, tx, native = provisioning_fixture()
    _, group, document = group_rows(native)
    single_member: Final = group.model_copy(update={"member_ids": [native.id]})
    request: Final = document.model_copy(update={"members": [SCIMMember(value=native.id)]})
    tx.litellm_scimresource.find_unique.return_value = None if operation == "create" else single_member
    tx.litellm_scimresource.create = AsyncMock(return_value=single_member)
    tx.litellm_scimresource.count.return_value = 1
    tx.litellm_scimresource.find_many.return_value = [native]
    writer: Final = MagicMock()
    writer.litellm_scimresource.count = AsyncMock(return_value=1)
    writer.litellm_scimresource.find_many = AsyncMock(return_value=[native])
    service.client.writer_db = writer
    result: Final = (
        await service.create_group(request) if operation == "create" else await service.update_group(group.id, request)
    )
    assert result.id == group.id
    assert result.members == [SCIMMember(value=native.id)]
    tx.litellm_scimresource.count.assert_awaited_once()
    writer.litellm_scimresource.count.assert_not_awaited()
    service.client.tx.assert_called_once()
    assert service.client.tx.call_args.kwargs["timeout"].total_seconds() == 30
    if operation == "create":
        tx.litellm_scimresource.create.assert_awaited_once()
    else:
        assert tx.litellm_scimresource.update_many.call_args.kwargs["where"] == {
            "id": group.id,
            "updated_at": group.updated_at,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("include_human", [True, False])
async def test_group_create_keeps_agents_out_of_human_teams(
    include_human: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    human, group, document = group_rows(native)
    rows: Final = [native, human] if include_human else [native]
    group: Final = group.model_copy(update={"member_ids": [row.id for row in rows]})
    tx.litellm_scimresource.find_unique.return_value = None
    tx.litellm_scimresource.find_many.return_value = rows
    tx.litellm_scimresource.count.return_value = len(rows)
    tx.litellm_scimresource.create = AsyncMock(return_value=group)
    tx.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    create_team: Final = AsyncMock(return_value=scim_v2.ProvisionedGroupWrite(team_id="local-team", created=None, removals=(), additions=()))
    monkeypatch.setattr(scim_v2, "write_provisioned_group", create_team)
    result: Final = await service.create_group(
        document.model_copy(update={"members": [SCIMMember(value=row.id) for row in rows]})
    )
    assert result.id == group.id
    assert {member.value for member in result.members} == set(group.member_ids)
    if include_human:
        create_team.assert_awaited_once()
        assert create_team.call_args.args[2].members == [SCIMMember(value=human.local_id)]
        assert tx.litellm_scimresource.update.call_args.kwargs == {
            "where": {"id": group.id},
            "data": {"local_id": "local-team"},
        }
    else:
        create_team.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_create_rejects_members_missing_from_its_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    _, _, document = group_rows(native)
    tx.litellm_scimresource.find_unique.return_value = None
    tx.litellm_scimresource.create = AsyncMock()
    create_team: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "write_provisioned_group", create_team)
    with pytest.raises(HTTPException) as failure:
        await service.create_group(document.model_copy(update={"members": [SCIMMember(value="foreign-member")]}))
    assert failure.value.status_code == 400
    tx.litellm_scimresource.create.assert_not_awaited()
    create_team.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_replay_reconciles_removed_humans_but_preserves_agent_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    _, group, document = group_rows(native)
    old: Final = group.model_copy(update={"local_id": "local-team"})
    updated: Final = old.model_copy(update={"member_ids": [native.id]})
    tx.litellm_scimresource.find_unique.side_effect = [old, old, updated]
    tx.litellm_scimresource.find_many.return_value = [native]
    tx.litellm_scimresource.count.return_value = 1
    tx.litellm_teamtable.find_unique = AsyncMock(return_value=SimpleNamespace(team_id="local-team"))
    update_team: Final = AsyncMock(return_value=scim_v2.ProvisionedGroupWrite(team_id="local-team", created=None, removals=(), additions=()))
    monkeypatch.setattr(scim_v2, "write_provisioned_group", update_team)
    result: Final = await service.create_group(document.model_copy(update={"members": [SCIMMember(value=native.id)]}))
    assert result.id == old.id
    assert result.members == [SCIMMember(value=native.id)]
    assert tx.litellm_scimresource.update_many.call_args.kwargs["data"]["member_ids"] == [native.id]
    update_team.assert_awaited_once()
    assert update_team.call_args.args[2].id == "local-team"
    assert update_team.call_args.args[2].members == []


@pytest.mark.asyncio
async def test_group_external_id_is_immutable() -> None:
    service, tx, native = provisioning_fixture()
    _, group, document = group_rows(native)
    tx.litellm_scimresource.find_unique.return_value = group
    with pytest.raises(HTTPException) as failure:
        await service.update_group(group.id, document.model_copy(update={"externalId": "other-directory-group"}))
    assert failure.value.status_code == 409
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create", "update"])
async def test_native_replay_accepts_equivalent_uppercase_object_id(method: str) -> None:
    service, tx, row = provisioning_fixture()
    subject: Final = "abcdef01-abcd-4abc-8abc-abcdef012345"
    user: Final = agent_user().model_copy(update={"externalId": subject.upper()})
    stored: Final = row.model_copy(
        update={"external_id": subject, "document": user.model_dump(by_alias=True, mode="json")}
    )
    tx.litellm_scimresource.find_unique.return_value = stored
    result: Final = await service.create_user(user) if method == "create" else await service.update_user(row.id, user)
    assert result.id == row.id
    assert result.externalId == subject.upper()
    tx.litellm_scimresource.update_many.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("external_id", [None, "invalid", PARENT])
async def test_native_put_cannot_remove_or_change_subject(external_id: str | None) -> None:
    service, tx, row = provisioning_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, agent_user().model_copy(update={"externalId": external_id}))
    assert failure.value.status_code == 409
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_patch_rejects_identity_mutation_without_writing() -> None:
    service, tx, row = provisioning_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.update_user(
            row.id, SCIMPatchOp(Operations=[{"op": "replace", "path": "externalId", "value": PARENT}])
        )
    assert failure.value.status_code == 400
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_source_lookup_uses_writer_and_rejects_disabled_source(enabled: bool) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.scim.agent_provisioning import source_for_auth

    service, _, _ = provisioning_fixture()
    service.client.writer_db.litellm_scimsource.find_unique = AsyncMock(
        return_value=service.source.model_copy(update={"enabled": enabled})
    )
    if enabled:
        result: Final = await source_for_auth(UserAPIKeyAuth(token="source-token-hash"), service.client)
        assert result.source_id == service.source.source_id
    else:
        with pytest.raises(HTTPException) as failure:
            await source_for_auth(UserAPIKeyAuth(token="source-token-hash"), service.client)
        assert failure.value.status_code == 403
    service.client.writer_db.litellm_scimsource.find_unique.assert_awaited_once_with(
        where={"key_hash": "source-token-hash"}
    )
    assert await source_for_auth(None, service.client) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["put", "patch"])
async def test_source_owned_human_cannot_be_reclassified_as_agent(operation: str) -> None:
    service, tx, row = provisioning_fixture()
    human: Final = SCIMUser(schemas=[], externalId=SUBJECT, userName="human@example.com")
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(
        update={"document": human.model_dump(mode="json")}
    )
    incoming: Final = (
        agent_user()
        if operation == "put"
        else SCIMPatchOp(
            Operations=[{"op": "add", "path": SCIM_AGENT_USER_SCHEMA + ":identityParentId", "value": PARENT}]
        )
    )
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, incoming)
    assert failure.value.status_code == 409
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_native_subject_cannot_be_recreated_by_post_replay() -> None:
    service, tx, row = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(update={"deleted": True})
    with pytest.raises(HTTPException) as failure:
        await service.create_user(agent_user())
    assert failure.value.status_code == 409
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["foreign", "missing", "wrong-kind"])
async def test_scoped_delete_rejects_unknown_or_foreign_resources(state: str) -> None:
    service, tx, row = provisioning_fixture()
    changed: Final = {"source_id": "other"} if state == "foreign" else {"kind": "Groups"}
    tx.litellm_scimresource.find_unique.return_value = None if state == "missing" else row.model_copy(update=changed)
    with pytest.raises(HTTPException) as failure:
        await service.delete("Users", row.id)
    assert failure.value.status_code == 404
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create-replay", "put"])
async def test_scoped_human_routes_preserve_reserved_identity_and_use_human_provisioner(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from litellm.proxy._types import LiteLLM_UserTable

    service, tx, row = provisioning_fixture()
    human: Final = SCIMUser(schemas=[], userName="human@example.com", externalId=SUBJECT)
    reserved: Final = row.model_copy(update={"document": human.model_dump(mode="json"), "local_id": "human"})
    tx.litellm_scimresource.find_unique.return_value = reserved
    tx.litellm_usertable.find_unique = AsyncMock(return_value=LiteLLM_UserTable(user_id="human"))
    tx.litellm_usertable.find_many = AsyncMock(return_value=[])
    tx.litellm_agentstable.create = AsyncMock()
    tx.litellm_usertable.update = AsyncMock(
        return_value=LiteLLM_UserTable(user_id="human", user_email=human.userName, metadata={"scim_active": True})
    )
    result: Final = (
        await service.create_user(human) if operation == "create-replay" else await service.update_user(row.id, human)
    )
    assert result.id == row.id
    assert result.externalId == SUBJECT
    assert result.agent_user is None
    assert tx.litellm_usertable.update.await_args.kwargs["where"] == {"user_id": "human"}
    assert tx.litellm_usertable.update.await_args.kwargs["data"]["user_email"] == human.userName
    service.client.tx.assert_called_once()
    tx.litellm_agentstable.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["missing-external-id", "deleted"])
async def test_group_create_cannot_recreate_deleted_group_or_omit_directory_id(state: str) -> None:
    service, tx, row = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(update={"kind": "Groups", "deleted": True})
    with pytest.raises(HTTPException) as failure:
        await service.create_group(
            SCIMGroup(schemas=[], displayName="Group", externalId=None if state == "missing-external-id" else SUBJECT)
        )
    assert failure.value.status_code == (400 if state == "missing-external-id" else 409)
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["Users", "Groups"])
async def test_get_returns_the_owned_directory_document(kind: str) -> None:
    service, tx, row = provisioning_fixture()
    if kind == "Groups":
        _, group, _ = group_rows(row)
        tx.litellm_scimresource.find_unique.return_value = group
    result: Final = await service.get(kind, tx.litellm_scimresource.find_unique.return_value.id)
    assert result.id == tx.litellm_scimresource.find_unique.return_value.id
    assert isinstance(result, SCIMUser if kind == "Users" else SCIMGroup)


@pytest.mark.asyncio
async def test_native_insert_unique_collision_is_a_conflict() -> None:
    from prisma.errors import UniqueViolationError

    service, tx, _ = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = None
    tx.litellm_agentidentity.find_unique = AsyncMock(return_value=None)
    tx.litellm_agentstable.create = AsyncMock(
        side_effect=UniqueViolationError({"user_facing_error": {"error_code": "P2002", "message": "Duplicate"}})
    )
    tx.litellm_scimresource.create = AsyncMock()
    tx.litellm_verifiedsubject.create = AsyncMock()
    with pytest.raises(HTTPException) as failure:
        await service.create_user(agent_user())
    assert failure.value.status_code == 409
    tx.litellm_verifiedsubject.create.assert_not_awaited()
    assert isinstance(failure.value.__cause__, UniqueViolationError)
    assert service.client.tx.return_value.__aexit__.call_args.args[1] is failure.value


@pytest.mark.asyncio
async def test_missing_native_local_id_rejects_profile_update() -> None:
    service, tx, row = provisioning_fixture()
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(update={"local_id": None})
    with pytest.raises(HTTPException) as failure:
        await service.update_user(row.id, agent_user())
    assert failure.value.status_code == 409
    tx.litellm_scimresource.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["invalid-patch", "concurrent"])
async def test_group_patch_failure_does_not_sync_members(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    _, group, _ = group_rows(native)
    tx.litellm_scimresource.find_unique.return_value = group.model_copy(update={"member_ids": []})
    tx.litellm_scimresource.update_many.return_value = 0
    change: Final = SCIMPatchOp(
        Operations=[
            {"op": "replace", "path": "externalId" if case == "invalid-patch" else "displayName", "value": "changed"}
        ]
    )
    sync: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "update_group", sync)
    with pytest.raises(HTTPException) as failure:
        await service.update_group(group.id, change)
    assert failure.value.status_code == (400 if case == "invalid-patch" else 409)
    sync.assert_not_awaited()
    if case == "invalid-patch":
        tx.litellm_scimresource.update_many.assert_not_awaited()




@pytest.mark.parametrize(
    "value, expected_name, expected_members",
    [
        ({"displayName": "Renamed"}, "Renamed", ["human"]),
        ({"displayName": "Renamed", "members": [{"value": "agent"}]}, "Renamed", ["agent"]),
        ({"members": [{"value": "agent"}]}, "Original", ["agent"]),
    ],
)
def test_pathless_group_replace_applies_display_name_and_members(
    value: dict[str, object], expected_name: str, expected_members: list[str]
) -> None:
    group: Final = SCIMGroup(schemas=[], displayName="Original", members=[SCIMMember(value="human")])
    result: Final = group_members_after_patch(group, SCIMPatchOp(Operations=[{"op": "replace", "value": value}]))
    assert isinstance(result, SCIMGroup), result
    assert result.displayName == expected_name
    assert [member.value for member in result.members or []] == expected_members


@pytest.mark.parametrize(
    "value",
    [
        {"externalId": "foreign"},
        {"displayName": ""},
        {"displayName": "Renamed", "members": [{"display": "no-id"}]},
        "x",
    ],
)
def test_pathless_group_replace_rejects_unsupported_or_invalid_attributes(value: object) -> None:
    group: Final = SCIMGroup(schemas=[], displayName="Original", members=[SCIMMember(value="human")])
    result: Final = group_members_after_patch(group, SCIMPatchOp(Operations=[{"op": "replace", "value": value}]))
    assert isinstance(result, SCIMProvisioningFailure)
    assert result.status == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("identity", ["missing", "ambiguous", "incomplete"])
async def test_source_group_rejects_unusable_human_identity_without_placeholder_creation(
    operation: str, identity: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    human, group, document = group_rows(native)
    member = human.model_copy(update={"local_id": None}) if identity == "incomplete" else human
    tx.litellm_scimresource.find_unique.return_value = None if operation == "create" else group
    tx.litellm_scimresource.create = AsyncMock(return_value=group)
    tx.litellm_scimresource.find_many.return_value = [member]
    tx.litellm_scimresource.count.return_value = 1
    tx.query_raw = AsyncMock(return_value=[] if identity == "missing" else [{"user_id": human.local_id}])
    tx.litellm_usertable.find_many = AsyncMock(return_value=[
        LiteLLM_UserTable(user_id=human.local_id), LiteLLM_UserTable(user_id="other")
    ])
    placeholder = AsyncMock()
    effects = AsyncMock()
    monkeypatch.setattr(scim_v2, "_create_user_if_not_exists", placeholder)
    monkeypatch.setattr(scim_v2, "finish_provisioned_group", effects)
    request = document.model_copy(update={"members": [SCIMMember(value=human.id)]})
    with pytest.raises(HTTPException) as failure:
        await service.create_group(request) if operation == "create" else await service.update_group(group.id, request)
    assert failure.value.status_code == (400 if identity == "ambiguous" else 409)
    assert service.client.tx.return_value.__aexit__.await_args.args[1] is failure.value
    placeholder.assert_not_awaited()
    effects.assert_not_awaited()
    tx.litellm_teamtable.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["directory", "team", "keys"])
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_group_replacement_rolls_back_local_and_directory_writes_then_retries(
    failure_stage: str, failure_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy
    from contextlib import asynccontextmanager

    from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    human, group, document = group_rows(native)
    group = group.model_copy(update={"local_id": group.id, "member_ids": [human.id]})
    state = {"directory": group, "members": [Member(user_id=human.local_id, role="user")],
             "alias": group.display_name, "teams": [group.id], "membership": True, "keys_deleted": False}
    initial = copy.deepcopy(state)
    fail = True
    transaction_open = False

    @asynccontextmanager
    async def transaction(**kwargs):
        nonlocal transaction_open
        snapshot = copy.deepcopy(state)
        transaction_open = True
        try:
            yield tx
        except BaseException:
            state.clear()
            state.update(snapshot)
            raise
        finally:
            transaction_open = False

    async def directory_update(*, where, data):
        state["directory"] = state["directory"].model_copy(update={"member_ids": data["member_ids"], "display_name": data["display_name"], "document": data["document"].data})
        if fail and failure_stage == "directory":
            raise failure_type("directory interrupted")
        return 1

    async def team_read(**kwargs):
        return LiteLLM_TeamTable(team_id=group.id, team_alias=state["alias"], members_with_roles=state["members"], metadata={})

    async def team_write(*, where, data):
        state["alias"] = data.get("team_alias", state["alias"])
        if "members_with_roles" in data:
            import json
            state["members"] = [Member.model_validate(value) for value in json.loads(data["members_with_roles"])]
        if fail and failure_stage == "team":
            raise failure_type("team interrupted")
        return await team_read()

    async def query(sql, *args):
        return [{"members_with_roles": [member.model_dump() for member in state["members"]]}] if "SELECT members_with_roles" in sql else []

    async def user_update(*, where, data):
        state["teams"] = data["teams"]["set"]

    async def membership_delete(**kwargs):
        state["membership"] = False
        return 1

    async def key_delete(**kwargs):
        state["keys_deleted"] = True
        if fail and failure_stage == "keys":
            raise failure_type("key deletion interrupted")
        return 0

    async def effects(result):
        assert not transaction_open
        assert state["directory"].member_ids == []
        assert state["members"] == state["teams"] == []
        assert state["membership"] is False and state["keys_deleted"] is True
        assert len(result.removals) == 1
        assert result.removals[0].user_ids == frozenset([human.local_id])

    service.client.tx.side_effect = transaction
    tx.query_raw = AsyncMock(side_effect=query)
    tx.litellm_scimresource.find_unique.side_effect = lambda **kwargs: state["directory"]
    tx.litellm_scimresource.update_many.side_effect = directory_update
    tx.litellm_scimresource.find_many.return_value = []
    tx.litellm_teamtable.find_unique = AsyncMock(side_effect=team_read)
    tx.litellm_teamtable.update = AsyncMock(side_effect=team_write)
    tx.litellm_usertable.find_many = AsyncMock(side_effect=lambda **kwargs: [LiteLLM_UserTable(user_id=human.local_id, teams=state["teams"])])
    tx.litellm_usertable.update = AsyncMock(side_effect=user_update)
    tx.litellm_teammembership.delete_many = AsyncMock(side_effect=membership_delete)
    tx.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    tx.litellm_verificationtoken.delete_many = AsyncMock(side_effect=key_delete)
    finish = AsyncMock(side_effect=effects)
    monkeypatch.setattr(scim_v2, "finish_provisioned_group", finish)
    monkeypatch.setattr(scim_v2, "provisioning_group_admin_role", AsyncMock(return_value=None))
    change = document.model_copy(update={"displayName": "Renamed", "members": []})
    with pytest.raises(failure_type):
        await service.update_group(group.id, change)
    assert state == initial
    finish.assert_not_awaited()
    fail = False
    result = await service.update_group(group.id, change)
    assert result.id == group.id and result.members == [] and result.displayName == "Renamed"
    finish.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_group_creation_link_failure_rolls_back_and_replays_one_identity(
    failure_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import asynccontextmanager

    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    human, group, document = group_rows(native)
    state = {"directory": None, "team": None}
    fail = True
    transaction_open = False

    @asynccontextmanager
    async def transaction(**kwargs):
        nonlocal transaction_open
        before = dict(state)
        transaction_open = True
        try:
            yield tx
        except BaseException:
            state.clear()
            state.update(before)
            raise
        finally:
            transaction_open = False

    async def create_resource(*, data):
        row = group.model_copy(update={"id": data["id"], "member_ids": [human.id], "document": data["document"].data})
        state["directory"] = row
        return row

    async def link_resource(*, where, data):
        state["directory"] = state["directory"].model_copy(update=data)
        if fail:
            raise failure_type("link interrupted")
        return state["directory"]

    async def write_team(writer, client, incoming, admin_group):
        assert writer is tx and transaction_open
        state["team"] = incoming.id
        return scim_v2.ProvisionedGroupWrite(team_id=incoming.id, created=None, removals=(), additions=())

    async def finish_group(result):
        assert not transaction_open
        assert state["directory"].local_id == state["team"] == result.team_id

    service.client.tx.side_effect = transaction
    tx.litellm_scimresource.find_unique.side_effect = lambda **kwargs: state["directory"]
    tx.litellm_scimresource.create = AsyncMock(side_effect=create_resource)
    tx.litellm_scimresource.update.side_effect = link_resource
    tx.litellm_scimresource.find_many.return_value = [human]
    tx.litellm_scimresource.count.return_value = 1
    effects = AsyncMock(side_effect=finish_group)
    monkeypatch.setattr(scim_v2, "write_provisioned_group", AsyncMock(side_effect=write_team))
    monkeypatch.setattr(scim_v2, "finish_provisioned_group", effects)
    monkeypatch.setattr(scim_v2, "provisioning_group_admin_role", AsyncMock(return_value=None))
    request = document.model_copy(update={"members": [SCIMMember(value=human.id)]})
    with pytest.raises(failure_type):
        await service.create_group(request)
    assert state == {"directory": None, "team": None}
    effects.assert_not_awaited()
    fail = False
    result = await service.create_group(request)
    replay = await service.create_group(request)
    assert result.id == replay.id == state["team"]
    assert effects.await_count == 2
    assert tx.litellm_scimresource.create.await_count == 2


@pytest.mark.asyncio
async def test_group_addition_reuses_member_writes_and_defers_existing_audit_and_cache_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy._types import LiteLLM_TeamMembership, LiteLLM_TeamTable, LiteLLM_UserTable
    from litellm.proxy.management_endpoints import team_endpoints
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, native = provisioning_fixture()
    human, group, document = group_rows(native)
    local_user = LiteLLM_UserTable(user_id=human.local_id, teams=[])
    local_team = LiteLLM_TeamTable(team_id=group.id, members_with_roles=[], metadata={"team_member_budget_id": "inherited"})
    tx.query_raw = AsyncMock(side_effect=lambda sql, *args: [{"members_with_roles": []}] if "SELECT members_with_roles" in sql else [{"user_id": human.local_id}])
    tx.litellm_budgettable.find_unique = AsyncMock(return_value=SimpleNamespace(budget_id="inherited"))
    tx.litellm_usertable.find_many = AsyncMock(return_value=[local_user])
    tx.litellm_usertable.upsert = AsyncMock(return_value=local_user)
    tx.litellm_usertable.update_many = AsyncMock(return_value=1)
    tx.litellm_teamtable.find_unique = AsyncMock(return_value=local_team)
    tx.litellm_teamtable.update = AsyncMock(return_value=local_team)
    tx.litellm_teammembership.upsert = AsyncMock(return_value=LiteLLM_TeamMembership(
        team_id=group.id, user_id=human.local_id, budget_id="inherited"
    ))
    invalidate = AsyncMock()
    audit = MagicMock()
    monkeypatch.setattr(team_endpoints, "evict_and_broadcast", invalidate)
    monkeypatch.setattr(team_endpoints, "_schedule_team_member_add_audit_logs", audit)
    request = document.model_copy(update={"id": group.id, "members": [SCIMMember(value=human.local_id)]})
    result = await scim_v2.write_provisioned_group(tx, service.client, request, None)
    assert result.created is None and result.removals == ()
    assert len(result.additions) == 1
    assert result.additions[0].users[0].user_id == human.local_id
    membership = tx.litellm_teammembership.upsert.await_args.kwargs["data"]
    assert membership["create"]["budget_id"] == "inherited"
    assert membership["create"]["user_id"] == human.local_id
    assert tx.litellm_usertable.update_many.await_count == 1
    service.client.tx.assert_not_called()
    invalidate.assert_not_awaited()
    audit.assert_not_called()
    await scim_v2.finish_provisioned_group(result)
    assert any(call.kwargs.get("cache_keys") == (human.local_id,) for call in invalidate.await_args_list)
    audit.assert_called_once()
    assert audit.call_args.kwargs["existing_user_ids"] == frozenset([human.local_id])

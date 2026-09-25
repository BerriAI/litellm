from typing import Final
from unittest.mock import AsyncMock

from fastapi import HTTPException

import pytest
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
    tx.execute_raw = AsyncMock(return_value=1)
    tx.litellm_scimsource.find_unique = AsyncMock(return_value=source)
    tx.litellm_scimresource.find_unique = AsyncMock(return_value=row)
    tx.litellm_scimresource.update_many = AsyncMock(return_value=1)
    tx.litellm_scimresource.update = AsyncMock(return_value=row)
    tx.litellm_scimresource.find_many = AsyncMock(return_value=[])
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
    tx.litellm_scimresource.create = AsyncMock(return_value=group)
    tx.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    create_team: Final = AsyncMock(return_value=document.model_copy(update={"id": "local-team"}))
    monkeypatch.setattr(scim_v2, "create_group", create_team)
    result: Final = await service.create_group(
        document.model_copy(update={"members": [SCIMMember(value=row.id) for row in rows]})
    )
    assert result.id == group.id
    assert {member.value for member in result.members} == set(group.member_ids)
    if include_human:
        create_team.assert_awaited_once()
        assert create_team.call_args.kwargs["group"].members == [SCIMMember(value=human.local_id)]
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
    monkeypatch.setattr(scim_v2, "create_group", create_team)
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
    tx.litellm_teamtable.find_unique = AsyncMock(return_value=SimpleNamespace(team_id="local-team"))
    update_team: Final = AsyncMock(return_value=document.model_copy(update={"id": "local-team"}))
    monkeypatch.setattr(scim_v2, "update_group", update_team)
    result: Final = await service.create_group(document.model_copy(update={"members": [SCIMMember(value=native.id)]}))
    assert result.id == old.id
    assert result.members == [SCIMMember(value=native.id)]
    assert tx.litellm_scimresource.update_many.call_args.kwargs["data"]["member_ids"] == [native.id]
    update_team.assert_awaited_once()
    assert update_team.call_args.kwargs["group_id"] == "local-team"
    assert update_team.call_args.kwargs["group"].members == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create-replay", "put"])
async def test_scoped_human_routes_preserve_reserved_identity_and_use_human_provisioner(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.scim import scim_v2

    service, tx, row = provisioning_fixture()
    human: Final = SCIMUser(schemas=[], userName="human@example.com", externalId=SUBJECT)
    reserved: Final = row.model_copy(update={"document": human.model_dump(mode="json"), "local_id": "human"})
    tx.litellm_scimresource.find_unique.return_value = reserved
    tx.litellm_usertable.find_unique = AsyncMock(return_value=LiteLLM_UserTable(user_id="human"))
    tx.litellm_agentstable.create = AsyncMock()
    updated: Final = AsyncMock(return_value=human.model_copy(update={"id": "human"}))
    monkeypatch.setattr(scim_v2, "update_user", updated)
    result: Final = (
        await service.create_user(human) if operation == "create-replay" else await service.update_user(row.id, human)
    )
    assert result.id == row.id
    assert result.externalId == SUBJECT
    assert result.agent_user is None
    assert updated.call_args.kwargs["user_id"] == "human"
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

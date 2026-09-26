from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.models import LiteLLM_SCIMResource, LiteLLM_SCIMSource

from litellm.proxy import proxy_server
from litellm.proxy.management_endpoints.scim import scim_v2
from litellm.proxy.management_endpoints.scim.human_provisioning import SourceHumanProvisioner, human_email
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMPatchOp, SCIMUser

TENANT: Final = "11111111-1111-4111-8111-111111111111"
SUBJECT: Final = "22222222-2222-4222-8222-222222222222"


def human_fixture():
    now: Final = datetime.now(timezone.utc)
    source: Final = LiteLLM_SCIMSource(
        source_id="source",
        display_name="Directory",
        tenant_id=TENANT,
        key_hash="hash",
        enabled=True,
        group_mappings="[]",
        created_at=now,
        updated_at=now,
    )
    user: Final = SCIMUser(schemas=[], userName="human@example.com", externalId=SUBJECT, displayName="Human")
    row: Final = LiteLLM_SCIMResource(
        id="stable-scim-id",
        source_id=source.source_id,
        kind="Users",
        external_id=SUBJECT,
        user_name=user.userName,
        display_name="Human",
        document=user.model_dump_json(),
        active=True,
        deleted=False,
        local_id="local-human",
        human_email=user.userName,
        member_ids=[],
        created_at=now,
        updated_at=now,
    )
    client: Final = MagicMock(spec=PrismaClient)
    tx: Final = client.tx.return_value.__aenter__.return_value
    tx.litellm_scimresource.find_unique = AsyncMock(return_value=None)
    tx.litellm_scimresource.create = AsyncMock(return_value=row)
    tx.litellm_scimresource.update = AsyncMock(return_value=row)
    tx.litellm_usertable.find_many = AsyncMock(return_value=[])
    tx.litellm_usertable.find_unique = AsyncMock(return_value=None)
    tx.litellm_usertable.create = AsyncMock()
    client.db = MagicMock()
    client.db.litellm_usertable.count = AsyncMock(return_value=0)
    return SourceHumanProvisioner(client, source), tx, row, user


@pytest.mark.parametrize(
    "emails,expected",
    [
        (None, "human@example.com"),
        ([{"value": "FIRST@example.com"}], "first@example.com"),
        ([{"value": "FIRST@example.com"}, {"value": "PRIMARY@example.com", "primary": True}], "primary@example.com"),
    ],
)
def test_human_ownership_email_uses_primary_and_normalizes_case(emails: object, expected: str) -> None:
    user: Final = SCIMUser.model_validate({"schemas": [], "userName": "human@example.com", "emails": emails})
    assert human_email(user) == expected


@pytest.mark.asyncio
async def test_reservation_replay_preserves_identity_before_creating_a_local_user() -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_scimresource.find_unique.return_value = row
    assert await service.reserve(user) == row
    tx.litellm_scimresource.create.assert_not_awaited()
    tx.litellm_usertable.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_reservation_claims_email_subject_and_local_identity() -> None:
    service, tx, _, user = human_fixture()
    await service.reserve(user)
    data: Final = tx.litellm_scimresource.create.call_args.kwargs["data"]
    assert data["local_id"] == user.userName
    assert data["human_email"] == user.userName
    assert data["human_subject_key"] == f"{TENANT}:{SUBJECT}"
    assert data["id"] == data["document"].data["id"]
    tx.litellm_usertable.create.assert_awaited_once_with(data={
        "user_id": user.userName, "user_email": user.userName,
        "user_role": "internal_user_viewer", "teams": [],
    })


@pytest.mark.asyncio
async def test_ambiguous_local_human_match_is_rejected_before_reserving() -> None:
    service, tx, _, user = human_fixture()
    tx.litellm_usertable.find_many.return_value = [SimpleNamespace(user_id="one"), SimpleNamespace(user_id="two")]
    with pytest.raises(HTTPException) as failure:
        await service.reserve(user)
    assert failure.value.status_code == 409
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["externalId", "userName"])
async def test_incomplete_identity_cannot_be_reserved(missing: str) -> None:
    service, tx, _, user = human_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.create(user.model_copy(update={missing: None}))
    assert failure.value.status_code == 400
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_local_human_is_not_recreated_or_email_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_scimresource.find_unique.return_value = row
    create: Final = AsyncMock(return_value=user.model_copy(update={"id": "unrelated-admin"}))
    update: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "create_user", create)
    monkeypatch.setattr(scim_v2, "update_user", update)
    with pytest.raises(HTTPException) as failure:
        await service.create(user)
    assert failure.value.status_code == 409
    create.assert_not_awaited()
    update.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_human_cannot_be_recreated_by_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(update={"deleted": True})
    create: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "create_user", create)
    with pytest.raises(HTTPException) as failure:
        await service.create(user)
    assert failure.value.status_code == 409
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("path,value", [("groups", []), ("externalId", "foreign"), (None, {"externalId": "foreign"})])
async def test_scoped_human_patch_cannot_modify_directory_owned_correspondence(path: str | None, value: object) -> None:
    service, tx, row, _ = human_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.update(row, SCIMPatchOp(Operations=[{"op": "replace", "path": path, "value": value}]))
    assert failure.value.status_code == 400
    tx.litellm_usertable.find_unique.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_insert_failure_aborts_resource_reservation() -> None:
    service, tx, _, user = human_fixture()
    tx.litellm_usertable.create.side_effect = RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        await service.reserve(user)
    tx.litellm_scimresource.create.assert_not_awaited()
    assert service.client.tx.return_value.__aexit__.call_args.args[0] is RuntimeError


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["reservation", "email-update"])
async def test_ownership_collision_is_a_conflict_before_legacy_user_mutation(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prisma.errors import UniqueViolationError

    service, tx, row, user = human_fixture()
    collision: Final = UniqueViolationError(
        {"user_facing_error": {"error_code": "P2002", "message": "Unique identity"}}
    )
    create: Final = AsyncMock()
    update: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "create_user", create)
    monkeypatch.setattr(scim_v2, "update_user", update)
    if phase == "reservation":
        tx.litellm_scimresource.create.side_effect = collision
    else:
        tx.litellm_scimresource.find_unique.return_value = row
        tx.litellm_scimresource.update.side_effect = collision
        tx.litellm_usertable.find_unique.return_value = SimpleNamespace(user_id=row.local_id)
    with pytest.raises(HTTPException) as failure:
        await service.create(user)
    assert failure.value.status_code == 409
    create.assert_not_awaited()
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_patch_preserves_scim_id_and_claims_the_changed_email(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import LiteLLM_UserTable

    service, tx, row, user = human_fixture()
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id=row.local_id, user_email="human@example.com"
    )
    updated: Final = user.model_copy(update={"id": row.local_id, "active": False})
    patch: Final = AsyncMock(return_value=updated)
    monkeypatch.setattr(scim_v2, "patch_user", patch)
    operations: Final = SCIMPatchOp(Operations=[{"op": "replace", "path": "active", "value": False}])
    result: Final = await service.update(row, operations)
    assert result.id == row.id and result.externalId == row.external_id
    assert result.active is False
    patch.assert_awaited_once_with(user_id=row.local_id, patch_ops=operations)
    assert tx.litellm_scimresource.update.call_args.kwargs["data"]["active"] is False


@pytest.mark.asyncio
async def test_unavailable_ownership_database_is_not_reported_as_a_conflict() -> None:
    service, tx, _, user = human_fixture()
    tx.litellm_scimresource.find_unique.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await service.create(user)
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_cannot_adopt_an_existing_local_or_sso_user() -> None:
    service, tx, _, user = human_fixture()
    tx.litellm_usertable.find_many.return_value = [SimpleNamespace(user_id="existing-admin")]
    with pytest.raises(HTTPException) as failure:
        await service.reserve(user)
    assert failure.value.status_code == 409
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["put", "display", "username", "object"])
async def test_source_username_is_independent_of_local_display_name(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy._types import LiteLLM_UserTable

    service, tx, row, user = human_fixture()
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id=row.local_id, user_email="human@example.com"
    )
    legacy: Final = user.model_copy(update={"userName": "Display Name", "displayName": "Display Name"})
    monkeypatch.setattr(scim_v2, "update_user", AsyncMock(return_value=legacy))
    monkeypatch.setattr(scim_v2, "patch_user", AsyncMock(return_value=legacy))
    changes: Final = {
        "put": user,
        "display": SCIMPatchOp(Operations=[{"op": "replace", "path": "displayName", "value": "Display Name"}]),
        "username": SCIMPatchOp(Operations=[{"op": "replace", "path": "userName", "value": "renamed@example.com"}]),
        "object": SCIMPatchOp(Operations=[{"op": "replace", "value": {"userName": "renamed@example.com"}}]),
    }
    result: Final = await service.update(row, changes[operation])
    expected: Final = "renamed@example.com" if operation in ("username", "object") else "human@example.com"
    assert result.userName == expected
    assert result.displayName == "Display Name"
    assert result.id == row.id
    assert tx.litellm_scimresource.update.call_args.kwargs["data"]["user_name"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,value", [("remove", None), ("replace", ""), ("replace", 123)])
async def test_invalid_username_is_rejected_before_local_mutation(operation: str, value: object) -> None:
    service, tx, row, _ = human_fixture()
    change: Final = SCIMPatchOp(Operations=[{"op": operation, "path": "userName", "value": value}])
    with pytest.raises(HTTPException, match="userName is required") as failure:
        await service.update(row, change)
    assert failure.value.status_code == 400
    tx.litellm_usertable.find_unique.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_guid_case_replays_the_same_reserved_identity() -> None:
    service, tx, row, user = human_fixture()
    guid: Final = "abcdefab-abcd-4abc-8abc-abcdefabcdef"
    tx.litellm_scimresource.find_unique.side_effect = lambda **query: (
        row if query["where"]["source_id_kind_external_id"]["external_id"] == guid else None
    )
    result: Final = await service.reserve(user.model_copy(update={"externalId": guid.upper()}))
    assert result.id == row.id
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_email_update_cannot_claim_an_unrelated_local_human(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_usertable.find_unique.return_value = SimpleNamespace(user_id=row.local_id)
    tx.litellm_usertable.find_many.return_value = [SimpleNamespace(user_id="unrelated-admin")]
    update: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "update_user", update)
    with pytest.raises(HTTPException) as failure:
        await service.update(row, SCIMUser.model_validate({**user.model_dump(), "emails": [{"value": "ADMIN@example.com"}]}))
    assert failure.value.status_code == 409
    update.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_human_creation_preserves_license_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, _, user = human_fixture()
    service.client.db.litellm_usertable.count.side_effect = [3, 0]
    license_check: Final = MagicMock()
    license_check.is_over_limit.return_value = True
    monkeypatch.setattr(proxy_server, "_license_check", license_check)
    with pytest.raises(HTTPException) as failure:
        await service.reserve(user)
    assert failure.value.status_code == 403
    license_check.is_over_limit.assert_called_once_with(total_users=3)
    tx.litellm_usertable.create.assert_not_awaited()
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_replayed_human_does_not_consume_another_license_seat(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_scimresource.find_unique.return_value = row
    license_check: Final = MagicMock()
    license_check.is_over_limit.return_value = True
    monkeypatch.setattr(proxy_server, "_license_check", license_check)
    assert await service.reserve(user) == row
    license_check.is_over_limit.assert_not_called()
    tx.litellm_usertable.create.assert_not_awaited()

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.models import LiteLLM_SCIMResource, LiteLLM_SCIMSource

from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim import scim_v2
from litellm.proxy.management_endpoints.scim.human_provisioning import SourceHumanProvisioner, human_email
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMPatchOp, SCIMUser, SCIMUserEmail

TENANT: Final = "11111111-1111-4111-8111-111111111111"
SUBJECT: Final = "22222222-2222-4222-8222-222222222222"


@pytest.mark.asyncio
async def test_source_ownership_collects_scim_and_local_ids_across_query_batches() -> None:
    from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

    client: Final = MagicMock(spec=PrismaClient)
    client.writer_db = MagicMock()
    client.db = MagicMock()
    local_ids: Final = tuple(f"subject-{index}" for index in range(IN_LIST_CHUNK_SIZE + 1))
    by_scim_id: Final = SimpleNamespace(id=local_ids[-1], local_id="local-human")
    by_local_id: Final = SimpleNamespace(id="scim-human", local_id=local_ids[0])
    client.writer_db.litellm_scimresource.find_many = AsyncMock(side_effect=[[], [by_scim_id], [], [by_local_id]])
    owned: Final = await scim_v2._source_owned_ids(client, "Users", local_ids)
    assert owned == frozenset((local_ids[-1], "local-human", "scim-human", local_ids[0]))
    client.db.litellm_scimresource.find_many.assert_not_called()


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

    async def save_local(*, where, data):
        before = tx.litellm_usertable.find_unique.return_value.model_dump()
        values = {**before, **data}
        if isinstance(values.get("metadata"), str):
            values["metadata"] = json.loads(values["metadata"])
        return LiteLLM_UserTable.model_validate(values)

    tx.litellm_usertable.update = AsyncMock(side_effect=save_local)
    tx.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    tx.litellm_verificationtoken.update = AsyncMock()

    client.db = MagicMock()
    client.db.litellm_usertable.count = AsyncMock(return_value=0)
    tx.litellm_usertable.count = client.db.litellm_usertable.count
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


def test_missing_human_ownership_identifier_is_rejected() -> None:
    with pytest.raises(HTTPException) as failure:
        human_email(SCIMUser(schemas=[]))
    assert failure.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["externalId", "userName"])
async def test_direct_reservation_requires_complete_directory_identity(missing: str) -> None:
    service, tx, _, user = human_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.reserve(user.model_copy(update={missing: None}))
    assert failure.value.status_code == 400
    tx.litellm_scimresource.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_create_does_not_hide_storage_failure() -> None:
    service, tx, _, user = human_fixture()
    tx.litellm_scimresource.find_unique.side_effect = ConnectionError("unavailable")
    with pytest.raises(ConnectionError, match="unavailable"):
        await service.create(user)
    tx.litellm_usertable.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_human_record_cannot_be_updated() -> None:
    service, tx, row, user = human_fixture()
    with pytest.raises(HTTPException) as failure:
        await service.update(row.model_copy(update={"local_id": None}), user)
    assert failure.value.status_code == 409
    tx.litellm_usertable.find_unique.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,value",
    [
        ("emails", [{"value": "NEW@example.com"}]),
        ('emails[type eq "work"].value', "NEW@example.com"),
        (None, {"emails": [{"value": "NEW@example.com"}]}),
    ],
)
async def test_scoped_email_patch_requires_put_before_mutation(path: str | None, value: object) -> None:
    service, tx, row, _ = human_fixture()
    change: Final = SCIMPatchOp(Operations=[{"op": "replace", "path": path, "value": value}])
    with pytest.raises(HTTPException, match="PUT") as failure:
        await service.update(row, change)
    assert failure.value.status_code == 400
    tx.litellm_usertable.find_unique.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "emails", [None, [{"value": "FIRST@example.com"}, {"value": "PRIMARY@example.com", "primary": True}]]
)
async def test_human_put_claims_the_same_email_it_writes(emails: object, monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(user_id=row.local_id, user_email=row.human_email, teams=[])
    change: Final = SCIMUser.model_validate({**user.model_dump(), "emails": emails})
    expected: Final = human_email(change)

    result: Final = await service.update(row, change)
    assert result.id == row.id
    assert result.emails and result.emails[0].value == expected
    assert tx.litellm_usertable.update.await_args.kwargs["data"]["user_email"] == expected
    assert tx.litellm_scimresource.update.await_args_list[0].kwargs["data"] == {"human_email": expected}


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
    tx.litellm_usertable.create.assert_awaited_once_with(
        data={
            "user_id": user.userName,
            "user_email": user.userName,
            "user_role": "internal_user_viewer",
            "teams": [],
        }
    )


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
async def test_disappearing_local_human_aborts_update_before_key_or_document_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, tx, row, user = human_fixture()
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id=row.local_id, user_email=row.human_email, metadata={"scim_active": True}
    )
    tx.litellm_usertable.update = AsyncMock(return_value=None)
    evict_user: Final = AsyncMock()
    evict_key: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "evict_and_broadcast", evict_user)
    monkeypatch.setattr(scim_v2, "_delete_cache_key_object", evict_key)

    with pytest.raises(HTTPException) as failure:
        await service.update(row, user.model_copy(update={"active": False}))

    assert failure.value.status_code == 409
    assert "automatic recreation is not permitted" in failure.value.detail
    assert service.client.tx.return_value.__aexit__.await_args.args[1] is failure.value
    tx.litellm_scimresource.update.assert_awaited_once()
    assert tx.litellm_scimresource.update.await_args.kwargs["data"] == {"human_email": row.human_email}
    tx.litellm_usertable.create.assert_not_awaited()
    tx.litellm_verificationtoken.find_many.assert_not_awaited()
    tx.litellm_verificationtoken.update.assert_not_awaited()
    evict_user.assert_not_awaited()
    evict_key.assert_not_awaited()


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
        tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(user_id=row.local_id, user_email=row.human_email, teams=[])
    with pytest.raises(HTTPException) as failure:
        await service.create(user)
    assert failure.value.status_code == 409
    create.assert_not_awaited()
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_patch_preserves_scim_id_and_updates_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import LiteLLM_UserTable

    service, tx, row, user = human_fixture()
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id=row.local_id, user_email="human@example.com"
    )
    operations: Final = SCIMPatchOp(Operations=[{"op": "replace", "path": "active", "value": False}])
    result: Final = await service.update(row, operations)
    assert result.id == row.id and result.externalId == row.external_id
    assert result.active is False
    assert json.loads(tx.litellm_usertable.update.await_args.kwargs["data"]["metadata"])["scim_active"] is False
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
    changes: Final = {
        "put": user.model_copy(update={"userName": "renamed@example.com", "emails": [SCIMUserEmail(value="human@example.com", primary=True)]}),
        "display": SCIMPatchOp(Operations=[{"op": "replace", "path": "displayName", "value": "Display Name"}]),
        "username": SCIMPatchOp(Operations=[{"op": "replace", "path": "userName", "value": "renamed@example.com"}]),
        "object": SCIMPatchOp(Operations=[{"op": "replace", "value": {"userName": "renamed@example.com"}}]),
    }
    result: Final = await service.update(row, changes[operation])
    expected: Final = "renamed@example.com" if operation in ("put", "username", "object") else "human@example.com"
    assert result.userName == expected
    assert result.displayName == "human@example.com"
    if operation == "display":
        assert tx.litellm_usertable.update.await_args.kwargs["data"]["user_alias"] == "Display Name"
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
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(user_id=row.local_id, user_email=row.human_email, teams=[])
    tx.litellm_usertable.find_many.return_value = [SimpleNamespace(user_id="unrelated-admin")]
    update: Final = AsyncMock()
    monkeypatch.setattr(scim_v2, "update_user", update)
    with pytest.raises(HTTPException) as failure:
        await service.update(
            row, SCIMUser.model_validate({**user.model_dump(), "emails": [{"value": "ADMIN@example.com"}]})
        )
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


@pytest.mark.asyncio
async def test_existing_native_agent_cannot_be_reclassified_as_human() -> None:
    from litellm.types.proxy.management_endpoints.scim_v2 import SCIM_AGENT_USER_SCHEMA

    service, tx, row, user = human_fixture()
    native: Final = SCIMUser.model_validate(
        {
            **user.model_dump(),
            SCIM_AGENT_USER_SCHEMA: {"identityParentId": TENANT},
        }
    )
    tx.litellm_scimresource.find_unique.return_value = row.model_copy(
        update={"document": native.model_dump(by_alias=True, mode="json")}
    )
    with pytest.raises(HTTPException) as failure:
        await service.create(user)
    assert failure.value.status_code == 409
    tx.litellm_usertable.find_unique.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_put_without_username_is_rejected_before_anything_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tx, row, user = human_fixture()
    legacy_put: Final = AsyncMock(return_value=user)
    monkeypatch.setattr(scim_v2, "update_user", legacy_put)
    with pytest.raises(HTTPException, match="userName is required") as failure:
        await service.update(
            row, user.model_copy(update={"userName": None, "emails": [SCIMUserEmail(value="human@example.com")]})
        )
    assert failure.value.status_code == 400
    legacy_put.assert_not_awaited()
    tx.litellm_usertable.find_unique.assert_not_awaited()
    tx.litellm_scimresource.update.assert_not_called()


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("failure_stage", ["local", "keys", "document"])
@pytest.mark.asyncio
async def test_human_update_rolls_back_and_retry_preserves_identity(
    failure_type, failure_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import asynccontextmanager

    service, tx, row, user = human_fixture()
    state = {"human_email": row.human_email, "user_email": row.human_email, "blocked": False}
    failure_enabled = True
    transaction_open = False

    @asynccontextmanager
    async def transaction():
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

    async def save_resource(*, where, data):
        state.update(data)
        if "document" in data and failure_stage == "document" and failure_enabled:
            raise failure_type("directory write interrupted")
        return row

    async def save_user(*, where, data):
        state["user_email"] = data["user_email"]
        if failure_stage == "local" and failure_enabled:
            raise failure_type("local write interrupted")
        return LiteLLM_UserTable(
            user_id=row.local_id, user_email=data["user_email"], metadata=json.loads(data["metadata"]), teams=[]
        )

    async def save_key(*, where, data):
        state["blocked"] = data["blocked"]
        if failure_stage == "keys" and failure_enabled:
            raise failure_type("key write interrupted")

    async def invalidate_key(**kwargs):
        assert not transaction_open
        assert state["blocked"] is True

    service.client.tx.side_effect = transaction
    tx.litellm_scimresource.update = AsyncMock(side_effect=save_resource)
    tx.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
        user_id=row.local_id, user_email=row.human_email, teams=[], metadata={"scim_active": True}
    )
    tx.litellm_usertable.update = AsyncMock(side_effect=save_user)
    tx.litellm_verificationtoken.find_many.return_value = [SimpleNamespace(token="human-key", metadata={})]
    tx.litellm_verificationtoken.update = AsyncMock(side_effect=save_key)
    evict_user = AsyncMock()
    evict_key = AsyncMock(side_effect=invalidate_key)
    monkeypatch.setattr(scim_v2, "evict_and_broadcast", evict_user)
    monkeypatch.setattr(scim_v2, "_delete_cache_key_object", evict_key)
    change: Final = user.model_copy(
        update={"emails": [SCIMUserEmail(value="replacement@example.com", primary=True)], "active": False}
    )
    with pytest.raises(failure_type):
        await service.update(row, change)
    assert state == {"human_email": row.human_email, "user_email": row.human_email, "blocked": False}
    evict_user.assert_not_awaited()
    evict_key.assert_not_awaited()

    failure_enabled = False
    result: Final = await service.update(row, change)
    assert result.id == row.id and result.externalId == row.external_id
    assert result.active is False
    assert state["human_email"] == state["user_email"] == "replacement@example.com"
    assert state["blocked"] is True
    assert state["active"] is False
    evict_user.assert_awaited_once()
    evict_key.assert_awaited_once()

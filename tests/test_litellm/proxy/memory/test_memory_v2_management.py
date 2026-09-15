"""Admin activation and record permissions, replacing only external database/cache edges."""

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prisma.models import LiteLLM_MemoryTable

from litellm.caching.caching import DualCache
from litellm.proxy._types import UI_TEAM_ID, KeyManagementRoutes, LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.record_permissions import can_read_team_records
from litellm.proxy.memory import management
from litellm.proxy.memory.continuation import MemoryContinuation, MemoryContinuations
from litellm.proxy.memory.policy import MemoryIdentity, resolve_memory_access
from litellm.proxy.memory.store import MemoryStore
from litellm.types.memory_v2 import (
    MemoryCapture,
    MemoryRecallRequest,
    MemorySearch,
    MemorySettings,
)

_NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
_CAPTURE = MemoryCapture(key="demo", title="Demo port", content="Use port 8123", evidence="User selected port 8123")


@pytest.fixture
def database() -> Iterator[MagicMock]:
    client = MagicMock()
    for name in (
        "litellm_config",
        "litellm_memorytable",
        "litellm_teamtable",
        "litellm_usertable",
        "litellm_memorycontinuation",
    ):
        table = getattr(client.db, name)
        table.find_unique = AsyncMock(return_value=None)
        table.find_first = AsyncMock(return_value=None)
        table.find_many = AsyncMock(return_value=[])
        table.upsert = AsyncMock()
        table.delete_many = AsyncMock(return_value=0)
        table.create = AsyncMock(return_value=row())
        table.count = AsyncMock(return_value=0)
        table.update_many = AsyncMock(return_value=1)
    client.db.tx.return_value.__aenter__.return_value = client.db
    client.db.execute_raw = AsyncMock()
    client.db.litellm_usertable.find_unique.return_value = {"user_id": "owner", "teams": ["team"]}
    with patch.multiple(  # test-quality-ok: Inject the external database/cache; run real handlers and authorization.
        "litellm.proxy.proxy_server", prisma_client=client, user_api_key_cache=DualCache()
    ):
        yield client


def auth(user: str | None = "owner", role: LitellmUserRoles = LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token="a" * 64, user_id=user, user_role=role, team_id="team", org_id="org")


def configure(database: MagicMock, **settings: object) -> None:
    database.db.litellm_config.find_unique.return_value = SimpleNamespace(
        param_value=MemorySettings.model_validate({"enabled": True, **settings}).model_dump()
    )


def team(
    *, member: str = "owner", role: str = "user", permissions: tuple[str, ...] = (), **changes: object
) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable.model_validate(
        {
            "team_id": "team",
            "team_alias": "Engineering",
            "organization_id": "org",
            "members_with_roles": [{"user_id": member, "role": role}],
            "team_member_permissions": list(permissions),
            **changes,
        }
    )


def row(**changes: object) -> LiteLLM_MemoryTable:
    namespace = MemoryIdentity.from_auth(auth()).namespace
    return LiteLLM_MemoryTable.model_validate(
        {
            "memory_id": "entry",
            "key": f"memory-v2:{namespace}:demo",
            "namespace": namespace,
            "value": _CAPTURE.content,
            "metadata": json.dumps({"title": _CAPTURE.title, "evidence": _CAPTURE.evidence}),
            "user_id": "owner",
            "team_id": "team",
            "organization_id": "org",
            "owner_key_id": "a" * 64,
            "created_by": "owner",
            "created_at": _NOW,
            "updated_at": _NOW,
            **changes,
        }
    )


@pytest.mark.asyncio
async def test_default_off_and_proxy_admin_can_enable_selected_users(database: MagicMock) -> None:
    admin = auth("admin", LitellmUserRoles.PROXY_ADMIN)
    assert not (await management.get_settings(admin)).enabled
    assert not (await management.get_status(auth())).active
    database.db.litellm_usertable.find_many.return_value = [
        SimpleNamespace(user_id="owner", user_alias="Alex", user_email="alex@example.test")
    ]
    saved = await management.set_settings(
        MemorySettings(enabled=True, everyone=False, user_ids=("owner", "owner")), admin
    )
    assert saved.user_ids == ("owner",)
    assert saved.user_names == {"owner": "Alex"}
    written = database.db.litellm_config.upsert.call_args.kwargs["data"]["update"]["param_value"]
    database.db.litellm_config.find_unique.return_value = SimpleNamespace(param_value=written)
    assert (await management.get_status(auth())).active
    assert (await management.get_status(auth().model_copy(update={"token": "b" * 64}))).active
    assert not (await management.get_status(auth("other"))).active
    assert not (await management.get_status(auth(None))).active


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ],
)
async def test_only_proxy_admin_can_change_activation(database: MagicMock, role: LitellmUserRoles) -> None:
    database.db.litellm_teamtable.find_many.return_value = [team(role="admin")]
    with pytest.raises(HTTPException) as exc:
        await management.set_settings(MemorySettings(enabled=True), auth(role=role))
    assert exc.value.status_code == 403
    database.db.litellm_config.upsert.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [(), ("missing",)])
async def test_invalid_selected_users_are_rejected_without_changing_config(
    database: MagicMock, selected: tuple[str, ...]
) -> None:
    with pytest.raises(HTTPException) as exc:
        await management.set_settings(
            MemorySettings(enabled=True, everyone=False, user_ids=selected), auth(role=LitellmUserRoles.PROXY_ADMIN)
        )
    assert exc.value.status_code == 422
    database.db.litellm_config.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_everyone_clears_stale_selection_and_includes_service_keys(database: MagicMock) -> None:
    settings = await management.set_settings(
        MemorySettings(enabled=True, user_ids=("deleted",)), auth(role=LitellmUserRoles.PROXY_ADMIN)
    )
    assert settings.user_ids == ()
    configure(database)
    assert (await resolve_memory_access(database, MemoryIdentity.from_auth(auth(None)))).active
    database.db.litellm_usertable.find_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "member,member_role,permission,global_role,expected",
    [
        ("owner", "user", None, LitellmUserRoles.INTERNAL_USER, ()),
        ("owner", "user", "/spend/logs", LitellmUserRoles.INTERNAL_USER, ()),
        ("owner", "user", "/memory/v2/entries", LitellmUserRoles.INTERNAL_USER, ("team",)),
        ("owner", "admin", None, LitellmUserRoles.INTERNAL_USER, ("team",)),
        ("other", "admin", "/memory/v2/entries", LitellmUserRoles.INTERNAL_USER, ()),
        ("owner", "user", None, LitellmUserRoles.ORG_ADMIN, ()),
    ],
)
async def test_team_memory_access_reuses_membership_but_not_log_permission(
    database: MagicMock,
    member: str,
    member_role: str,
    permission: str | None,
    global_role: LitellmUserRoles,
    expected: tuple[str, ...],
) -> None:
    database.db.litellm_teamtable.find_many.return_value = [
        team(member=member, role=member_role, permissions=(permission,) if permission else ())
    ]
    access = await resolve_memory_access(database, MemoryIdentity.from_auth(auth(role=global_role)))
    assert access.team_ids == expected
    assert not access.admin_view


@pytest.mark.parametrize(
    "permission,log_read,memory_read", [("/spend/logs", True, False), ("/memory/v2/entries", False, True)]
)
def test_common_helper_keeps_resource_permissions_independent(
    permission: str, log_read: bool, memory_read: bool
) -> None:
    context = team(permissions=(permission,))
    assert can_read_team_records(auth(), context, KeyManagementRoutes.SPEND_LOGS) is log_read
    assert can_read_team_records(auth(), context, KeyManagementRoutes.MEMORY_READ) is memory_read
    assert not can_read_team_records(auth("outsider"), context, KeyManagementRoutes.MEMORY_READ)


@pytest.mark.asyncio
async def test_key_context_restricts_org_while_dashboard_combines_permitted_teams(database: MagicMock) -> None:
    database.db.litellm_usertable.find_unique.return_value = {"user_id": "owner", "teams": ["team", "elsewhere"]}
    database.db.litellm_teamtable.find_many.return_value = [
        team(role="admin"),
        team(role="admin", team_id="elsewhere", organization_id="other-org"),
    ]
    key = await resolve_memory_access(database, MemoryIdentity.from_auth(auth()))
    dashboard = await resolve_memory_access(
        database, MemoryIdentity.from_auth(auth().model_copy(update={"is_session_token": True, "team_id": UI_TEAM_ID}))
    )
    assert key.team_ids == ("team",)
    assert dashboard.team_ids == ("team", "elsewhere")
    assert key.visible_rows()["organization_id"] == "org"
    assert "organization_id" not in dashboard.visible_rows()
    assert dashboard.identity.key_id is None and dashboard.identity.team_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_authenticated_proxy_roles_inspect_all_without_reinterpreting_auth(
    database: MagicMock, role: LitellmUserRoles
) -> None:
    database.db.litellm_usertable.find_unique.return_value = None
    access = await resolve_memory_access(database, MemoryIdentity.from_auth(auth(role=role)))
    assert access.admin_view
    assert access.visible_rows() == {"namespace": {"not": None}}
    assert access.identity.read_only == (role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)


@pytest.mark.asyncio
async def test_team_read_does_not_enable_human_edit_or_delete(database: MagicMock) -> None:
    configure(database)
    database.db.litellm_teamtable.find_many.return_value = [team(role="admin")]
    store = await management.memory_store(auth())
    assert not store.entry(row(user_id="teammate")).can_edit
    assert store.entry(row()).can_edit
    assert "team_id" not in str(store.access.visible_rows(write=True))
    with pytest.raises(HTTPException) as exc:
        await store.update("teammate-record", _CAPTURE)
    assert exc.value.status_code == 404
    assert not await store.delete("teammate-record")
    database.db.litellm_memorytable.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_service_key_write_ownership_never_matches_all_unowned_rows(database: MagicMock) -> None:
    identity = MemoryIdentity.from_auth(auth(None))
    store = MemoryStore(database, await resolve_memory_access(database, identity))
    assert store.entry(row(user_id=None)).can_edit
    assert not store.entry(row(user_id=None, owner_key_id="b" * 64)).can_edit
    assert not store.entry(row(user_id="someone")).can_edit
    assert store.access.visible_rows(write=True)["OR"] == [{"owner_key_id": "a" * 64, "user_id": None}]


@pytest.mark.asyncio
async def test_revocation_blocks_existing_store_and_private_continuation(database: MagicMock) -> None:
    configure(database)
    database.db.litellm_teamtable.find_many.return_value = [team(permissions=("/memory/v2/entries",))]
    original = await management.memory_store(auth())
    patch = MemoryContinuation(
        response={"output": [{"role": "assistant", "content": "Team secret"}]},
        permission_revision=original.access.permission_revision,
    )
    database.db.litellm_teamtable.find_many.return_value = [team()]
    with pytest.raises(HTTPException) as exc:
        await original.read("team-record")
    assert exc.value.status_code == 403
    fresh = await management.memory_store(auth())
    database.db.litellm_memorycontinuation.find_first = AsyncMock(
        return_value=SimpleNamespace(payload=patch.model_dump())
    )
    with pytest.raises(HTTPException, match="Memory permissions changed"):
        await MemoryContinuations(fresh).load_response("resp_litellm_memory_private")
    database.db.litellm_memorytable.find_first.assert_not_awaited()


@pytest.mark.asyncio
async def test_disable_stops_tools_but_keeps_dashboard_and_ownership_checks(database: MagicMock) -> None:
    database.db.litellm_memorytable.find_first.return_value = row()
    store = await management.memory_store(auth())
    assert (await store.read("entry", require_active=False)).content == _CAPTURE.content
    with pytest.raises(HTTPException) as exc:
        await store.read("entry")
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException):
        await store.capture(_CAPTURE)
    viewer = await management.memory_store(auth(role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
    with pytest.raises(HTTPException):
        await viewer.delete("entry")
    database.db.litellm_memorytable.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_names_and_team_attribution_are_loaded_for_returned_page(database: MagicMock) -> None:
    database.db.litellm_memorytable.find_many.return_value = [row()]
    database.db.litellm_teamtable.find_many.return_value = [team()]
    database.db.litellm_usertable.find_many.return_value = [
        SimpleNamespace(user_id="owner", user_alias="Alex", user_email="alex@example.test")
    ]
    entries = await management.list_entries(
        query="",
        limit=20,
        offset=0,
        before_updated_at=None,
        before_memory_id=None,
        team_id=None,
        user_id=None,
        auth=auth(),
    )
    assert entries[0].actor_name == "Alex" and entries[0].team_name == "Engineering"
    assert entries[0].actor == "owner" and entries[0].user_id == "owner"


@pytest.mark.asyncio
async def test_capture_records_authenticated_contributor_and_tenant(database: MagicMock) -> None:
    configure(database)
    await management.capture_entry(_CAPTURE, auth("admin", LitellmUserRoles.PROXY_ADMIN))
    data = database.db.litellm_memorytable.create.call_args.kwargs["data"]
    assert data["user_id"] == "admin" and data["created_by"] == "admin"
    assert data["organization_id"] == "org" and data["team_id"] == "team"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential", "secret"),
    (
        ("AKIA" + "A" * 16, "A" * 16),
        ("aws_secret_access_key=" + "B" * 40, "B" * 40),
        ("AIza" + "C" * 35, "C" * 35),
        ("postgres://test-user:memory-test-password@db.example.test/app", "memory-test-password"),
        ("Authorization: Basic " + "D" * 24, "D" * 24),
        ("Authorization: Bearer " + "E" * 24, "E" * 24),
        ("ghp_" + "F" * 24, "F" * 24),
        ("github_pat_" + "G" * 24, "G" * 24),
    ),
)
async def test_capture_redacts_credentials_before_persisting_content_and_metadata(
    database: MagicMock, credential: str, secret: str
) -> None:
    configure(database)
    text = "Deployment uses staging port 8123; credential: " + credential
    fields = ("title", "content", "evidence", "when_to_use", "scope", "source")
    await management.capture_entry(_CAPTURE.model_copy(update={field: text for field in fields}), auth())
    data = database.db.litellm_memorytable.create.call_args.kwargs["data"]
    stored = {"content": data["value"], **json.loads(data["metadata"])}
    for field in fields:
        assert secret not in stored[field]
        assert "REDACTED" in stored[field]
        assert "staging port 8123" in stored[field]


@pytest.mark.asyncio
async def test_search_finds_an_old_record_beyond_the_first_thousand(database: MagicMock) -> None:
    configure(database)
    newer = [
        row(
            memory_id=f"new-{index:04}",
            value="Unrelated",
            metadata="{}",
            updated_at=_NOW + timedelta(seconds=2000 - index),
        )
        for index in range(1152)
    ]
    old = row(memory_id="old", value="The rare quokka uses port 9187")
    table = database.db.litellm_memorytable
    table.find_many.side_effect = [newer[start : start + 128] for start in range(0, len(newer), 128)] + [[old]]
    store = await management.memory_store(auth())
    matches, total = await store.recall(MemoryRecallRequest(query="quokka", limit=1))
    assert total == 1 and matches[0][0].memory_id == "old"
    assert table.find_many.await_count == 10


@pytest.mark.asyncio
async def test_browse_and_search_recheck_permissions_after_fetch(database: MagicMock) -> None:
    configure(database)
    table = database.db.litellm_memorytable
    store = await management.memory_store(auth())

    async def revoke(**kwargs: object) -> list[LiteLLM_MemoryTable]:
        configure(database, enabled=False)
        return [row()]

    table.find_many.side_effect = revoke
    for operation in (lambda: store.recall(MemoryRecallRequest(query="")), lambda: store.search(MemorySearch())):
        configure(database)
        with pytest.raises(HTTPException) as exc:
            await operation()
        assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "params",
    [
        {"query": ",".join(f"term{i}" for i in range(17))},
        {"before_memory_id": "only-half"},
        {"before_updated_at": "2026-09-12T00:00:00Z"},
        {"before_updated_at": "2026-09-12T00:00:00", "before_memory_id": "entry"},
        {"offset": "10001"},
    ],
)
def test_invalid_queries_are_rejected_before_database_work(database: MagicMock, params: dict[str, str]) -> None:
    app = FastAPI()
    app.include_router(management.router)
    app.dependency_overrides[user_api_key_auth] = auth
    with TestClient(app) as client:
        response = client.get("/memory/v2/entries", params=params)
    assert response.status_code == 422
    database.db.litellm_memorytable.find_many.assert_not_awaited()

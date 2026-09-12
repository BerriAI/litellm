"""Exercise real policy administration and self-service logic at the database edge."""

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UI_TEAM_ID, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.memory import management
from litellm.proxy.memory.policy import MemoryIdentity, memory_digest
from litellm.types.memory_v2 import MemoryCapture, MemoryPolicy, MemoryPolicyInput, MemoryPreference


@pytest.fixture
def database() -> Iterator[MagicMock]:
    client = MagicMock()
    for name in (
        "litellm_memorypolicy",
        "litellm_memorypreference",
        "litellm_memorytable",
        "litellm_teamtable",
        "litellm_projecttable",
        "litellm_organizationtable",
        "litellm_organizationmembership",
        "litellm_verificationtoken",
        "litellm_usertable",
    ):
        table = getattr(client.db, name)
        table.find_unique = AsyncMock(return_value=None)
        table.find_first = AsyncMock(return_value=None)
        table.find_many = AsyncMock(return_value=[])
        table.upsert = AsyncMock()
        table.delete = AsyncMock()
        table.delete_many = AsyncMock(return_value=0)
        table.create = AsyncMock()
        table.count = AsyncMock(return_value=0)
    client.db.tx.return_value.__aenter__.return_value = client.db
    client.db.execute_raw = AsyncMock()
    client.db.litellm_teamtable.find_unique.return_value = {
        "team_id": "team",
        "organization_id": None,
        "members_with_roles": [{"user_id": "team-admin", "role": "admin"}],
    }
    client.db.litellm_projecttable.find_unique.return_value = {"project_id": "project", "team_id": "team"}
    client.db.litellm_verificationtoken.find_unique.return_value = {
        "token": "a" * 64,
        "user_id": "owner",
        "team_id": "team",
        "org_id": "explicit-org",
    }
    client.db.litellm_usertable.find_unique.return_value = {"user_id": "owner"}
    client.db.litellm_organizationtable.find_unique.return_value = {
        "organization_id": "org",
        "budget_id": "budget",
        "created_by": "admin",
        "updated_by": "admin",
    }
    with patch.multiple(  # test-quality-ok: Replace only the external database and cache; exercise actual authorization and repositories.
        "litellm.proxy.proxy_server", prisma_client=client, user_api_key_cache=MagicMock(async_delete_cache=AsyncMock())
    ):
        yield client


def auth(user: str = "owner", role: LitellmUserRoles = LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token="a" * 64, user_id=user, user_role=role, team_id="team", org_id="explicit-org")


def policy(**changes: object) -> MemoryPolicy:
    return MemoryPolicy.model_validate(
        {
            "policy_id": memory_digest("gateway", "*"),
            "target_type": "gateway",
            "target_id": "*",
            "activation": "automatic",
            "scope": "key",
            "updated_by": "admin",
            "updated_at": datetime(2026, 9, 12, tzinfo=timezone.utc),
            **changes,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target,target_id",
    [
        ("gateway", "*"),
        ("organization", "org"),
        ("team", "team"),
        ("project", "project"),
        ("key", "a" * 64),
        ("user", "owner"),
    ],
)
async def test_proxy_admin_can_configure_each_existing_target(database: MagicMock, target: str, target_id: str) -> None:
    request = MemoryPolicyInput.model_validate(
        {"target_type": target, "target_id": target_id, "activation": "opt_in", "scope": "key"}
    )
    database.db.litellm_memorypolicy.upsert.return_value = policy(**request.model_dump())
    result = await management.set_policy(request, auth("admin", LitellmUserRoles.PROXY_ADMIN))
    assert result.activation == "opt_in"
    written = database.db.litellm_memorypolicy.upsert.call_args.kwargs
    assert written["where"]["policy_id"] == memory_digest(target, target_id)
    assert written["data"]["create"]["updated_by"] == "admin"
    assert written["data"]["update"]["target_id"] == target_id


@pytest.mark.asyncio
@pytest.mark.parametrize("target,target_id", [("team", "team"), ("project", "project"), ("key", "a" * 64)])
async def test_team_admin_can_configure_owned_targets_but_cannot_broaden_to_user_scope(
    database: MagicMock, target: str, target_id: str
) -> None:
    request = MemoryPolicyInput.model_validate(
        {"target_type": target, "target_id": target_id, "activation": "automatic", "scope": "team"}
    )
    database.db.litellm_memorypolicy.upsert.return_value = policy(**request.model_dump())
    assert (await management.set_policy(request, auth("team-admin"))).scope == "team"
    with pytest.raises(HTTPException) as denied:
        await management.set_policy(request.model_copy(update={"scope": "user"}), auth("team-admin"))
    assert denied.value.status_code == 403
    database.db.litellm_memorypolicy.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_org_membership_is_checked_for_the_selected_organization(database: MagicMock) -> None:
    request = MemoryPolicyInput(
        target_type="organization", target_id="org", activation="automatic", scope="organization"
    )
    database.db.litellm_organizationmembership.find_first.return_value = SimpleNamespace(user_role="org_admin")
    database.db.litellm_memorypolicy.upsert.return_value = policy(**request.model_dump())
    assert (await management.set_policy(request, auth("org-admin"))).scope == "organization"
    assert database.db.litellm_organizationmembership.find_first.call_args.kwargs["where"] == {
        "organization_id": "org",
        "user_id": "org-admin",
        "user_role": "org_admin",
    }
    database.db.litellm_organizationmembership.find_first.return_value = None
    with pytest.raises(HTTPException) as denied:
        await management.set_policy(request, auth("org-admin"))
    assert denied.value.status_code == 403
    database.db.litellm_memorypolicy.upsert.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
async def test_non_administrator_cannot_change_gateway_policy(database: MagicMock, role: LitellmUserRoles) -> None:
    with pytest.raises(HTTPException) as denied:
        await management.set_policy(
            MemoryPolicyInput(target_type="gateway", target_id="*", activation="automatic"), auth(role=role)
        )
    assert denied.value.status_code == 403
    database.db.litellm_memorypolicy.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_listing_requires_owned_target_and_preserves_pagination(database: MagicMock) -> None:
    database.db.litellm_memorypolicy.find_many.return_value = [policy(target_type="team", target_id="team")]
    assert len(await management.list_policies("team", "team", 100, auth("team-admin"))) == 1
    assert database.db.litellm_memorypolicy.find_many.call_args.kwargs == {
        "where": {"target_type": "team", "target_id": "team"},
        "take": 100,
        "skip": 100,
        "order": {"policy_id": "asc"},
    }
    for caller, target, expected in [(auth(), None, 403), (auth("admin", LitellmUserRoles.PROXY_ADMIN), "team", 400)]:
        with pytest.raises(HTTPException) as denied:
            await management.list_policies(target, None, 0, caller)
        assert denied.value.status_code == expected
    assert (
        len(await management.list_policies(None, None, 0, auth("admin", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))) == 1
    )


@pytest.mark.asyncio
async def test_policy_delete_rechecks_administrator_and_reports_missing(database: MagicMock) -> None:
    table = database.db.litellm_memorypolicy
    table.find_unique.return_value = policy()
    with pytest.raises(HTTPException) as denied:
        await management.delete_policy("policy", auth())
    assert denied.value.status_code == 403
    table.delete.assert_not_awaited()
    assert (await management.delete_policy("policy", auth("admin", LitellmUserRoles.PROXY_ADMIN))).status_code == 204
    table.find_unique.return_value = None
    with pytest.raises(HTTPException) as missing:
        await management.delete_policy("policy", auth("admin", LitellmUserRoles.PROXY_ADMIN))
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_preference_updates_are_bound_to_authenticated_subject(database: MagicMock) -> None:
    assert not (await management.get_preference(auth())).enabled
    table = database.db.litellm_memorypreference
    assert (await management.set_preference(MemoryPreference(enabled=True), auth())).enabled
    assert table.upsert.call_args.kwargs["where"] == {"subject": memory_digest("user", "owner")}
    table.find_unique.return_value = SimpleNamespace(enabled=True)
    assert (await management.get_preference(auth())).enabled
    assert not (await management.set_preference(MemoryPreference(enabled=False), auth())).enabled
    assert table.delete_many.call_args.kwargs["where"] == {"subject": memory_digest("user", "owner")}
    with pytest.raises(HTTPException) as denied:
        await management.set_preference(
            MemoryPreference(enabled=False), auth(role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
        )
    assert denied.value.status_code == 403


@pytest.mark.asyncio
async def test_dashboard_session_can_select_own_key_but_not_another_users_key(database: MagicMock) -> None:
    database.db.litellm_memorypolicy.find_many.return_value = [policy()]
    session = auth().model_copy(update={"token": "session", "team_id": UI_TEAM_ID})
    own = await management.access_for_key(session, "a" * 64)
    assert own.active and own.identity.organization_id == "explicit-org"
    assert own.namespace == MemoryIdentity.from_auth(auth()).namespace("key")
    assert (await management.get_status(None, auth())).active
    database.db.litellm_verificationtoken.find_unique.return_value["user_id"] = "someone-else"
    with pytest.raises(HTTPException) as denied:
        await management.access_for_key(session, "a" * 64)
    assert denied.value.status_code == 403
    database.db.litellm_verificationtoken.find_unique.return_value = None
    with pytest.raises(HTTPException) as missing:
        await management.access_for_key(auth("admin", LitellmUserRoles.PROXY_ADMIN), "a" * 64)
    assert missing.value.status_code == 403


@pytest.mark.asyncio
async def test_entry_endpoints_apply_namespace_and_delete_after_disable(database: MagicMock) -> None:
    database.db.litellm_memorypolicy.find_many.return_value = [policy()]
    namespace = MemoryIdentity.from_auth(auth()).namespace("key")
    table = database.db.litellm_memorytable
    assert not await management.list_entries("demo", 2, 4, None, auth())
    assert table.find_many.call_args.kwargs["where"]["namespace"] == namespace
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)
    table.create.return_value = SimpleNamespace(
        memory_id="entry",
        key=f"memory-v2:{namespace}:demo",
        namespace=namespace,
        value="Use port 8347",
        metadata={"title": "Demo", "evidence": "User said so"},
        updated_at=now,
        created_at=now,
        created_by="owner",
    )
    saved = await management.capture_entry(
        MemoryCapture(key="demo", title="Demo", content="Use port 8347", evidence="User said so"), None, auth()
    )
    assert saved.content == "Use port 8347" and saved.key == "demo"
    database.db.litellm_memorypolicy.find_many.return_value = [policy(activation="disabled")]
    table.delete_many.return_value = 1
    assert (await management.delete_entry("entry", None, auth())).status_code == 204
    assert table.delete_many.call_args.kwargs["where"] == {"namespace": namespace, "memory_id": "entry"}
    table.delete_many.return_value = 0
    with pytest.raises(HTTPException) as missing:
        await management.delete_entry("entry", None, auth())
    assert missing.value.status_code == 404

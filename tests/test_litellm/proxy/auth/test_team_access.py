from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.team_access import (
    is_org_admin_for_team,
    is_team_admin,
    require_key_access,
    require_team_access,
    resolve_team_access,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.proxy.auth.auth_checks import UserNotFoundError

NOW: Final = datetime.now(timezone.utc)
ADMIN: Final = Member(user_id="admin", role="admin")
MEMBER: Final = Member(user_id="member", role="user")


def team(*members: Member, organization_id: str | None = None) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id="team-1", organization_id=organization_id, members_with_roles=list(members))


def caller(user_id: str | None, role: LitellmUserRoles = LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, api_key="sk-x", user_role=role)


def org_member(user_id: str, organization_id: str, user_role: str) -> LiteLLM_UserTable:
    membership: Final = LiteLLM_OrganizationMembershipTable(
        user_id=user_id, organization_id=organization_id, user_role=user_role, created_at=NOW, updated_at=NOW
    )
    return LiteLLM_UserTable(user_id=user_id, organization_memberships=[membership])


@contextmanager
def user_lookup(user: LiteLLM_UserTable | None) -> Iterator[AsyncMock]:
    """Stand in for the proxy-wide singletons the org-admin check imports lazily, and for the user read it makes."""
    get_user: Final = AsyncMock(return_value=user)
    with (
        patch("litellm.proxy.auth.auth_checks.get_user_object", get_user),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock(), create=True),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock(), create=True),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock(), create=True),
    ):
        yield get_user


def prisma_with_key(row: SimpleNamespace | None) -> SimpleNamespace:
    table: Final = SimpleNamespace(find_unique=AsyncMock(return_value=row))
    return SimpleNamespace(db=SimpleNamespace(litellm_verificationtoken=table))


@pytest.mark.parametrize(
    ("members", "user_id", "expected"),
    [
        ((ADMIN,), "admin", True),
        ((MEMBER,), "member", False),
        ((MEMBER, ADMIN), "admin", True),
        ((), "admin", False),
        ((ADMIN,), "someone-else", False),
    ],
)
def test_is_team_admin(members: tuple[Member, ...], user_id: str, expected: bool) -> None:
    assert is_team_admin(caller(user_id), team(*members)) is expected


async def test_is_org_admin_for_team_needs_an_org_and_a_user() -> None:
    with user_lookup(org_member("u1", "org-1", "org_admin")) as get_user:
        assert await is_org_admin_for_team(caller("u1"), team(organization_id=None)) is False
        assert await is_org_admin_for_team(caller(None), team(organization_id="org-1")) is False
    get_user.assert_not_awaited()


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        (org_member("u1", "org-1", "org_admin"), True),
        (org_member("u1", "org-2", "org_admin"), False),
        (org_member("u1", "org-1", "user"), False),
        (None, False),
    ],
)
async def test_is_org_admin_for_team_reads_the_callers_role_in_the_teams_org(
    user: LiteLLM_UserTable | None, expected: bool
) -> None:
    with user_lookup(user) as get_user:
        assert await is_org_admin_for_team(caller("u1"), team(organization_id="org-1")) is expected
    assert get_user.await_args is not None
    assert get_user.await_args.kwargs["user_id"] == "u1"
    assert get_user.await_args.kwargs["user_id_upsert"] is False


async def test_resolve_team_access_proxy_admin_without_any_lookup() -> None:
    with user_lookup(None) as get_user:
        role = await resolve_team_access(team(organization_id="org-1"), caller("root", LitellmUserRoles.PROXY_ADMIN))
    assert role == "proxy_admin"
    get_user.assert_not_awaited()


async def test_resolve_team_access_org_admin_outranks_team_admin() -> None:
    with user_lookup(org_member("admin", "org-1", "org_admin")):
        assert await resolve_team_access(team(ADMIN, organization_id="org-1"), caller("admin")) == "org_admin"


async def test_resolve_team_access_team_admin_and_member() -> None:
    assert await resolve_team_access(team(ADMIN, MEMBER), caller("admin")) == "team_admin"
    assert await resolve_team_access(team(ADMIN, MEMBER), caller("member")) is None


async def test_require_team_access_rejects_outsiders_with_403() -> None:
    await require_team_access(team(ADMIN), caller("admin"))
    with pytest.raises(HTTPException) as denied:
        await require_team_access(team(ADMIN), caller("outsider"))
    assert denied.value.status_code == 403
    assert denied.value.detail == "You do not have access to this team"


async def test_require_key_access_proxy_admin_skips_the_key_lookup() -> None:
    await require_key_access(
        caller("root", LitellmUserRoles.PROXY_ADMIN), "hash", None, UserApiKeyCache(), "/key/block"
    )


async def test_require_key_access_unknown_key_is_404() -> None:
    with pytest.raises(HTTPException) as missing:
        await require_key_access(caller("u1"), "hash", prisma_with_key(None), UserApiKeyCache(), "/key/block")
    assert missing.value.status_code == 404
    assert missing.value.detail == {"error": "Key not found: hash"}


async def test_require_key_access_teamless_key_is_403_for_non_admins() -> None:
    prisma: Final = prisma_with_key(SimpleNamespace(team_id=None))
    with pytest.raises(HTTPException) as denied:
        await require_key_access(caller("u1"), "hash", prisma, UserApiKeyCache(), "/key/block")
    assert denied.value.status_code == 403
    assert isinstance(denied.value.detail, dict)
    assert "/key/block" in denied.value.detail["error"]
    assert "user_id=u1" in denied.value.detail["error"]


async def test_require_key_access_admits_only_admins_of_the_keys_team() -> None:
    team_row: Final = LiteLLM_TeamTableCachedObj(team_id="team-1", members_with_roles=[ADMIN, MEMBER])
    cache: Final = UserApiKeyCache()
    with patch("litellm.proxy.auth.auth_checks.get_team_object", AsyncMock(return_value=team_row)) as get_team:
        await require_key_access(
            caller("admin"), "hash", prisma_with_key(SimpleNamespace(team_id="team-1")), cache, "/key/block"
        )
        with pytest.raises(HTTPException) as denied:
            await require_key_access(
                caller("member"), "hash", prisma_with_key(SimpleNamespace(team_id="team-1")), cache, "/key/block"
            )
    assert denied.value.status_code == 403
    assert get_team.await_args is not None
    assert get_team.await_args.kwargs == {
        "team_id": "team-1",
        "prisma_client": get_team.await_args.kwargs["prisma_client"],
        "user_api_key_cache": cache,
        "check_db_only": True,
    }


@pytest.mark.parametrize(
    ("user_id", "user_lookup_outcome"),
    [
        pytest.param("admin", UserNotFoundError(user_id="admin"), id="team-admin-without-user-row"),
        pytest.param(
            "boss", org_member("boss", "org-1", LitellmUserRoles.ORG_ADMIN.value), id="org-admin-off-the-roster"
        ),
    ],
)
async def test_require_key_access_admits_team_admins_and_org_admins_of_an_org_owned_team(
    user_id: str, user_lookup_outcome: LiteLLM_UserTable | UserNotFoundError
) -> None:
    team_row: Final = LiteLLM_TeamTableCachedObj(
        team_id="team-1", organization_id="org-1", members_with_roles=[ADMIN, MEMBER]
    )
    prisma: Final = prisma_with_key(SimpleNamespace(team_id="team-1"))
    with (
        user_lookup(None) as get_user,
        patch("litellm.proxy.auth.auth_checks.get_team_object", AsyncMock(return_value=team_row)),
    ):
        get_user.side_effect = [user_lookup_outcome]
        await require_key_access(caller(user_id), "hash", prisma, UserApiKeyCache(), "/key/block")

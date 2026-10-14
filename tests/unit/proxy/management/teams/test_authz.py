from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.management.teams.authz import (
    TEAM_ADMIN_ONLY,
    TEAM_OR_ORG_ADMIN,
    TeamAccess,
    TeamRole,
    is_team_admin,
    team_access_denied,
)

ADMIN: Final = Member(user_id="admin", role="admin")
MEMBER: Final = Member(user_id="member", role="user")


@dataclass(frozen=True, slots=True)
class OrgAdmins:
    of: frozenset[tuple[str, str]]

    async def is_org_admin(self, user_id: str, organization_id: str) -> bool:
        return (user_id, organization_id) in self.of


class NoOrgLookup:
    async def is_org_admin(self, user_id: str, organization_id: str) -> bool:
        raise AssertionError(f"org lookup ran for {user_id} in {organization_id}")


def team(*members: Member, organization_id: str | None = "org-1") -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id="team-1", organization_id=organization_id, members_with_roles=list(members))


def caller(user_id: str | None, role: LitellmUserRoles = LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, api_key="sk-x", user_role=role)


BOSS_OF_ORG_1: Final = OrgAdmins(of=frozenset({("boss", "org-1")}))


@pytest.mark.parametrize(
    ("who", "allow", "expected"),
    [
        (caller("root", LitellmUserRoles.PROXY_ADMIN), TEAM_ADMIN_ONLY, True),
        (caller("root", LitellmUserRoles.PROXY_ADMIN), TEAM_OR_ORG_ADMIN, True),
        (caller("root", LitellmUserRoles.PROXY_ADMIN), frozenset({"team_admin"}), False),
        (caller("admin"), TEAM_ADMIN_ONLY, True),
        (caller("admin"), frozenset({"proxy_admin"}), False),
        (caller("member"), TEAM_ADMIN_ONLY, False),
    ],
)
async def test_allows_answers_proxy_and_team_admins_without_an_org_lookup(
    who: UserAPIKeyAuth, allow: frozenset[TeamRole], expected: bool
) -> None:
    assert await TeamAccess(org_roles=NoOrgLookup()).allows(who, team(ADMIN, MEMBER), allow) is expected


async def test_allows_checks_the_roster_before_the_org_lookup() -> None:
    assert await TeamAccess(org_roles=NoOrgLookup()).allows(caller("admin"), team(ADMIN), TEAM_OR_ORG_ADMIN)


@pytest.mark.parametrize(
    ("who", "on_team", "allow", "expected"),
    [
        (caller("boss"), team(ADMIN, organization_id="org-1"), TEAM_OR_ORG_ADMIN, True),
        (caller("boss"), team(ADMIN, organization_id="org-1"), TEAM_ADMIN_ONLY, False),
        (caller("boss"), team(ADMIN, organization_id="org-2"), TEAM_OR_ORG_ADMIN, False),
        (caller("member"), team(MEMBER, organization_id="org-1"), TEAM_OR_ORG_ADMIN, False),
    ],
)
async def test_allows_admits_org_admins_only_of_the_teams_org_and_only_when_asked(
    who: UserAPIKeyAuth, on_team: LiteLLM_TeamTable, allow: frozenset[TeamRole], expected: bool
) -> None:
    assert await TeamAccess(org_roles=BOSS_OF_ORG_1).allows(who, on_team, allow) is expected


@pytest.mark.parametrize(
    ("who", "on_team"),
    [
        pytest.param(caller(None), team(organization_id="org-1"), id="caller-without-user-id"),
        pytest.param(caller(""), team(organization_id="org-1"), id="caller-with-empty-user-id"),
        pytest.param(caller("boss"), team(organization_id=None), id="team-without-org"),
        pytest.param(caller("boss"), team(organization_id=""), id="team-with-empty-org"),
    ],
)
async def test_allows_skips_the_org_lookup_without_a_user_and_an_org(
    who: UserAPIKeyAuth, on_team: LiteLLM_TeamTable
) -> None:
    assert await TeamAccess(org_roles=NoOrgLookup()).allows(who, on_team, TEAM_OR_ORG_ADMIN) is False


@pytest.mark.parametrize(
    ("who", "on_team", "org_roles", "expected"),
    [
        (caller("root", LitellmUserRoles.PROXY_ADMIN), team(), NoOrgLookup(), "proxy_admin"),
        (caller("boss"), team(Member(user_id="boss", role="admin")), BOSS_OF_ORG_1, "org_admin"),
        (caller("boss"), team(), BOSS_OF_ORG_1, "org_admin"),
        (caller("admin"), team(ADMIN), BOSS_OF_ORG_1, "team_admin"),
        (caller("member"), team(ADMIN, MEMBER), BOSS_OF_ORG_1, None),
    ],
)
async def test_strongest_role_ranks_org_admin_above_team_admin(
    who: UserAPIKeyAuth,
    on_team: LiteLLM_TeamTable,
    org_roles: OrgAdmins | NoOrgLookup,
    expected: TeamRole | None,
) -> None:
    assert await TeamAccess(org_roles=org_roles).strongest_role(who, on_team) == expected


@pytest.mark.parametrize(
    ("members", "user_id", "expected"),
    [
        ((ADMIN,), "admin", True),
        ((MEMBER,), "member", False),
        ((MEMBER, ADMIN), "admin", True),
        ((), "admin", False),
        ((ADMIN,), "someone-else", False),
        ((Member(user_id=None, user_email="a@b.c", role="admin"),), None, False),
    ],
)
def test_is_team_admin_reads_the_roster(members: tuple[Member, ...], user_id: str | None, expected: bool) -> None:
    assert is_team_admin(caller(user_id), team(*members)) is expected


def test_team_access_denied_is_the_403_management_routes_have_always_raised() -> None:
    with pytest.raises(HTTPException) as denied:
        team_access_denied()
    assert denied.value.status_code == 403
    assert denied.value.detail == "You do not have access to this team"


def team_key(team_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=None, team_id=team_id, api_key="sk-team", user_role=LitellmUserRoles.INTERNAL_USER)


@pytest.mark.parametrize(
    ("who", "expected"),
    [
        pytest.param(caller("root", LitellmUserRoles.PROXY_ADMIN), True, id="proxy-admin"),
        pytest.param(caller("viewer", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), True, id="admin-viewer"),
        pytest.param(team_key("team-1"), True, id="key-issued-to-the-team"),
        pytest.param(caller("admin"), True, id="team-admin"),
        pytest.param(caller("member"), True, id="plain-member"),
        pytest.param(team_key("team-2"), False, id="key-issued-to-another-team"),
        pytest.param(caller(None), False, id="no-user-no-team"),
    ],
)
async def test_reads_roster_admits_team_info_readers_without_an_org_lookup(who: UserAPIKeyAuth, expected: bool) -> None:
    assert await TeamAccess(org_roles=NoOrgLookup()).reads_roster(who, team(ADMIN, MEMBER)) is expected


async def test_reads_roster_never_matches_a_userless_caller_to_an_email_only_member() -> None:
    email_only: Final = Member(user_id=None, user_email="invitee@example.com", role="user")
    assert await TeamAccess(org_roles=NoOrgLookup()).reads_roster(caller(None), team(email_only)) is False


@pytest.mark.parametrize(
    ("who", "on_team", "expected"),
    [
        pytest.param(caller("boss"), team(ADMIN, organization_id="org-1"), True, id="org-admin-of-the-team"),
        pytest.param(caller("boss"), team(ADMIN, organization_id="org-2"), False, id="org-admin-elsewhere"),
        pytest.param(caller("stranger"), team(ADMIN, organization_id="org-1"), False, id="off-the-roster"),
    ],
)
async def test_reads_roster_falls_back_to_the_teams_org_admins(
    who: UserAPIKeyAuth, on_team: LiteLLM_TeamTable, expected: bool
) -> None:
    assert await TeamAccess(org_roles=BOSS_OF_ORG_1).reads_roster(who, on_team) is expected


def test_the_old_access_module_still_serves_the_published_enterprise_wheel() -> None:
    from litellm.proxy.management.teams import access, authz

    assert access.is_team_admin is authz.is_team_admin
    assert set(access.__all__) <= set(dir(authz))

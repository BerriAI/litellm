from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.auth.roles import Role
from litellm.proxy.management.teams.authz import is_team_admin, roles_on, team_access_denied

ADMIN: Final = Member(user_id="admin", role="admin")
MEMBER: Final = Member(user_id="member", role="user")
BOSS_ON_ROSTER: Final = Member(user_id="boss", role="admin")


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
    ("who", "on_team", "expected"),
    [
        pytest.param(
            caller("root", LitellmUserRoles.PROXY_ADMIN),
            team(ADMIN),
            frozenset({Role.PLATFORM_ADMIN}),
            id="proxy-admin",
        ),
        pytest.param(caller("admin"), team(ADMIN, MEMBER), frozenset({Role.TEAM_ADMIN}), id="team-admin"),
        pytest.param(caller("boss"), team(ADMIN), frozenset({Role.ORG_ADMIN}), id="org-admin-off-roster"),
        pytest.param(
            caller("boss"), team(BOSS_ON_ROSTER), frozenset({Role.ORG_ADMIN, Role.TEAM_ADMIN}), id="org-and-team-admin"
        ),
        pytest.param(caller("member"), team(ADMIN, MEMBER), frozenset(), id="plain-member"),
        pytest.param(caller("boss"), team(ADMIN, organization_id="org-2"), frozenset(), id="org-admin-of-another-org"),
        pytest.param(caller("stranger"), team(ADMIN), frozenset(), id="stranger"),
    ],
)
async def test_roles_on_is_the_union_of_every_admin_role_held(
    who: UserAPIKeyAuth, on_team: LiteLLM_TeamTable, expected: frozenset[Role]
) -> None:
    assert await roles_on(on_team, who, BOSS_OF_ORG_1) == expected


@pytest.mark.parametrize(
    ("who", "on_team"),
    [
        pytest.param(caller(None), team(organization_id="org-1"), id="caller-without-user-id"),
        pytest.param(caller(""), team(organization_id="org-1"), id="caller-with-empty-user-id"),
        pytest.param(caller("boss"), team(organization_id=None), id="team-without-org"),
        pytest.param(caller("boss"), team(organization_id=""), id="team-with-empty-org"),
    ],
)
async def test_roles_on_skips_the_org_lookup_without_a_user_and_an_org(
    who: UserAPIKeyAuth, on_team: LiteLLM_TeamTable
) -> None:
    assert await roles_on(on_team, who, NoOrgLookup()) == frozenset()


async def test_roles_on_answers_a_proxy_admin_without_the_roster_or_org_lookup() -> None:
    who: Final = caller("boss", LitellmUserRoles.PROXY_ADMIN)
    assert await roles_on(team(BOSS_ON_ROSTER), who, NoOrgLookup()) == frozenset({Role.PLATFORM_ADMIN})


async def test_roles_on_asks_the_org_lookup_for_the_caller_on_the_teams_org() -> None:
    asked: Final[list[tuple[str, str]]] = []

    class Recording:
        async def is_org_admin(self, user_id: str, organization_id: str) -> bool:
            asked.append((user_id, organization_id))
            return False

    await roles_on(team(organization_id="org-9"), caller("someone"), Recording())
    assert asked == [("someone", "org-9")]


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

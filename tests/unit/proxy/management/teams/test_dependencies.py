from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.list_api.common import ManagementProblem
from litellm.proxy.management.teams.authz import TeamAccess
from litellm.proxy.management.teams.dependencies import get_readable_team

TEAM: Final = LiteLLM_TeamTable(
    team_id="team-1", organization_id=None, members_with_roles=[Member(user_id="member", role="user")]
)


@dataclass(frozen=True, slots=True)
class OneTeam:
    team: LiteLLM_TeamTable

    async def find_by_id(self, team_id: str) -> LiteLLM_TeamTable | None:
        return self.team if team_id == self.team.team_id else None


class NoOrgAdmins:
    async def is_org_admin(self, user_id: str, organization_id: str) -> bool:
        return False


def caller(user_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, api_key="sk-x", user_role=LitellmUserRoles.INTERNAL_USER)


async def readable_team(team_id: str, who: UserAPIKeyAuth) -> LiteLLM_TeamTable:
    return await get_readable_team(team_id, who, OneTeam(TEAM), TeamAccess(org_roles=NoOrgAdmins()))


async def test_get_readable_team_hands_a_member_their_team() -> None:
    assert await readable_team("team-1", caller("member")) == TEAM


@pytest.mark.parametrize(
    ("team_id", "who", "status"),
    [
        pytest.param("team-1", caller("stranger"), 403, id="stranger-is-refused"),
        pytest.param("team-2", caller("member"), 404, id="unknown-team"),
    ],
)
async def test_get_readable_team_stops_the_request_before_the_handler(
    team_id: str, who: UserAPIKeyAuth, status: int
) -> None:
    with pytest.raises(ManagementProblem) as refused:
        await readable_team(team_id, who)

    assert refused.value.problem.status == status

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem
from litellm.proxy.list_api.list_framework import QueryPlan
from litellm.proxy.management.teams.authz import TeamAccess
from litellm.proxy.management.teams.dependencies import get_readable_team, get_team_members_plan

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


def plan_client(as_caller: UserAPIKeyAuth) -> TestClient:
    app: Final = FastAPI()
    app.dependency_overrides[user_api_key_auth] = lambda: as_caller

    @app.get("/plan")
    def planned(plan: Annotated[QueryPlan, Depends(get_team_members_plan)]) -> tuple[int, int]:
        return plan.skip, plan.take

    return TestClient(app)


def test_get_team_members_plan_turns_paging_params_into_a_window_of_rows() -> None:
    planned: Final = plan_client(caller("member")).get("/plan", params={"page": "3", "page_size": "10"})

    assert planned.json() == [20, 10]


def test_get_team_members_plan_refuses_a_field_members_cannot_be_sorted_by() -> None:
    with pytest.raises(ManagementProblem) as refused:
        plan_client(caller("member")).get("/plan", params={"sort": "budget_id"})

    assert refused.value.problem.status == 400


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"page": "abc"}, id="non-integer-page"),
        pytest.param({"filter[role]": "owner"}, id="unknown-role"),
        pytest.param({"team_id": "other"}, id="undeclared-param"),
    ],
)
def test_get_team_members_plan_is_not_reached_by_params_the_query_model_rejects(params: dict[str, str]) -> None:
    assert plan_client(caller("member")).get("/plan", params=params).is_client_error


def test_get_team_members_plan_refuses_a_param_sent_twice() -> None:
    with pytest.raises(ManagementProblem) as refused:
        plan_client(caller("member")).get("/plan?page=1&page=2")

    assert refused.value.problem.status == 400

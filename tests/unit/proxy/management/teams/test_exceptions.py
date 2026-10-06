from __future__ import annotations

import pytest

from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE
from litellm.proxy.management.teams.exceptions import roster_problem
from litellm.proxy.management.teams.service import RosterHidden, TeamNotFound


@pytest.mark.parametrize(
    ("refusal", "status", "problem_type"),
    [
        pytest.param(TeamNotFound(team_id="team-404"), 404, f"{PROBLEM_TYPE_BASE}team-not-found", id="unknown-team"),
        pytest.param(RosterHidden(team_id="team-403"), 403, f"{PROBLEM_TYPE_BASE}forbidden", id="hidden-roster"),
    ],
)
def test_roster_problem_answers_each_refusal_with_its_own_status_and_names_the_team(
    refusal: TeamNotFound | RosterHidden, status: int, problem_type: str
) -> None:
    problem = roster_problem(refusal).problem

    assert (problem.status, problem.type) == (status, problem_type)
    assert refusal.team_id in problem.detail

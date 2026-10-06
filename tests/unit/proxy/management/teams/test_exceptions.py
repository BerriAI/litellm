from __future__ import annotations

from collections.abc import Callable

import pytest

from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management.teams.exceptions import members_not_readable, team_not_found


@pytest.mark.parametrize(
    ("refuse", "status", "problem_type"),
    [
        pytest.param(team_not_found, 404, f"{PROBLEM_TYPE_BASE}team-not-found", id="unknown-team"),
        pytest.param(members_not_readable, 403, f"{PROBLEM_TYPE_BASE}forbidden", id="hidden-members"),
    ],
)
def test_each_refusal_answers_with_its_own_status_and_names_the_team(
    refuse: Callable[[str], ManagementProblem], status: int, problem_type: str
) -> None:
    problem = refuse("team-x").problem

    assert (problem.status, problem.type) == (status, problem_type)
    assert "team-x" in problem.detail

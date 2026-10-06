from typing_extensions import assert_never

from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management.teams.service import RosterHidden, TeamNotFound
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail


def roster_problem(refusal: TeamNotFound | RosterHidden) -> ManagementProblem:
    match refusal:
        case TeamNotFound(team_id=team_id):
            return ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}team-not-found",
                    title="Team not found",
                    status=404,
                    detail=f"Team id={team_id} does not exist in db",
                )
            )
        case RosterHidden(team_id=team_id):
            return ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}forbidden",
                    title="Forbidden",
                    status=403,
                    detail=f"Not allowed to read the members of team={team_id}",
                )
            )
        case _:
            assert_never(refusal)

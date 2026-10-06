from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail


def team_not_found(team_id: str) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}team-not-found",
            title="Team not found",
            status=404,
            detail=f"Team id={team_id} does not exist in db",
        )
    )


def members_not_readable(team_id: str) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}forbidden",
            title="Forbidden",
            status=403,
            detail=f"Not allowed to read the members of team={team_id}",
        )
    )

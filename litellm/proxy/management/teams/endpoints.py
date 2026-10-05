from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.list_api.list_framework import handle_list
from litellm.proxy.management.teams.access import TeamAccess
from litellm.proxy.management.teams.dependencies import get_roster_db, get_team_access, get_teams
from litellm.proxy.management.teams.repository import RawQuery, TeamMemberRows
from litellm.proxy.management.teams.schemas import TeamMemberListItem
from litellm.proxy.management.teams.service import (
    TEAM_MEMBERS_LIST_SPEC,
    ReadableRoster,
    RosterHidden,
    TeamNotFound,
    Teams,
    find_readable_roster,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.types.proxy.management_endpoints.management_v1 import ListResponse, ProblemDetail

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


def _roster_problem(refusal: TeamNotFound | RosterHidden) -> ManagementProblem:
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


@router.get(
    "/teams/{team_id}/members",
    tags=["team management"],
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[TeamMemberListItem],
)
async def list_team_members(
    team_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    teams: Annotated[Teams, Depends(get_teams)],
    roster_db: Annotated[RawQuery, Depends(get_roster_db)],
    team_access: Annotated[TeamAccess, Depends(get_team_access)],
) -> ListResponse[TeamMemberListItem]:
    """
    One page of a team's members, with each member's spend in the team and the limits on their budget row.

    Readable by whoever can read `/team/info` for the team: proxy admins and admin viewers, a key issued to
    the team, any member of the team, and admins of the team's organization. An unknown team is a 404.

    Members come in the order they joined unless `sort` names a comma-separated list of `user_alias`,
    `user_email`, `user_id`, `role`, `spend`, `total_spend`, `max_budget_in_team` or `budget_reset_at`,
    each optionally prefixed with `-` for descending. `q` is a case-insensitive substring match on
    `user_id` or `user_email`, and `filter[role]=admin` or `filter[role][in]=admin,user` filters by role.
    `page_size` defaults to 50 and is capped at 100. `meta.total_count` counts the members matching `q`
    and the filters.

    `budget_source` is `custom` when the member has their own budget row, `team_default` when they follow
    the team's member budget, and `none` when the team has no member budget.

    Example curl:
    ```
    curl --location --globoff 'http://0.0.0.0:4000/management/v1/teams/team-1/members?q=acme&filter[role]=admin&page_size=25' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    try:
        lookup: Final = await find_readable_roster(team_id, user_api_key_dict, teams, team_access)
        match lookup:
            case ReadableRoster(team_id=readable_team_id):
                return await handle_list(
                    spec=TEAM_MEMBERS_LIST_SPEC,
                    executor=TeamMemberRows(db=roster_db, team_id=readable_team_id),
                    request=request,
                    caller=user_api_key_dict,
                )
            case TeamNotFound() | RosterHidden():
                raise _roster_problem(lookup)
            case _:
                assert_never(lookup)
    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management.teams.endpoints.list_team_members(): Exception occured - %s", e
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to list team members.",
            )
        )

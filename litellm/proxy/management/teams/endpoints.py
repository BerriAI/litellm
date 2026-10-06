from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management.teams.dependencies import get_readable_team, get_roster_db
from litellm.proxy.management.teams.repository import RawQuery
from litellm.proxy.management.teams.schemas import TeamMemberListItem
from litellm.proxy.management.teams.service import get_team_members_list
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.types.proxy.management_endpoints.management_v1 import ListResponse, ProblemDetail

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


@router.get(
    "/teams/{team_id}/members",
    tags=["team management"],
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[TeamMemberListItem],
)
async def list_team_members(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    team: Annotated[LiteLLM_TeamTable, Depends(get_readable_team)],
    roster_db: Annotated[RawQuery, Depends(get_roster_db)],
) -> ListResponse[TeamMemberListItem]:
    """
    List a team's members one page at a time, with each member's spend and budget limits in the team.

    Anyone who can read `/team/info` for the team can call this: proxy admins, admin viewers, the team's keys,
    its members and its org admins. Anyone else gets a 403, and a team that does not exist is a 404.

    Query parameters, all optional:
    - `q`: search by `user_id` or `user_email`. Matches any part of the value and ignores case
    - `filter[role]`: `admin` or `user`. Use `filter[role][in]=admin,user` to match several roles
    - `sort`: `user_alias`, `user_email`, `user_id`, `role`, `spend`, `total_spend`, `max_budget_in_team` or
      `budget_reset_at`. Put `-` in front to sort descending. Defaults to the order members were added
    - `page`: the page to return, starting at 1
    - `page_size`: members per page. Defaults to 50, max 100

    `budget_source` says where a member's budget comes from:
    - `custom`: the member has a budget of their own
    - `team_default`: the member follows the team's member budget
    - `none`: the team has no member budget

    ```
    curl --globoff 'http://0.0.0.0:4000/management/v1/teams/team-1/members?q=acme&filter[role]=admin' -H 'Authorization: Bearer sk-1234'
    ```
    """
    try:
        return await get_team_members_list(team.team_id, request, user_api_key_dict, roster_db)
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

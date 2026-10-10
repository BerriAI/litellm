from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request

from litellm.proxy._types import LiteLLM_TeamTable
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.list_framework import QueryPlan, list_response
from litellm.proxy.management.teams.dependencies import get_readable_team, get_roster_db, get_team_members_plan
from litellm.proxy.management.teams.repository import RawQuery
from litellm.proxy.management.teams.schemas import TeamMemberResponse
from litellm.proxy.management.teams.service import get_team_members_list
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.types.proxy.management_endpoints.management_v1 import ListResponse

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


@router.get(
    "/teams/{team_id}/members",
    tags=["team management"],
    dependencies=(Depends(user_api_key_auth),),
    response_model=ListResponse[TeamMemberResponse],
)
async def list_team_members(
    request: Request,
    team: Annotated[LiteLLM_TeamTable, Depends(get_readable_team)],
    plan: Annotated[QueryPlan, Depends(get_team_members_plan)],
    roster_db: Annotated[RawQuery, Depends(get_roster_db)],
) -> ListResponse[TeamMemberResponse]:
    """
    List a team's members one page at a time, with each member's spend and budget limits in the team.

    Anyone who can read `/team/info` for the team can call this: proxy admins, admin viewers, the team's keys,
    its members and its org admins. Anyone else gets a 403, and a team that does not exist is a 404.

    `budget_source` says where a member's budget comes from:
    - `custom`: the member has a budget of their own
    - `team_default`: the member follows the team's member budget
    - `none`: the team has no member budget

    ```
    curl --globoff 'http://0.0.0.0:4000/management/v1/teams/team-1/members?q=acme&filter[role]=admin' -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    """
    page: Final = await get_team_members_list(team.team_id, plan, roster_db)
    return list_response(request, plan, page.members, page.total_count)

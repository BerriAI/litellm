# What is this?
## Helper utils for the management endpoints (keys/users/teams)

from litellm.proxy._types import LiteLLM_TeamTable, Member, UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient
import uuid
from typing import Optional


async def add_new_member(
    new_member: Member,
    max_budget_in_team: Optional[float],
    prisma_client: PrismaClient,
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
):
    """
    Add a new member to a team

    - add team id to user table
    - add team member w/ budget to team member table
    """
    ## ADD TEAM ID, to USER TABLE IF NEW ##
    if new_member.user_id is not None:
        await prisma_client.db.litellm_usertable.update(
            where={"user_id": new_member.user_id},
            data={"teams": {"push": [team_id]}},
        )
    elif new_member.user_email is not None:
        user_data = {"user_id": str(uuid.uuid4()), "user_email": new_member.user_email}
        ## user email is not unique acc. to prisma schema -> future improvement
        ### for now: check if it exists in db, if not - insert it
        existing_user_row = await prisma_client.get_data(
            key_val={"user_email": new_member.user_email},
            table_name="user",
            query_type="find_all",
        )
        if existing_user_row is None or (
            isinstance(existing_user_row, list) and len(existing_user_row) == 0
        ):

            await prisma_client.insert_data(data=user_data, table_name="user")

    # Check if trying to set a budget for team member
    if max_budget_in_team is not None and new_member.user_id is not None:
        # create a new budget item for this member
        response = await prisma_client.db.litellm_budgettable.create(
            data={
                "max_budget": max_budget_in_team,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )

        _budget_id = response.budget_id
        await prisma_client.db.litellm_teammembership.create(
            data={
                "team_id": team_id,
                "user_id": new_member.user_id,
                "budget_id": _budget_id,
            }
        )

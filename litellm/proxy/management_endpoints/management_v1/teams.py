"""`POST /management/v1/teams/{team_id}/members/bulk_delete` and `.../members/bulk_update`."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem, reject_unknown_query_params
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.bulk_team_member_budgets import bulk_update_team_member_budgets
from litellm.proxy.management_helpers.bulk_user_deletion import bulk_remove_team_members
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberBudgetUpdateRequest,
    BulkTeamMemberBudgetUpdateResponse,
    BulkTeamMemberDeleteRequest,
    BulkTeamMemberDeleteResponse,
)

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


@router.post(
    "/teams/{team_id}/members/bulk_delete",
    tags=["team management"],  # mutable-ok: FastAPI types `tags` as list[str], not Sequence
    dependencies=(Depends(user_api_key_auth), Depends(reject_unknown_query_params)),
    response_model=BulkTeamMemberDeleteResponse,
)
@management_endpoint_wrapper
async def bulk_delete_team_members_action(
    team_id: str,
    data: BulkTeamMemberDeleteRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> BulkTeamMemberDeleteResponse:
    """
    Remove up to 500 members from one team in one call. Same authorization as
    `/team/member_delete`: proxy admins, the team's admins, and admins of the team's
    organization. Each member is named by exactly one of `user_id` or `user_email`;
    unknown body fields are a 422 and an unknown team is a 404.

    `data` holds one result per requested member, in request order. A row is
    `success: false` with an `error` when it names nobody on the team or repeats an
    earlier row. The roster is rewritten once, under the team's advisory lock, so a
    concurrent member_add is never overwritten from a stale read.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/management/v1/teams/team-1/members/bulk_delete' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{"members": [{"user_id": "user-1"}, {"user_email": "user-2@example.com"}]}'
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        results: Final = await bulk_remove_team_members(
            team_id=team_id,
            data=data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        return BulkTeamMemberDeleteResponse(data=results)

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.teams.bulk_delete_team_members_action(): "
            "Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to remove team members.",
            )
        )


@router.post(
    "/teams/{team_id}/members/bulk_update",
    tags=["team management"],  # mutable-ok: FastAPI types `tags` as list[str], not Sequence
    dependencies=(Depends(user_api_key_auth), Depends(reject_unknown_query_params)),
    response_model=BulkTeamMemberBudgetUpdateResponse,
)
@management_endpoint_wrapper
async def bulk_update_team_member_budgets_action(
    team_id: str,
    data: BulkTeamMemberBudgetUpdateRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    litellm_changed_by: Annotated[
        str | None,
        Header(
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ] = None,
) -> BulkTeamMemberBudgetUpdateResponse:
    """
    Set per-member limits for up to 500 members of one team in one call. Same
    authorization and member addressing as `/team/member_update`: proxy admins, the team's
    admins, and admins of the team's organization, with each member named by exactly one of
    `user_id` or `user_email`. Unknown body fields are a 422 and an unknown team is a 404.

    Each row is a merge patch of that member's limits: a field left out is untouched, a
    field sent as null is cleared, and clearing the last limit drops the member back to the
    team default. A budget row shared by several memberships, the team default included, is
    copied for the member being patched rather than written in place, so one member's new
    cap never lands on anybody else.

    `data` holds one result per requested member, in request order, carrying the limits in
    force after the write. A row is `success: false` with an `error` when it names nobody on
    the team or repeats an earlier row. Roles are not part of this route; `/team/member_update`
    still owns them.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/management/v1/teams/team-1/members/bulk_update' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{"members": [{"user_id": "user-1", "max_budget_in_team": 10}, {"user_email": "user-2@example.com", "max_budget_in_team": 10, "budget_duration": "30d"}]}'
    ```
    """
    try:
        from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client, user_api_key_cache

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        results: Final = await bulk_update_team_member_budgets(
            team_id=team_id,
            data=data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            litellm_changed_by=litellm_changed_by,
        )
        return BulkTeamMemberBudgetUpdateResponse(data=results)

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.teams.bulk_update_team_member_budgets_action(): "
            "Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to update team member budgets.",
            )
        )

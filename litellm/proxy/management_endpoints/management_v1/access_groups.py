"""`PATCH /management/v1/access-groups/{access_group_id}`."""

from typing import Annotated, Final, assert_never

from fastapi import APIRouter, Depends

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    problem_responses,
)
from litellm.types.access_group import AccessGroupPatchRequest, AccessGroupResponse
from litellm.types.proxy.management_endpoints.management_v1 import ItemResponse, ProblemDetail

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


def _forbidden(caller: UserAPIKeyAuth) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}forbidden",
            title="Forbidden",
            status=403,
            detail=f"Only proxy admins can update access groups, your role={caller.user_role}",
        )
    )


@router.patch(
    "/access-groups/{access_group_id}",
    tags=("access group management",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ItemResponse[AccessGroupResponse],
    responses=problem_responses(403, 404, 409, 422, 500, 503),
)
async def patch_access_group(
    access_group_id: str,
    body: AccessGroupPatchRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ItemResponse[AccessGroupResponse]:
    """
    Update one access group as a JSON merge patch: a key that is sent is written, `null` clears
    it (a cleared list becomes `[]`), and a key that is omitted keeps its value. Unknown keys
    are refused with a 422 so a typo is never a silent no-op.

    Proxy admins only. Errors are RFC 9457 problem documents.

    Example curl:
    ```
    curl --location --request PATCH 'http://0.0.0.0:4000/management/v1/access-groups/<access_group_id>' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{"description": "Production models", "access_model_names": ["gpt-5.2"]}'
    ```
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise _forbidden(user_api_key_dict)

    # access_group_endpoints is a lazy feature: importing it at module load would mark it warm in
    # /openapi.json before its legacy routes are registered, so it is imported per request instead
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        AccessGroupNameTaken,
        AccessGroupNotFound,
        AccessGroupUpdated,
        apply_access_group_update,
        propagate_access_group_update,
    )
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                title="Database not connected",
                status=503,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )
        )

    try:
        outcome: Final = await apply_access_group_update(
            prisma_client, access_group_id, body, user_api_key_dict.user_id
        )
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.access_groups.patch_access_group(): Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to update the access group.",
            )
        )

    match outcome:
        case AccessGroupNotFound():
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}not-found",
                    title="Not found",
                    status=404,
                    detail=f"Access group '{access_group_id}' not found.",
                )
            )
        case AccessGroupNameTaken(access_group_name=access_group_name):
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}conflict",
                    title="Conflict",
                    status=409,
                    detail=f"Access group '{access_group_name}' already exists.",
                )
            )
        case AccessGroupUpdated():
            await propagate_access_group_update(outcome, access_group_id)
            return ItemResponse(data=AccessGroupResponse.model_validate(outcome.record.dict()))
        case _:
            assert_never(outcome)

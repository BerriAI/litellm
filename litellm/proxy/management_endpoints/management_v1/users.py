"""`POST /management/v1/users/bulk_delete`."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem, reject_unknown_query_params
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.bulk_user_deletion import bulk_delete_users
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkDeleteUserRequest,
    BulkDeleteUsersResponse,
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


@router.post(
    "/users/bulk_delete",
    tags=["Internal User management"],  # mutable-ok: FastAPI types `tags` as list[str], not Sequence
    dependencies=(Depends(user_api_key_auth), Depends(reject_unknown_query_params)),
    response_model=BulkDeleteUsersResponse,
)
@management_endpoint_wrapper
async def bulk_delete_users_action(
    data: BulkDeleteUserRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    litellm_changed_by: Annotated[
        str | None,
        Header(description="Who the caller is acting for; recorded on the audit log entries this call writes."),
    ] = None,
) -> BulkDeleteUsersResponse:
    """
    Delete up to 500 users in one call, taking each out of every team it belongs to.
    Same authorization as `/user/delete`: proxy admins may delete anyone, org admins
    only users inside organizations they administer. Unknown body fields are a 422.

    `data` holds one result per requested `user_id`, in request order. A row is
    `success: false` with an `error` when the id is unknown, repeated in the request,
    or outside the caller's scope. Rows that pass those checks are deleted together,
    in one transaction, so either all of them go or none does.

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/management/v1/users/bulk_delete' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{"user_ids": ["user-1", "user-2"]}'
    ```
    """
    try:
        from litellm.proxy.proxy_server import (
            litellm_proxy_admin_name,
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise ManagementProblem(
                ProblemDetail(
                    type=f"{PROBLEM_TYPE_BASE}database-not-connected",
                    title="Database not connected",
                    status=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        results: Final = await bulk_delete_users(
            data=data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            litellm_changed_by=litellm_changed_by,
        )
        return BulkDeleteUsersResponse(data=results)

    except ManagementProblem:
        raise
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.users.bulk_delete_users_action(): Exception occured - %s",
            e,
        )
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to delete users.",
            )
        )

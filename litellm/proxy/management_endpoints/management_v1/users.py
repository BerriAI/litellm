"""`POST /management/v1/users/bulk`."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.bulk_user_creation import bulk_create_users
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy untyped decorator
)
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkNewUserRequest,
    BulkNewUserResponse,
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


@router.post(
    "/users/bulk",
    tags=["Internal User management"],  # mutable-ok: fastapi types tags as list[str | Enum]
    dependencies=(Depends(user_api_key_auth),),
    response_model=BulkNewUserResponse,
)
@management_endpoint_wrapper
async def bulk_create_users_route(
    data: BulkNewUserRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> BulkNewUserResponse:
    """
    Create up to 500 internal users in one request, optionally adding each one to teams.

    Every entry in `users` takes the same fields as `/user/new`, with two differences: `auto_create_key`
    defaults to `false` (opt in per user to also get a virtual key back) and `send_invite_email` is not
    supported. Unknown fields are rejected with 422. Rows are validated together (duplicate ids or emails,
    unknown teams, roles the caller may not grant), inserted in one statement, and each referenced team is
    written once for all of its new members.

    Rows fail independently: a bad row is reported in `data` with `success: false` and an `error`, and the
    other rows still get created. A user that was created but could not be added to one of its teams is
    reported with `success: true`, `teams` listing where they did land, and `error` naming the failed team.
    The whole request is refused with a 403 problem document only if creating the valid rows would exceed
    the license seat limit.

    Example curl:
    ```
    curl -X POST "http://localhost:4000/management/v1/users/bulk" \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer sk-1234" \\
    -d '{
        "users": [
            {"user_email": "a@example.com", "user_role": "internal_user", "teams": ["team-1"]},
            {"user_email": "b@example.com", "user_role": "internal_user", "auto_create_key": true}
        ]
    }'
    ```

    Returns `data` (one entry per input row, in order, with `user_id`, `user_email`, `success`, `teams`,
    `key`, `error`) and `meta` with `total_requested`, `created` and `failed`.
    """
    try:
        from litellm.proxy.proxy_server import (
            _license_check,  # pyright: ignore[reportPrivateUsage]  # same proxy license singleton /user/new reads
            litellm_proxy_admin_name,
            prisma_client,
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

        return await bulk_create_users(
            users=data.users,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            license_check=_license_check,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            user_api_key_cache=user_api_key_cache,
        )

    except ManagementProblem:
        raise
    except Exception:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception("/management/v1/users/bulk: Exception occurred")
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}internal-server-error",
                title="Internal server error",
                status=500,
                detail="Failed to create users.",
            )
        )

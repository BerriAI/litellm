"""`PATCH /management/v1/users/{user_id}`."""

from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_ObjectPermissionBase,
    LitellmUserRoles,
    UpdateUserRequest,
    UpdateUserRequestNoUserIDorEmail,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management_endpoints.common_utils import validate_budget_duration
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    problem_responses,
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.management_endpoints.management_v1 import (
    ItemResponse,
    ProblemDetail,
)

if TYPE_CHECKING:
    from prisma import models as prisma_models

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)

# Starlette leaves `HTTPException.status_code` untyped; validate rather than widen the call site.
_HTTP_STATUS: Final = TypeAdapter(int)

# Columns the schema declares NOT NULL with a default. A merge-patch null on one of these means
# "back to empty", which is the default, not SQL NULL; writing NULL would be rejected by the engine.
# Held as factories so each call gets its own empty container to hand to the query engine.
_CLEARS_TO_EMPTY: Final[Mapping[str, Callable[[], object]]] = MappingProxyType(
    {"models": list, "metadata": dict, "model_max_budget": dict}
)


class UserPatchRequest(BaseModel):
    """Body of `PATCH /management/v1/users/{user_id}`, read as an RFC 7396 JSON merge patch.

    Every field is optional and nullable, and the two are not the same thing: an omitted field is
    left alone, an explicit `null` clears the setting. `extra="forbid"` is what makes that promise
    keepable, since a misspelled key would otherwise read as "omitted" and silently do nothing.
    """

    model_config = ConfigDict(extra="forbid")

    user_email: str | None = None
    user_alias: str | None = None
    user_role: (
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
        | None
    ) = None
    models: list[str] | None = None
    max_budget: float | None = None
    budget_duration: str | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    max_parallel_requests: int | None = None
    metadata: dict[str, JsonValue] | None = None
    model_max_budget: dict[str, JsonValue] | None = None
    object_permission: LiteLLM_ObjectPermissionBase | None = None


class UserItem(BaseModel):
    """One internal user as the control plane returns it, read back off the row the write produced.

    Re-reading rather than echoing the request is the point of the endpoint: a caller can tell a
    clear that landed from one that was dropped by looking at the response.
    """

    model_config = ConfigDict(from_attributes=True)

    user_id: str
    user_email: str | None = None
    user_alias: str | None = None
    user_role: str | None = None
    models: list[str] = Field(default_factory=list)
    spend: float = 0.0
    max_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    max_parallel_requests: int | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    model_max_budget: dict[str, JsonValue] = Field(default_factory=dict)
    object_permission_id: str | None = None
    teams: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


def resolve_user_patch_fields(
    data_json: dict,
    data: UpdateUserRequest | UpdateUserRequestNoUserIDorEmail,
) -> dict:
    """Merge-patch field resolution: whatever the caller sent is written, nulls included.

    The inverse of `_update_internal_user_params`, which drops nulls and so cannot express a clear.
    Only three adjustments are made to the raw body, and each is a property of the schema rather
    than a policy choice: NOT NULL columns clear to their default instead of SQL NULL,
    `budget_reset_at` follows `budget_duration` because it is derived from it, and an
    `object_permission` clear is left for the caller of this resolver, which drops the entitlement
    link rather than writing a column.
    """
    resolved: Final[dict] = {
        key: (_CLEARS_TO_EMPTY[key]() if value is None and key in _CLEARS_TO_EMPTY else value)
        for key, value in data_json.items()
        if not (key == "object_permission" and value is None)
    }
    derived: Final = _budget_reset_fields(resolved["budget_duration"]) if "budget_duration" in resolved else {}
    return {**resolved, **derived, **_internal_user_role_defaults(data, resolved)}


def _budget_reset_fields(budget_duration: str | None) -> dict:
    """`budget_reset_at` is derived from `budget_duration`, so the two only ever move together."""
    from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

    if budget_duration is None:
        return {"budget_reset_at": None}
    validate_budget_duration(budget_duration)
    return {"budget_reset_at": get_budget_reset_time(budget_duration=budget_duration)}


def _internal_user_role_defaults(
    data: UpdateUserRequest | UpdateUserRequestNoUserIDorEmail,
    resolved: dict,
) -> dict:
    """Apply the proxy-wide internal-user budget caps, but only to fields the caller left alone.

    Mirrors `/user/update`, so promoting someone to internal user still lands them under
    `max_internal_user_budget`. Guarded on absence rather than falsiness, so an explicit clear is
    never overwritten by a global default the caller was trying to get out from under.
    """
    if data.user_role != LitellmUserRoles.INTERNAL_USER:
        return {}
    budget: Final = (
        {"max_budget": litellm.max_internal_user_budget}
        if "max_budget" not in resolved and litellm.max_internal_user_budget is not None
        else {}
    )
    if "budget_duration" in resolved or litellm.internal_user_budget_duration is None:
        return budget
    return {
        **budget,
        "budget_duration": litellm.internal_user_budget_duration,
        **_budget_reset_fields(litellm.internal_user_budget_duration),
    }


def _problem(status: int, slug: str, title: str, detail: str) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(type=f"{PROBLEM_TYPE_BASE}{slug}", title=title, status=status, detail=detail)
    )


_PROBLEM_BY_STATUS: Final[Mapping[int, tuple[str, str]]] = MappingProxyType(
    {403: ("forbidden", "Forbidden"), 404: ("user-not-found", "Not found")}
)


def _problem_from_http_exception(exc: HTTPException) -> ManagementProblem:
    """Re-dress the shared write path's `HTTPException` as a problem document.

    `update_single_user` is shared with `/user/update` and raises that endpoint's error
    shape. Translating here keeps the control plane's contract without forking the authorization
    checks, which is the one thing that must not drift between the two surfaces.
    """
    status: Final[int] = _HTTP_STATUS.validate_python(exc.status_code)
    detail: Final[object] = exc.detail
    slug, title = _PROBLEM_BY_STATUS.get(status, ("user-update-failed", "User update failed"))
    return _problem(
        status=status,
        slug=slug,
        title=title,
        detail=str(detail.get("error", detail)) if isinstance(detail, Mapping) else str(detail),
    )


@router.patch(
    "/users/{user_id}",
    tags=["Internal User management"],
    dependencies=(Depends(user_api_key_auth),),
    response_model=ItemResponse[UserItem],
    responses=problem_responses(403, 404, 422, 500, 503),
)
async def patch_user(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    body: UserPatchRequest,
    user_id: Annotated[str, Path(description="The id of the user to update.")],
) -> ItemResponse[UserItem]:
    """
    Partially update one internal user, as an RFC 7396 JSON merge patch.

    An omitted field is left alone and an explicit `null` clears the setting, which is the whole
    reason this route exists: `POST /user/update` drops nulls, so it answers `200` to a clear it
    silently discarded, and only `max_budget` was ever made clearable. Unknown body keys are
    refused with a `422` rather than ignored. `null` on `models`, `metadata` or `model_max_budget`
    resets the column to empty, since the schema declares those NOT NULL.

    Requires a proxy admin: the route is in no non-admin allowlist, so everyone else is refused at
    the route gate, and the shared write path's self-service guards stand behind that as defense in
    depth. Unlike `/user/update`, a user id that does not exist is a `404` rather than a silent
    create, since the underlying write is an upsert.

    Example curl, clearing a rate limit and setting another:
    ```
    curl --location --request PATCH 'http://0.0.0.0:4000/management/v1/users/user123' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{"tpm_limit": null, "rpm_limit": 60}'
    ```
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        update_single_user,
    )
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise _problem(
            503,
            "database-not-connected",
            "Database not connected",
            CommonProxyErrors.db_not_connected_error.value,
        )

    # `update_data` upserts, so a missing row would otherwise be created by an admin's PATCH.
    if await _find_user(prisma_client, user_id) is None:
        raise _problem(404, "user-not-found", "Not found", f"No user with id {user_id!r} exists.")

    try:
        await update_single_user(
            user_request=UpdateUserRequest(user_id=user_id, **body.model_dump(exclude_unset=True)),
            user_api_key_dict=user_api_key_dict,
            resolve_fields=resolve_user_patch_fields,
        )
    except ManagementProblem:
        raise
    except HTTPException as e:
        raise _problem_from_http_exception(e) from e
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.users.patch_user(): Exception occured - %s", e
        )
        raise _problem(500, "internal-server-error", "Internal server error", str(e)) from e

    updated: Final = await _find_user(prisma_client, user_id)
    if updated is None:
        raise _problem(
            500,
            "internal-server-error",
            "Internal server error",
            "The updated user could not be read back.",
        )
    return ItemResponse[UserItem](data=UserItem.model_validate(updated))


async def _find_user(prisma_client: PrismaClient, user_id: str) -> "prisma_models.LiteLLM_UserTable | None":
    table: Final[TableActions[prisma_models.LiteLLM_UserTable]] = UserRepository(prisma_client).table
    return await table.find_first(where={"user_id": user_id})

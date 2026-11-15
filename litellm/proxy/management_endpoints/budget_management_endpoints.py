"""
BUDGET MANAGEMENT

All /budget management endpoints

/budget/new
/budget/info
/budget/update
/budget/delete
/budget/settings
/budget/list
"""

#### BUDGET TABLE MANAGEMENT ####
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import TypeAdapter

from litellm._internal_context import service_target
from litellm._logging import verbose_proxy_logger
from litellm.models.budget import initialize_budget_limits
from litellm.proxy._types import *
from litellm.proxy.auth.customer_hierarchy_auth import forget_customer_hierarchies, forget_customer_organization_aliases
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_settings, get_budget_reset_time
from litellm.proxy.management_endpoints.common_utils import (  # noqa: F401  # legacy module exports
    _user_has_admin_view,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    user_api_key_has_admin_view,
    validate_budget_duration,
)
from litellm.proxy.management_helpers.customer_hierarchy import budget_limit_json
from litellm.proxy.management_helpers.utils import validate_budget_windows_update
from litellm.proxy.utils import jsonify_object
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE
from litellm.repositories.table_repositories import (
    CustomerMemberRepository,
    CustomerOrganizationRepository,
    CustomerTeamRepository,
)

router: Final = APIRouter()
_WRITE_VALUES: Final = TypeAdapter(Mapping[str, object])


async def new_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new budget object. Can apply this to teams, orgs, end-users, keys.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - budget_limits: Optional[list] - Concurrent budget windows with independent caps and reset times.
    - budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
    - max_budget: Optional[float] - The max budget for the budget.
    - soft_budget: Optional[float] - The soft budget for the budget.
    - max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
    - tpm_limit: Optional[int] - The tokens per minute limit for the budget.
    - rpm_limit: Optional[int] - The requests per minute limit for the budget.
    - tpd_limit: Optional[int] - The tokens per day limit for the budget. Charged by batch submissions instead of tpm_limit/rpm_limit.
    - model_max_budget: Optional[dict] - Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}}
    - model_tpm_limit: Optional[dict] - Max tokens per minute per model, shared by everything on this budget. Example: {"gpt-4o": 400000}
    - model_rpm_limit: Optional[dict] - Max requests per minute per model, shared by everything on this budget. Example: {"gpt-4o": 200}
    - budget_reset_at: Optional[datetime] - Datetime when the initial budget is reset. Default is now.
    """
    from prisma.errors import UniqueViolationError

    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Validate budget values are not negative
    if budget_obj.max_budget is not None and (not math.isfinite(budget_obj.max_budget) or budget_obj.max_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"max_budget must be a non-negative finite number. Received: {budget_obj.max_budget}"},
        )
    if budget_obj.soft_budget is not None and (not math.isfinite(budget_obj.soft_budget) or budget_obj.soft_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"soft_budget must be a non-negative finite number. Received: {budget_obj.soft_budget}"},
        )

    validate_budget_duration(budget_obj.budget_duration)

    # Validate model_max_budget if present
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    # if no budget_reset_at date is set, but a budget_duration is given, then set budget_reset_at initially to the first completed duration interval in future
    reset: Final = (
        {"budget_reset_at": get_budget_reset_time(budget_duration=budget_obj.budget_duration)}
        if budget_obj.budget_reset_at is None and budget_obj.budget_duration is not None
        else {}
    )

    budget_obj_json: Final = {**budget_obj.model_dump(exclude_none=True), **reset}
    budget_obj_jsonified: Final[dict[str, object]] = jsonify_object(budget_obj_json)  # mutable-ok: prisma create input
    window_data: Final = (
        {
            "budget_limits": budget_limit_json(
                initialize_budget_limits(
                    budget_obj.budget_limits, get_budget_reset_settings(), datetime.now(timezone.utc)
                )
            )
        }
        if isinstance(budget_obj, BudgetWriteRequest) and budget_obj.budget_limits is not None
        else {}
    )
    try:
        response: Final = await BudgetRepository(prisma_client).table.create(
            data={
                **budget_obj_jsonified,
                **window_data,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )
    except Exception as e:
        if not isinstance(e, UniqueViolationError):
            raise
        raise HTTPException(
            status_code=400,
            detail={"error": f"Budget with id '{budget_obj.budget_id}' already exists."},
        )

    return response


async def update_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing budget object.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - budget_limits: Optional[list] - Replace concurrent budget windows. Null or an empty list clears them.
    - budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
    - max_budget: Optional[float] - The max budget for the budget.
    - soft_budget: Optional[float] - The soft budget for the budget.
    - max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
    - tpm_limit: Optional[int] - The tokens per minute limit for the budget.
    - rpm_limit: Optional[int] - The requests per minute limit for the budget.
    - tpd_limit: Optional[int] - The tokens per day limit for the budget. Charged by batch submissions instead of tpm_limit/rpm_limit.
    - model_max_budget: Optional[dict] - Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}}
    - model_tpm_limit: Optional[dict] - Max tokens per minute per model, shared by everything on this budget. Example: {"gpt-4o": 400000}
    - model_rpm_limit: Optional[dict] - Max requests per minute per model, shared by everything on this budget. Example: {"gpt-4o": 200}
    - budget_reset_at: Optional[datetime] - Update the Datetime when the budget was last reset.
    """
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    if budget_obj.budget_id is None:
        raise HTTPException(status_code=400, detail={"error": "budget_id is required"})

    # Validate budget values are not negative
    if budget_obj.max_budget is not None and (not math.isfinite(budget_obj.max_budget) or budget_obj.max_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"max_budget must be a non-negative finite number. Received: {budget_obj.max_budget}"},
        )
    if budget_obj.soft_budget is not None and (not math.isfinite(budget_obj.soft_budget) or budget_obj.soft_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"soft_budget must be a non-negative finite number. Received: {budget_obj.soft_budget}"},
        )

    validate_budget_duration(budget_obj.budget_duration)

    # Validate model_max_budget if present in update
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    # recompute budget_reset_at when the duration changes, unless the caller pinned a reset time explicitly
    recomputed_reset_at: Final = (
        {"budget_reset_at": get_budget_reset_time(budget_duration=budget_obj.budget_duration)}
        if budget_obj.budget_duration is not None and "budget_reset_at" not in budget_obj.model_fields_set
        else MappingProxyType({"budget_reset_at": None})
        if "budget_duration" in budget_obj.model_fields_set
        and budget_obj.budget_duration is None
        and "budget_reset_at" not in budget_obj.model_fields_set
        else {}
    )

    budget_obj_jsonified: Final = _WRITE_VALUES.validate_python(
        jsonify_object(
            {
                **budget_obj.model_dump(exclude_unset=True),
                **recomputed_reset_at,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )
    )

    window_data: Final = (
        {
            "budget_limits": budget_limit_json(
                initialize_budget_limits(
                    budget_obj.budget_limits, get_budget_reset_settings(), datetime.now(timezone.utc)
                )
            )
        }
        if isinstance(budget_obj, BudgetWriteRequest) and budget_obj.budget_limits is not None
        else {}
    )
    if isinstance(budget_obj, BudgetWriteRequest) and budget_obj.budget_limits:
        await validate_budget_windows_update(prisma_client.db, budget_obj.budget_id)
    response: Final = await BudgetRepository(prisma_client).table.update(
        where={"budget_id": budget_obj.budget_id}, data={**budget_obj_jsonified, **window_data}
    )
    members: Final = await CustomerMemberRepository(prisma_client).table.find_many(
        where={
            "OR": (
                {"budget_id": budget_obj.budget_id},
                {"budget_override_id": budget_obj.budget_id},
                {"parent_org": {"is": {"budget_id": budget_obj.budget_id}}},
            )
        }
    )
    await forget_customer_hierarchies(tuple(member.customer_member_id for member in members), user_api_key_cache)
    teams: Final = await CustomerTeamRepository(prisma_client).table.find_many(
        where={"OR": ({"budget_id": budget_obj.budget_id}, {"budget_override_id": budget_obj.budget_id})}
    )
    team_ids: Final = tuple(team.customer_team_id for team in teams)
    for offset in range(0, len(team_ids), IN_LIST_CHUNK_SIZE):
        await forget_customer_hierarchies(
            tuple(
                member.customer_member_id
                for member in await CustomerMemberRepository(prisma_client).table.find_many(
                    where={"parent_team_ids": {"has_some": team_ids[offset : offset + IN_LIST_CHUNK_SIZE]}}
                )
            ),
            user_api_key_cache,
        )
    organizations: Final = await CustomerOrganizationRepository(prisma_client).table.find_many(
        where={"budget_id": budget_obj.budget_id}
    )
    await forget_customer_organization_aliases(
        tuple(organization.alias for organization in organizations), user_api_key_cache
    )
    with service_target("team_member_budgets"):
        try:
            await user_api_key_cache.async_delete_cache(key=f"team_member_default_budget:{budget_obj.budget_id}")
        except Exception:  # noqa: BLE001  # Cache failure must not fail a committed budget update.
            verbose_proxy_logger.warning("Failed to evict the team member default budget cache", exc_info=True)

    return response


@router.post("/budget/new", tags=["budget management"], description=new_budget.__doc__)
async def new_budget_endpoint(
    budget_obj: BudgetWriteRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> object:
    return await new_budget(budget_obj, user_api_key_dict)


@router.post("/budget/update", tags=["budget management"], description=update_budget.__doc__)
async def update_budget_endpoint(
    budget_obj: BudgetWriteRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> object:
    return await update_budget(budget_obj, user_api_key_dict)


@router.post(
    "/budget/info",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_budget(data: BudgetRequest):
    """
    Get the budget id specific information

    Parameters:
    - budgets: List[str] - The list of budget ids to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if len(data.budgets) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Specify list of budget id's to query. Passed in={data.budgets}"},
        )
    response: Final = await BudgetRepository(prisma_client).table.find_many(
        where={"budget_id": {"in": data.budgets}},
    )

    return response


@router.get(
    "/budget/settings",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def budget_settings(
    budget_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get list of configurable params + current value for a budget item + description of each field

    Used on Admin UI.

    Query Parameters:
    - budget_id: str - The budget id to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not user_api_key_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    ## get budget item from db
    db_budget_row: Final = await BudgetRepository(prisma_client).table.find_first(where={"budget_id": budget_id})

    if db_budget_row is not None:
        db_budget_row_dict = db_budget_row.model_dump(exclude_none=True)
    else:
        db_budget_row_dict = {}

    allowed_args: Final = {
        "max_parallel_requests": {"type": "Integer"},
        "tpm_limit": {"type": "Integer"},
        "rpm_limit": {"type": "Integer"},
        "tpd_limit": {"type": "Integer"},
        "budget_duration": {"type": "String"},
        "max_budget": {"type": "Float"},
        "soft_budget": {"type": "Float"},
        "model_max_budget": {"type": "Object"},
        "model_tpm_limit": {"type": "Object"},
        "model_rpm_limit": {"type": "Object"},
        "budget_limits": {"type": "Object"},
    }

    return_val: Final = []

    for field_name, field_info in BudgetWriteRequest.model_fields.items():
        if field_name in allowed_args:
            _stored_in_db = True

            _response_obj = ConfigList(
                field_name=field_name,
                field_type=allowed_args[field_name]["type"],
                field_description=field_info.description or "",
                field_value=db_budget_row_dict.get(field_name, None),
                stored_in_db=_stored_in_db,
                field_default_value=field_info.default,
            )
            return_val.append(_response_obj)

    return return_val


@router.get(
    "/budget/list",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_budget(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all the created budgets in proxy db. Used on Admin UI."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not user_api_key_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    response: Final = await BudgetRepository(prisma_client).table.find_many()

    return response


@router.post(
    "/budget/delete",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_budget(
    data: BudgetDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete budget

    Parameters:
    - id: str - The budget id to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    customer_org_count: Final = await CustomerOrganizationRepository(prisma_client).table.count(
        where={"budget_template_ids": {"has": data.id}}
    )
    if customer_org_count:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Budget is a customer organization default budget. Delete it through "
                "/management/v1/customer_organizations/{customer_org_id}/default_budgets/{budget_id}"
            },
        )

    response: Final = await BudgetRepository(prisma_client).table.delete(where={"budget_id": data.id})

    return response

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
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, Depends, HTTPException

from litellm._internal_context import service_target
from litellm._logging import verbose_proxy_logger
from litellm.models.budget import validate_person_budget_limits
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management_endpoints.common_utils import (  # noqa: F401  # legacy module exports
    _user_has_admin_view,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    user_api_key_has_admin_view,
    validate_budget_duration,
)
from litellm.proxy.utils import jsonify_object
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import CustomerOrganizationRepository

router: Final = APIRouter()


def validate_budget_profile(budget: BudgetNewRequest) -> None:
    try:
        validate_person_budget_limits(budget.budget_limits)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if (budget.temp_budget_increase is None) != (budget.temp_budget_expiry is None):
        raise HTTPException(status_code=400, detail="A temporary increase requires both an amount and an expiry")
    if budget.is_profile and not budget.budget_limits:
        raise HTTPException(status_code=400, detail="A profile requires a monthly or annual budget limit")
    if budget.is_profile and budget.temp_budget_increase is not None:
        raise HTTPException(status_code=400, detail="Apply temporary increases to a person's budget override")
    if budget.is_profile and any(
        value is not None
        for value in (
            budget.max_budget,
            budget.soft_budget,
            budget.max_parallel_requests,
            budget.budget_reset_at,
            budget.budget_duration,
            budget.tpm_limit,
            budget.rpm_limit,
            budget.tpd_limit,
            budget.model_max_budget,
            budget.model_tpm_limit,
            budget.model_rpm_limit,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Profiles configure monthly and annual limits; keep other limits on the person, key or team",
        )


def serialized_budget_limits(budget: BudgetNewRequest) -> Mapping[str, object]:
    if "budget_limits" not in budget.model_fields_set:
        return {}
    return {
        "budget_limits": json.dumps(tuple(limit.model_dump(mode="json") for limit in budget.budget_limits))
        if budget.budget_limits is not None
        else None
    }


@router.post(
    "/budget/new",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new budget object. Can apply this to teams, orgs, end-users, keys.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
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
    validate_budget_profile(budget_obj)

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
    if budget_obj.budget_reset_at is None and budget_obj.budget_duration is not None:
        budget_obj.budget_reset_at = get_budget_reset_time(budget_duration=budget_obj.budget_duration)

    budget_obj_json: Final = {**budget_obj.model_dump(exclude_none=True), **serialized_budget_limits(budget_obj)}
    budget_obj_jsonified: Final[dict[str, object]] = jsonify_object(budget_obj_json)  # mutable-ok: prisma create input
    try:
        response: Final = await BudgetRepository(prisma_client).table.create(
            data={
                **budget_obj_jsonified,
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


@router.post(
    "/budget/update",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing budget object.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
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
    stored_budget: Final = await BudgetRepository(prisma_client).table.find_unique(
        where={"budget_id": budget_obj.budget_id}
    )
    if stored_budget is None:
        raise HTTPException(status_code=404, detail="Budget not found")
    stored_values: Final = stored_budget.model_dump()
    if "is_profile" in budget_obj.model_fields_set and budget_obj.is_profile != stored_values.get("is_profile", False):
        raise HTTPException(status_code=400, detail="A budget's profile type cannot be changed after creation")
    validate_budget_profile(
        BudgetNewRequest.model_validate(
            {
                **stored_values,
                **budget_obj.model_dump(exclude_unset=True),
            }
        )
    )

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

    budget_obj_jsonified: Final[Mapping[str, object]] = jsonify_object(
        {
            **budget_obj.model_dump(exclude_unset=True),
            **serialized_budget_limits(budget_obj),
            **recomputed_reset_at,
            "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        }
    )

    response: Final = await BudgetRepository(prisma_client).table.update(
        where={"budget_id": budget_obj.budget_id},
        data=budget_obj_jsonified,
    )
    with service_target("team_member_budgets"):
        try:
            await user_api_key_cache.async_delete_cache(key=f"team_member_default_budget:{budget_obj.budget_id}")
        except Exception:  # noqa: BLE001  # Cache failure must not fail a committed budget update.
            verbose_proxy_logger.warning("Failed to evict the team member default budget cache", exc_info=True)

    return response


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
    }

    return_val: Final = []

    for field_name, field_info in BudgetNewRequest.model_fields.items():
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

    async with prisma_client.tx() as tx:
        await tx.query_raw('SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = $1 FOR UPDATE', data.id)
        profile_row: Final = await tx.litellm_budgettable.find_unique(where={"budget_id": data.id})
        if profile_row is not None and profile_row.model_dump().get("is_profile") is True:
            attached_users: Final = await tx.litellm_usertable.count(where={"budget_template_id": data.id})
            attached_members: Final = await tx.litellm_customermembertable.count(where={"budget_id": data.id})
            if attached_users or attached_members:
                raise HTTPException(status_code=409, detail="Unassign this profile before deleting it")
            return await tx.litellm_budgettable.delete(where={"budget_id": data.id})

    customer_org_count: Final = await CustomerOrganizationRepository(prisma_client).table.count(
        where={"budget_ids": {"has": data.id}}
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

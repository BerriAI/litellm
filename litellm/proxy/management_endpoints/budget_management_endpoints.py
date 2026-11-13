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
from typing import Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, TypeAdapter

from litellm._internal_context import service_target
from litellm._logging import verbose_proxy_logger
from litellm.models.person_budget import PersonBudgetDefinition
from litellm.proxy._types import (
    BudgetDeleteRequest,
    BudgetNewRequest,
    BudgetRequest,
    CommonProxyErrors,
    ConfigList,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management_endpoints.common_utils import (  # noqa: F401  # legacy module exports
    _user_has_admin_view,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    user_api_key_has_admin_view,
    validate_budget_duration,
)
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchyStore,
    PrismaRecord,
    budget_revision,
    invalidate_person_budget_members,
    invalidate_person_budget_sources,
    utc_now,
)
from litellm.proxy.utils import jsonify_object
from litellm.repositories.budget_repository import BudgetRepository

router: Final = APIRouter()


class BudgetWriteRequest(BudgetNewRequest):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BudgetResponse(PersonBudgetDefinition):
    """Preserve budget row attributes for internal callers and audit columns on the wire."""

    model_config = ConfigDict(frozen=True, extra="allow")
    revision: str


_WRITE_VALUES: Final = TypeAdapter(Mapping[str, object])
_SETTING_VALUE: Final = TypeAdapter(object)
_EMPTY_FILTER: Final[Mapping[str, object]] = {}
OWNED_BUDGET: Final[Mapping[str, object]] = {
    "OR": ({"customer_member_overrides": {"some": _EMPTY_FILTER}}, {"user_overrides": {"some": _EMPTY_FILTER}}),
}
REUSABLE_BUDGET: Final[Mapping[str, object]] = {"NOT": OWNED_BUDGET}


def validate_budget_kind(request: BudgetNewRequest, existing: PersonBudgetDefinition | None = None) -> str | None:
    kind: Final = existing.is_profile if existing is not None else request.is_profile is True
    if existing is not None and "is_profile" in request.model_fields_set and request.is_profile != existing.is_profile:
        return "A budget's profile kind cannot be changed."
    if not kind:
        if request.budget_limits is not None or request.cap_policy is not None:
            return "Personal budget limits and cap policy require is_profile=true."
        return None
    limits: Final = (
        request.budget_limits
        if "budget_limits" in request.model_fields_set
        else existing.budget_limits
        if existing is not None
        else None
    )
    if not limits:
        return "A reusable profile requires at least one monthly or yearly limit."
    if len({limit.budget_duration for limit in limits}) != len(limits):
        return "Each profile duration may appear only once."
    scalar_fields: Final = frozenset(BudgetNewRequest.model_fields) - {
        "budget_id",
        "revision",
        "is_profile",
        "budget_limits",
        "cap_policy",
    }
    if any(getattr(request, field) is not None for field in scalar_fields):
        return "Reusable profiles accept personal budget limits and cap policy only."
    return None


def budget_response(row: PrismaRecord) -> BudgetResponse:
    values: Final = row.model_dump()
    definition: Final = PersonBudgetDefinition.model_validate(values)
    return BudgetResponse.model_validate({**values, "revision": budget_revision(definition)})


@router.post(
    "/budget/new",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_budget(
    budget_obj: BudgetWriteRequest,
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

    # Validate model_max_budget if present
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    invalid_kind: Final = validate_budget_kind(budget_obj)
    if invalid_kind is not None:
        raise HTTPException(status_code=400, detail={"error": invalid_kind})
    if budget_obj.is_profile and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail={"error": "Only proxy admins can create budget profiles."})
    from prisma import Json

    reset: Final = (
        {"budget_reset_at": get_budget_reset_time(budget_duration=budget_obj.budget_duration)}
        if budget_obj.budget_reset_at is None and budget_obj.budget_duration is not None
        else {}
    )
    limits: Final = (
        {"budget_limits": Json([limit.model_dump() for limit in budget_obj.budget_limits])}
        if budget_obj.budget_limits is not None
        else {}
    )
    budget_obj_json: Final = {**budget_obj.model_dump(exclude_none=True, exclude={"revision"}), **reset}
    budget_obj_jsonified: Final = _WRITE_VALUES.validate_python(jsonify_object(budget_obj_json))
    try:
        response: Final = await BudgetRepository(prisma_client).table.create(
            data={
                **budget_obj_jsonified,
                **limits,
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

    saved: Final = budget_response(response)
    await invalidate_person_budget_sources((saved.budget_id,))
    return saved


@router.post(
    "/budget/update",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_budget(
    budget_obj: BudgetWriteRequest,
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
        else {"budget_reset_at": None}
        if "budget_duration" in budget_obj.model_fields_set
        and budget_obj.budget_duration is None
        and "budget_reset_at" not in budget_obj.model_fields_set
        else {}
    )

    budget_obj_jsonified: Final = _WRITE_VALUES.validate_python(
        jsonify_object(
            {
                **budget_obj.model_dump(exclude_unset=True, exclude={"revision"}),
                **recomputed_reset_at,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )
    )

    from prisma import Json

    async with prisma_client.tx() as tx:
        store: Final = CustomerHierarchyStore(tx, utc_now())
        await tx.query_raw(
            'SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = $1 FOR UPDATE', budget_obj.budget_id
        )
        existing_row: Final = await store.budgets.find_unique(where={"budget_id": budget_obj.budget_id})
        if existing_row is None:
            raise HTTPException(status_code=404, detail={"error": "Budget not found."})
        if await store.budgets.count(where={"budget_id": budget_obj.budget_id, **OWNED_BUDGET}):
            raise HTTPException(
                status_code=409, detail={"error": "Edit an owned override through its person's assignment."}
            )
        existing: Final = PersonBudgetDefinition.model_validate(existing_row.model_dump())
        invalid_kind: Final = validate_budget_kind(budget_obj, existing)
        if invalid_kind is not None:
            raise HTTPException(status_code=400, detail={"error": invalid_kind})
        if existing.is_profile:
            if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
                raise HTTPException(status_code=403, detail={"error": "Only proxy admins can edit budget profiles."})
            if budget_obj.revision != budget_revision(existing):
                raise HTTPException(status_code=409, detail={"error": "Budget changed. Reload before saving."})
        limits: Final = (
            {"budget_limits": Json([limit.model_dump() for limit in budget_obj.budget_limits])}
            if budget_obj.budget_limits is not None
            else {}
        )
        response: Final = await store.budgets.update(
            where={"budget_id": budget_obj.budget_id},
            data={**budget_obj_jsonified, **limits},
        )
        saved: Final = budget_response(response) if response is not None else None
        affected_members: Final = await store.member_ids_using_template(budget_obj.budget_id)
    await invalidate_person_budget_sources((budget_obj.budget_id,))
    if affected_members:
        await invalidate_person_budget_members(affected_members)
    with service_target("team_member_budgets"):
        try:
            await user_api_key_cache.async_delete_cache(key=f"team_member_default_budget:{budget_obj.budget_id}")
        except Exception:  # noqa: BLE001  # Cache failure must not fail a committed budget update.
            verbose_proxy_logger.warning("Failed to evict the team member default budget cache", exc_info=True)

    return saved


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
        where={"budget_id": {"in": data.budgets}, **REUSABLE_BUDGET},
    )

    return tuple(budget_response(row) for row in response)


@router.get(
    "/budget/settings",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def budget_settings(
    budget_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> tuple[ConfigList, ...]:
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
    db_budget_row: Final = await BudgetRepository(prisma_client).table.find_first(
        where={"budget_id": budget_id, **REUSABLE_BUDGET}
    )

    db_budget_row_dict: Final = db_budget_row.model_dump(exclude_none=True) if db_budget_row is not None else {}
    defaults: Final = _WRITE_VALUES.validate_python(BudgetNewRequest().model_dump())
    allowed_args: Final = {
        "max_parallel_requests": "Integer",
        "tpm_limit": "Integer",
        "rpm_limit": "Integer",
        "tpd_limit": "Integer",
        "budget_duration": "String",
        "max_budget": "Float",
        "soft_budget": "Float",
        "model_max_budget": "Object",
        "model_tpm_limit": "Object",
        "model_rpm_limit": "Object",
    }
    return tuple(
        ConfigList(
            field_name=field_name,
            field_type=allowed_args[field_name],
            field_description=field_info.description or "",
            field_value=_SETTING_VALUE.validate_python(db_budget_row_dict.get(field_name)),
            stored_in_db=True,
            field_default_value=defaults[field_name],
        )
        for field_name, field_info in BudgetNewRequest.model_fields.items()
        if field_name in allowed_args
    )


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

    response: Final = await BudgetRepository(prisma_client).table.find_many(
        where={**REUSABLE_BUDGET, "is_profile": False}
    )

    return tuple(budget_response(row) for row in response)


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
        store: Final = CustomerHierarchyStore(tx, utc_now())
        await tx.query_raw('SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = $1 FOR UPDATE', data.id)
        customer_org_count: Final = await store.organizations.count(where={"budget_ids": {"has": data.id}})
        if customer_org_count:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Delete organization templates through "
                    "/management/v1/customer_organizations/{customer_org_id}/default_budgets/{budget_id}."
                },
            )
        if await store.budgets.count(where={"budget_id": data.id, **OWNED_BUDGET}):
            raise HTTPException(
                status_code=409, detail={"error": "Remove an owned override through its person's assignment."}
            )
        referenced_profile: Final[Mapping[str, object]] = {
            "budget_id": data.id,
            "is_profile": True,
            "OR": ({"customer_members": {"some": _EMPTY_FILTER}}, {"user_profiles": {"some": _EMPTY_FILTER}}),
        }
        if await store.budgets.count(where=referenced_profile):
            raise HTTPException(
                status_code=409, detail={"error": "Detach every assignment before deleting this profile."}
            )
        response: Final = await store.budgets.delete(where={"budget_id": data.id})
    await invalidate_person_budget_sources((data.id,))

    return response

from datetime import datetime
from typing import Annotated, Final, Literal, TypeAlias

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict

from litellm.proxy.auth.person_budget import (
    PersonBudgetRecord,
    load_person_budget,
    person_budget_record,
    person_budget_windows,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    customer_actor,
    get_customer_hierarchy,
    require_customer_admin,
    resolve,
)
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerHierarchyStore,
    ErrorCode,
    Failure,
    PrismaRecord,
    effective_member_budget,
    related_budget,
    utc_datetime,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.carried_budget_state import PersonBudgetWindow
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, MemberPatch

SubjectType: TypeAlias = Literal["user", "customer_member"]
router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX,
    tags=["budget management"],
    dependencies=(Depends(require_customer_admin),),
)


class BudgetAssignment(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    subject_type: SubjectType
    subject_id: str
    budget_template_id: str | None
    budget_override: Budget | None
    budget_override_expires_at: datetime | None
    effective_budget: Budget
    windows: tuple[PersonBudgetWindow, ...]


def budget_assignment(
    record: PersonBudgetRecord, subject_type: SubjectType, subject_id: str, now: datetime
) -> BudgetAssignment:
    effective: Final = record.effective_budget(now)
    return BudgetAssignment(
        subject_type=subject_type,
        subject_id=subject_id,
        budget_template_id=record.budget_template_id,
        budget_override=related_budget(record.budget),
        budget_override_expires_at=record.budget_override_expires_at,
        effective_budget=effective,
        windows=person_budget_windows(effective, subject_type, subject_id, now),
    )


def assignment_patch_error(
    current: PersonBudgetRecord, template: Budget | None, patch: MemberPatch, now: datetime
) -> str | None:
    if patch.budget_override is None:
        return None
    changed: Final = patch.budget_override.model_fields_set
    if not changed.intersection({"temp_budget_increase", "temp_budget_expiry", "temp_budget_duration"}):
        return None
    stored: Final = related_budget(current.budget) or Budget()
    override: Final = Budget.model_validate(
        {**stored.model_dump(), **patch.budget_override.model_dump(exclude_unset=True)}
    )
    if override.temp_budget_increase is None:
        return None
    if override.temp_budget_expiry is None or override.temp_budget_duration is None:
        return "A personal temporary increase requires a duration and expiry"
    if utc_datetime(override.temp_budget_expiry) <= utc_datetime(now):
        return "A temporary increase must expire in the future"
    expiry: Final = (
        patch.budget_override_expires_at
        if "budget_override_expires_at" in patch.model_fields_set
        else current.budget_override_expires_at
    )
    if expiry is not None and utc_datetime(expiry) <= utc_datetime(now):
        return "Clear or extend the expired personal override before increasing it"
    effective: Final = effective_member_budget(template, override, expiry, now)
    if not any(limit.budget_duration == override.temp_budget_duration for limit in effective.budget_limits or ()):
        return "Configure a limit for the selected duration before increasing it"
    return None


async def _validate_assignment_patch(
    store: CustomerHierarchyStore, current: PersonBudgetRecord, patch: MemberPatch
) -> None:
    template_id: Final = patch.budget_id if "budget_id" in patch.model_fields_set else current.budget_template_id
    template: Final = (
        await store.budgets.find_unique(where={"budget_id": template_id}) if template_id is not None else None
    )
    error: Final = assignment_patch_error(
        current, related_budget(template.model_dump()) if template is not None else None, patch, store.now
    )
    if error is not None:
        raise HTTPException(status_code=422, detail=error)


async def update_user_budget(
    store: CustomerHierarchyStore,
    user_id: str,
    patch: MemberPatch,
    actor: str,
) -> None | Failure:
    users: Final[TableActions[PrismaRecord]] = store.db.litellm_usertable
    await store.db.query_raw('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = $1 FOR UPDATE', user_id)
    row: Final = await users.find_unique(where={"user_id": user_id}, include={"budget": True, "budget_template": True})
    if row is None:
        return Failure(ErrorCode.NOT_FOUND)
    current: Final = PersonBudgetRecord.model_validate(row.model_dump())
    await _validate_assignment_patch(store, current, patch)
    template_id: Final = patch.budget_id if "budget_id" in patch.model_fields_set else current.budget_template_id
    if template_id is not None:
        await store.db.query_raw(
            'SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = $1 FOR KEY SHARE', template_id
        )
        template: Final = await store.budgets.find_unique(where={"budget_id": template_id})
        if template is None or template.model_dump().get("is_profile") is not True:
            return Failure(ErrorCode.NOT_FOUND)
    budget_id: Final = (
        current.budget_id
        if "budget_override" not in patch.model_fields_set
        else await store.update_budget(current.budget_id, patch.budget_override, actor)
        if patch.budget_override is not None
        else None
    )
    expiry: Final = (
        patch.budget_override_expires_at
        if "budget_override_expires_at" in patch.model_fields_set
        else current.budget_override_expires_at
        if budget_id is not None
        else None
    )
    await users.update(
        where={"user_id": user_id},
        data={"budget_template_id": template_id, "budget_id": budget_id, "budget_override_expires_at": expiry},
    )
    await store.delete_released_budget(current.budget_id, budget_id)
    return None


async def _update_customer_budget(
    store: CustomerHierarchyStore, member_id: str, patch: MemberPatch, actor: str
) -> None | Failure:
    await store.db.query_raw(
        'SELECT customer_member_id FROM "LiteLLM_CustomerMemberTable" WHERE customer_member_id = $1 FOR UPDATE',
        member_id,
    )
    row: Final = await store.members.find_unique(
        where={"customer_member_id": member_id}, include={"budget": True, "budget_override": True}
    )
    if row is None:
        return Failure(ErrorCode.NOT_FOUND)
    await _validate_assignment_patch(store, person_budget_record(row.model_dump(), "customer_member"), patch)
    result: Final = await store.update_member(member_id, patch, actor)
    return result if isinstance(result, Failure) else None


@router.get(
    "/budget_assignments/{subject_type}/{subject_id}",
    response_model=BudgetAssignment,
    description="Read a user's or customer member's profile, personal overrides and calendar limits. Proxy admins only.",
)
async def get_budget_assignment(
    subject_type: SubjectType,
    subject_id: str,
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> BudgetAssignment:
    async def read(store: CustomerHierarchyStore) -> BudgetAssignment | Failure:
        record: Final = await load_person_budget(store.db, subject_type, subject_id)
        return (
            budget_assignment(record, subject_type, subject_id, store.now)
            if record is not None
            else Failure(ErrorCode.NOT_FOUND)
        )

    return resolve(await hierarchy.transaction(read))


@router.patch(
    "/budget_assignments/{subject_type}/{subject_id}",
    response_model=BudgetAssignment,
    description="Assign a profile or update a person's overrides, including a direct temporary increase. Proxy admins only.",
)
async def update_budget_assignment(
    subject_type: SubjectType,
    subject_id: str,
    data: MemberPatch,
    actor: Annotated[str, Depends(customer_actor)],
    hierarchy: Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)],
) -> BudgetAssignment:
    override: Final = data.budget_override
    if data.alias is not None:
        raise HTTPException(status_code=422, detail="Update the person's alias through their management endpoint")
    if subject_type == "user" and override is not None:
        unsupported: Final = override.model_dump(
            exclude_none=True,
            exclude={
                "budget_limits",
                "cap_policy",
                "temp_budget_increase",
                "temp_budget_expiry",
                "temp_budget_duration",
            },
        )
        if unsupported:
            raise HTTPException(
                status_code=422, detail="Personal user overrides support monthly and annual limits only"
            )
    if subject_type == "customer_member":
        resolve(await hierarchy.transaction(lambda store: _update_customer_budget(store, subject_id, data, actor)))
    else:
        resolve(await hierarchy.transaction(lambda store: update_user_budget(store, subject_id, data, actor)))
    return await get_budget_assignment(subject_type, subject_id, hierarchy)

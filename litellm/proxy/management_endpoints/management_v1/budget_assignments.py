"""Admin-only reads and conditional writes of one person's budget assignment."""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Final, cast

from fastapi import APIRouter, Depends
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from litellm.models.person_budget import (
    PersonBudgetAssignmentRecord,
    PersonBudgetCapPolicy,
    PersonBudgetDuration,
    PersonBudgetPolicy,
    PersonBudgetSubject,
)
from litellm.proxy.auth.person_budget import person_budget_windows
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings, get_budget_reset_settings
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    customer_actor,
    raise_public,
    require_customer_admin,
    resolve,
)
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchyStore,
    ErrorCode,
    Failure,
    MemberBudgetRecord,
    PrismaRecord,
    TransactionSource,
    assignment_revision,
    invalidate_person_budget_members,
    invalidate_person_budget_sources,
    merged_member_override,
    utc_now,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.management_endpoints.customer_hierarchy import MemberPatch


class BudgetAssignmentPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: str = Field(min_length=1, description="Revision returned by the last assignment read.")
    budget_id: str | None = Field(default=None, min_length=1)
    budget_override: PersonBudgetPolicy | None = None
    budget_override_expires_at: AwareDatetime | None = None


class BudgetAssignmentWindow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    budget_duration: PersonBudgetDuration
    max_budget: float
    reset_at: AwareDatetime
    cap_policy: PersonBudgetCapPolicy


class BudgetAssignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject_type: PersonBudgetSubject
    subject_id: str
    budget_id: str | None
    budget_override_id: str | None
    budget_override: PersonBudgetPolicy | None
    budget_override_expires_at: datetime | None
    effective_budget: PersonBudgetPolicy
    windows: tuple[BudgetAssignmentWindow, ...]
    revision: str


def assignment_response(
    subject_type: PersonBudgetSubject,
    subject_id: str,
    record: PersonBudgetAssignmentRecord,
    now: datetime,
    settings: BudgetResetSettings,
) -> BudgetAssignment:
    effective: Final = record.effective_budget(now)
    return BudgetAssignment(
        subject_type=subject_type,
        subject_id=subject_id,
        budget_id=record.budget_id,
        budget_override_id=record.budget_override_id,
        budget_override=(
            PersonBudgetPolicy.model_validate(record.budget_override, from_attributes=True)
            if record.budget_override is not None
            else None
        ),
        budget_override_expires_at=record.budget_override_expires_at,
        effective_budget=effective,
        windows=tuple(
            BudgetAssignmentWindow.model_validate(window)
            for window in person_budget_windows(effective, subject_type, subject_id, now, settings)
        ),
        revision=assignment_revision(record),
    )


class BudgetAssignments:
    def __init__(
        self,
        db: TransactionSource,
        clock: Callable[[], datetime] = utc_now,
        invalidate: Callable[[tuple[str, ...], str | None], Awaitable[None]] = invalidate_person_budget_sources,
        settings: BudgetResetSettings = BudgetResetSettings(),
        invalidate_members: Callable[[tuple[str, ...]], Awaitable[None]] = invalidate_person_budget_members,
    ) -> None:
        self.db = db
        self.clock = clock
        self.invalidate = invalidate
        self.settings = settings
        self.invalidate_members = invalidate_members

    @staticmethod
    def table(store: CustomerHierarchyStore, subject_type: PersonBudgetSubject) -> TableActions[PrismaRecord]:
        if subject_type == "customer_member":
            return store.members
        return cast(  # cast-ok: Prisma actions implement TableActions with narrower generated mapping inputs
            TableActions[PrismaRecord], store.db.litellm_usertable
        )

    async def read(
        self, store: CustomerHierarchyStore, subject_type: PersonBudgetSubject, subject_id: str
    ) -> PersonBudgetAssignmentRecord | Failure:
        table_name, id_field = (
            ("LiteLLM_UserTable", "user_id")
            if subject_type == "user"
            else ("LiteLLM_CustomerMemberTable", "customer_member_id")
        )
        await store.db.query_raw(
            f'SELECT "{id_field}" FROM "{table_name}" WHERE "{id_field}" = $1 FOR UPDATE', subject_id
        )
        table: Final = self.table(store, subject_type)
        row: Final = await table.find_unique(where={id_field: subject_id})
        if row is None:
            return Failure(ErrorCode.NOT_FOUND)
        initial: Final = PersonBudgetAssignmentRecord.model_validate(row.model_dump())
        await store.lock_budgets(
            tuple(item for item in (initial.budget_id, initial.budget_override_id) if item is not None)
        )
        related: Final = await table.find_unique(
            where={id_field: subject_id}, include={"budget": True, "budget_override": True}
        )
        return (
            PersonBudgetAssignmentRecord.model_validate(related.model_dump())
            if related is not None
            else Failure(ErrorCode.NOT_FOUND)
        )

    async def get(self, subject_type: PersonBudgetSubject, subject_id: str) -> BudgetAssignment | Failure:
        async with self.db.tx() as tx:
            now: Final = self.clock()
            record: Final = await self.read(CustomerHierarchyStore(tx, now), subject_type, subject_id)
            return (
                record
                if isinstance(record, Failure)
                else assignment_response(subject_type, subject_id, record, now, self.settings)
            )

    async def patch(
        self, subject_type: PersonBudgetSubject, subject_id: str, payload: BudgetAssignmentPatch, actor: str
    ) -> BudgetAssignment | Failure:
        from prisma.errors import ForeignKeyViolationError

        try:
            async with self.db.tx() as tx:
                store: Final = CustomerHierarchyStore(tx, self.clock())
                existing: Final = await self.read(store, subject_type, subject_id)
                if isinstance(existing, Failure):
                    return existing
                if payload.revision != assignment_revision(existing):
                    return Failure(ErrorCode.REVISION_CONFLICT)
                if payload.budget_id is not None:
                    await store.lock_budgets((payload.budget_id,))
                    template: Final = await store.person_definition(payload.budget_id)
                    if template is None or not template.is_profile:
                        return Failure(ErrorCode.INVALID_BUDGET, "Select an existing reusable budget profile.")
                member_patch: Final = MemberPatch.model_validate(
                    payload.model_dump(exclude_unset=True, exclude={"revision"})
                )
                result: Final = (
                    await store.update_member(subject_id, member_patch, actor)
                    if subject_type == "customer_member"
                    else await self.update_user(store, subject_id, existing, member_patch, actor)
                )
                if isinstance(result, Failure):
                    return result
                updated: Final = await self.read(store, subject_type, subject_id)
                if isinstance(updated, Failure):
                    return updated
                response: Final = assignment_response(subject_type, subject_id, updated, store.now, self.settings)
            await self.invalidate(store.invalidated_budget_ids, subject_id if subject_type == "user" else None)
            if store.invalidated_member_ids:
                await self.invalidate_members(store.invalidated_member_ids)
            return response
        except ForeignKeyViolationError:
            return Failure(ErrorCode.REVISION_CONFLICT)

    async def update_user(
        self,
        store: CustomerHierarchyStore,
        user_id: str,
        existing: PersonBudgetAssignmentRecord,
        patch: MemberPatch,
        actor: str,
    ) -> None | Failure:
        selected: Final = patch.budget_id if "budget_id" in patch.model_fields_set else existing.budget_id
        current_override: Final = (
            MemberBudgetRecord.model_validate(existing.budget_override, from_attributes=True)
            if existing.budget_override is not None
            else None
        )
        override: Final = (
            merged_member_override(current_override, patch.budget_override)
            if patch.budget_override is not None
            else None
            if "budget_override" in patch.model_fields_set
            else current_override
        )
        if (
            override is None
            and patch.budget_override_expires_at is not None
            and "budget_override" not in patch.model_fields_set
        ):
            return Failure(ErrorCode.INVALID_BUDGET, "An override expiration requires an override.")
        expiry: Final = (
            None
            if override is None
            else patch.budget_override_expires_at
            if "budget_override_expires_at" in patch.model_fields_set
            else existing.budget_override_expires_at
        )
        invalid: Final = await store.validate_member_override(
            selected,
            override,
            expiry,
            frozenset(patch.budget_override.model_fields_set) if patch.budget_override is not None else frozenset(),
        )
        if invalid is not None:
            return invalid
        override_id: Final = (
            await store.update_budget(existing.budget_override_id, override, actor)
            if override is not None and "budget_override" in patch.model_fields_set
            else existing.budget_override_id
            if override is not None
            else None
        )
        await self.table(store, "user").update(
            where={"user_id": user_id},
            data={
                "budget_id": selected,
                "budget_override_id": override_id,
                "budget_override_expires_at": expiry,
            },
        )
        await store.delete_released_budget(existing.budget_override_id, override_id)
        return None


def get_budget_assignments() -> BudgetAssignments:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise_public(Failure(ErrorCode.DATABASE_UNAVAILABLE))
    return BudgetAssignments(prisma_client, settings=get_budget_reset_settings())


router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX, tags=["budget management"], dependencies=(Depends(require_customer_admin),)
)


@router.get("/budget_assignments/{subject_type}/{subject_id}", response_model=BudgetAssignment)
async def get_budget_assignment(
    subject_type: PersonBudgetSubject,
    subject_id: str,
    assignments: Annotated[BudgetAssignments, Depends(get_budget_assignments)],
) -> BudgetAssignment:
    """Read a person's profile, owned override, effective calendar windows and revision. Proxy admins only."""
    return resolve(await assignments.get(subject_type, subject_id))


@router.patch("/budget_assignments/{subject_type}/{subject_id}", response_model=BudgetAssignment)
async def patch_budget_assignment(
    subject_type: PersonBudgetSubject,
    subject_id: str,
    data: BudgetAssignmentPatch,
    actor: Annotated[str, Depends(customer_actor)],
    assignments: Annotated[BudgetAssignments, Depends(get_budget_assignments)],
) -> BudgetAssignment:
    """Conditionally patch an assignment. Omitted fields preserve values; null clears. Stale revisions return 409."""
    return resolve(await assignments.patch(subject_type, subject_id, data, actor))

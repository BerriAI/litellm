from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Final, Literal, NoReturn, TypeAlias, TypeVar
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import assert_never

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.skill_submission import SubmissionStatus
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.person_budget import PersonBudgetRecord, load_person_budget
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.discovery_endpoints.skill_hub_mutations import (
    SkillMutationError,
    submission_editor,
    submission_owner,
)
from litellm.proxy.management_endpoints.management_v1.budget_assignments import (
    BudgetAssignment,
    budget_assignment,
    update_user_budget,
)
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import get_customer_hierarchy, resolve
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerHierarchyStore,
    Failure,
    utc_datetime,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, MemberPatch

router: Final = APIRouter(prefix=f"{MANAGEMENT_V1_PREFIX}/budget_increase_requests", tags=["budget management"])


class BudgetIncreaseInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    amount: float = Field(gt=0, allow_inf_nan=False, description="Temporary additional USD, without resetting spend.")
    duration: Literal["1mo", "12mo"] = Field(description="Personal calendar-month or calendar-year limit to increase.")
    expires_at: AwareDatetime = Field(
        description="Absolute expiry with timezone, which must still be future at approval."
    )
    reason: str = Field(min_length=1, max_length=2000, description="Reason shown to the reviewing proxy administrator.")


class BudgetIncreaseRequest(BudgetIncreaseInput):
    id: str
    submitted_by: str
    status: SubmissionStatus | Literal["expired"]
    created_at: datetime
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    rejection_reason: str | None = None


class BudgetIncreaseList(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: tuple[BudgetIncreaseRequest, ...]
    has_more: bool


class BudgetIncreaseRejection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    reason: str = Field(min_length=1, max_length=2000, description="Explanation visible to the requester.")


_ROWS: Final = TypeAdapter(tuple[BudgetIncreaseRequest, ...])
_COUNTS: Final = TypeAdapter(tuple[Mapping[str, int], ...])
_HOURLY_REQUEST_LIMIT: Final = 10
_COLUMNS: Final = "id, submitted_by, amount, duration, expires_at, reason, status, created_at, reviewed_by, reviewed_at, rejection_reason"
_SELECT: Final = f'SELECT {_COLUMNS} FROM "LiteLLM_BudgetIncreaseRequest"'
Caller: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
Hierarchy: TypeAlias = Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)]
T: Final = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RequestFailure:
    kind: Literal["forbidden", "missing", "conflict", "limited"]
    detail: str


def raise_public(failure: RequestFailure) -> NoReturn:
    match failure.kind:
        case "forbidden":
            raise HTTPException(403, failure.detail)
        case "missing":
            raise HTTPException(404, failure.detail)
        case "conflict":
            raise HTTPException(409, failure.detail)
        case "limited":
            raise HTTPException(429, failure.detail)
        case _:
            assert_never(failure.kind)


def request_result(result: T | RequestFailure | Failure) -> T:
    if isinstance(result, RequestFailure):
        raise_public(result)
    return resolve(result)


def _identity(caller: UserAPIKeyAuth) -> str | RequestFailure:
    identity: Final = submission_editor(caller)
    if isinstance(identity, SkillMutationError):
        return RequestFailure("forbidden", identity.message)
    return identity


def _owner(caller: UserAPIKeyAuth) -> str | None | RequestFailure:
    owner: Final = submission_owner(caller)
    if isinstance(owner, SkillMutationError):
        return RequestFailure("forbidden", owner.message)
    return owner


def _visible(row: BudgetIncreaseRequest, now: datetime) -> BudgetIncreaseRequest:
    return (
        row.model_copy(update={"status": "expired"})
        if row.status == "pending_review" and utc_datetime(row.expires_at) <= utc_datetime(now)
        else row
    )


async def _person(
    store: CustomerHierarchyStore, user_id: str, *, lock: bool = False
) -> PersonBudgetRecord | RequestFailure:
    if lock:
        await store.db.query_raw('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = $1 FOR UPDATE', user_id)
    record: Final = await load_person_budget(store.db, "user", user_id)
    if record is None:
        return RequestFailure("missing", "Internal user budget not found")
    return record


def _eligible(record: PersonBudgetRecord, data: BudgetIncreaseInput, now: datetime) -> RequestFailure | None:
    if data.expires_at <= utc_datetime(now):
        return RequestFailure("conflict", "The requested increase has expired. Submit a new request.")
    override_expiry: Final = record.budget_override_expires_at
    if (
        record.budget is not None
        and override_expiry is not None
        and utc_datetime(now) < utc_datetime(override_expiry) < data.expires_at
    ):
        return RequestFailure("conflict", "The requested increase cannot outlast the personal budget override.")
    effective: Final = record.effective_budget(now)
    if effective.cap_policy != "pause_allow_increase":
        return RequestFailure(
            "conflict", "This budget requires waiting until reset and does not allow increase requests."
        )
    if not any(limit.budget_duration == data.duration for limit in effective.budget_limits or ()):
        return RequestFailure("conflict", "The selected personal budget limit is no longer configured.")
    if LiteLLM_BudgetTable.model_validate(effective.model_dump()).active_temp_budget_increase(utc_datetime(now)) > 0:
        return RequestFailure(
            "conflict", "A temporary increase is already active. Wait until it expires or ask an admin to edit it."
        )
    return None


def _grant_patch(record: PersonBudgetRecord, row: BudgetIncreaseRequest, now: datetime) -> MemberPatch:
    increase: Final = Budget(
        temp_budget_increase=row.amount,
        temp_budget_expiry=row.expires_at,
        temp_budget_duration=row.duration,
    )
    expiry: Final = record.budget_override_expires_at
    if record.budget is None or (expiry is not None and utc_datetime(expiry) <= utc_datetime(now)):
        return MemberPatch(
            budget_override=Budget.model_validate(increase.model_dump()),
            budget_override_expires_at=None,
        )
    return MemberPatch(budget_override=increase)


async def _get(
    store: CustomerHierarchyStore, request_id: str, owner: str | None, *, lock: bool = False
) -> BudgetIncreaseRequest | RequestFailure:
    rows: Final = _ROWS.validate_python(
        await store.db.query_raw(
            _SELECT + " WHERE id = $1 AND ($2::text IS NULL OR submitted_by = $2)" + (" FOR UPDATE" if lock else ""),
            request_id,
            owner,
        )
    )
    if not rows:
        return RequestFailure("missing", "Budget increase request not found")
    return rows[0]


async def submit_increase(
    store: CustomerHierarchyStore, user_id: str, data: BudgetIncreaseInput
) -> BudgetIncreaseRequest | RequestFailure:
    person: Final = await _person(store, user_id, lock=True)
    if isinstance(person, RequestFailure):
        return person
    eligibility: Final = _eligible(person, data, store.now)
    if eligibility is not None:
        return eligibility
    await store.db.execute_raw(
        """UPDATE "LiteLLM_BudgetIncreaseRequest" SET status = 'expired', pending_user_id = NULL
        WHERE submitted_by = $1 AND status = 'pending_review' AND expires_at <= $2::text::timestamptz""",
        user_id,
        utc_datetime(store.now).isoformat(),
    )
    pending: Final = _ROWS.validate_python(
        await store.db.query_raw(
            _SELECT + " WHERE pending_user_id = $1",
            user_id,
        )
    )
    if pending:
        previous: Final = BudgetIncreaseInput.model_validate(
            pending[0].model_dump(include={field: True for field in BudgetIncreaseInput.model_fields})
        )
        if previous == data:
            return pending[0]
        return RequestFailure("conflict", "You already have a pending budget increase request.")
    recent: Final = _COUNTS.validate_python(
        await store.db.query_raw(
            """SELECT COUNT(*) AS count FROM "LiteLLM_BudgetIncreaseRequest"
            WHERE submitted_by = $1 AND created_at > $2::text::timestamptz""",
            user_id,
            (utc_datetime(store.now) - timedelta(hours=1)).isoformat(),
        )
    )[0]["count"]
    if recent >= _HOURLY_REQUEST_LIMIT:
        return RequestFailure("limited", "You can submit up to 10 budget increase requests per hour. Try again later.")
    rows: Final = _ROWS.validate_python(
        await store.db.query_raw(
            f"""INSERT INTO "LiteLLM_BudgetIncreaseRequest"
        (id, submitted_by, pending_user_id, amount, duration, expires_at, reason, status, created_at)
        VALUES ($1, $2, $2, $3, $4, $5::text::timestamptz, $6, 'pending_review', $7::text::timestamptz)
        RETURNING {_COLUMNS}""",
            str(uuid4()),
            user_id,
            data.amount,
            data.duration,
            data.expires_at.isoformat(),
            data.reason,
            utc_datetime(store.now).isoformat(),
        )
    )
    return rows[0]


async def decide_increase(
    store: CustomerHierarchyStore,
    request_id: str,
    reviewer: str,
    approve: bool,
    reason: str | None,
) -> BudgetIncreaseRequest | RequestFailure | Failure:
    snapshot: Final = await _get(store, request_id, None)
    if isinstance(snapshot, RequestFailure):
        return snapshot
    person: Final = await _person(store, snapshot.submitted_by, lock=True)
    if isinstance(person, RequestFailure):
        return person
    row: Final = await _get(store, request_id, None, lock=True)
    if isinstance(row, RequestFailure):
        return row
    if row.submitted_by == reviewer:
        return RequestFailure("forbidden", "You cannot review your own budget increase request.")
    target: Final = "approved" if approve else "rejected"
    if row.status == target:
        return row
    if row.status != "pending_review":
        return RequestFailure("conflict", "This request already has a decision.")
    if row.expires_at <= utc_datetime(store.now):
        return RequestFailure("conflict", "This request has expired.")
    if approve:
        eligibility: Final = _eligible(person, row, store.now)
        if eligibility is not None:
            return eligibility
        applied: Final = await update_user_budget(
            store,
            row.submitted_by,
            _grant_patch(person, row, store.now),
            reviewer,
        )
        if isinstance(applied, Failure):
            return applied
    rows: Final = _ROWS.validate_python(
        await store.db.query_raw(
            f"""UPDATE "LiteLLM_BudgetIncreaseRequest" SET status = $2, reviewed_by = $3,
        reviewed_at = $4::text::timestamptz, rejection_reason = $5, pending_user_id = NULL WHERE id = $1 RETURNING {_COLUMNS}""",
            request_id,
            target,
            reviewer,
            utc_datetime(store.now).isoformat(),
            reason,
        )
    )
    return rows[0]


@router.get("/me", response_model=BudgetAssignment)
async def own_budget(caller: Caller, hierarchy: Hierarchy) -> BudgetAssignment:
    """Read the authenticated internal user's personal limits, even when inference is paused."""
    identity: Final = request_result(_identity(caller))

    async def read(store: CustomerHierarchyStore) -> BudgetAssignment | RequestFailure:
        person: Final = await _person(store, identity)
        return person if isinstance(person, RequestFailure) else budget_assignment(person, "user", identity, store.now)

    return request_result(await hierarchy.transaction(read))


@router.post("", response_model=BudgetIncreaseRequest, status_code=201)
async def request_increase(data: BudgetIncreaseInput, caller: Caller, hierarchy: Hierarchy) -> BudgetIncreaseRequest:
    """Request a temporary increase for the authenticated internal user. Only one request may be pending."""
    identity: Final = request_result(_identity(caller))
    return request_result(await hierarchy.transaction(lambda store: submit_increase(store, identity, data)))


@router.get("", response_model=BudgetIncreaseList)
async def list_increases(
    caller: Caller,
    hierarchy: Hierarchy,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum requests in this page.")] = 20,
    offset: Annotated[int, Query(ge=0, description="Number of requests to skip.")] = 0,
) -> BudgetIncreaseList:
    """List owned requests. Proxy admins and admin viewers can read all requests."""
    owner: Final = request_result(_owner(caller))

    async def read(store: CustomerHierarchyStore) -> BudgetIncreaseList:
        rows: Final = _ROWS.validate_python(
            await store.db.query_raw(
                _SELECT
                + " WHERE ($1::text IS NULL OR submitted_by = $1) ORDER BY created_at DESC, id DESC LIMIT $2 OFFSET $3",
                owner,
                limit + 1,
                offset,
            )
        )
        return BudgetIncreaseList(
            requests=tuple(_visible(row, store.now) for row in rows[:limit]), has_more=len(rows) > limit
        )

    return resolve(await hierarchy.transaction(read))


@router.get("/{request_id}", response_model=BudgetIncreaseRequest)
async def get_increase(request_id: str, caller: Caller, hierarchy: Hierarchy) -> BudgetIncreaseRequest:
    """Read an owned request and its decision. Proxy admins and admin viewers may read any request."""
    owner: Final = request_result(_owner(caller))

    async def read(store: CustomerHierarchyStore) -> BudgetIncreaseRequest | RequestFailure:
        row: Final = await _get(store, request_id, owner)
        return row if isinstance(row, RequestFailure) else _visible(row, store.now)

    return request_result(await hierarchy.transaction(read))


async def _review(
    request_id: str,
    caller: UserAPIKeyAuth,
    hierarchy: CustomerHierarchy,
    approve: bool,
    reason: str | None,
) -> BudgetIncreaseRequest | RequestFailure | Failure:
    if not is_proxy_admin(caller):
        return RequestFailure("forbidden", "Only proxy admins may review budget increase requests.")
    reviewer: Final = _identity(caller)
    if isinstance(reviewer, RequestFailure):
        return reviewer
    return await hierarchy.transaction(lambda store: decide_increase(store, request_id, reviewer, approve, reason))


@router.put("/{request_id}/approve", response_model=BudgetIncreaseRequest)
async def approve_increase(request_id: str, caller: Caller, hierarchy: Hierarchy) -> BudgetIncreaseRequest:
    """Approve a pending request and apply its temporary increase atomically. A different proxy admin must review."""
    return request_result(await _review(request_id, caller, hierarchy, True, None))


@router.put("/{request_id}/reject", response_model=BudgetIncreaseRequest)
async def reject_increase(
    request_id: str,
    data: BudgetIncreaseRejection,
    caller: Caller,
    hierarchy: Hierarchy,
) -> BudgetIncreaseRequest:
    """Reject a pending request with an explanation, without changing its budget. Proxy admins only."""
    return request_result(await _review(request_id, caller, hierarchy, False, data.reason))

"""Temporary budget increases for paused customer members.

Same request and approval shape as MCP server and skill submissions: a key acting for the member files a request that waits
in ``pending_review``, a proxy admin approves or rejects it, and an approval lands as the member's budget override.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Final, Literal, TypeAlias, TypeVar
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request
from pydantic import TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import forget_customer_hierarchies
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    customer_actor,
    get_customer_hierarchy,
    get_user_api_key_cache,
    list_response,
    resolve,
)
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerHierarchyStore,
    ErrorCode,
    Failure,
    Page,
    PrismaRecord,
)
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator is untyped
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    BudgetIncreaseRejection,
    BudgetIncreaseRequest,
    BudgetIncreaseRequestCreate,
    CustomerMember,
    IncreaseRequestStatus,
    MemberPatch,
)
from litellm.types.proxy.management_endpoints.management_v1 import ListResponse

router: Final = APIRouter(
    prefix=f"{MANAGEMENT_V1_PREFIX}/customer_budget_increase_requests", tags=["customer management"]
)

Hierarchy: TypeAlias = Annotated[CustomerHierarchy, Depends(get_customer_hierarchy)]
Caller: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
Admin: TypeAlias = Annotated[str, Depends(customer_actor)]
Cache: TypeAlias = Annotated[UserApiKeyCache, Depends(get_user_api_key_cache)]

T: Final = TypeVar("T")
_REQUEST: Final = TypeAdapter(BudgetIncreaseRequest)
_REQUESTS: Final = TypeAdapter(tuple[BudgetIncreaseRequest, ...])


def _requests_table(store: CustomerHierarchyStore) -> TableActions[PrismaRecord]:
    return store.db.litellm_customerbudgetincreaserequest  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]  # prisma tables are generated at build time


async def _in_transaction(
    hierarchy: CustomerHierarchy, operation: Callable[[CustomerHierarchyStore], Awaitable[T]]
) -> T:
    """Every request-table read and write runs through the hierarchy's transaction, which also maps Prisma errors."""
    return resolve(await hierarchy.transaction(operation))


def _allows_increase(member: CustomerMember) -> bool:
    return member.effective_budget.cap_policy == "pause_allow_increase"


def _active_override(member: CustomerMember, now: datetime) -> tuple[Budget, datetime | None]:
    expires_at: Final = member.budget_override_expires_at
    if member.budget_override is None or (expires_at is not None and expires_at <= now):
        return Budget(), None
    return member.budget_override, expires_at


def _grant(member: CustomerMember, request: BudgetIncreaseRequest, now: datetime) -> MemberPatch:
    """The approved cap rides on the member's active override and ends no later than that override already did,
    so model and rate grants are never renewed or revived by a budget approval."""
    current, current_expiry = _active_override(member, now)
    requested_expiry: Final = request.expires_at if request.expires_at is not None else current_expiry
    expiry: Final = (
        requested_expiry
        if current_expiry is None or requested_expiry is None
        else min(current_expiry, requested_expiry)
    )
    return MemberPatch(
        budget_override=Budget.model_validate({**current.model_dump(), "max_budget": request.requested_max_budget}),
        budget_override_expires_at=expiry,
    )


def _check_key_scope(caller: UserAPIKeyAuth, member: CustomerMember) -> None:
    """A key whose metadata pins ``customer_org_id`` may only file for members of that organization."""
    key_customer_org_id: Final = (caller.metadata or {}).get("customer_org_id")  # pyright: ignore[reportUnknownMemberType]  # UserAPIKeyAuth.metadata is an untyped dict
    if key_customer_org_id is not None and key_customer_org_id != member.parent_org_id:
        resolve(Failure(ErrorCode.KEY_NOT_IN_ORG))


@router.post("", response_model=BudgetIncreaseRequest, status_code=201)
@management_endpoint_wrapper
async def request_budget_increase(
    data: BudgetIncreaseRequestCreate, caller: Caller, hierarchy: Hierarchy
) -> BudgetIncreaseRequest:
    """File a temporary increase for a member whose budget is set to pause_allow_increase. Waits for an admin."""
    member: Final = resolve(await hierarchy.get_member(data.customer_member_id))
    _check_key_scope(caller, member)
    if not _allows_increase(member):
        resolve(Failure(ErrorCode.INCREASE_NOT_ALLOWED))
    now: Final = hierarchy.clock()
    if data.expires_at <= now:
        resolve(Failure(ErrorCode.EXPIRY_IN_PAST))
    pending_count: Final = await _in_transaction(
        hierarchy,
        lambda store: _requests_table(store).count(
            where={"customer_member_id": data.customer_member_id, "status": "pending_review"}
        ),
    )
    if pending_count > 0:
        resolve(Failure(ErrorCode.REQUEST_ALREADY_PENDING))
    row: Final = await _in_transaction(
        hierarchy,
        lambda store: _requests_table(store).create(
            data={
                "request_id": str(uuid4()),
                "customer_member_id": data.customer_member_id,
                "requested_max_budget": data.requested_max_budget,
                "expires_at": data.expires_at,
                "reason": data.reason,
                "status": "pending_review",
                "submitted_by": caller.user_id or caller.key_alias or "unknown",
                "submitted_at": now,
            }
        ),
    )
    return _REQUEST.validate_python(row, from_attributes=True)


@router.get("", response_model=ListResponse[BudgetIncreaseRequest])
@management_endpoint_wrapper
async def list_budget_increase_requests(
    request: Request,
    actor: Admin,
    hierarchy: Hierarchy,
    status: Annotated[IncreaseRequestStatus | None, Query()] = None,
    customer_member_id: Annotated[str | None, Query(max_length=256)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ListResponse[BudgetIncreaseRequest]:
    """The admin review queue. Proxy admins only."""
    where: Final = {
        **({"status": status} if status is not None else {}),
        **({"customer_member_id": customer_member_id} if customer_member_id is not None else {}),
    }
    rows: Final = await _in_transaction(
        hierarchy,
        lambda store: _requests_table(store).find_many(
            where=where, order=[{"submitted_at": "desc"}], skip=(page - 1) * page_size, take=page_size
        ),
    )
    total: Final = await _in_transaction(hierarchy, lambda store: _requests_table(store).count(where=where))
    return list_response(
        Page(data=_REQUESTS.validate_python(rows, from_attributes=True), total_count=total), request, page, page_size
    )


async def _claim(
    store: CustomerHierarchyStore,
    request_id: str,
    actor: str,
    status: Literal["approved", "rejected"],
    notes: str | None,
) -> BudgetIncreaseRequest:
    """Flip the request out of ``pending_review`` only if it is still there, so two reviewers cannot both win."""
    claimed: Final = await _requests_table(store).update_many(
        where={"request_id": request_id, "status": "pending_review"},
        data={"status": status, "reviewed_by": actor, "reviewed_at": store.now, "review_notes": notes},
    )
    row: Final = await _requests_table(store).find_unique(where={"request_id": request_id})
    if row is None:
        resolve(Failure(ErrorCode.NOT_FOUND))
    if claimed != 1:
        resolve(Failure(ErrorCode.REQUEST_NOT_PENDING))
    return _REQUEST.validate_python(row, from_attributes=True)


async def _approve(store: CustomerHierarchyStore, request_id: str, actor: str) -> BudgetIncreaseRequest:
    approved: Final = await _claim(store, request_id, actor, "approved", None)
    member: Final = resolve(await store.get_member(approved.customer_member_id))
    resolve(await store.update_member(approved.customer_member_id, _grant(member, approved, store.now), actor))
    return approved


@router.get("/{request_id}", response_model=BudgetIncreaseRequest)
@management_endpoint_wrapper
async def get_budget_increase_request(request_id: str, actor: Admin, hierarchy: Hierarchy) -> BudgetIncreaseRequest:
    """One request. Proxy admins only."""
    row: Final = await _in_transaction(
        hierarchy, lambda store: _requests_table(store).find_unique(where={"request_id": request_id})
    )
    if row is None:
        resolve(Failure(ErrorCode.NOT_FOUND))
    return _REQUEST.validate_python(row, from_attributes=True)


@router.put("/{request_id}/approve", response_model=BudgetIncreaseRequest)
@management_endpoint_wrapper
async def approve_budget_increase_request(
    request_id: str, actor: Admin, hierarchy: Hierarchy, cache: Cache
) -> BudgetIncreaseRequest:
    """Grant the increase: the member's override takes the requested cap until the request's expiry. Proxy admins only.

    The claim and the override land in one transaction, so a racing reject sees ``request_not_pending`` instead of
    leaving a rejected request with the cap still granted."""
    approved: Final = await _in_transaction(hierarchy, lambda store: _approve(store, request_id, actor))
    await forget_customer_hierarchies((approved.customer_member_id,), cache)
    return approved


@router.put("/{request_id}/reject", response_model=BudgetIncreaseRequest)
@management_endpoint_wrapper
async def reject_budget_increase_request(
    request_id: str, actor: Admin, hierarchy: Hierarchy, body: BudgetIncreaseRejection | None = None
) -> BudgetIncreaseRequest:
    """Decline the increase and leave the member paused, with optional review notes. Proxy admins only."""
    notes: Final = None if body is None else body.review_notes
    return await _in_transaction(hierarchy, lambda store: _claim(store, request_id, actor, "rejected", notes))

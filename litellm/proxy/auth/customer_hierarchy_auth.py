"""Request-path resolution of the customer organization, team and member a request runs under.

A request names its customer member with the ``x-litellm-customer-member-id`` header. The member row
carries its team and that team's organization, so one lookup yields every level whose budget and rate
limits apply to the request.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final

from fastapi import status
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_checks import can_object_call_model
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, get_management_object_ttl
from litellm.proxy.management_helpers.customer_hierarchy import (
    PrismaRecord,
    effective_member_budget,
    related_budget,
    utc_now,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth, CustomerScopeKey
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router

CUSTOMER_MEMBER_ID_HEADER: Final = "x-litellm-customer-member-id"
_MEMBER_INCLUDE: Final = {
    "budget": True,
    "budget_template": True,
    "parent_team": {"include": {"budget": True, "parent_org": {"include": {"budget": True}}}},
}
_EXPIRY: Final[TypeAdapter[datetime | None]] = TypeAdapter(datetime | None)


class _OrganizationRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_org_id: str
    spend: float = 0.0
    budget: Mapping[str, object] | None = None


class _TeamRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_team_id: str
    spend: float = 0.0
    budget: Mapping[str, object] | None = None
    parent_org: _OrganizationRow


class _MemberRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_member_id: str
    spend: float = 0.0
    budget: Mapping[str, object] | None = None
    budget_template: Mapping[str, object] | None = None
    budget_override_expires_at: datetime | None = None
    parent_team: _TeamRow


def customer_member_cache_key(customer_member_id: str) -> str:
    return f"customer_member_id:{customer_member_id}"


def customer_member_id_from_headers(headers: Mapping[str, object] | None) -> str | None:
    if headers is None:
        return None
    values: Final = tuple(
        value.strip()
        for name, value in headers.items()
        if name.lower() == CUSTOMER_MEMBER_ID_HEADER and isinstance(value, str) and value.strip()
    )
    return values[0] if values else None


def _scope(key: CustomerScopeKey, entity_id: str, spend: float, budget: Budget | None) -> CustomerScopeAuth:
    if budget is None:
        return CustomerScopeAuth(key=key, entity_id=entity_id, spend=spend)
    return CustomerScopeAuth(
        key=key,
        entity_id=entity_id,
        spend=spend,
        max_budget=budget.max_budget,
        tpm_limit=budget.tpm_limit,
        rpm_limit=budget.rpm_limit,
        allowed_models=budget.allowed_models or None,
        model_tpm_limit=budget.model_tpm_limit,
        model_rpm_limit=budget.model_rpm_limit,
    )


def customer_hierarchy_from_member_row(row: Mapping[str, object], now: datetime) -> CustomerHierarchyAuth:
    """The org, team and member scopes of one member row loaded with ``_MEMBER_INCLUDE``. The member's limits are
    its effective budget: the selected organization template under any unexpired override."""
    member: Final = _MemberRow.model_validate(row)
    team: Final = member.parent_team
    organization: Final = team.parent_org
    template: Final = related_budget(member.budget_template, now)
    override: Final = related_budget(member.budget, now, template.max_budget if template is not None else None)
    return CustomerHierarchyAuth(
        customer_member_id=member.customer_member_id,
        customer_team_id=team.customer_team_id,
        customer_org_id=organization.customer_org_id,
        scopes=(
            _scope(
                "customer_org",
                organization.customer_org_id,
                organization.spend,
                related_budget(organization.budget, now),
            ),
            _scope("customer_team", team.customer_team_id, team.spend, related_budget(team.budget, now)),
            _scope(
                "customer_member",
                member.customer_member_id,
                member.spend,
                effective_member_budget(template, override, member.budget_override_expires_at, now),
            ),
        ),
    )


def _member_table(prisma_client: "PrismaClient") -> TableActions[PrismaRecord]:
    return prisma_client.db.litellm_customermembertable  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]  # prisma tables are generated at build time


async def get_customer_hierarchy(
    customer_member_id: str,
    prisma_client: "PrismaClient",
    user_api_key_cache: UserApiKeyCache,
    now: datetime | None = None,
) -> CustomerHierarchyAuth | None:
    cache_key: Final = customer_member_cache_key(customer_member_id)
    cached: Final = await user_api_key_cache.async_get_cache(key=cache_key, model_type=CustomerHierarchyAuth)
    if cached is not None:
        return cached
    row: Final = await _member_table(prisma_client).find_unique(
        where={"customer_member_id": customer_member_id}, include=_MEMBER_INCLUDE
    )
    if row is None:
        return None
    hierarchy: Final = customer_hierarchy_from_member_row(row.model_dump(), now if now is not None else utc_now())
    await user_api_key_cache.async_set_cache(
        key=cache_key, value=hierarchy, ttl=get_management_object_ttl(user_api_key_cache)
    )
    return hierarchy


def check_customer_hierarchy_budget(hierarchy: CustomerHierarchyAuth) -> None:
    """Raises ``BudgetExceededError`` for the first level, org first, whose spend has reached its ``max_budget``."""
    for scope in hierarchy.scopes:
        if scope.max_budget is not None and scope.spend >= scope.max_budget:
            raise litellm.BudgetExceededError(
                current_cost=scope.spend,
                max_budget=scope.max_budget,
                message=(
                    f"ExceededBudget: {scope.key}={scope.entity_id} over budget. "
                    f"Spend={scope.spend}, Budget={scope.max_budget}"
                ),
                entity_type=scope.key,
                entity_id=scope.entity_id,
            )


def check_customer_hierarchy_model_access(
    hierarchy: CustomerHierarchyAuth, model: str, llm_router: "Router | None"
) -> None:
    """Every level with an ``allowed_models`` list must permit ``model``; a level without one permits every model."""
    for scope in hierarchy.scopes:
        if scope.allowed_models:
            can_object_call_model(
                model=model, llm_router=llm_router, models=list(scope.allowed_models), object_type="customer"
            )


async def resolve_customer_hierarchy(
    customer_member_id: str,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    llm_router: "Router | None",
    request_model: object,
    route: str,
) -> CustomerHierarchyAuth:
    """Load the hierarchy the header names and admit the request against it. An unknown member is refused
    rather than treated as unrestricted: the caller asked for a budgeted identity that does not exist."""
    hierarchy: Final = (
        await get_customer_hierarchy(customer_member_id, prisma_client, user_api_key_cache)
        if prisma_client is not None
        else None
    )
    if hierarchy is None:
        raise ProxyException(
            message=f"Authentication Error - customer member not found: {customer_member_id}",
            type=ProxyErrorTypes.auth_error,
            param=CUSTOMER_MEMBER_ID_HEADER,
            code=status.HTTP_401_UNAUTHORIZED,
        )
    if not RouteChecks.is_info_route(route):
        check_customer_hierarchy_budget(hierarchy)
    if isinstance(request_model, str) and request_model:
        check_customer_hierarchy_model_access(hierarchy, request_model, llm_router)
    return hierarchy


def check_customer_hierarchy_key_scope(hierarchy: CustomerHierarchyAuth, key_metadata: Mapping[str, object]) -> None:
    """A key whose metadata pins ``customer_org_id`` may only act for members of that organization."""
    key_customer_org_id: Final = key_metadata.get("customer_org_id")
    if key_customer_org_id is None or key_customer_org_id == hierarchy.customer_org_id:
        return
    raise ProxyException(
        message="API key is not allowed to act for this customer member's organization",
        type=ProxyErrorTypes.auth_error,
        param=CUSTOMER_MEMBER_ID_HEADER,
        code=status.HTTP_403_FORBIDDEN,
    )

"""Request-path resolution of the customer organization, team and member a request runs under.

A request names its customer member with the ``x-litellm-customer-member-id`` header. The member row
carries its team and that team's organization, so one lookup yields every level whose budget and rate
limits apply to the request. A request may instead, or additionally, name the organization by its alias
with ``x-litellm-customer-org-alias``: alone it charges the organization level only, next to the member
header it must agree with the member's organization.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, TypeAlias

from fastapi import status
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_checks import can_object_call_model
from litellm.proxy.auth.auth_object_prefetch import AUTH_OBJECTS_TARGET
from litellm.proxy.auth.customer_member_auto_create import ensure_customer_member
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, get_management_object_ttl
from litellm.proxy.management_helpers.customer_hierarchy import (
    Failure,
    PrismaRecord,
    effective_member_budget,
    related_budget,
    utc_now,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerRequestIdentity,
    CustomerScopeAuth,
    CustomerScopeKey,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, CustomerMember

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router_api import RegistryReader

CUSTOMER_MEMBER_ID_HEADER: Final = "x-litellm-customer-member-id"
CUSTOMER_ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
_ORGANIZATION_INCLUDE: Final = {"budget": True}
_MEMBER_INCLUDE: Final = {
    "budget": True,
    "budget_override": True,
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
    budget_override: Mapping[str, object] | None = None
    budget_override_expires_at: datetime | None = None
    parent_team: _TeamRow


@dataclass(frozen=True, slots=True)
class AmbiguousOrganizationAlias:
    """More than one customer organization carries the alias, so the request cannot be charged to one of them."""

    alias: str


def customer_member_cache_key(customer_member_id: str) -> str:
    return f"customer_member_id:{customer_member_id}"


def customer_org_alias_cache_key(customer_org_alias: str) -> str:
    return f"customer_org_alias:{customer_org_alias}"


def _header_value(headers: Mapping[str, object] | None, header: str) -> str | None:
    if headers is None:
        return None
    values: Final = tuple(
        value.strip()
        for name, value in headers.items()
        if name.lower() == header and isinstance(value, str) and value.strip()
    )
    return values[0] if values else None


def customer_member_id_from_headers(headers: Mapping[str, object] | None) -> str | None:
    return _header_value(headers, CUSTOMER_MEMBER_ID_HEADER)


def customer_org_alias_from_headers(headers: Mapping[str, object] | None) -> str | None:
    return _header_value(headers, CUSTOMER_ORG_ALIAS_HEADER)


def customer_identity_from_headers(headers: Mapping[str, object] | None) -> CustomerRequestIdentity | None:
    """The customer the request named, or None when it carried neither customer header."""
    identity: Final = CustomerRequestIdentity(
        customer_member_id=customer_member_id_from_headers(headers),
        customer_org_alias=customer_org_alias_from_headers(headers),
    )
    if identity.customer_member_id is None and identity.customer_org_alias is None:
        return None
    return identity


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
    template: Final = related_budget(member.budget, now)
    override: Final = related_budget(member.budget_override, now, template.max_budget if template is not None else None)
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


def customer_hierarchy_from_organization_row(row: Mapping[str, object], now: datetime) -> CustomerHierarchyAuth:
    """The organization scope alone, for a request that named an organization alias and no member."""
    organization: Final = _OrganizationRow.model_validate(row)
    return CustomerHierarchyAuth(
        customer_org_id=organization.customer_org_id,
        scopes=(
            _scope(
                "customer_org",
                organization.customer_org_id,
                organization.spend,
                related_budget(organization.budget, now),
            ),
        ),
    )


def _member_table(prisma_client: "PrismaClient") -> TableActions[PrismaRecord]:
    return prisma_client.db.litellm_customermembertable  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]  # prisma tables are generated at build time


def _organization_table(prisma_client: "PrismaClient") -> TableActions[PrismaRecord]:
    return prisma_client.db.litellm_customerorganizationtable  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]  # prisma tables are generated at build time


@with_service_target(AUTH_OBJECTS_TARGET)
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
    loaded_at: Final = now if now is not None else utc_now()
    row_values: Final = row.model_dump()
    hierarchy: Final = customer_hierarchy_from_member_row(row_values, loaded_at)
    await user_api_key_cache.async_set_cache(
        key=cache_key,
        value=hierarchy,
        ttl=_cache_ttl(row_values, loaded_at, get_management_object_ttl(user_api_key_cache)),
    )
    return hierarchy


@with_service_target(AUTH_OBJECTS_TARGET)
async def get_customer_organization_hierarchy(
    customer_org_alias: str,
    prisma_client: "PrismaClient",
    user_api_key_cache: UserApiKeyCache,
    now: datetime | None = None,
) -> CustomerHierarchyAuth | AmbiguousOrganizationAlias | None:
    """The organization-only hierarchy behind an alias, None when no organization carries it."""
    cache_key: Final = customer_org_alias_cache_key(customer_org_alias)
    cached: Final = await user_api_key_cache.async_get_cache(key=cache_key, model_type=CustomerHierarchyAuth)
    if cached is not None:
        return cached
    rows: Final = await _organization_table(prisma_client).find_many(
        where={"alias": customer_org_alias}, include=_ORGANIZATION_INCLUDE, take=2
    )
    if not rows:
        return None
    if len(rows) > 1:
        return AmbiguousOrganizationAlias(customer_org_alias)
    hierarchy: Final = customer_hierarchy_from_organization_row(
        rows[0].model_dump(), now if now is not None else utc_now()
    )
    await user_api_key_cache.async_set_cache(
        key=cache_key, value=hierarchy, ttl=get_management_object_ttl(user_api_key_cache)
    )
    return hierarchy


def _cache_ttl(row: Mapping[str, object], now: datetime, default_ttl: float) -> float:
    """A cached hierarchy may not outlive the member's budget override, or an expired override would keep
    granting its limits until the management-object TTL ran out."""
    expires_at: Final = _MemberRow.model_validate(row).budget_override_expires_at
    if expires_at is None or expires_at <= now:
        return default_ttl
    return max(1.0, min(default_ttl, (expires_at - now).total_seconds()))


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
    hierarchy: CustomerHierarchyAuth, model: str, llm_router: "RegistryReader | None"
) -> None:
    """Every level with an ``allowed_models`` list must permit ``model``; a level without one permits every model."""
    for scope in hierarchy.scopes:
        if scope.allowed_models:
            can_object_call_model(
                model=model, llm_router=llm_router, models=list(scope.allowed_models), object_type="customer"
            )


def _not_found(message: str, param: str) -> ProxyException:
    return ProxyException(
        message=f"Authentication Error - {message}",
        type=ProxyErrorTypes.auth_error,
        param=param,
        code=status.HTTP_401_UNAUTHORIZED,
    )


async def _organization_named_by_alias(
    customer_org_alias: str, prisma_client: "PrismaClient | None", user_api_key_cache: UserApiKeyCache
) -> CustomerHierarchyAuth:
    organization: Final = (
        await get_customer_organization_hierarchy(customer_org_alias, prisma_client, user_api_key_cache)
        if prisma_client is not None
        else None
    )
    if organization is None:
        raise _not_found(f"customer organization alias not found: {customer_org_alias}", CUSTOMER_ORG_ALIAS_HEADER)
    if isinstance(organization, AmbiguousOrganizationAlias):
        raise ProxyException(
            message=f"Customer organization alias is not unique: {customer_org_alias}",
            type=ProxyErrorTypes.bad_request_error,
            param=CUSTOMER_ORG_ALIAS_HEADER,
            code=status.HTTP_400_BAD_REQUEST,
        )
    return organization


MemberRegistrar: TypeAlias = Callable[[str, str, "PrismaClient"], Awaitable[CustomerMember | Failure]]


async def _register_member(
    customer_member_id: str,
    organization: CustomerHierarchyAuth | None,
    prisma_client: "PrismaClient",
    user_api_key_cache: UserApiKeyCache,
    register: MemberRegistrar,
) -> CustomerHierarchyAuth | None:
    """A member the request named for the first time joins the aliased organization's default team. Without an
    alias there is no organization to place it in, and the member stays unknown."""
    if organization is None:
        return None
    registered: Final = await register(organization.customer_org_id, customer_member_id, prisma_client)
    if isinstance(registered, Failure):
        verbose_proxy_logger.warning(
            "Could not register customer member %s in organization %s: %s",
            customer_member_id,
            organization.customer_org_id,
            registered.code.value,
        )
        return None
    return await get_customer_hierarchy(customer_member_id, prisma_client, user_api_key_cache)


async def _hierarchy_named_by_member(
    customer_member_id: str,
    organization: CustomerHierarchyAuth | None,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    register: MemberRegistrar,
) -> CustomerHierarchyAuth:
    if prisma_client is None:
        raise _not_found(f"customer member not found: {customer_member_id}", CUSTOMER_MEMBER_ID_HEADER)
    known: Final = await get_customer_hierarchy(customer_member_id, prisma_client, user_api_key_cache)
    hierarchy: Final = (
        known
        if known is not None
        else await _register_member(customer_member_id, organization, prisma_client, user_api_key_cache, register)
    )
    if hierarchy is None:
        raise _not_found(f"customer member not found: {customer_member_id}", CUSTOMER_MEMBER_ID_HEADER)
    return hierarchy


async def load_customer_hierarchy(
    identity: CustomerRequestIdentity,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    register: MemberRegistrar = ensure_customer_member,
) -> CustomerHierarchyAuth:
    """The hierarchy the headers name. An unknown alias is refused rather than treated as unrestricted: the caller
    asked for a budgeted identity that does not exist. A member header next to an alias header must name a member
    of that organization; a member the organization has never seen is registered in its default team first."""
    organization: Final = (
        await _organization_named_by_alias(identity.customer_org_alias, prisma_client, user_api_key_cache)
        if identity.customer_org_alias is not None
        else None
    )
    if identity.customer_member_id is None:
        if organization is None:
            raise _not_found("request named no customer member or organization", CUSTOMER_MEMBER_ID_HEADER)
        return organization
    hierarchy: Final = await _hierarchy_named_by_member(
        identity.customer_member_id, organization, prisma_client, user_api_key_cache, register
    )
    if organization is not None and organization.customer_org_id != hierarchy.customer_org_id:
        raise ProxyException(
            message=(
                f"Customer member {identity.customer_member_id} does not belong to the organization "
                f"aliased {identity.customer_org_alias}"
            ),
            type=ProxyErrorTypes.auth_error,
            param=CUSTOMER_ORG_ALIAS_HEADER,
            code=status.HTTP_403_FORBIDDEN,
        )
    return hierarchy


async def resolve_customer_hierarchy(
    identity: CustomerRequestIdentity,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    llm_router: "RegistryReader | None",
    request_model: str | Sequence[str] | None,
    route: str,
    key_metadata: Mapping[str, object] | None = None,
    register: MemberRegistrar = ensure_customer_member,
) -> CustomerHierarchyAuth:
    """Load the hierarchy the headers name and admit the request against it. ``request_model`` is the model the
    route will actually serve, which may differ from the body."""
    hierarchy: Final = await load_customer_hierarchy(identity, prisma_client, user_api_key_cache, register)
    check_customer_hierarchy_key_scope(hierarchy, key_metadata or {})
    if not RouteChecks.is_info_route(route):
        check_customer_hierarchy_budget(hierarchy)
    requested: Final = (request_model,) if isinstance(request_model, str) else tuple(request_model or ())
    for model in requested:
        if model:
            check_customer_hierarchy_model_access(hierarchy, model, llm_router)
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


@with_service_target(AUTH_OBJECTS_TARGET)
async def forget_customer_hierarchies(customer_member_ids: Sequence[str], user_api_key_cache: UserApiKeyCache) -> None:
    """Drop the cached hierarchies of members whose own, team, org or template budget changed so auth reloads them."""
    await evict_and_broadcast(tuple(customer_member_cache_key(m) for m in customer_member_ids), user_api_key_cache)


@with_service_target(AUTH_OBJECTS_TARGET)
async def forget_customer_organization_aliases(
    aliases: Sequence[str | None], user_api_key_cache: UserApiKeyCache
) -> None:
    """Drop the cached organization hierarchies behind aliases whose organization changed so request auth reloads them."""
    keys: Final = tuple(customer_org_alias_cache_key(alias) for alias in dict.fromkeys(aliases) if alias)
    if keys:
        await evict_and_broadcast(keys, user_api_key_cache)

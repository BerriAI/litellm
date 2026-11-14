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
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_checks import can_object_call_model
from litellm.proxy.auth.auth_object_prefetch import AUTH_OBJECTS_TARGET
from litellm.proxy.auth.customer_member_auto_create import ensure_customer_member, ensure_customer_organization
from litellm.proxy.auth.customer_spend import current_customer_spend
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.customer_usage import CustomerIdentifier
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, get_management_object_ttl
from litellm.proxy.management_helpers.customer_hierarchy import (
    ErrorCode,
    Failure,
    PrismaRecord,
    customer_row_reset_at,
    effective_member_budget,
    related_budget,
    utc_now,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import CustomerTeamRepository
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerHierarchyIds,
    CustomerRequestIdentity,
    CustomerScopeAuth,
    CustomerScopeKey,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    INCREASE_REQUEST_ROUTE,
    Budget,
    CustomerMember,
    Organization,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router_api import RegistryReader

CUSTOMER_MEMBER_ID_HEADER: Final = "x-litellm-customer-member-id"
CUSTOMER_ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
CUSTOMER_TEAM_ID_HEADER: Final = "x-litellm-customer-team-id"
_ORGANIZATION_INCLUDE: Final = {"budget": True}
_TEAM_INCLUDE: Final = {"budget": True, "budget_override": True}
_MEMBER_INCLUDE: Final = {"budget": True, "budget_override": True, "parent_org": {"include": {"budget": True}}}
_EXPIRY: Final[TypeAdapter[datetime | None]] = TypeAdapter(datetime | None)
_TEAM_IDS: Final = TypeAdapter(tuple[str, ...])
_SPEND: Final = TypeAdapter(float)


class _OrganizationRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_org_id: str
    alias: str | None = None
    spend: float = 0.0
    budget: Mapping[str, object] | None = None


class _TeamRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_team_id: str
    alias: str | None = None
    spend: float = 0.0
    budget: Mapping[str, object] | None = None
    budget_override: Mapping[str, object] | None = None
    budget_override_expires_at: datetime | None = None


class _MemberRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_member_id: str
    alias: str | None = None
    spend: float = 0.0
    budget: Mapping[str, object] | None = None
    budget_override: Mapping[str, object] | None = None
    budget_override_expires_at: datetime | None = None
    parent_org: _OrganizationRow
    parent_team_ids: tuple[str, ...] = ()
    teams: tuple[_TeamRow, ...] = ()


@dataclass(frozen=True, slots=True)
class AmbiguousOrganizationAlias:
    """More than one customer organization carries the alias, so the request cannot be charged to one of them."""

    alias: str


class CustomerBudgetExceededError(litellm.BudgetExceededError):
    def __init__(self, hierarchy: CustomerHierarchyAuth, error: litellm.BudgetExceededError) -> None:
        super().__init__(
            current_cost=error.current_cost,
            max_budget=error.max_budget,
            message=error.message,
            llm_provider=error.llm_provider,
            entity_type=error.entity_type,
            entity_id=error.entity_id,
        )
        self.customer_hierarchy: Final = hierarchy


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


def customer_team_id_from_headers(headers: Mapping[str, object] | None) -> str | None:
    return _header_value(headers, CUSTOMER_TEAM_ID_HEADER)


def customer_identity_from_headers(headers: Mapping[str, object] | None) -> CustomerRequestIdentity | None:
    """The customer the request named, or None when it carried neither customer header."""
    identity: Final = CustomerRequestIdentity(
        customer_member_id=customer_member_id_from_headers(headers),
        customer_org_alias=customer_org_alias_from_headers(headers),
        customer_team_id=customer_team_id_from_headers(headers),
    )
    if identity.customer_member_id is None and identity.customer_org_alias is None:
        return None
    return identity


def _scope(
    key: CustomerScopeKey, entity_id: str, spend: float, budget: Budget | None, budget_reset_at: datetime | None = None
) -> CustomerScopeAuth:
    if budget is None:
        return CustomerScopeAuth(key=key, entity_id=entity_id, spend=spend, budget_reset_at=budget_reset_at)
    return CustomerScopeAuth(
        key=key,
        entity_id=entity_id,
        spend=spend,
        budget_reset_at=budget_reset_at,
        max_budget=budget.max_budget,
        budget_limits=budget.budget_limits or (),
        tpm_limit=budget.tpm_limit,
        rpm_limit=budget.rpm_limit,
        allowed_models=budget.allowed_models or None,
        model_tpm_limit=budget.model_tpm_limit,
        model_rpm_limit=budget.model_rpm_limit,
        cap_policy=budget.cap_policy,
    )


def _team_scope(team: _TeamRow, now: datetime) -> CustomerScopeAuth:
    budget: Final = related_budget(team.budget, now)
    override: Final = related_budget(team.budget_override, now, budget.max_budget if budget is not None else None)
    return _scope(
        "customer_team",
        team.customer_team_id,
        team.spend,
        effective_member_budget(budget, override, team.budget_override_expires_at, now),
        customer_row_reset_at(team, now),
    ).model_copy(update={"alias": team.alias})


def customer_hierarchy_from_member_row(row: Mapping[str, object], now: datetime) -> CustomerHierarchyAuth:
    """The org, team and member scopes of one member row loaded with ``_MEMBER_INCLUDE`` plus its ``teams`` rows.
    The member's limits are its effective budget: the selected organization template under any unexpired override.
    ``scopes`` charges the first team in ``parent_team_ids``; ``select_customer_team`` swaps in another."""
    member: Final = _MemberRow.model_validate(row)
    organization: Final = member.parent_org
    template: Final = related_budget(member.budget, now)
    override: Final = related_budget(member.budget_override, now, template.max_budget if template is not None else None)
    teams_by_id: Final = {team.customer_team_id: team for team in member.teams}
    team_scopes: Final = tuple(
        _team_scope(teams_by_id[team_id], now) for team_id in member.parent_team_ids if team_id in teams_by_id
    )
    return CustomerHierarchyAuth(
        customer_member_id=member.customer_member_id,
        customer_member_alias=member.alias,
        customer_team_id=team_scopes[0].entity_id if team_scopes else None,
        customer_team_alias=team_scopes[0].alias if team_scopes else None,
        customer_org_id=organization.customer_org_id,
        customer_org_alias=organization.alias,
        scopes=(
            _scope(
                "customer_org",
                organization.customer_org_id,
                organization.spend,
                related_budget(organization.budget, now),
                customer_row_reset_at(organization, now),
            ),
            *team_scopes[:1],
            _scope(
                "customer_member",
                member.customer_member_id,
                member.spend,
                effective_member_budget(template, override, member.budget_override_expires_at, now),
                customer_row_reset_at(member, now),
            ),
        ),
        team_scopes=team_scopes,
    )


def select_customer_team(hierarchy: CustomerHierarchyAuth, customer_team_id: str) -> CustomerHierarchyAuth | None:
    """The same hierarchy charged to ``customer_team_id``, or None when the member is not in that team."""
    selected: Final = next((scope for scope in hierarchy.team_scopes if scope.entity_id == customer_team_id), None)
    if selected is None:
        return None
    return hierarchy.model_copy(
        update={
            "customer_team_id": customer_team_id,
            "customer_team_alias": selected.alias,
            "scopes": tuple(selected if scope.key == "customer_team" else scope for scope in hierarchy.scopes),
        }
    )


def customer_hierarchy_from_organization_row(row: Mapping[str, object], now: datetime) -> CustomerHierarchyAuth:
    """The organization scope alone, for a request that named an organization alias and no member."""
    organization: Final = _OrganizationRow.model_validate(row)
    return CustomerHierarchyAuth(
        customer_org_id=organization.customer_org_id,
        customer_org_alias=organization.alias,
        scopes=(
            _scope(
                "customer_org",
                organization.customer_org_id,
                organization.spend,
                related_budget(organization.budget, now),
                customer_row_reset_at(organization, now),
            ),
        ),
    )


def _member_table(prisma_client: "PrismaClient") -> TableActions[PrismaRecord]:
    return prisma_client.db.litellm_customermembertable  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]  # prisma tables are generated at build time


def _team_table(prisma_client: "PrismaClient") -> TableActions[PrismaRecord]:
    return prisma_client.db.litellm_customerteamtable  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAny]  # generated client


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
    lease: Final = await user_api_key_cache.lease(cache_key)
    row: Final = await _member_table(prisma_client).find_unique(
        where={"customer_member_id": customer_member_id}, include=_MEMBER_INCLUDE
    )
    if row is None:
        return None
    loaded_at: Final = now if now is not None else utc_now()
    team_ids: Final = _TEAM_IDS.validate_python(row.model_dump().get("parent_team_ids") or ())
    teams: Final = (
        await _team_table(prisma_client).find_many(
            # bounded-ok: team_ids is one member's parent_team_ids, a handful of teams written by the management API
            where={"customer_team_id": {"in": list(team_ids)}},
            include=_TEAM_INCLUDE,
        )
        if team_ids
        else ()
    )
    row_values: Final = {**row.model_dump(), "teams": tuple(team.model_dump() for team in teams)}
    hierarchy: Final = customer_hierarchy_from_member_row(row_values, loaded_at)
    _ = await user_api_key_cache.async_set_leased(
        lease, hierarchy, _cache_ttl(row_values, loaded_at, get_management_object_ttl(user_api_key_cache))
    )
    return hierarchy


async def get_customer_scopes_for_accounting(
    ids: CustomerHierarchyIds,
    prisma_client: "PrismaClient",
    user_api_key_cache: UserApiKeyCache,
) -> tuple[CustomerScopeAuth, ...]:
    """Resolve policies for the stamped charge identities without running admission checks."""
    now: Final = utc_now()
    hierarchy: Final = (
        await get_customer_hierarchy(ids.customer_member_id, prisma_client, user_api_key_cache, now)
        if ids.customer_member_id is not None
        else None
    )
    known: Final = () if hierarchy is None else hierarchy.scopes
    identities: Final[tuple[tuple[CustomerScopeKey, str | None], ...]] = (
        ("customer_org", ids.customer_org_id),
        ("customer_team", ids.customer_team_id),
        ("customer_member", ids.customer_member_id),
    )
    scopes: Final = tuple(
        [
            await _accounting_scope(key, entity_id, known, prisma_client, now)
            for key, entity_id in identities
            if entity_id is not None
        ]
    )
    return tuple(scope for scope in scopes if scope is not None)


async def _accounting_scope(
    key: CustomerScopeKey,
    entity_id: str,
    known: tuple[CustomerScopeAuth, ...],
    prisma_client: "PrismaClient",
    now: datetime,
) -> CustomerScopeAuth | None:
    cached: Final = next((scope for scope in known if scope.key == key and scope.entity_id == entity_id), None)
    if cached is not None or key == "customer_member":
        return cached
    # Organization-only requests and a member moved since admission still charge
    # the original team/organization, never the member's new parents.
    table: Final[TableActions[PrismaRecord]] = (
        _organization_table(prisma_client) if key == "customer_org" else CustomerTeamRepository(prisma_client).table
    )
    row: Final = await table.find_unique(where={f"{key}_id": entity_id}, include=_ORGANIZATION_INCLUDE)
    if row is None:
        return None
    values: Final = row.model_dump()
    return _scope(
        key, entity_id, _SPEND.validate_python(values.get("spend", 0)), related_budget(values.get("budget"), now)
    )


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
    lease: Final = await user_api_key_cache.lease(cache_key)
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
    _ = await user_api_key_cache.async_set_leased(lease, hierarchy, get_management_object_ttl(user_api_key_cache))
    return hierarchy


def _cache_ttl(row: Mapping[str, object], now: datetime, default_ttl: float) -> float:
    """A cached hierarchy may not outlive the member's or any of its teams' budget override, or an expired
    override would keep granting its limits until the management-object TTL ran out."""
    member: Final = _MemberRow.model_validate(row)
    expiries: Final = tuple(
        expires_at
        for expires_at in (
            member.budget_override_expires_at,
            *(team.budget_override_expires_at for team in member.teams),
        )
        if expires_at is not None and expires_at > now
    )
    if not expiries:
        return default_ttl
    return max(1.0, min(default_ttl, (min(expiries) - now).total_seconds()))


def cap_policy_hint(hierarchy: CustomerHierarchyAuth, key: str, entity_id: str) -> str:
    """What the paused caller can do, from the budget of the level that blocked it."""
    policy: Final = next(
        (scope.cap_policy for scope in hierarchy.scopes if scope.key == key and scope.entity_id == entity_id), None
    )
    if policy == "pause_allow_increase":
        return f" Request a temporary increase through POST {INCREASE_REQUEST_ROUTE}."
    return " Spend resumes when the budget window resets."


def check_customer_hierarchy_budget(hierarchy: CustomerHierarchyAuth) -> None:
    """Raises ``BudgetExceededError`` for the first level, org first, whose spend has reached its ``max_budget``."""
    for scope in hierarchy.scopes:
        if scope.max_budget is not None and scope.spend >= scope.max_budget:
            raise litellm.BudgetExceededError(
                current_cost=scope.spend,
                max_budget=scope.max_budget,
                message=(
                    f"ExceededBudget: {scope.key}={scope.entity_id} over budget. "
                    f"Spend={scope.spend}, Budget={scope.max_budget}."
                    f"{cap_policy_hint(hierarchy, scope.key, scope.entity_id)}"
                ),
                entity_type=scope.key,
                entity_id=scope.entity_id,
            )


async def check_customer_hierarchy_windows(
    hierarchy: CustomerHierarchyAuth,
    now: datetime | None = None,
) -> None:
    from itertools import chain

    from litellm.proxy.proxy_server import get_current_spend, spend_counter_store
    from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
    from litellm.proxy.spend_tracking.spend_counter_batch import spend_counter_batch_scope

    counters: Final = tuple(
        chain.from_iterable(
            customer_window_counters(scope.key, scope.entity_id, scope.budget_limits, now=now)
            for scope in hierarchy.scopes
        )
    )
    with spend_counter_batch_scope(
        spend_counter_store, counter_keys=frozenset(counter.counter_key for counter in counters)
    ):
        for counter in counters:
            if counter.window_start is None:
                raise ProxyException(
                    message="Customer budget window is unavailable.",
                    type=ProxyErrorTypes.internal_server_error,
                    param="budget_limits",
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            try:
                spend = await get_current_spend(
                    counter_key=counter.counter_key,
                    fallback_spend=0.0,
                    max_budget=counter.max_budget,
                    window_entity_type=counter.entity_type,
                    window_entity_id=counter.entity_id,
                    window_duration=counter.window_duration,
                    window_start=counter.window_start,
                    require_authoritative_window=True,
                )
            except HTTPException as error:
                raise ProxyException(
                    message="Customer budget window spend is unavailable.",
                    type=ProxyErrorTypes.internal_server_error,
                    param="budget_limits",
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                ) from error
            if spend >= counter.max_budget:
                raise litellm.BudgetExceededError(
                    current_cost=spend,
                    max_budget=counter.max_budget,
                    message=(
                        f"ExceededBudget: {counter.entity_type}={counter.entity_id} over {counter.window_duration} budget."
                        f"{cap_policy_hint(hierarchy, counter.entity_type, counter.entity_id)}"
                    ),
                    entity_type=counter.entity_type,
                    entity_id=counter.entity_id,
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


def _untracked(message: str, header: str) -> None:
    """An id the proxy does not know is logged and the request runs without that customer level, never refused:
    the headers are attribution, and a customer whose rows are not set up yet must keep working."""
    verbose_proxy_logger.warning("Customer %s not tracked, %s", header, message)


OrganizationRegistrar: TypeAlias = Callable[[str, "PrismaClient"], Awaitable[Organization | Failure]]


def _organization_from_lookup(
    customer_org_alias: str, organization: CustomerHierarchyAuth | AmbiguousOrganizationAlias | None
) -> CustomerHierarchyAuth | None:
    if organization is None:
        return _untracked(f"organization alias not found: {customer_org_alias}", CUSTOMER_ORG_ALIAS_HEADER)
    if isinstance(organization, AmbiguousOrganizationAlias):
        raise ProxyException(
            message=f"Customer organization alias is not unique: {customer_org_alias}",
            type=ProxyErrorTypes.bad_request_error,
            param=CUSTOMER_ORG_ALIAS_HEADER,
            code=status.HTTP_400_BAD_REQUEST,
        )
    return organization


async def _register_organization(
    customer_org_alias: str,
    prisma_client: "PrismaClient",
    user_api_key_cache: UserApiKeyCache,
    register: OrganizationRegistrar,
    key_metadata: Mapping[str, object],
) -> CustomerHierarchyAuth | None:
    """An alias the proxy has never seen becomes an organization with that id and alias, so a customer's first
    request is charged to it instead of running untracked. A key pinned to another organization is refused first."""
    if not _registrable(customer_org_alias):
        verbose_proxy_logger.warning(
            "Could not register customer organization: alias is not a valid customer identifier"
        )
        return _untracked(f"organization alias not found: {customer_org_alias}", CUSTOMER_ORG_ALIAS_HEADER)
    _check_key_scope(customer_org_alias, key_metadata)
    registered: Final = await register(customer_org_alias, prisma_client)
    if isinstance(registered, Failure):
        if registered.code is ErrorCode.FREE_TIER_LIMIT:
            verbose_proxy_logger.warning(
                "CUSTOMER TRACKING OFF for organization %s because the free customer portal limit was reached. "
                "Upgrade to Enterprise to remove the limits: https://www.litellm.ai/enterprise",
                customer_org_alias,
            )
            return None
        verbose_proxy_logger.warning(
            "Could not register customer organization %s: %s", customer_org_alias, registered.code.value
        )
        return _untracked(f"organization alias not found: {customer_org_alias}", CUSTOMER_ORG_ALIAS_HEADER)
    return _organization_from_lookup(
        customer_org_alias,
        await get_customer_organization_hierarchy(customer_org_alias, prisma_client, user_api_key_cache),
    )


async def _organization_named_by_alias(
    customer_org_alias: str,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    register: OrganizationRegistrar,
    key_metadata: Mapping[str, object],
) -> CustomerHierarchyAuth | None:
    if prisma_client is None:
        return _untracked(f"organization alias not found: {customer_org_alias}", CUSTOMER_ORG_ALIAS_HEADER)
    found: Final = await get_customer_organization_hierarchy(customer_org_alias, prisma_client, user_api_key_cache)
    if found is None:
        return await _register_organization(
            customer_org_alias, prisma_client, user_api_key_cache, register, key_metadata
        )
    return _organization_from_lookup(customer_org_alias, found)


MemberRegistrar: TypeAlias = Callable[[str, str, "PrismaClient"], Awaitable[CustomerMember | Failure]]
_MEMBER_ID: Final[TypeAdapter[str]] = TypeAdapter(CustomerIdentifier)


def _registrable(customer_member_id: str) -> bool:
    """A header value becomes a member row only within the identifier bound the customer tables share, so the spend
    ledger can index every row the proxy creates."""
    try:
        _MEMBER_ID.validate_python(customer_member_id)
    except ValidationError:
        return False
    return True


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
    if not _registrable(customer_member_id):
        verbose_proxy_logger.warning(
            "Could not register customer member in organization %s: id is not a valid customer identifier",
            organization.customer_org_id,
        )
        return None
    registered: Final = await register(organization.customer_org_id, customer_member_id, prisma_client)
    if isinstance(registered, Failure):
        if registered.code is ErrorCode.FREE_TIER_LIMIT:
            verbose_proxy_logger.warning(
                "CUSTOMER TRACKING OFF for member %s because the free customer portal limit was reached. "
                "Upgrade to Enterprise to remove the limits: https://www.litellm.ai/enterprise",
                customer_member_id,
            )
            return None
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
) -> CustomerHierarchyAuth | None:
    """The member's hierarchy, or the aliased organization alone when the member is unknown and cannot be
    registered, or None when neither is known."""
    if prisma_client is None:
        return _untracked(f"member not found: {customer_member_id}", CUSTOMER_MEMBER_ID_HEADER)
    known: Final = await get_customer_hierarchy(customer_member_id, prisma_client, user_api_key_cache)
    hierarchy: Final = (
        known
        if known is not None
        else await _register_member(customer_member_id, organization, prisma_client, user_api_key_cache, register)
    )
    if hierarchy is None:
        _untracked(f"member not found: {customer_member_id}", CUSTOMER_MEMBER_ID_HEADER)
        return organization
    return hierarchy


async def load_customer_hierarchy(
    identity: CustomerRequestIdentity,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    register: MemberRegistrar = ensure_customer_member,
    key_metadata: Mapping[str, object] = MappingProxyType({}),
    register_organization: OrganizationRegistrar = ensure_customer_organization,
) -> CustomerHierarchyAuth | None:
    """The hierarchy the headers name, or None when the proxy knows none of the ids, in which case the request runs
    untracked. An alias the proxy has never seen is registered as a new organization first. A member header next to an alias header must name a member of that organization; a member the
    organization has never seen is registered in its default team first, and when that fails the organization
    alone is charged. A key pinned to another organization is refused before any row is created."""
    organization: Final = (
        await _organization_named_by_alias(
            identity.customer_org_alias, prisma_client, user_api_key_cache, register_organization, key_metadata
        )
        if identity.customer_org_alias is not None
        else None
    )
    if organization is not None:
        check_customer_hierarchy_key_scope(organization, key_metadata)
    if identity.customer_member_id is None:
        if identity.customer_team_id is not None:
            raise _forbidden("a customer team header needs a customer member header", CUSTOMER_TEAM_ID_HEADER)
        return organization
    hierarchy: Final = await _hierarchy_named_by_member(
        identity.customer_member_id, organization, prisma_client, user_api_key_cache, register
    )
    if hierarchy is None:
        return None
    if organization is not None and organization.customer_org_id != hierarchy.customer_org_id:
        raise _forbidden(
            f"Customer member {identity.customer_member_id} does not belong to the organization "
            f"aliased {identity.customer_org_alias}",
            CUSTOMER_ORG_ALIAS_HEADER,
        )
    if identity.customer_team_id is None or hierarchy.customer_member_id is None:
        return hierarchy
    selected: Final = select_customer_team(hierarchy, identity.customer_team_id)
    if selected is None:
        raise _forbidden(
            f"Customer member {identity.customer_member_id} is not in customer team {identity.customer_team_id}",
            CUSTOMER_TEAM_ID_HEADER,
        )
    return selected


def _forbidden(message: str, param: str) -> ProxyException:
    return ProxyException(message=message, type=ProxyErrorTypes.auth_error, param=param, code=status.HTTP_403_FORBIDDEN)


async def resolve_customer_hierarchy(
    identity: CustomerRequestIdentity,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: UserApiKeyCache,
    llm_router: "RegistryReader | None",
    request_model: str | Sequence[str] | None,
    route: str,
    key_metadata: Mapping[str, object] | None = None,
    register: MemberRegistrar = ensure_customer_member,
    register_organization: OrganizationRegistrar = ensure_customer_organization,
) -> CustomerHierarchyAuth | None:
    """Load the hierarchy the headers name and admit the request against it, or None when no id is known.
    ``request_model`` is the model the route will actually serve, which may differ from the body."""
    from litellm.proxy.proxy_server import get_current_spend

    metadata: Final = key_metadata or {}
    cached: Final = await load_customer_hierarchy(
        identity, prisma_client, user_api_key_cache, register, metadata, register_organization
    )
    if cached is None:
        return None
    hierarchy: Final = await current_customer_spend(cached, get_current_spend)
    try:
        await check_customer_hierarchy_access(hierarchy, llm_router, request_model, route, key_metadata)
    except litellm.BudgetExceededError as error:
        raise CustomerBudgetExceededError(hierarchy, error) from error
    return hierarchy


async def check_customer_hierarchy_access(
    hierarchy: CustomerHierarchyAuth,
    llm_router: "RegistryReader | None",
    request_model: str | Sequence[str] | None,
    route: str,
    key_metadata: Mapping[str, object] | None = None,
) -> None:
    check_customer_hierarchy_key_scope(hierarchy, key_metadata or {})
    requested: Final = (request_model,) if isinstance(request_model, str) else tuple(request_model or ())
    for model in requested:
        if model:
            check_customer_hierarchy_model_access(hierarchy, model, llm_router)
    if RouteChecks.is_info_route(route):
        return
    check_customer_hierarchy_budget(hierarchy)
    if any(scope.budget_limits for scope in hierarchy.scopes):
        await check_customer_hierarchy_windows(hierarchy)


def check_customer_hierarchy_key_scope(hierarchy: CustomerHierarchyAuth, key_metadata: Mapping[str, object]) -> None:
    """A key whose metadata pins ``customer_org_id`` may only act for members of that organization."""
    _check_key_scope(hierarchy.customer_org_id, key_metadata)


def _check_key_scope(customer_org_id: str, key_metadata: Mapping[str, object]) -> None:
    key_customer_org_id: Final = key_metadata.get("customer_org_id")
    if key_customer_org_id is None or key_customer_org_id == customer_org_id:
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

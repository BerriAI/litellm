"""Resolve every budget that can gate requests made with one virtual key.

Enforcement is spread over a dozen checks in ``auth_checks`` that each raise the first
time they trip, so a caller who gets a 429 cannot tell which scope produced it. This
module answers the same question up front: for one key, every applicable scope, its
live spend, and the operator the enforcing check compares with.

The limit and counter for each scope come from the same helpers enforcement uses, so
the two cannot disagree about which budget applies or which counter it is measured
against.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_BUDGET_NAME
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team import BudgetLimitEntry, LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import (
    Litellm_EntityType,
    LiteLLM_ProjectTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    TeamMemberBudget,
    get_default_end_user_budget,
    get_end_user_object,
    get_org_object,
    get_project_object,
    get_tag_objects_batch,
    get_team_object,
    get_user_object,
    resolve_budget_org_id,
    resolve_team_member_budget,
    user_budget_applies_to_key,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start
from litellm.proxy.spend_tracking.spend_counter_keys import (
    end_user_spend_counter,
    key_spend_counter,
    key_window_spend_counter,
    org_spend_counter,
    tag_spend_counter,
    team_member_spend_counter,
    team_spend_counter,
    team_window_spend_counter,
    user_spend_counter,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.management_endpoints.key_management_endpoints import (
    BudgetComparison,
    BudgetEnforcement,
    BudgetScope,
    BudgetSpendState,
    BudgetStatus,
    KeyBudgetEntry,
    KeyBudgetNote,
)

_ENTITY_TYPE_BY_SCOPE: Final[Mapping[BudgetScope, Litellm_EntityType]] = MappingProxyType(
    {
        "proxy": Litellm_EntityType.PROXY,
        "key": Litellm_EntityType.KEY,
        "key_window": Litellm_EntityType.KEY,
        "team": Litellm_EntityType.TEAM,
        "team_window": Litellm_EntityType.TEAM,
        "team_member": Litellm_EntityType.TEAM_MEMBER,
        "user": Litellm_EntityType.USER,
        "organization": Litellm_EntityType.ORGANIZATION,
        "project": Litellm_EntityType.PROJECT,
        "tag": Litellm_EntityType.TAG,
        "end_user": Litellm_EntityType.END_USER,
    }
)

_RESERVATION_COVERED_SCOPES: Final[frozenset[BudgetScope]] = frozenset(
    {"key", "key_window", "team", "team_window", "team_member", "user", "organization", "tag", "end_user"}
)

_ROLLING_WINDOW_NOTE: Final = KeyBudgetNote(
    code="rolling_window",
    severity="info",
    text="rolling window; the start moves with reset_at so consecutive windows can overlap",
)
_RESERVATION_NOTE: Final = KeyBudgetNote(
    code="reservation_blocks_at_limit",
    severity="info",
    text=(
        "the reservation layer usually blocks this scope as soon as spend reaches the limit, ahead of the "
        "read-time check; requests it cannot price up front are still gated by the read-time check alone"
    ),
)
_PROJECT_SPEND_NOTE: Final = KeyBudgetNote(
    code="project_spend_not_tracked",
    severity="warning",
    text="project spend is never incremented today, so this budget cannot trip",
)
_TAG_NOTE: Final = KeyBudgetNote(
    code="request_tags_add_budgets",
    severity="warning",
    text="key tags are attached to every request; request-supplied tags add budgets not listed here",
)
_END_USER_ROUTE_NOTE: Final = KeyBudgetNote(
    code="end_user_route_only",
    severity="warning",
    text="applies only to requests that name this end user; nothing else on this row says so",
)
_CUSTOM_AUTH_END_USER_NOTE: Final = KeyBudgetNote(
    code="custom_auth_may_override_end_user_cap",
    severity="warning",
    text="a custom auth callable can set a request-scoped end user cap that overrides this one and is not visible here",
)
_CUSTOM_AUTH_SKIPS_CHECKS_NOTE: Final = KeyBudgetNote(
    code="custom_auth_skips_read_time_checks",
    severity="warning",
    text=(
        "a custom auth callable authenticates requests on this proxy and "
        "general_settings.custom_auth_run_common_checks is not set, so none of these budgets are checked "
        "at request time for the requests it authenticates; only the reservation layer still applies"
    ),
)
_ENTITY_UNAVAILABLE_NOTE: Final = KeyBudgetNote(
    code="entity_unavailable",
    severity="warning",
    text=(
        "this entity could not be read, so its budgets are unknown rather than unset; nothing on this "
        "scope can be ruled out until it can be read again"
    ),
)
_USER_ON_TEAM_KEY_NOTE: Final = KeyBudgetNote(
    code="user_budget_not_applied_to_team_key",
    severity="info",
    text=(
        "the owner's personal budget is not applied to team keys unless "
        "general_settings.apply_user_budget_to_team_keys is enabled"
    ),
)
_THROTTLE_NOTE: Final = KeyBudgetNote(
    code="throttled_instead_of_blocked",
    severity="info",
    text=(
        "past the limit the key's own tpm and rpm limits are scaled down by "
        "litellm.budget_exceeded_throttle_percentage; no other budget on this key throttles"
    ),
)


class SpendReader(Protocol):
    async def __call__(
        self,
        *,
        counter_key: str,
        fallback_spend: float,
        max_budget: float | None,
        window_entity_type: str | None,
        window_entity_id: str | None,
        window_start: datetime | None,
        fallback_authoritative: bool,
    ) -> float: ...


async def _read_counter_spend(
    *,
    counter_key: str,
    fallback_spend: float,
    max_budget: float | None,
    window_entity_type: str | None,
    window_entity_id: str | None,
    window_start: datetime | None,
    fallback_authoritative: bool,
) -> float:
    from litellm.proxy.proxy_server import get_current_spend

    return await get_current_spend(
        counter_key=counter_key,
        fallback_spend=fallback_spend,
        max_budget=max_budget,
        window_entity_type=window_entity_type,
        window_entity_id=window_entity_id,
        window_start=window_start,
        fallback_authoritative=fallback_authoritative,
    )


@dataclass(frozen=True, slots=True)
class KeyBudgetResolverDeps:
    prisma_client: PrismaClient
    user_api_key_cache: UserApiKeyCache
    proxy_logging_obj: ProxyLogging
    general_settings: Mapping[str, object]
    custom_auth_enabled: bool = False
    read_spend: SpendReader = field(default=_read_counter_spend)


@dataclass(frozen=True, slots=True)
class _Unavailable:
    """An entity that could not be read, which is not the same as an entity that is not there."""


_UNAVAILABLE: Final = _Unavailable()


@dataclass(frozen=True, slots=True)
class _UnknownSpend:
    """Stands in for the counter of an entity that could not be read, so no number is invented for it."""


@dataclass(frozen=True, slots=True)
class _CounterSpend:
    counter_key: str
    fallback_spend: float
    window_entity_type: str | None = None
    window_entity_id: str | None = None
    window_start: datetime | None = None
    fallback_authoritative: bool = False


@dataclass(frozen=True, slots=True)
class _RecordedSpend:
    value: float


_SpendSource = _CounterSpend | _RecordedSpend | _UnknownSpend


@dataclass(frozen=True, slots=True)
class _SpendReading:
    """A spend number and why it is missing when it is, so a blank cell never reads as zero."""

    value: float | None
    state: BudgetSpendState


@dataclass(frozen=True, slots=True)
class _PlannedBudget:
    scope: BudgetScope
    entity_id: str | None
    entity_label: str | None
    enforcement: BudgetEnforcement
    max_budget: float | None
    comparison: BudgetComparison
    source: str
    spend_source: _SpendSource
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    window_start: datetime | None = None
    notes: tuple[KeyBudgetNote, ...] = ()


_ReadPlan = tuple[_PlannedBudget, _SpendReading]


class _MetadataTags(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tags: tuple[str, ...] = ()


class _MetadataFields(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: _MetadataTags = _MetadataTags()


class _WindowFields(BaseModel):
    """Entries stay unparsed here so that one malformed window cannot discard the rest."""

    model_config = ConfigDict(extra="ignore")

    budget_limits: tuple[object, ...] = ()


_T = TypeVar("_T")

_METADATA_FIELDS: Final = TypeAdapter(_MetadataFields)
_WINDOW_FIELDS: Final = TypeAdapter(_WindowFields)
_BUDGET_LIMIT_ENTRY: Final = TypeAdapter(BudgetLimitEntry)


def _validated(adapter: TypeAdapter[_T], value: object, field_name: str) -> _T | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        verbose_proxy_logger.exception("Skipping malformed %s entry during budget resolution", field_name)
        return None


@dataclass(frozen=True, slots=True)
class _TokenBudgetInputs:
    """Key columns the token model leaves as bare dicts, re-validated once into the shapes enforcement uses."""

    tags: tuple[str, ...]
    budget_limits: tuple[BudgetLimitEntry, ...]


def _token_budget_inputs(valid_token: UserAPIKeyAuth) -> _TokenBudgetInputs:
    dumped: Final = valid_token.model_dump()
    return _TokenBudgetInputs(tags=_key_tags(dumped), budget_limits=_budget_windows(dumped))


def _key_tags(dumped: Mapping[str, object]) -> tuple[str, ...]:
    """Key metadata tags ride every request this key makes, so their budgets always apply."""
    try:
        tags: Final = _METADATA_FIELDS.validate_python(dumped).metadata.tags
    except ValidationError:
        verbose_proxy_logger.exception("Skipping malformed key metadata tags during budget resolution")
        return ()
    return tuple(dict.fromkeys(tag for tag in tags if tag))


def _budget_windows(dumped: Mapping[str, object]) -> tuple[BudgetLimitEntry, ...]:
    """Enforcement still applies the windows that parse, so a bad neighbour must not hide them."""
    container: Final = _validated(_WINDOW_FIELDS, dumped, "budget_limits")
    if container is None:
        return ()
    parsed: Final = (_validated(_BUDGET_LIMIT_ENTRY, entry, "budget_limits") for entry in container.budget_limits)
    return tuple(entry for entry in parsed if entry is not None)


@dataclass(frozen=True, slots=True)
class _ProxyBudget:
    spend: float
    budget_duration: str | None
    budget_reset_at: datetime | None


@dataclass(frozen=True, slots=True)
class _BudgetRowMeta:
    budget_duration: str | None
    budget_reset_at: datetime | None
    max_budget: float | None = None


@dataclass(frozen=True, slots=True)
class _KeyBudgetContext:
    valid_token: UserAPIKeyAuth
    token_inputs: _TokenBudgetInputs
    end_user_id: str | None
    custom_auth_enabled: bool
    custom_auth_skips_checks: bool
    general_settings: Mapping[str, object]
    proxy: _ProxyBudget | _Unavailable | None
    team: LiteLLM_TeamTable | None
    user: LiteLLM_UserTable | None
    project: LiteLLM_ProjectTableCachedObj | None
    organization: LiteLLM_OrganizationTable | None
    team_member: TeamMemberBudget | None
    tags: tuple[tuple[str, LiteLLM_TagTable | _Unavailable | None], ...]
    end_user: LiteLLM_EndUserTable | None
    default_end_user_budget: LiteLLM_BudgetTable | None
    budget_meta: Mapping[str, _BudgetRowMeta]
    unresolved: Mapping[BudgetScope, str | None]


async def resolve_key_budgets(
    valid_token: UserAPIKeyAuth,
    end_user_id: str | None,
    deps: KeyBudgetResolverDeps,
) -> tuple[KeyBudgetEntry, ...]:
    """Every budget that applies to ``valid_token``, configured or not, with live spend."""
    context: Final = await _load_context(valid_token=valid_token, end_user_id=end_user_id, deps=deps)
    plans: Final = _plan_budgets(context)
    readings: Final = await asyncio.gather(*(_read_spend(plan=plan, deps=deps) for plan in plans))
    reservation_enabled: Final = deps.general_settings.get("disable_budget_reservation") is not True
    proxy_notes: Final = (_CUSTOM_AUTH_SKIPS_CHECKS_NOTE,) if context.custom_auth_skips_checks else ()
    return tuple(
        _to_entry(
            plan=plan,
            reading=reading,
            reservation_enabled=reservation_enabled,
            proxy_notes=proxy_notes,
        )
        for plan, reading in zip(plans, readings, strict=True)
    )


async def _read_spend(plan: _PlannedBudget, deps: KeyBudgetResolverDeps) -> _SpendReading:
    source: Final = plan.spend_source
    try:
        match source:
            case _UnknownSpend():
                return _SpendReading(value=None, state="unavailable")
            case _RecordedSpend():
                return _SpendReading(value=source.value, state="live")
            case _CounterSpend():
                return _SpendReading(
                    value=await deps.read_spend(
                        counter_key=source.counter_key,
                        fallback_spend=source.fallback_spend,
                        max_budget=plan.max_budget,
                        window_entity_type=source.window_entity_type,
                        window_entity_id=source.window_entity_id,
                        window_start=source.window_start,
                        fallback_authoritative=source.fallback_authoritative,
                    ),
                    state="live",
                )
            case _:
                assert_never(source)
    except Exception:  # noqa: BLE001  # one unreadable counter must not blank the whole report
        verbose_proxy_logger.exception("Unable to read live spend for budget scope %s", plan.scope)
        return _SpendReading(value=None, state="unavailable")


def _effective_comparison(plan: _PlannedBudget, reservation_enabled: bool) -> BudgetComparison:
    """
    Reservation runs ahead of the read-time check and blocks once spend reaches the cap, so for
    every scope it covers it, not the read-time comparison, decides when a request stops going through.

    It builds no counter at all for a non-positive cap, and it releases the one it built for a key
    that throttles instead of blocking, so neither of those is tightened by it.
    """
    if plan.enforcement != "hard" or not reservation_enabled:
        return plan.comparison
    if plan.max_budget is None or plan.max_budget <= 0:
        return plan.comparison
    if plan.scope not in _RESERVATION_COVERED_SCOPES:
        return plan.comparison
    return ">="


def _entry_notes(
    plan: _PlannedBudget, comparison: BudgetComparison, proxy_notes: tuple[KeyBudgetNote, ...]
) -> tuple[KeyBudgetNote, ...]:
    tightened: Final = () if comparison == plan.comparison else (_RESERVATION_NOTE,)
    return (*tightened, *plan.notes, *proxy_notes)


def _status(plan: _PlannedBudget, spend: float | None, exceeded: bool) -> BudgetStatus:
    """
    A row nobody could evaluate reports ``unknown``, never ``unlimited``.

    Reading it as unset is the failure this endpoint exists to prevent: an unreadable entity and an
    unreadable counter both leave a budget that may well be the one blocking the key.
    """
    if isinstance(plan.spend_source, _UnknownSpend):
        return "unknown"
    if plan.max_budget is None:
        return "unlimited"
    if spend is None:
        return "unknown"
    return "exceeded" if exceeded else "ok"


def _to_entry(
    plan: _PlannedBudget,
    reading: _SpendReading,
    reservation_enabled: bool,
    proxy_notes: tuple[KeyBudgetNote, ...],
) -> KeyBudgetEntry:
    comparison: Final = _effective_comparison(plan=plan, reservation_enabled=reservation_enabled)
    spend: Final = reading.value
    exceeded: Final = (
        plan.max_budget is not None
        and spend is not None
        and (spend >= plan.max_budget if comparison == ">=" else spend > plan.max_budget)
    )
    status: Final[BudgetStatus] = _status(plan=plan, spend=spend, exceeded=exceeded)
    return KeyBudgetEntry(
        scope=plan.scope,
        entity_type=_ENTITY_TYPE_BY_SCOPE[plan.scope],
        entity_id=plan.entity_id,
        entity_label=plan.entity_label,
        enforcement=plan.enforcement,
        max_budget=plan.max_budget,
        spend=spend,
        spend_state=reading.state,
        remaining=(plan.max_budget - spend) if plan.max_budget is not None and spend is not None else None,
        comparison=comparison,
        budget_duration=plan.budget_duration,
        budget_reset_at=plan.budget_reset_at,
        window_start=plan.window_start,
        source=plan.source,
        status=status,
        notes=_entry_notes(plan=plan, comparison=comparison, proxy_notes=proxy_notes),
    )


async def _load_context(
    valid_token: UserAPIKeyAuth,
    end_user_id: str | None,
    deps: KeyBudgetResolverDeps,
) -> _KeyBudgetContext:
    token_inputs: Final = _token_budget_inputs(valid_token)
    loaded_proxy, loaded_team, loaded_user, loaded_project, tags, loaded_end_user = await asyncio.gather(
        _load_proxy_budget(deps),
        _load_team(valid_token.team_id, deps),
        _load_user(valid_token.user_id, deps),
        _load_project(valid_token.project_id, deps),
        _load_tags(token_inputs.tags, deps),
        _load_end_user(end_user_id, deps),
    )
    team: Final = _resolved(loaded_team)
    org_id: Final = resolve_budget_org_id(valid_token=valid_token, team_object=team)
    loaded_organization, loaded_team_member, loaded_default_end_user_budget = await asyncio.gather(
        _load_organization(org_id, deps),
        _load_team_member(valid_token=valid_token, team=team, deps=deps),
        _load_default_end_user_budget(deps),
    )
    project: Final = _resolved(loaded_project)
    organization: Final = _resolved(loaded_organization)
    end_user: Final = _resolved(loaded_end_user)
    default_end_user_budget: Final = _resolved(loaded_default_end_user_budget)
    budget_ids: Final = _referenced_budget_ids(
        organization=organization,
        project=project,
        tags=tags,
        end_user=end_user,
        default_end_user_budget=default_end_user_budget,
        key_budget_id=valid_token.budget_id,
    )
    return _KeyBudgetContext(
        valid_token=valid_token,
        token_inputs=token_inputs,
        end_user_id=end_user_id,
        custom_auth_enabled=deps.custom_auth_enabled,
        custom_auth_skips_checks=(
            deps.custom_auth_enabled and deps.general_settings.get("custom_auth_run_common_checks") is not True
        ),
        general_settings=deps.general_settings,
        proxy=loaded_proxy,
        team=team,
        user=_resolved(loaded_user),
        project=project,
        organization=organization,
        team_member=_resolved(loaded_team_member),
        tags=tags,
        end_user=end_user,
        default_end_user_budget=default_end_user_budget,
        budget_meta=await _load_budget_meta(budget_ids, deps),
        unresolved=_unresolved_scopes(
            valid_token=valid_token,
            end_user_id=end_user_id,
            org_id=org_id,
            loaded_team=loaded_team,
            loaded_user=loaded_user,
            loaded_project=loaded_project,
            loaded_organization=loaded_organization,
            loaded_team_member=loaded_team_member,
            loaded_end_user=loaded_end_user,
            loaded_default_end_user_budget=loaded_default_end_user_budget,
        ),
    )


def _resolved(value: _T | _Unavailable | None) -> _T | None:
    return None if isinstance(value, _Unavailable) else value


def _unresolved_scopes(
    valid_token: UserAPIKeyAuth,
    end_user_id: str | None,
    org_id: str | None,
    loaded_team: LiteLLM_TeamTable | _Unavailable | None,
    loaded_user: LiteLLM_UserTable | _Unavailable | None,
    loaded_project: LiteLLM_ProjectTableCachedObj | _Unavailable | None,
    loaded_organization: LiteLLM_OrganizationTable | _Unavailable | None,
    loaded_team_member: TeamMemberBudget | _Unavailable | None,
    loaded_end_user: LiteLLM_EndUserTable | _Unavailable | None,
    loaded_default_end_user_budget: LiteLLM_BudgetTable | _Unavailable | None,
) -> Mapping[BudgetScope, str | None]:
    """
    The scopes whose budgets could not be read, mapped to the entity they belong to.

    A failed lookup reaches the planners as an absent entity, which would drop the scope from the
    report or leave it looking unconfigured, so each one is recorded here and reported on its own row.
    A team carries three scopes and is also where an organization is inherited from, so losing it
    leaves all four unknown.
    """
    team_unavailable: Final = isinstance(loaded_team, _Unavailable)
    end_user_object: Final = _resolved(loaded_end_user)
    candidates: Final[tuple[tuple[BudgetScope, bool, str | None], ...]] = (
        ("team", team_unavailable, valid_token.team_id),
        ("team_window", team_unavailable, valid_token.team_id),
        (
            "team_member",
            isinstance(loaded_team_member, _Unavailable) or (team_unavailable and valid_token.user_id is not None),
            f"{valid_token.user_id}:{valid_token.team_id}",
        ),
        ("user", isinstance(loaded_user, _Unavailable), valid_token.user_id),
        (
            "organization",
            isinstance(loaded_organization, _Unavailable) or (team_unavailable and valid_token.org_id is None),
            org_id,
        ),
        ("project", isinstance(loaded_project, _Unavailable), valid_token.project_id),
        (
            "end_user",
            isinstance(loaded_end_user, _Unavailable)
            or (
                end_user_id is not None
                and isinstance(loaded_default_end_user_budget, _Unavailable)
                and (end_user_object is None or end_user_object.budget_id is None)
            ),
            end_user_id,
        ),
    )
    return MappingProxyType({scope: entity_id for scope, unavailable, entity_id in candidates if unavailable})


def _referenced_budget_ids(
    organization: LiteLLM_OrganizationTable | None,
    project: LiteLLM_ProjectTableCachedObj | None,
    tags: Sequence[tuple[str, LiteLLM_TagTable | _Unavailable | None]],
    end_user: LiteLLM_EndUserTable | None,
    default_end_user_budget: LiteLLM_BudgetTable | None,
    key_budget_id: str | None,
) -> frozenset[str]:
    candidates: Final = (
        key_budget_id,
        organization.budget_id if organization is not None else None,
        project.budget_id if project is not None else None,
        end_user.budget_id if end_user is not None else None,
        default_end_user_budget.budget_id if default_end_user_budget is not None else None,
        *(tag.budget_id for _, tag in tags if isinstance(tag, LiteLLM_TagTable)),
    )
    return frozenset(budget_id for budget_id in candidates if budget_id is not None)


async def _load_budget_meta(
    budget_ids: frozenset[str],
    deps: KeyBudgetResolverDeps,
) -> Mapping[str, _BudgetRowMeta]:
    """Reset schedules live on the budget row but are absent from LiteLLM_BudgetTable, so read them once."""
    if not budget_ids:
        return MappingProxyType({})
    try:
        rows: Final = await BudgetRepository(deps.prisma_client).find_full_by_ids(sorted(budget_ids))
    except Exception:  # noqa: BLE001  # missing reset metadata degrades the report, it must not fail it
        verbose_proxy_logger.exception("Unable to load budget reset metadata for %s", sorted(budget_ids))
        return MappingProxyType({})
    return MappingProxyType(
        {
            row.budget_id: _BudgetRowMeta(
                budget_duration=row.budget_duration,
                budget_reset_at=row.budget_reset_at,
                max_budget=row.max_budget,
            )
            for row in rows
            if row.budget_id is not None
        }
    )


async def _load_proxy_budget(deps: KeyBudgetResolverDeps) -> _ProxyBudget | _Unavailable | None:
    try:
        row: Final = await UserRepository(deps.prisma_client).find_by_id(LITELLM_PROXY_BUDGET_NAME)
    except Exception:  # noqa: BLE001  # every entity load degrades to "unknown" rather than failing the report
        verbose_proxy_logger.exception("Unable to load the proxy budget row")
        return _UNAVAILABLE
    if row is None:
        return None
    return _ProxyBudget(
        spend=row.spend,
        budget_duration=row.budget_duration,
        budget_reset_at=row.budget_reset_at,
    )


async def _load_team(team_id: str | None, deps: KeyBudgetResolverDeps) -> LiteLLM_TeamTable | _Unavailable | None:
    if team_id is None:
        return None
    try:
        return await get_team_object(
            team_id=team_id,
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            proxy_logging_obj=deps.proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load team %s for budget resolution", team_id)
        return _UNAVAILABLE


async def _load_user(user_id: str | None, deps: KeyBudgetResolverDeps) -> LiteLLM_UserTable | _Unavailable | None:
    if user_id is None:
        return None
    try:
        return await get_user_object(
            user_id=user_id,
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            user_id_upsert=False,
            proxy_logging_obj=deps.proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load user %s for budget resolution", user_id)
        return _UNAVAILABLE


async def _load_project(
    project_id: str | None, deps: KeyBudgetResolverDeps
) -> LiteLLM_ProjectTableCachedObj | _Unavailable | None:
    if project_id is None:
        return None
    try:
        return await get_project_object(
            project_id=project_id,
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            proxy_logging_obj=deps.proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load project %s for budget resolution", project_id)
        return _UNAVAILABLE


async def _load_organization(
    org_id: str | None, deps: KeyBudgetResolverDeps
) -> LiteLLM_OrganizationTable | _Unavailable | None:
    if org_id is None:
        return None
    try:
        return await get_org_object(
            org_id=org_id,
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            proxy_logging_obj=deps.proxy_logging_obj,
            include_budget_table=True,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load organization %s for budget resolution", org_id)
        return _UNAVAILABLE


async def _load_tags(
    tag_names: Sequence[str],
    deps: KeyBudgetResolverDeps,
) -> tuple[tuple[str, LiteLLM_TagTable | _Unavailable | None], ...]:
    if not tag_names:
        return ()
    try:
        tag_objects: Final = await get_tag_objects_batch(
            tag_names=list(tag_names),
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            proxy_logging_obj=deps.proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load tags %s for budget resolution", tag_names)
        return tuple((tag_name, _UNAVAILABLE) for tag_name in tag_names)
    return tuple((tag_name, tag_objects.get(tag_name)) for tag_name in tag_names)


async def _load_end_user(
    end_user_id: str | None, deps: KeyBudgetResolverDeps
) -> LiteLLM_EndUserTable | _Unavailable | None:
    if end_user_id is None:
        return None
    try:
        return await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            proxy_logging_obj=deps.proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load end user %s for budget resolution", end_user_id)
        return _UNAVAILABLE


async def _load_default_end_user_budget(deps: KeyBudgetResolverDeps) -> LiteLLM_BudgetTable | _Unavailable | None:
    try:
        return await get_default_end_user_budget(
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to load the default end user budget")
        return _UNAVAILABLE


async def _load_team_member(
    valid_token: UserAPIKeyAuth,
    team: LiteLLM_TeamTable | None,
    deps: KeyBudgetResolverDeps,
) -> TeamMemberBudget | _Unavailable | None:
    if team is None or valid_token.user_id is None:
        return None
    try:
        return await resolve_team_member_budget(
            team_object=team,
            user_id=valid_token.user_id,
            prisma_client=deps.prisma_client,
            user_api_key_cache=deps.user_api_key_cache,
            proxy_logging_obj=deps.proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # see _load_proxy_budget
        verbose_proxy_logger.exception("Unable to resolve the team-member budget for team %s", team.team_id)
        return _UNAVAILABLE


def _plan_budgets(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    return (
        *_plan_proxy(context),
        *_plan_key(context),
        *_plan_key_windows(context),
        *_plan_team(context),
        *_plan_team_windows(context),
        *_plan_team_member(context),
        *_plan_user(context),
        *_plan_organization(context),
        *_plan_project(context),
        *_plan_tags(context),
        *_plan_end_user(context),
        *_unresolved_plans(context),
    )


_UNRESOLVED_COMPARISON: Final[Mapping[BudgetScope, BudgetComparison]] = MappingProxyType(
    {
        "team": ">",
        "team_window": ">=",
        "team_member": ">=",
        "user": ">=",
        "organization": ">=",
        "project": ">",
        "tag": ">",
        "end_user": ">",
    }
)


def _unresolved_plan(scope: BudgetScope, entity_id: str | None) -> _PlannedBudget:
    return _PlannedBudget(
        scope=scope,
        entity_id=entity_id,
        entity_label=None,
        enforcement="hard",
        max_budget=None,
        comparison=_UNRESOLVED_COMPARISON[scope],
        source="unavailable",
        spend_source=_UnknownSpend(),
        notes=(_ENTITY_UNAVAILABLE_NOTE,),
    )


def _unresolved_plans(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    return tuple(_unresolved_plan(scope, entity_id) for scope, entity_id in context.unresolved.items())


def _budget_meta(context: _KeyBudgetContext, budget_id: str | None) -> _BudgetRowMeta:
    if budget_id is None:
        return _BudgetRowMeta(budget_duration=None, budget_reset_at=None)
    return context.budget_meta.get(budget_id, _BudgetRowMeta(budget_duration=None, budget_reset_at=None))


def _plan_proxy(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    proxy: Final = context.proxy
    max_budget: Final = litellm.max_budget if litellm.max_budget > 0 else None
    match proxy:
        case _Unavailable():
            return (
                _PlannedBudget(
                    scope="proxy",
                    entity_id=None,
                    entity_label=None,
                    enforcement="hard",
                    max_budget=max_budget,
                    comparison=">",
                    source="litellm_settings.max_budget",
                    spend_source=_UnknownSpend(),
                    notes=(_ENTITY_UNAVAILABLE_NOTE,),
                ),
            )
        case _:
            return (
                _PlannedBudget(
                    scope="proxy",
                    entity_id=None,
                    entity_label=None,
                    enforcement="hard",
                    max_budget=max_budget,
                    comparison=">",
                    source="litellm_settings.max_budget",
                    spend_source=_RecordedSpend(proxy.spend if proxy is not None else 0.0),
                    budget_duration=proxy.budget_duration if proxy is not None else None,
                    budget_reset_at=proxy.budget_reset_at if proxy is not None else None,
                ),
            )


def _key_max_budget_source(context: _KeyBudgetContext) -> str:
    """The key column wins over its linked budget row, so only name the row when the row is what applies."""
    budget_id: Final = context.valid_token.budget_id
    if budget_id is None:
        return "key.max_budget"
    linked: Final = context.budget_meta.get(budget_id)
    if linked is not None and linked.max_budget is not None and linked.max_budget == context.valid_token.max_budget:
        return f"budget_table:{budget_id}"
    return "key.max_budget"


def _plan_key(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    from litellm.proxy.auth.budget_throttle import should_throttle_budget_exceeded

    token: Final = context.valid_token
    throttled: Final = should_throttle_budget_exceeded(token)
    hard: Final = _PlannedBudget(
        scope="key",
        entity_id=token.key_alias,
        entity_label=token.key_alias,
        enforcement="throttled" if throttled else "hard",
        max_budget=token.max_budget,
        comparison=">=",
        source=_key_max_budget_source(context),
        spend_source=_CounterSpend(counter_key=key_spend_counter(token.token), fallback_spend=token.spend or 0.0),
        budget_duration=token.budget_duration,
        budget_reset_at=token.budget_reset_at,
        notes=(_THROTTLE_NOTE,) if throttled else (),
    )
    return (hard,)


def _plan_key_windows(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    token: Final = context.valid_token
    return tuple(
        _PlannedBudget(
            scope="key_window",
            entity_id=window.budget_duration,
            entity_label=token.key_alias,
            enforcement="hard",
            max_budget=window.max_budget,
            comparison=">=",
            source=f"key.budget_limits[{window.budget_duration}]",
            spend_source=_CounterSpend(
                counter_key=key_window_spend_counter(token.token, window.budget_duration),
                fallback_spend=0.0,
                window_entity_type="Key",
                window_entity_id=token.token,
                window_start=get_budget_window_start(window.model_dump()),
            ),
            budget_duration=window.budget_duration,
            budget_reset_at=window.reset_at,
            window_start=get_budget_window_start(window.model_dump()),
            notes=(_ROLLING_WINDOW_NOTE,),
        )
        for window in context.token_inputs.budget_limits
    )


def _plan_team(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    team: Final = context.team
    if team is None:
        return ()
    hard: Final = _PlannedBudget(
        scope="team",
        entity_id=team.team_id,
        entity_label=team.team_alias,
        enforcement="hard",
        max_budget=team.max_budget,
        comparison=">",
        source="team.max_budget",
        spend_source=_CounterSpend(counter_key=team_spend_counter(team.team_id), fallback_spend=team.spend or 0.0),
        budget_duration=team.budget_duration,
        budget_reset_at=team.budget_reset_at,
    )
    return (hard,)


def _plan_team_windows(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    team: Final = context.team
    if team is None:
        return ()
    return tuple(
        _PlannedBudget(
            scope="team_window",
            entity_id=window.budget_duration,
            entity_label=team.team_alias,
            enforcement="hard",
            max_budget=window.max_budget,
            comparison=">=",
            source=f"team.budget_limits[{window.budget_duration}]",
            spend_source=_CounterSpend(
                counter_key=team_window_spend_counter(team.team_id, window.budget_duration),
                fallback_spend=0.0,
                window_entity_type="Team",
                window_entity_id=team.team_id,
                window_start=get_budget_window_start(window.model_dump()),
            ),
            budget_duration=window.budget_duration,
            budget_reset_at=window.reset_at,
            window_start=get_budget_window_start(window.model_dump()),
            notes=(_ROLLING_WINDOW_NOTE,),
        )
        for window in (team.budget_limits or ())
    )


def _plan_team_member(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    team: Final = context.team
    member: Final = context.team_member
    user_id: Final = context.valid_token.user_id
    if team is None or member is None or user_id is None:
        return ()
    meta: Final = _budget_meta(context, _budget_id_from_source(member.source))
    return (
        _PlannedBudget(
            scope="team_member",
            entity_id=f"{user_id}:{team.team_id}",
            entity_label=team.team_alias,
            enforcement="hard",
            max_budget=member.max_budget,
            comparison=">=",
            source=member.source,
            spend_source=_CounterSpend(
                counter_key=team_member_spend_counter(user_id, team.team_id),
                fallback_spend=member.recorded_spend,
            ),
            budget_duration=meta.budget_duration,
            budget_reset_at=meta.budget_reset_at,
        ),
    )


def _budget_id_from_source(source: str) -> str | None:
    prefix: Final = "budget_table:"
    return source[len(prefix) :] if source.startswith(prefix) else None


def _plan_user(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    user: Final = context.user
    if user is None:
        return ()
    applies: Final = user_budget_applies_to_key(team_object=context.team, general_settings=context.general_settings)
    return (
        _PlannedBudget(
            scope="user",
            entity_id=user.user_id,
            entity_label=user.user_email or user.user_alias,
            enforcement="hard",
            max_budget=user.max_budget if applies else None,
            comparison=">=",
            source="user.max_budget",
            spend_source=_CounterSpend(counter_key=user_spend_counter(user.user_id), fallback_spend=user.spend or 0.0),
            budget_duration=user.budget_duration,
            budget_reset_at=user.budget_reset_at,
            notes=() if applies else (_USER_ON_TEAM_KEY_NOTE,),
        ),
    )


def _positive_or_none(max_budget: float | None) -> float | None:
    """Several checks treat a non-positive cap as 'unset' rather than as an immediate block."""
    return max_budget if max_budget is not None and max_budget > 0 else None


def _plan_organization(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    org: Final = context.organization
    if org is None or org.organization_id is None:
        return ()
    meta: Final = _budget_meta(context, org.budget_id)
    linked: Final = org.litellm_budget_table
    return (
        _PlannedBudget(
            scope="organization",
            entity_id=org.organization_id,
            entity_label=org.organization_alias,
            enforcement="hard",
            max_budget=_positive_or_none(linked.max_budget if linked is not None else None),
            comparison=">=",
            source=f"budget_table:{org.budget_id}",
            spend_source=_CounterSpend(
                counter_key=org_spend_counter(org.organization_id), fallback_spend=org.spend or 0.0
            ),
            budget_duration=meta.budget_duration,
            budget_reset_at=meta.budget_reset_at,
        ),
    )


def _plan_project(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    project: Final = context.project
    if project is None:
        return ()
    meta: Final = _budget_meta(context, project.budget_id)
    linked: Final = project.litellm_budget_table
    hard: Final = _PlannedBudget(
        scope="project",
        entity_id=project.project_id,
        entity_label=project.project_alias,
        enforcement="hard",
        max_budget=linked.max_budget if linked is not None else None,
        comparison=">",
        source=f"budget_table:{project.budget_id}" if project.budget_id else "project.budget_id",
        spend_source=_RecordedSpend(project.spend or 0.0),
        budget_duration=meta.budget_duration,
        budget_reset_at=meta.budget_reset_at,
        notes=(_PROJECT_SPEND_NOTE,),
    )
    return (hard,)


def _plan_tags(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    return tuple(_plan_tag(context, tag_name, tag) for tag_name, tag in context.tags)


def _plan_tag(context: _KeyBudgetContext, tag_name: str, tag: LiteLLM_TagTable | _Unavailable | None) -> _PlannedBudget:
    if isinstance(tag, _Unavailable):
        return _unresolved_plan("tag", tag_name)
    return _PlannedBudget(
        scope="tag",
        entity_id=tag_name,
        entity_label=None,
        enforcement="hard",
        max_budget=(
            tag.litellm_budget_table.max_budget if tag is not None and tag.litellm_budget_table is not None else None
        ),
        comparison=">",
        source=f"budget_table:{tag.budget_id}" if tag is not None and tag.budget_id else "tag.budget_id",
        spend_source=_CounterSpend(
            counter_key=tag_spend_counter(tag_name),
            fallback_spend=(tag.spend or 0.0) if tag is not None else 0.0,
            fallback_authoritative=True,
        ),
        budget_duration=_budget_meta(context, tag.budget_id if tag is not None else None).budget_duration,
        budget_reset_at=_budget_meta(context, tag.budget_id if tag is not None else None).budget_reset_at,
        notes=(_TAG_NOTE,),
    )


def _plan_end_user(context: _KeyBudgetContext) -> tuple[_PlannedBudget, ...]:
    """
    An end user with no row still gets `max_end_user_budget_id`; one that could not be read gets nothing.

    `get_end_user_object` swallows its own read failures into the same None auth falls back on, so the
    default cap below covers a failed read too. Reaching `unresolved` takes a raise, and both auth call
    sites wrap that fallback in the `try` a raise escapes, leaving the request with no end user cap.
    """
    end_user_id: Final = context.end_user_id
    if end_user_id is None or "end_user" in context.unresolved:
        return ()
    end_user: Final = context.end_user
    budget: Final = end_user.litellm_budget_table if end_user is not None else context.default_end_user_budget
    budget_id: Final = budget.budget_id if budget is not None else None
    meta: Final = _budget_meta(context, budget_id)
    row_source: Final = (
        f"budget_table:{budget_id}"
        if end_user is not None and end_user.budget_id is not None
        else "litellm_settings.max_end_user_budget_id"
    )
    primary: Final = _PlannedBudget(
        scope="end_user",
        entity_id=end_user_id,
        entity_label=end_user.alias if end_user is not None else None,
        enforcement="hard",
        max_budget=budget.max_budget if budget is not None else None,
        comparison=">",
        source=row_source,
        spend_source=_CounterSpend(
            counter_key=end_user_spend_counter(end_user_id),
            fallback_spend=(end_user.spend or 0.0) if end_user is not None else 0.0,
            fallback_authoritative=True,
        ),
        budget_duration=meta.budget_duration,
        budget_reset_at=meta.budget_reset_at,
        notes=(
            (_END_USER_ROUTE_NOTE, _CUSTOM_AUTH_END_USER_NOTE)
            if context.custom_auth_enabled
            else (_END_USER_ROUTE_NOTE,)
        ),
    )
    return (primary,)

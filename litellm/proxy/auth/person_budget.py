import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from typing import Final

import litellm
from litellm.litellm_core_utils.duration_parser import get_budget_window_start
from litellm.models.person_budget import (
    PersonBudgetAssignmentRecord,
    PersonBudgetDefinition,
    PersonBudgetDuration,
    PersonBudgetPolicy,
    PersonBudgetSubject,
)
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings, compute_budget_reset_at
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, get_management_object_ttl
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.spend_counter_batch import bind_spend_counter_keys
from litellm.proxy.utils import PrismaClient
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import CustomerMemberRepository
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth
from litellm.types.proxy.carried_budget_state import PersonBudgetWindow


class PersonBudgetFailure(str, Enum):
    UNAVAILABLE = "Personal budget configuration or spend is unavailable."


@dataclass(frozen=True, slots=True)
class PersonBudgetExceeded:
    window: PersonBudgetWindow
    spend: float


def person_budget_definition_cache_key(budget_id: str) -> str:
    return f"person_budget:{budget_id}"


def person_budget_assignment_cache_key(customer_member_id: str) -> str:
    return f"person_budget_assignment:customer_member:{customer_member_id}"


async def _member_assignment(
    customer: CustomerHierarchyAuth | None, prisma_client: PrismaClient | None, cache: UserApiKeyCache
) -> PersonBudgetAssignmentRecord | PersonBudgetFailure | None:
    if customer is None:
        return None
    cache_key: Final = person_budget_assignment_cache_key(customer.customer_member_id)
    try:
        cached: Final = await cache.async_get_cache(key=cache_key, model_type=PersonBudgetAssignmentRecord)
        if cached is not None:
            return cached
        if prisma_client is None:
            return PersonBudgetFailure.UNAVAILABLE
        row: Final = await CustomerMemberRepository(prisma_client).table.find_unique(
            where={"customer_member_id": customer.customer_member_id}
        )
        if row is None:
            return PersonBudgetFailure.UNAVAILABLE
        assignment: Final = PersonBudgetAssignmentRecord.model_validate(row)
        await cache.async_set_cache(
            key=cache_key,
            value=assignment,
            model_type=PersonBudgetAssignmentRecord,
            ttl=get_management_object_ttl(cache),
        )
    except Exception:  # noqa: BLE001  # an unreadable assignment must not silently remove personal caps
        return PersonBudgetFailure.UNAVAILABLE
    return assignment


async def load_person_budget_definition(
    budget_id: str | None, prisma_client: PrismaClient | None, cache: UserApiKeyCache
) -> PersonBudgetDefinition | PersonBudgetFailure | None:
    if budget_id is None:
        return None
    cache_key: Final = person_budget_definition_cache_key(budget_id)
    try:
        cached: Final = await cache.async_get_cache(key=cache_key, model_type=PersonBudgetDefinition)
        if cached is not None:
            return cached
        if prisma_client is None:
            return PersonBudgetFailure.UNAVAILABLE
        row: Final = await BudgetRepository(prisma_client).table.find_unique(where={"budget_id": budget_id})
        if row is None:
            return PersonBudgetFailure.UNAVAILABLE
        budget: Final = PersonBudgetDefinition.model_validate(row)
        await cache.async_set_cache(
            key=cache_key, value=budget, model_type=PersonBudgetDefinition, ttl=get_management_object_ttl(cache)
        )
    except Exception:  # noqa: BLE001  # configuration failures deny only people with assigned budgets
        return PersonBudgetFailure.UNAVAILABLE
    return budget


def person_budget_windows(
    budget: PersonBudgetPolicy,
    subject_type: PersonBudgetSubject,
    subject_id: str,
    now: datetime,
    settings: BudgetResetSettings,
) -> tuple[PersonBudgetWindow, ...]:
    return tuple(
        _window(budget, subject_type, subject_id, limit.budget_duration, limit.max_budget, now, settings)
        for limit in budget.budget_limits or ()
    )


def _window(
    budget: PersonBudgetPolicy,
    subject_type: PersonBudgetSubject,
    subject_id: str,
    duration: PersonBudgetDuration,
    maximum: float,
    now: datetime,
    settings: BudgetResetSettings,
) -> PersonBudgetWindow:
    reset: Final = compute_budget_reset_at(duration, settings, now)
    return PersonBudgetWindow(
        entity_type=subject_type,
        entity_id=subject_id,
        budget_duration=duration,
        window_start=get_budget_window_start(duration, reset).astimezone(timezone.utc),
        reset_at=reset.astimezone(timezone.utc),
        max_budget=maximum + budget.increase_for(duration, now),
        cap_policy=budget.cap_policy or "pause_until_reset",
    )


async def _assignment_windows(
    assignment: PersonBudgetAssignmentRecord,
    subject_type: PersonBudgetSubject,
    subject_id: str,
    prisma_client: PrismaClient | None,
    cache: UserApiKeyCache,
    now: datetime,
    settings: BudgetResetSettings,
) -> tuple[PersonBudgetWindow, ...] | PersonBudgetFailure:
    profile, override = await asyncio.gather(
        load_person_budget_definition(assignment.budget_id, prisma_client, cache),
        load_person_budget_definition(
            assignment.budget_override_id if assignment.override_is_active(now) else None, prisma_client, cache
        ),
    )
    if isinstance(profile, PersonBudgetFailure) or isinstance(override, PersonBudgetFailure):
        return PersonBudgetFailure.UNAVAILABLE
    resolved: Final = assignment.model_copy(update={"budget": profile, "budget_override": override})
    return person_budget_windows(resolved.effective_budget(now), subject_type, subject_id, now, settings)


async def prepare_person_budget_windows(
    *,
    user: LiteLLM_UserTable | None,
    customer: CustomerHierarchyAuth | None,
    prisma_client: PrismaClient | None,
    cache: UserApiKeyCache,
    store: SpendCounterStore,
    now: datetime,
    settings: BudgetResetSettings,
) -> tuple[PersonBudgetWindow, ...] | PersonBudgetFailure | PersonBudgetExceeded:
    subjects: Final[tuple[tuple[PersonBudgetSubject, str, PersonBudgetAssignmentRecord], ...]] = (
        (
            (
                "user",
                user.user_id,
                PersonBudgetAssignmentRecord.model_validate(user),
            ),
        )
        if user is not None
        else ()
    )
    member: Final = await _member_assignment(customer, prisma_client, cache)
    if isinstance(member, PersonBudgetFailure):
        return member
    members: Final[tuple[tuple[PersonBudgetSubject, str, PersonBudgetAssignmentRecord], ...]] = (
        (("customer_member", customer.customer_member_id, member),)
        if customer is not None and member is not None
        else ()
    )
    assignments: Final[tuple[tuple[PersonBudgetSubject, str, PersonBudgetAssignmentRecord], ...]] = (
        *subjects,
        *members,
    )
    results: Final = await asyncio.gather(
        *(
            _assignment_windows(record, subject_type, subject_id, prisma_client, cache, now, settings)
            for subject_type, subject_id, record in assignments
        )
    )
    if any(isinstance(result, PersonBudgetFailure) for result in results):
        return PersonBudgetFailure.UNAVAILABLE
    windows: Final = tuple(
        chain.from_iterable(result for result in results if not isinstance(result, PersonBudgetFailure))
    )
    bind_spend_counter_keys(frozenset(window.counter_key for window in windows))
    checks: Final = await asyncio.gather(*(_check_window(prisma_client, store, window) for window in windows))
    failure: Final = next((result for result in checks if result is not None), None)
    return failure if failure is not None else windows


async def _check_window(
    prisma_client: PrismaClient | None, store: SpendCounterStore, window: PersonBudgetWindow
) -> PersonBudgetFailure | PersonBudgetExceeded | None:
    spend: Final = await SpendCounterReseed.coalesced_window(
        prisma_client,
        store,
        window.counter_key,
        window.counter_entity_type,
        window.entity_id,
        window.window_duration,
        window.window_start,
    )
    if spend is None:
        return PersonBudgetFailure.UNAVAILABLE
    if spend >= window.max_budget:
        return PersonBudgetExceeded(window, spend)
    return None


def require_person_budget_windows(
    result: tuple[PersonBudgetWindow, ...] | PersonBudgetFailure | PersonBudgetExceeded,
) -> tuple[PersonBudgetWindow, ...]:
    if isinstance(result, PersonBudgetFailure):
        raise litellm.ServiceUnavailableError(message=result.value, model="", llm_provider="")
    if isinstance(result, PersonBudgetExceeded):
        action: Final = (
            "Contact your admin for a temporary budget increase."
            if result.window.cap_policy == "pause_allow_increase"
            else "Wait until the budget resets."
        )
        raise litellm.BudgetExceededError(
            current_cost=result.spend,
            max_budget=result.window.max_budget,
            message=f"{result.window.entity_type} budget exceeded for {result.window.budget_duration}. "
            f"Resets at {result.window.reset_at.isoformat()}. {action}",
            entity_type=result.window.entity_type,
            entity_id=result.window.entity_id,
        )
    return result

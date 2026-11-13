from datetime import datetime, time, timedelta, timezone
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from litellm.models.person_budget import PersonBudgetAssignmentRecord, PersonBudgetDefinition, PersonBudgetLimit
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.person_budget import (
    PersonBudgetExceeded,
    PersonBudgetFailure,
    person_budget_assignment_cache_key,
    person_budget_definition_cache_key,
    person_budget_windows,
    prepare_person_budget_windows,
)
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth

NOW: Final = datetime(2028, 2, 29, 12, tzinfo=timezone.utc)
SETTINGS: Final = BudgetResetSettings()
PROFILE: Final = PersonBudgetDefinition(
    budget_id="shared",
    is_profile=True,
    budget_limits=(
        PersonBudgetLimit(budget_duration="1mo", max_budget=50),
        PersonBudgetLimit(budget_duration="12mo", max_budget=500),
    ),
)


@pytest.mark.parametrize(
    ("now", "start", "reset"),
    (
        (
            datetime(2028, 2, 29, 12, tzinfo=timezone.utc),
            datetime(2028, 2, 1, tzinfo=timezone.utc),
            datetime(2028, 3, 1, tzinfo=timezone.utc),
        ),
        (
            datetime(2029, 1, 1, tzinfo=timezone.utc),
            datetime(2029, 1, 1, tzinfo=timezone.utc),
            datetime(2029, 2, 1, tzinfo=timezone.utc),
        ),
    ),
)
def test_calendar_windows_include_leap_day_and_rotate_at_the_exact_boundary(
    now: datetime,
    start: datetime,
    reset: datetime,
) -> None:
    monthly, annual = person_budget_windows(PROFILE, "user", "person", now, SETTINGS)

    assert (monthly.window_start, monthly.reset_at) == (start, reset)
    assert (annual.window_start, annual.reset_at) == (
        datetime(now.year, 1, 1, tzinfo=timezone.utc),
        datetime(now.year + 1, 1, 1, tzinfo=timezone.utc),
    )


def test_configured_timezone_and_reset_time_define_both_window_boundaries() -> None:
    settings: Final = BudgetResetSettings(timezone="America/New_York", reset_time_of_day=time(6, 30))
    windows: Final = person_budget_windows(
        PROFILE, "customer_member", "member", datetime(2028, 3, 20, tzinfo=timezone.utc), settings
    )

    assert (windows[0].window_start, windows[0].reset_at) == (
        datetime(2028, 3, 1, 6, 30, tzinfo=ZoneInfo(settings.timezone)).astimezone(timezone.utc),
        datetime(2028, 4, 1, 6, 30, tzinfo=ZoneInfo(settings.timezone)).astimezone(timezone.utc),
    )


def test_profile_edits_and_temporary_increase_expiry_keep_the_same_person_spend_identity() -> None:
    increased: Final = PROFILE.model_copy(
        update={"temp_budget_duration": "1mo", "temp_budget_increase": 25, "temp_budget_expiry": NOW}
    )
    before: Final = person_budget_windows(increased, "user", "person", NOW - timedelta(seconds=1), SETTINGS)
    after: Final = person_budget_windows(PROFILE, "user", "person", NOW, SETTINGS)
    reassigned: Final = person_budget_windows(
        PROFILE.model_copy(update={"budget_id": "another-profile"}), "user", "person", NOW, SETTINGS
    )

    assert before[0].max_budget == 75
    assert after[0].max_budget == 50
    assert tuple(window.counter_key for window in before) == tuple(window.counter_key for window in after)
    assert tuple(window.counter_key for window in reassigned) == tuple(window.counter_key for window in after)


@pytest.mark.asyncio
async def test_unassigned_user_keeps_existing_behavior_without_a_database() -> None:
    result: Final = await prepare_person_budget_windows(
        user=LiteLLM_UserTable(user_id="unassigned"),
        customer=None,
        prisma_client=None,
        cache=UserApiKeyCache(),
        store=SpendCounterStore(connect(None).spend_counters()),
        now=NOW,
        settings=SETTINGS,
    )

    assert result == ()


@pytest.mark.asyncio
async def test_known_assignment_without_readable_policy_does_not_become_unlimited() -> None:
    result: Final = await prepare_person_budget_windows(
        user=LiteLLM_UserTable(user_id="person", budget_id="missing"),
        customer=None,
        prisma_client=None,
        cache=UserApiKeyCache(),
        store=SpendCounterStore(connect(None).spend_counters()),
        now=NOW,
        settings=SETTINGS,
    )

    assert result == PersonBudgetFailure.UNAVAILABLE


@pytest.mark.asyncio
async def test_people_sharing_a_profile_have_independent_live_spend() -> None:
    cache: Final = UserApiKeyCache()
    store: Final = SpendCounterStore(connect(None).spend_counters())
    await cache.async_set_cache(
        key=person_budget_definition_cache_key(PROFILE.budget_id),
        value=PROFILE,
        model_type=PersonBudgetDefinition,
    )
    for person, amount in (("first", 50.0), ("second", 0.0)):
        for window in person_budget_windows(PROFILE, "user", person, NOW, SETTINGS):
            await store.seed(window.counter_key, amount, increment_if_lost=False)
    first: Final = await prepare_person_budget_windows(
        user=LiteLLM_UserTable(user_id="first", budget_id=PROFILE.budget_id),
        customer=None,
        prisma_client=None,
        cache=cache,
        store=store,
        now=NOW,
        settings=SETTINGS,
    )
    second: Final = await prepare_person_budget_windows(
        user=LiteLLM_UserTable(user_id="second", budget_id=PROFILE.budget_id),
        customer=None,
        prisma_client=None,
        cache=cache,
        store=store,
        now=NOW,
        settings=SETTINGS,
    )

    assert isinstance(first, PersonBudgetExceeded)
    assert first.spend == 50
    assert second == person_budget_windows(PROFILE, "user", "second", NOW, SETTINGS)


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", ("1mo", "12mo"))
async def test_zero_cap_denies_even_when_the_person_has_no_spend(duration: str) -> None:
    profile: Final = PersonBudgetDefinition.model_validate(
        {"budget_id": "zero", "is_profile": True, "budget_limits": [{"budget_duration": duration, "max_budget": 0}]}
    )
    cache: Final = UserApiKeyCache()
    store: Final = SpendCounterStore(connect(None).spend_counters())
    await cache.async_set_cache(
        key=person_budget_definition_cache_key(profile.budget_id),
        value=profile,
        model_type=PersonBudgetDefinition,
    )
    windows: Final = person_budget_windows(profile, "user", "person", NOW, SETTINGS)
    await store.seed(windows[0].counter_key, 0, increment_if_lost=False)
    result: Final = await prepare_person_budget_windows(
        user=LiteLLM_UserTable(user_id="person", budget_id=profile.budget_id),
        customer=None,
        prisma_client=None,
        cache=cache,
        store=store,
        now=NOW,
        settings=SETTINGS,
    )

    assert result == PersonBudgetExceeded(windows[0], 0)


@pytest.mark.asyncio
async def test_customer_member_uses_its_assignment_and_current_counters() -> None:
    cache: Final = UserApiKeyCache()
    store: Final = SpendCounterStore(connect(None).spend_counters())
    customer: Final = CustomerHierarchyAuth(
        customer_member_id="member", customer_team_id="team", customer_org_id="org", scopes=()
    )
    await cache.async_set_cache(
        key=person_budget_assignment_cache_key("member"),
        value=PersonBudgetAssignmentRecord(budget_id=PROFILE.budget_id),
        model_type=PersonBudgetAssignmentRecord,
    )
    await cache.async_set_cache(
        key=person_budget_definition_cache_key(PROFILE.budget_id),
        value=PROFILE,
        model_type=PersonBudgetDefinition,
    )
    monthly, annual = person_budget_windows(PROFILE, "customer_member", "member", NOW, SETTINGS)
    await store.seed(monthly.counter_key, 51, increment_if_lost=False)
    await store.seed(annual.counter_key, 51, increment_if_lost=False)

    result: Final = await prepare_person_budget_windows(
        user=None,
        customer=customer,
        prisma_client=None,
        cache=cache,
        store=store,
        now=NOW,
        settings=SETTINGS,
    )

    assert result == PersonBudgetExceeded(monthly, 51)


@pytest.mark.asyncio
async def test_expired_override_does_not_require_its_definition_to_be_available() -> None:
    result: Final = await prepare_person_budget_windows(
        user=LiteLLM_UserTable(user_id="person", budget_override_id="expired", budget_override_expires_at=NOW),
        customer=None,
        prisma_client=None,
        cache=UserApiKeyCache(),
        store=SpendCounterStore(connect(None).spend_counters()),
        now=NOW,
        settings=SETTINGS,
    )

    assert result == ()

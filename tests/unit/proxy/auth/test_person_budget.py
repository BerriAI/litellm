from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.person_budget import (
    PersonBudgetAssignmentRecord,
    PersonBudgetDefinition,
    PersonBudgetDuration,
    PersonBudgetLimit,
    PersonBudgetSubject,
)
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


def test_profile_edits_and_temporary_increase_expiry_keep_the_same_person_spend_identity() -> None:
    increased: Final = PROFILE.model_copy(
        update={"temp_budget_duration": "1mo", "temp_budget_increase": 25, "temp_budget_expiry": NOW}
    )
    before: Final = person_budget_windows(increased, "user", "person", NOW - timedelta(seconds=1), SETTINGS)
    after: Final = person_budget_windows(increased, "user", "person", NOW, SETTINGS)
    reassigned: Final = person_budget_windows(
        PROFILE.model_copy(update={"budget_id": "another-profile"}), "user", "person", NOW, SETTINGS
    )

    assert before[0].max_budget == 75
    assert before[1].max_budget == 500
    assert after[0].max_budget == 50
    assert tuple(window.counter_key for window in before) == tuple(window.counter_key for window in after)
    assert tuple(window.counter_key for window in reassigned) == tuple(window.counter_key for window in after)


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", ("user", "customer_member"))
@pytest.mark.parametrize("duration", ("1mo", "12mo"))
@pytest.mark.parametrize("cap", (0, 50))
async def test_each_person_spends_independently_and_zero_caps_deny(
    subject: PersonBudgetSubject,
    duration: PersonBudgetDuration,
    cap: int,
) -> None:
    profile: Final = PersonBudgetDefinition(
        budget_id="shared",
        is_profile=True,
        budget_limits=(PersonBudgetLimit(budget_duration=duration, max_budget=cap),),
    )
    cache: Final = UserApiKeyCache()
    store: Final = SpendCounterStore(connect(None).spend_counters())
    await cache.async_set_cache(
        key=person_budget_definition_cache_key(profile.budget_id),
        value=profile,
        model_type=PersonBudgetDefinition,
    )
    for person, spent in (("first", 50), ("second", 0)):
        (window,) = person_budget_windows(profile, subject, person, NOW, SETTINGS)
        await store.seed(window.counter_key, spent, increment_if_lost=False)
    for person, spent in (("first", 50), ("second", 0)):
        (window,) = person_budget_windows(profile, subject, person, NOW, SETTINGS)
        await cache.async_set_cache(
            key=person_budget_assignment_cache_key(person),
            value=PersonBudgetAssignmentRecord(budget_id=profile.budget_id),
            model_type=PersonBudgetAssignmentRecord,
        )
        assert await prepare_person_budget_windows(
            user=LiteLLM_UserTable(user_id=person, budget_id=profile.budget_id) if subject == "user" else None,
            customer=CustomerHierarchyAuth(
                customer_member_id=person,
                customer_team_id="team",
                customer_org_id="org",
                scopes=(),
            )
            if subject == "customer_member"
            else None,
            prisma_client=None,
            cache=cache,
            store=store,
            now=NOW,
            settings=SETTINGS,
        ) == (PersonBudgetExceeded(window, spent) if spent >= cap else (window,))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user", "expected"),
    (
        (LiteLLM_UserTable(user_id="unassigned"), ()),
        (LiteLLM_UserTable(user_id="person", budget_id="missing"), PersonBudgetFailure.UNAVAILABLE),
        (LiteLLM_UserTable(user_id="person", budget_override_id="expired", budget_override_expires_at=NOW), ()),
    ),
)
async def test_only_active_assignments_require_a_readable_policy(
    user: LiteLLM_UserTable,
    expected: tuple[()] | PersonBudgetFailure,
) -> None:
    result: Final = await prepare_person_budget_windows(
        user=user,
        customer=None,
        prisma_client=None,
        cache=UserApiKeyCache(),
        store=SpendCounterStore(connect(None).spend_counters()),
        now=NOW,
        settings=SETTINGS,
    )
    assert result == expected

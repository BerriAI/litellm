from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.models.budget import LiteLLM_BudgetTable
import litellm
from litellm.proxy.auth.person_budget import PersonBudgetRecord, person_budget_windows, prepare_person_budget_windows
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.management_helpers.customer_hierarchy import effective_member_budget
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

NOW: Final = datetime(2028, 2, 15, tzinfo=timezone.utc)


def profile() -> Budget:
    return Budget.model_validate({
        "budget_limits": (
            {"budget_duration": "1mo", "max_budget": 100},
            {"budget_duration": "12mo", "max_budget": 1000},
        ),
    })


@pytest.mark.parametrize("subject_type", ("user", "customer_member"))
def test_calendar_caps_have_independent_person_and_period_counters(subject_type: str) -> None:
    from typing import Literal
    from pydantic import TypeAdapter

    kind: Final = TypeAdapter(Literal["user", "customer_member"]).validate_python(subject_type)
    first: Final = person_budget_windows(profile(), kind, "first", NOW)
    second: Final = person_budget_windows(profile(), kind, "second", NOW)
    march: Final = person_budget_windows(profile(), kind, "first", datetime(2028, 3, 1, tzinfo=timezone.utc))
    assert tuple(window.max_budget for window in first) == (100, 1000)
    assert first[0].window_start == datetime(2028, 2, 1, tzinfo=timezone.utc)
    assert first[0].reset_at == datetime(2028, 3, 1, tzinfo=timezone.utc)
    assert first[1].window_start == datetime(2028, 1, 1, tzinfo=timezone.utc)
    assert first[1].reset_at == datetime(2029, 1, 1, tzinfo=timezone.utc)
    assert first[0].counter_key != second[0].counter_key
    assert first[0].counter_key != march[0].counter_key
    assert first[1].counter_key == march[1].counter_key


def test_personal_monthly_override_keeps_annual_profile_cap_and_spend_identity() -> None:
    override: Final = Budget.model_validate({"budget_limits": ({"budget_duration": "1mo", "max_budget": 50},)})
    inherited: Final = person_budget_windows(profile(), "user", "person", NOW)
    effective: Final = effective_member_budget(profile(), override, None, NOW)
    adjusted: Final = person_budget_windows(effective, "user", "person", NOW)
    assert tuple(window.max_budget for window in adjusted) == (50, 1000)
    assert tuple(window.counter_key for window in adjusted) == tuple(window.counter_key for window in inherited)
    expired: Final = effective_member_budget(profile(), override, NOW, NOW)
    assert expired == profile()


@pytest.mark.parametrize("duration,expected", (("1mo", (125, 1000)), ("12mo", (100, 1025))))
def test_direct_increase_expires_without_changing_profile_or_usage_identity(
    duration: str, expected: tuple[int, int],
) -> None:
    record: Final = PersonBudgetRecord(
        budget_template=profile().model_dump(),
        budget={
            "temp_budget_increase": 25,
            "temp_budget_expiry": NOW + timedelta(hours=1),
            "temp_budget_duration": duration,
        },
    )
    active: Final = person_budget_windows(record.effective_budget(NOW), "user", "person", NOW)
    expired_at: Final = NOW + timedelta(hours=1)
    expired: Final = person_budget_windows(record.effective_budget(expired_at), "user", "person", expired_at)
    assert tuple(window.max_budget for window in active) == expected
    assert tuple(window.max_budget for window in expired) == (100, 1000)
    assert tuple(window.counter_key for window in active) == tuple(window.counter_key for window in expired)
    assert profile().budget_limits is not None
    assert tuple(limit.max_budget for limit in profile().budget_limits or ()) == (100, 1000)


def test_zero_limit_is_a_real_cap_and_annual_reset_opens_a_new_window() -> None:
    budget: Final = Budget.model_validate({"budget_limits": ({"budget_duration": "12mo", "max_budget": 0},)})
    before: Final = person_budget_windows(budget, "customer_member", "person", NOW)[0]
    after: Final = person_budget_windows(budget, "customer_member", "person", before.reset_at)[0]
    assert before.max_budget == 0
    assert before.counter_key != after.counter_key
    assert after.window_start == before.reset_at


@pytest.mark.parametrize("limits", (
    ({"budget_duration": "1mo", "max_budget": -1},),
    ({"budget_duration": "1mo", "max_budget": float("inf")},),
    ({"budget_duration": "1mo", "max_budget": 1}, {"budget_duration": "1mo", "max_budget": 2}),
    ({"budget_duration": "1mo", "max_budget": 1, "reset_at": NOW},),
))
def test_invalid_personal_limits_are_rejected(limits: tuple[dict[str, object], ...]) -> None:
    with pytest.raises(ValueError, match="budget_limits"):
        Budget.model_validate({"budget_limits": limits})


def test_existing_scalar_temporary_increase_is_unchanged() -> None:
    budget: Final = LiteLLM_BudgetTable(
        max_budget=100, temp_budget_increase=25, temp_budget_expiry=NOW + timedelta(hours=1),
    )
    assert budget.effective_max_budget(NOW) == 125
    assert budget.effective_max_budget(NOW + timedelta(hours=1)) == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_type", ("user", "customer_member"))
@pytest.mark.parametrize("spend,blocked", ((0, False), (100, True)))
async def test_admission_reseeds_persisted_spend_and_denies_at_the_cap(
    monkeypatch: pytest.MonkeyPatch, subject_type: Literal["user", "customer_member"], spend: float, blocked: bool
) -> None:
    from litellm.proxy import proxy_server

    database: Final = MagicMock()
    record: Final = PersonBudgetRecord(budget_template=profile().model_dump())
    database.db.litellm_usertable.find_unique = AsyncMock(return_value=record)
    database.db.litellm_customermembertable.find_unique = AsyncMock(return_value=record)
    database.db.litellm_budgetwindowspend.find_unique = AsyncMock(
        return_value=SimpleNamespace(window_start=datetime(2028, 2, 1, tzinfo=timezone.utc), spend=spend)
    )
    store: Final = FakeSpendCounters().store()
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    client: Final = cast("PrismaClient", database)  # cast-ok: mocked database boundary
    user_id: Final = "person" if subject_type == "user" else None
    member_id: Final = "person" if subject_type == "customer_member" else None
    if blocked:
        with pytest.raises(litellm.BudgetExceededError):
            await prepare_person_budget_windows(client, user_id, member_id, "/chat/completions", NOW)
    else:
        windows: Final = await prepare_person_budget_windows(client, user_id, member_id, "/chat/completions", NOW)
        assert tuple(window.max_budget for window in windows) == (100, 1000)
        assert await SpendCounterReseed.from_db(client, windows[0].counter_key) == spend


@pytest.mark.asyncio
async def test_assignment_lookup_failure_denies_inference_but_keeps_management_reachable() -> None:
    database: Final = MagicMock()
    database.db.litellm_usertable.find_unique = AsyncMock(side_effect=ConnectionError("unavailable"))
    client: Final = cast("PrismaClient", database)  # cast-ok: mocked database boundary
    with pytest.raises(litellm.ServiceUnavailableError):
        await prepare_person_budget_windows(client, "person", None, "/chat/completions", NOW)
    assert await prepare_person_budget_windows(client, "person", None, "/budget/update", NOW) == ()

from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.models.person_budget import (
    PersonBudgetAssignmentRecord,
    PersonBudgetDefinition,
    PersonBudgetLimit,
    PersonBudgetPolicy,
)

NOW: Final = datetime(2028, 2, 29, 12, tzinfo=timezone.utc)
PROFILE: Final = PersonBudgetDefinition(
    budget_id="profile",
    is_profile=True,
    budget_limits=(
        PersonBudgetLimit(budget_duration="1mo", max_budget=50),
        PersonBudgetLimit(budget_duration="12mo", max_budget=500),
    ),
    cap_policy="pause_allow_increase",
)


@pytest.mark.parametrize("limits", (None, ()))
def test_clearing_duration_overrides_restores_profile_limits(
    limits: tuple[PersonBudgetLimit, ...] | None,
) -> None:
    assignment: Final = PersonBudgetAssignmentRecord(
        budget=PROFILE,
        budget_override=PersonBudgetDefinition(budget_id="override", budget_limits=limits),
    )

    assert assignment.effective_budget(NOW) == PersonBudgetPolicy(
        budget_limits=PROFILE.budget_limits, cap_policy="pause_allow_increase"
    )


def test_zero_override_blocks_only_its_duration_and_preserves_annual_inheritance() -> None:
    assignment: Final = PersonBudgetAssignmentRecord(
        budget=PROFILE,
        budget_override=PersonBudgetDefinition(
            budget_id="override", budget_limits=(PersonBudgetLimit(budget_duration="1mo", max_budget=0),),
        ),
    )

    assert assignment.effective_budget(NOW).budget_limits == (
        PersonBudgetLimit(budget_duration="1mo", max_budget=0),
        PersonBudgetLimit(budget_duration="12mo", max_budget=500),
    )
    assert PROFILE.budget_limits == (
        PersonBudgetLimit(budget_duration="1mo", max_budget=50),
        PersonBudgetLimit(budget_duration="12mo", max_budget=500),
    )


@pytest.mark.parametrize("expiry", (NOW, NOW - timedelta(seconds=1)))
def test_expired_whole_override_restores_profile_and_discards_its_increase(expiry: datetime) -> None:
    assignment: Final = PersonBudgetAssignmentRecord(
        budget=PROFILE,
        budget_override_expires_at=expiry,
        budget_override=PersonBudgetDefinition(
            budget_id="override",
            budget_limits=(PersonBudgetLimit(budget_duration="1mo", max_budget=100),),
            temp_budget_duration="1mo", temp_budget_increase=20, temp_budget_expiry=NOW + timedelta(days=1),
        ),
    )

    assert assignment.effective_budget(NOW) == PersonBudgetPolicy(
        budget_limits=PROFILE.budget_limits, cap_policy="pause_allow_increase"
    )


def test_temporary_increase_expires_without_mutating_stored_policy() -> None:
    policy: Final = PersonBudgetPolicy(
        temp_budget_duration="12mo", temp_budget_increase=25, temp_budget_expiry=NOW
    )

    assert policy.increase_for("12mo", NOW - timedelta(seconds=1)) == 25
    assert policy.increase_for("1mo", NOW - timedelta(seconds=1)) == 0
    assert policy.increase_for("12mo", NOW) == 0
    assert policy.temp_budget_increase == 25


@pytest.mark.parametrize(
    "payload",
    (
        {"budget_duration": "30d", "max_budget": 50},
        {"budget_duration": "1mo", "max_budget": -1},
        {"budget_duration": "12mo", "max_budget": float("inf")},
        {"budget_duration": "1mo", "max_budget": 50, "reset_at": NOW},
    ),
)
def test_personal_limits_reject_rolling_periods_invalid_amounts_and_manual_boundaries(payload: object) -> None:
    with pytest.raises(ValidationError):
        PersonBudgetLimit.model_validate(payload)


def test_a_duration_cannot_have_two_competing_caps() -> None:
    with pytest.raises(ValidationError, match="only once"):
        PersonBudgetPolicy(
            budget_limits=(
                PersonBudgetLimit(budget_duration="1mo", max_budget=50),
                PersonBudgetLimit(budget_duration="1mo", max_budget=100),
            ),
        )

"""A team member's effective budget and reset time agree with what auth enforces."""

from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.budget import LiteLLM_BudgetTable, LiteLLM_BudgetTableFull
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy.spend_tracking.team_member_budget import TeamMemberBudget, hours_until, team_member_budget

NOW: Final = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
RESET_AT: Final = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)
OWN_RESET_AT: Final = datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc)


def _membership(budget: LiteLLM_BudgetTable | None, rollover_credit: float | None = None) -> LiteLLM_TeamMembership:
    return LiteLLM_TeamMembership(
        user_id="u1",
        team_id="t1",
        budget_id=budget.budget_id if budget is not None else None,
        litellm_budget_table=budget,
        rollover_credit=rollover_credit,
    )


def _budget(max_budget: float | None, **fields: object) -> LiteLLM_BudgetTable:
    return LiteLLM_BudgetTable.model_validate({"budget_id": "b1", "max_budget": max_budget, **fields})


def _full_budget(max_budget: float | None, budget_reset_at: datetime | None) -> LiteLLM_BudgetTableFull:
    return LiteLLM_BudgetTableFull(
        budget_id="b1", max_budget=max_budget, budget_reset_at=budget_reset_at, created_at=NOW - timedelta(days=30)
    )


ACTIVE_TEMP_INCREASE: Final = {"temp_budget_increase": 3.0, "temp_budget_expiry": NOW + timedelta(hours=1)}
EXPIRED_TEMP_INCREASE: Final = {"temp_budget_increase": 3.0, "temp_budget_expiry": NOW - timedelta(seconds=1)}


@pytest.mark.parametrize(
    ("membership", "team_default", "expected"),
    [
        pytest.param(_membership(_budget(20.0)), _budget(5.0), 20.0, id="own-cap-wins-over-default"),
        pytest.param(_membership(_budget(5.0)), None, 5.0, id="linked-to-default-row"),
        pytest.param(_membership(None), _budget(5.0), 5.0, id="null-link-inherits-default"),
        pytest.param(
            _membership(_budget(None, **ACTIVE_TEMP_INCREASE)), _budget(5.0), 8.0, id="uncapped-row-temp-on-default"
        ),
        pytest.param(
            _membership(_budget(None, **EXPIRED_TEMP_INCREASE)), _budget(5.0), 5.0, id="expired-temp-is-ignored"
        ),
        pytest.param(
            _membership(_budget(None, **ACTIVE_TEMP_INCREASE)), _budget(0.0), None, id="temp-without-default-is-none"
        ),
        pytest.param(_membership(_budget(0.0)), _budget(5.0), 0.0, id="own-zero-cap-blocks"),
        pytest.param(_membership(None), _budget(0.0), None, id="zero-default-is-no-budget"),
        pytest.param(_membership(None), None, None, id="no-default-no-cap"),
        pytest.param(
            _membership(_budget(10.0, rollover_max_budget=15.0, budget_duration="1d"), rollover_credit=2.0),
            None,
            12.0,
            id="own-cap-with-rollover-credit",
        ),
        pytest.param(None, _budget(5.0), 5.0, id="no-membership-row-inherits-default"),
    ],
)
def test_effective_member_budget_matches_enforcement(
    membership: LiteLLM_TeamMembership | None, team_default: LiteLLM_BudgetTable | None, expected: float | None
) -> None:
    result: Final = team_member_budget(membership=membership, team_default_budget=team_default, now=NOW)

    assert result == (None if expected is None else TeamMemberBudget(max_budget=expected, budget_reset_at=None))


@pytest.mark.parametrize(
    ("membership", "expected"),
    [
        pytest.param(_membership(_full_budget(5.0, RESET_AT)), RESET_AT, id="linked-default-row-reset"),
        pytest.param(_membership(_full_budget(None, OWN_RESET_AT)), OWN_RESET_AT, id="own-uncapped-row-reset"),
        pytest.param(_membership(None), None, id="null-link-never-resets"),
        pytest.param(_membership(_budget(5.0)), None, id="row-without-reset-field"),
    ],
)
def test_reset_time_is_the_linked_rows(membership: LiteLLM_TeamMembership, expected: datetime | None) -> None:
    team_default: Final = _full_budget(5.0, RESET_AT + timedelta(days=3))

    result: Final = team_member_budget(membership=membership, team_default_budget=team_default, now=NOW)

    assert result is not None
    assert result.budget_reset_at == expected


@pytest.mark.parametrize(
    ("budget_reset_at", "expected_hours"),
    [
        pytest.param(NOW + timedelta(minutes=90), 1.5, id="aware"),
        pytest.param((NOW + timedelta(hours=3)).replace(tzinfo=None), 3.0, id="naive-is-utc"),
        pytest.param(NOW - timedelta(minutes=30), -0.5, id="overdue-is-negative"),
    ],
)
def test_hours_until_counts_from_the_given_now(budget_reset_at: datetime, expected_hours: float) -> None:
    assert hours_until(budget_reset_at, NOW) == pytest.approx(expected_hours)

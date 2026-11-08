from collections.abc import Callable
from dataclasses import replace
from math import inf, nan
from typing import Final, get_args

import pytest

from litellm._auth import common_budgets as native
from litellm._auth.common_budgets import (
    AlertFacts,
    BudgetDecision,
    BudgetFacts,
    BudgetScope,
    MemberLimitFacts,
    WindowFacts,
)

from .budget_reference import common as reference


@pytest.mark.parametrize("scope", get_args(BudgetScope))
@pytest.mark.parametrize("limit", (None, -1.0, 0.0, 1.0, nan, inf, -inf))
@pytest.mark.parametrize("spend", (None, -1.0, 0.0, 1.0, 2.0, nan, inf, -inf))
def test_numeric_budget_case_table_matches_reference(
    scope: BudgetScope, limit: float | None, spend: float | None
) -> None:
    facts: Final = BudgetFacts(scope, limit, spend, token_present=True, alert_recipients=("owner",))
    assert native.check(facts) == reference(facts)


@pytest.mark.parametrize("policy", (reference, native.check), ids=("reference", "native"))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (BudgetFacts("Team", 1, 1), BudgetDecision()),
        (BudgetFacts("Key", 1, 1), BudgetDecision(denied=True, alert="token_budget")),
        (BudgetFacts("Team", 1, 2, token_present=True), BudgetDecision(denied=True, alert="team_budget")),
        (BudgetFacts("User", 1, 2, team_key=True), BudgetDecision()),
        (
            BudgetFacts("User", 1, 2, team_key=True, apply_user_budget_to_team_keys=True),
            BudgetDecision(denied=True, alert="user_budget"),
        ),
        (BudgetFacts("EndUser", 1, 2, info_route=True), BudgetDecision()),
        (BudgetFacts("EndUser", 1, 1), BudgetDecision()),
        (BudgetFacts("Project", inf), BudgetDecision()),
        (BudgetFacts("Team", inf), BudgetDecision(read_spend=True)),
        (BudgetFacts("Proxy", 1, 2, skip=True), BudgetDecision()),
        (BudgetFacts("Proxy", 1, 2, llm_route=False), BudgetDecision()),
        (BudgetFacts("Proxy", 1, 2, model_list_route=True), BudgetDecision()),
        (BudgetFacts("Proxy", 1, 2), BudgetDecision(denied=True)),
        (
            BudgetFacts("TeamSoft", 0, 0, token_present=True, alert_recipients=("a",)),
            BudgetDecision(alert="soft_budget"),
        ),
        (BudgetFacts("TeamSoft", 1, 2, token_present=True), BudgetDecision()),
        (BudgetFacts("ProjectSoft", 1, 2, token_present=True), BudgetDecision(alert="soft_budget")),
        (BudgetFacts("KeySoft", 0, 1), BudgetDecision()),
        (BudgetFacts("KeySoft", -1, 0), BudgetDecision(alert="soft_budget")),
        (BudgetFacts("Tag", 0, nan), BudgetDecision(denied=True)),
        (BudgetFacts("AccessGroup", nan, 0), BudgetDecision(denied=True)),
    ),
)
def test_reference_and_native_obey_the_same_policy_table(
    policy: Callable[[BudgetFacts], BudgetDecision],
    facts: BudgetFacts,
    expected: BudgetDecision,
) -> None:
    assert policy(facts) == expected


@pytest.mark.parametrize("percentage", (None, -1.0, 0.0, 0.5, 1.0, 1.1, nan, inf))
@pytest.mark.parametrize("opt_in", (False, True))
@pytest.mark.parametrize(("tpm", "rpm"), ((None, None), (0, None), (None, 0), (100, 100)))
def test_throttles_only_an_exceeded_opted_in_key(
    percentage: float | None, opt_in: bool, tpm: int | None, rpm: int | None
) -> None:
    facts: Final = BudgetFacts(
        "Key", 1, 2, throttle_percentage=percentage, throttle_opt_in=opt_in, tpm_limit=tpm, rpm_limit=rpm
    )
    assert native.check(facts) == reference(facts)
    assert native.check(replace(facts, spend=0)) == BudgetDecision(alert="token_budget")


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (AlertFacts(8, 10), True),
        (AlertFacts(10, 10), False),
        (AlertFacts(0, 0), False),
        (AlertFacts(nan, 10), False),
        (AlertFacts(nan, 10, True, 80, team_member=True), True),
        (AlertFacts(0, 0, True, 0, team_member=True), False),
        (AlertFacts(10, 10, True, 80), True),
        (AlertFacts(7, 10, True, 80), False),
        (AlertFacts(10, 10, True, None), False),
        (AlertFacts(10, 10, False, team_member=True), False),
    ),
)
def test_threshold_alert_policy(facts: AlertFacts, expected: bool) -> None:
    assert native.alert(facts) is expected


@pytest.mark.parametrize(
    ("member", "default", "increase", "expiry", "expected"),
    (
        (1, 5, 2, 101, 3),
        (1, 5, 2, 100, 1),
        (1, 5, 2, 99, 1),
        (0, 5, 2, 101, 2),
        (None, 5, 2, 101, 7),
        (None, 0, 2, 101, None),
        (None, -1, 2, 101, None),
        (None, None, 2, 101, None),
        (1, 5, None, 101, 1),
        (1, 5, 2, None, 1),
    ),
)
def test_temporary_member_budget_expiry_is_exclusive(
    member: float | None, default: float | None, increase: float | None, expiry: float | None, expected: float | None
) -> None:
    assert native.member_limit(MemberLimitFacts(member, default, increase, expiry, 100)) == expected


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (WindowFacts(60, 100, 200), 40),
        (WindowFacts(60, None, 200), 140),
        (WindowFacts(None, 100, 200), None),
        (WindowFacts(0, 100, 200), 100),
    ),
)
def test_counter_window_start(facts: WindowFacts, expected: float | None) -> None:
    assert native.window_start(facts) == expected

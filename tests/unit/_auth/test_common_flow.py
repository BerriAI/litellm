from itertools import product
from typing import Final

import pytest

from litellm._auth import common_flow

_FACTS: Final = tuple(
    common_flow.Facts(model, skip, team, user, token, user_id, agent_id, customer, models)
    for model, skip, team, user, token, user_id, agent_id, customer, models in product(
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (None, "", "user"),
        (None, "", "agent"),
        (False, True),
        ((), ("model",)),
    )
)
_BUDGETS: Final = tuple(
    common_flow.BudgetFacts(token, groups, customer, table)
    for token, groups, customer, table in product((False, True), (0, 1), (False, True), (False, True))
)
_CUSTOM: Final = tuple(
    common_flow.CustomFacts(end_user_id, end_user, common, model, router)
    for end_user_id, end_user, common, model, router in product(
        (None, "", "customer"), (False, True), (False, True), (False, True), (False, True)
    )
)


@pytest.mark.parametrize("facts", _FACTS)
def test_common_model_plan_matches_reference(facts: common_flow.Facts) -> None:
    assert common_flow.plan(facts) == common_flow.Plan(
        membership=facts.token_present
        and (facts.model_present or not facts.skip_budgets)
        and facts.team_present
        and facts.user_id is not None,
        team_models=facts.model_present and facts.team_present,
        member_models=facts.model_present and facts.team_present and facts.token_present and bool(facts.user_id),
        user_models=facts.model_present and not facts.team_present and facts.user_present,
        customer_models=facts.customer_present and bool(facts.customer_models),
        agent_trace=facts.token_present and bool(facts.agent_id),
    )


@pytest.mark.parametrize("facts", _BUDGETS)
def test_budget_scopes_preserve_reference_priority(facts: common_flow.BudgetFacts) -> None:
    assert common_flow.budgets(facts) == (
        "Team",
        "TeamWindows",
        *(("KeyWindows",) if facts.token_present else ()),
        "TeamSoft",
        "Organization",
        "Tags",
        *(("Groups",) if facts.group_count else ()),
        "User",
        "Member",
        *(("Customer",) if facts.customer_present and facts.customer_budget_present else ()),
    )


@pytest.mark.parametrize("facts", _CUSTOM)
def test_custom_flow_matches_reference(facts: common_flow.CustomFacts) -> None:
    assert common_flow.custom(facts) == common_flow.CustomPlan(
        facts.end_user_id is not None,
        facts.end_user_present and not facts.common_checks,
        facts.common_checks,
        facts.model_present and facts.router_present,
    )

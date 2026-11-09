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


@pytest.mark.parametrize("entity", (False, True))
@pytest.mark.parametrize("budget", (False, True))
@pytest.mark.parametrize("mapping", (False, True))
@pytest.mark.parametrize("team", (False, True))
def test_model_budget_scope_matches_reference(entity: bool, budget: bool, mapping: bool, team: bool) -> None:
    expected: Final = "CheckModelBudgets" if entity and budget and mapping and team else "Continue"
    assert common_flow.read(common_flow.ModelBudgets("Member", entity, budget, mapping, team)) == expected


@pytest.mark.parametrize("router", (False, True))
@pytest.mark.parametrize("disabled", (False, True))
def test_fallback_eligibility_matches_reference(router: bool, disabled: bool) -> None:
    expected: Final = "AllowFallback" if router and not disabled else "Continue"
    assert common_flow.read(common_flow.FallbackAllowed(router, disabled)) == expected


@pytest.mark.parametrize(
    "facts, expected",
    (
        (common_flow.ModelBudgets("Key"), "ReadModelBudgetPresence"),
        (common_flow.ModelBudgets("Key", budget_present=False), "Continue"),
        (common_flow.ModelBudgets("Key", budget_present=True), "ReadModelBudgetToken"),
        (common_flow.ModelBudgets("User"), "ReadModelBudgetUser"),
        (common_flow.ModelBudgets("User", entity_present=False), "Continue"),
        (common_flow.ModelBudgets("User", entity_present=True), "ReadModelBudgetShape"),
        (common_flow.ModelBudgets("User", entity_present=True, mapping=False), "Continue"),
        (common_flow.ModelBudgets("Member", entity_present=True), "ReadModelBudgetTeam"),
        (common_flow.ModelBudgets("Member", entity_present=True, team_present=False), "Continue"),
        (common_flow.FallbackAllowed(False), "Continue"),
        (common_flow.FallbackAllowed(True), "ReadFallbackDisabled"),
    ),
)
def test_scope_reads_preserve_reference_short_circuit(
    facts: common_flow.ModelBudgets | common_flow.FallbackAllowed, expected: common_flow.ReadPlan
) -> None:
    assert common_flow.read(facts) == expected


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


@pytest.mark.parametrize("skip", (False, True))
@pytest.mark.parametrize("exempt", (None, False, True))
def test_budget_reads_preserve_short_circuit(skip: bool, exempt: bool | None) -> None:
    expected: Final = "SkipBudgets" if skip or exempt else "ReadExemption" if exempt is None else "CheckBudgets"
    assert common_flow.read(common_flow.BudgetSkip(skip, exempt)) == expected


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("route", (None, False, True))
def test_pricing_read_order(enabled: bool, route: bool | None) -> None:
    expected: Final = "Continue" if not enabled or route is False else "ReadRoute" if route is None else "ReadUnpriced"
    assert common_flow.read(common_flow.Pricing(enabled, route)) == expected


@pytest.mark.parametrize("count", (0, 1, 2))
def test_unpriced_denial(count: int) -> None:
    assert common_flow.read(common_flow.Unpriced(count)) == ("RejectUnpriced" if count else "Continue")


@pytest.mark.parametrize("router", (False, True))
@pytest.mark.parametrize("multipart", (None, False, True))
@pytest.mark.parametrize("resource", (None, False, True))
def test_fallback_reads_preserve_body_binding_guards(
    router: bool, multipart: bool | None, resource: bool | None
) -> None:
    expected: Final = (
        "Continue"
        if not router or multipart or resource
        else "ReadContentType"
        if multipart is None
        else "ReadResourceBinding"
        if resource is None
        else "ReadFallbackSettings"
    )
    assert common_flow.read(common_flow.MemberFallback(router, multipart, resource)) == expected

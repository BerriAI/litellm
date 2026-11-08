from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.budget_pricing import ModelPriceFacts
from litellm._auth.common_budgets import BudgetFacts
from litellm._auth.failure import NativeAuthFailure
from litellm._auth.fallback_access import FallbackAuthorizationFacts, GroupedModelScope
from litellm._auth.fallback_candidate import CandidateBudgetRead, CandidateBudgetScope, fallback_candidate
from litellm._auth.fallback_candidate import FallbackCandidateDecision as Decision
from litellm._auth.fallback_candidate import FallbackCandidateFacts as Facts
from litellm._auth.model_access import ModelAccessFacts
from litellm._auth.model_attribution import ModelGroupAttributionFacts
from litellm._auth.model_budget_data import ModelBudgetEntity, ModelBudgetSnapshot
from litellm._auth.routes import RouteCatalog, RouteContext
from litellm._auth.unpriced_models import UnpricedModelFacts
from tests.unit._auth.support.fallback_reference import authorization
from tests.unit._auth.support.model_reference import model_access
from tests.unit._auth.test_model_attribution import reference as attribution

_ENTITIES: Final[tuple[ModelBudgetEntity, ...]] = ("Key", "User", "EndUser", "Team", "TeamMember")


def reference(facts: Facts) -> Decision:
    if not all((facts.router_present, facts.user_present, facts.team_present)) or facts.membership == "Failed":
        return Decision("Complete")
    if facts.membership == "Unread":
        return Decision("ReadMembership")
    if facts.io_failed:
        return Decision("Complete")
    if facts.block_unpriced:
        if facts.route is None:
            return Decision("ReadRoute")
        if facts.route.route in facts.route.catalog.groups.get("openai_routes", ()):
            if facts.unpriced is None:
                return Decision("ReadUnpriced")
            if (
                facts.unpriced.group_found
                and facts.unpriced.deployments == ()
                and facts.unpriced.cost_deployments == ()
            ):
                return Decision("Complete")
    if facts.customer_present:
        if facts.customer_grants is None:
            return Decision("ReadCustomerGrants")
        if facts.customer_grants:
            if facts.customer is None:
                return Decision("ReadCustomer")
            if model_access(facts.customer).kind != "Allow":
                return Decision("Complete", log_failure=True)
    if facts.price is None:
        return Decision("ReadPrice")
    if not facts.price.cached:
        if facts.budget_models is None:
            return Decision("ReadBudgetModels")
        if facts.budget_scopes is None:
            return Decision("ReadBudgetScopes")
        for index, model in enumerate(facts.budget_models):
            if model in facts.budget_models[:index]:
                continue
            for entity in _ENTITIES:
                if not any(
                    scope.entity == entity and scope.identity_present and scope.configured
                    for scope in facts.budget_scopes
                ):
                    continue
                reading = next(
                    (reading for reading in facts.budgets if reading.model_index == index and reading.entity == entity),
                    None,
                )
                if reading is None:
                    return Decision("ReadBudget", model_index=index, entity=entity)
                snapshot = reading.snapshot
                spend = (snapshot.local_spend or 0.0) + (snapshot.legacy_spend or 0.0)
                if snapshot.max_budget is not None and snapshot.max_budget >= 0 and spend >= snapshot.max_budget:
                    return Decision("Complete")
    if facts.authorization is None:
        return Decision("ReadAuthorization")
    if authorization(facts.authorization).kind == "Deny":
        return Decision("Complete")
    if not facts.attribution_reset:
        return Decision("ResetAttribution")
    if facts.attribution is None:
        return Decision("ReadAttribution")
    groups: Final = attribution(facts.attribution).groups
    if not facts.attribution_stamped:
        return Decision("StampAttribution", groups=groups)
    if groups:
        if not facts.group_budgets_read:
            return Decision("ReadGroupBudgets", groups=groups)
        if any(
            budget.limit is not None and budget.limit > 0 and budget.spend is not None and budget.spend >= budget.limit
            for budget in facts.group_budgets
        ):
            return Decision("Complete", log_failure=True)
    return Decision("Complete", usable=True)


_INITIAL: Final = Facts(True, True, True)
_MEMBER: Final = replace(_INITIAL, membership="Loaded")
_FREE: Final = replace(_MEMBER, price=ModelPriceFacts(cached=True))
_PAID: Final = replace(_MEMBER, price=ModelPriceFacts(cached=False))
_SCOPE: Final = ModelAccessFacts(("model",), ("model",), object_type="key")
_AUTH: Final = replace(_FREE, authorization=FallbackAuthorizationFacts(key=GroupedModelScope(_SCOPE)))
_RESET: Final = replace(_AUTH, attribution_reset=True)
_GROUPS: Final = ModelGroupAttributionFacts(
    model_present=True,
    token_present=True,
    router_present=True,
    database_present=True,
    registry_loaded=True,
    registry=("group",),
    covering=("group",),
    grants=(("group",), (), (), (), ()),
)
_ATTRIBUTED: Final = replace(_RESET, attribution=_GROUPS)
_STAMPED: Final = replace(_ATTRIBUTED, attribution_stamped=True)
_SCOPES: Final = tuple(CandidateBudgetScope(entity, True, True) for entity in _ENTITIES)
_BUDGET: Final = replace(_PAID, budget_models=("model",), budget_scopes=_SCOPES)
_ROUTE: Final = RouteContext(
    "/chat/completions", catalog=RouteCatalog(groups={"openai_routes": ("/chat/completions",)})
)


@pytest.mark.parametrize("policy", (reference, fallback_candidate))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_INITIAL, Decision("ReadMembership")),
        (replace(_INITIAL, router_present=False), Decision("Complete")),
        (replace(_INITIAL, user_present=False), Decision("Complete")),
        (replace(_INITIAL, team_present=False), Decision("Complete")),
        (replace(_INITIAL, membership="Failed"), Decision("Complete")),
        (replace(_MEMBER, io_failed=True), Decision("Complete")),
        (replace(_MEMBER, block_unpriced=True), Decision("ReadRoute")),
        (replace(_MEMBER, block_unpriced=True, route=_ROUTE), Decision("ReadUnpriced")),
        (
            replace(
                _MEMBER,
                block_unpriced=True,
                route=_ROUTE,
                unpriced=UnpricedModelFacts("model", True, group_found=True, deployments=(), cost_deployments=()),
                customer_present=True,
            ),
            Decision("Complete"),
        ),
        (replace(_MEMBER, block_unpriced=True, route=RouteContext("/health")), Decision("ReadPrice")),
        (replace(_MEMBER, customer_present=True), Decision("ReadCustomerGrants")),
        (replace(_MEMBER, customer_present=True, customer_grants=()), Decision("ReadPrice")),
        (replace(_MEMBER, customer_present=True, customer_grants=("model",)), Decision("ReadCustomer")),
        (
            replace(
                _MEMBER, customer_present=True, customer_grants=("other",), customer=replace(_SCOPE, grants=("other",))
            ),
            Decision("Complete", log_failure=True),
        ),
        (_MEMBER, Decision("ReadPrice")),
        (_FREE, Decision("ReadAuthorization")),
        (_PAID, Decision("ReadBudgetModels")),
        (replace(_PAID, budget_models=("model",)), Decision("ReadBudgetScopes")),
        (_BUDGET, Decision("ReadBudget", entity="Key")),
        (
            replace(
                _BUDGET, budgets=(CandidateBudgetRead(0, "Key", ModelBudgetSnapshot(max_budget=1, local_spend=1)),)
            ),
            Decision("Complete"),
        ),
        (replace(_BUDGET, budget_scopes=()), Decision("ReadAuthorization")),
        (
            replace(_FREE, authorization=FallbackAuthorizationFacts(refuse_agent_keys=True, agent_id="")),
            Decision("Complete"),
        ),
        (_AUTH, Decision("ResetAttribution")),
        (_RESET, Decision("ReadAttribution")),
        (_ATTRIBUTED, Decision("StampAttribution", groups=("group",))),
        (_STAMPED, Decision("ReadGroupBudgets", groups=("group",))),
        (
            replace(_STAMPED, group_budgets_read=True, group_budgets=(BudgetFacts("AccessGroup", 1, 1),)),
            Decision("Complete", log_failure=True),
        ),
        (
            replace(_STAMPED, group_budgets_read=True, group_budgets=(BudgetFacts("AccessGroup", 1, 0),)),
            Decision("Complete", usable=True),
        ),
        (replace(_STAMPED, attribution=ModelGroupAttributionFacts(io_failed=True)), Decision("Complete", usable=True)),
    ),
)
def test_candidate_staged_case_table(policy: Callable[[Facts], Decision], facts: Facts, expected: Decision) -> None:
    assert policy(facts) == expected


@pytest.mark.parametrize("policy", (reference, fallback_candidate))
@pytest.mark.parametrize("index", range(5))
def test_candidate_model_budget_order(policy: Callable[[Facts], Decision], index: int) -> None:
    reads: Final = tuple(CandidateBudgetRead(0, entity, ModelBudgetSnapshot()) for entity in _ENTITIES[:index])
    assert policy(replace(_BUDGET, budgets=reads)) == Decision("ReadBudget", entity=_ENTITIES[index])


@pytest.mark.parametrize("policy", (reference, fallback_candidate))
def test_candidate_deduplicates_model_names_without_reordering(policy: Callable[[Facts], Decision]) -> None:
    reads: Final = tuple(CandidateBudgetRead(0, entity, ModelBudgetSnapshot()) for entity in _ENTITIES)
    assert policy(replace(_BUDGET, budget_models=("model", "model", "other"), budgets=reads)) == Decision(
        "ReadBudget", model_index=2, entity="Key"
    )


def test_candidate_native_decode_failure_is_not_a_failed_candidate() -> None:
    with pytest.raises(NativeAuthFailure):
        fallback_candidate(replace(_BUDGET, budgets=(CandidateBudgetRead(-1, "Key", ModelBudgetSnapshot()),)))

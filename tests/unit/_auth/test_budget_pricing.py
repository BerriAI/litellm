from math import inf, nan
from typing import Final

import pytest

from litellm._auth import budget_pricing as native
from litellm._auth.budget_pricing import DeploymentPrice, ModelPriceFacts, PriceDecision, RouteBudgetFacts


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (ModelPriceFacts(cached=True, lookup_failed=True), PriceDecision(True, None)),
        (ModelPriceFacts(cached=False), PriceDecision(False, None)),
        (ModelPriceFacts(lookup_failed=True), PriceDecision(False, None)),
        (ModelPriceFacts(), PriceDecision(False, False)),
        (ModelPriceFacts(group_found=True), PriceDecision(False, False)),
        (ModelPriceFacts(group_found=True, input_price=0, output_price=0), PriceDecision(False, None, True)),
        (ModelPriceFacts(group_found=True, input_price=0, output_price=0, deployments=()), PriceDecision(False, False)),
    ),
)
def test_pricing_fail_closed_and_read_plan(facts: ModelPriceFacts, expected: PriceDecision) -> None:
    assert native.price(facts) == expected


@pytest.mark.parametrize(
    ("price", "expected"), ((None, False), (-1, True), (0, True), (1, False), (inf, False), (nan, True))
)
def test_zero_price_keeps_reference_nonfinite_semantics(price: float | None, expected: bool) -> None:
    facts: Final = ModelPriceFacts(
        group_found=True, input_price=price, output_price=0, deployments=(DeploymentPrice(has_input_price=True),)
    )
    assert native.price(facts) == PriceDecision(expected, expected)


@pytest.mark.parametrize(
    ("deployment", "expected"),
    (
        (DeploymentPrice(), False),
        (DeploymentPrice(has_input_price=True), True),
        (DeploymentPrice(has_output_price=True), True),
        (DeploymentPrice(has_input_price=True, input_price=1), False),
        (DeploymentPrice(has_input_price=True, output_price=1), False),
        (DeploymentPrice(has_input_price=True, query_price=1), False),
        (DeploymentPrice(has_input_price=True, ptu_count=0, cost_per_ptu_per_hour=0), False),
        (DeploymentPrice(has_input_price=True, ptu_count=0), True),
        (DeploymentPrice(has_input_price=True, input_price=nan), True),
    ),
)
def test_explicit_deployment_cost_and_ptu_override_zero_group_price(
    deployment: DeploymentPrice, expected: bool
) -> None:
    facts: Final = ModelPriceFacts(group_found=True, input_price=0, output_price=0, deployments=(deployment,))
    assert native.price(facts) == PriceDecision(expected, expected)


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (RouteBudgetFacts(side_effect_route=True, discovery_route=True), "ReadPricing"),
        (RouteBudgetFacts(discovery_route=True, llm_route=True), "Skip"),
        (RouteBudgetFacts(llm_route=True), "ReadPricing"),
        (RouteBudgetFacts(), "ReadPassThrough"),
        (RouteBudgetFacts(registered_pass_through=False), "Skip"),
        (RouteBudgetFacts(registered_pass_through=True), "ReadPricing"),
        (RouteBudgetFacts(llm_route=True, route="/mcp", method="tools/list", body_present=True), "Skip"),
        (RouteBudgetFacts(llm_route=True, route="/mcp", method="tools/list", body_present=False), "ReadPricing"),
        (
            RouteBudgetFacts(llm_route=True, route="/mcp/tools/call", method="tools/list", body_present=True),
            "ReadPricing",
        ),
        (RouteBudgetFacts(llm_route=True, route="/mcp-rest/tools/list", body_present=True), "Skip"),
        (RouteBudgetFacts(llm_route=True, route="/mcp/team", method="initialize", body_present=True), "Skip"),
        (RouteBudgetFacts(llm_route=True, route="/mcp", method="tools/call", body_present=True), "ReadPricing"),
    ),
)
def test_route_and_mcp_exemption_policy(facts: RouteBudgetFacts, expected: str) -> None:
    assert native.route(facts) == expected

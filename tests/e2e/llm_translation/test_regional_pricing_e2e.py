"""Live e2e: regional model variants resolve their own uplifted cost-map pricing.

Bedrock publishes a regional variant of each Claude model (`us.`, `eu.`) priced
above the `global.` variant, and the cost map carries all three. The mappings
regressed so regional deployments resolved the global rate (LIT-3912), which
silently under-bills every regional call and makes a customer's cost reporting
disagree with their AWS invoice.

`/model/info` is where the resolved rate is observable without spending money on
a call, so this registers the three variants of one model and pins each rate. It
asserts the exact expected uplift rather than merely `regional > global`: a
mapping that resolved regional to some other model's rate would still satisfy an
inequality while being just as wrong.

No provider call is made, so this needs no AWS credentials.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoEntry

pytestmark = pytest.mark.e2e

BASE_MODEL = "anthropic.claude-opus-4-7"
GLOBAL_INPUT_COST = 5e-06
GLOBAL_OUTPUT_COST = 2.5e-05
REGIONAL_UPLIFT = 1.1
RELATIVE_TOLERANCE = 1e-9


def _matches(actual: float, expected: float) -> bool:
    """Float-safe equality for rates; the uplift arithmetic is not exact in binary."""
    return abs(actual - expected) <= abs(expected) * RELATIVE_TOLERANCE


def _register(
    client: EndpointsClient, resources: ResourceManager, prefix: str
) -> str:
    model = f"e2e-regional-{prefix}-{unique_marker()}"
    model_id = client.create_model(
        model,
        LiteLLMParamsBody(
            model=f"bedrock/{prefix}.{BASE_MODEL}", aws_region_name="us-east-1"
        ),
    )
    resources.defer(lambda: client.delete_model(model_id))
    return model


def _entry(entries: list[ModelInfoEntry], model_name: str) -> ModelInfoEntry:
    for entry in entries:
        if entry.model_name == model_name:
            return entry
    pytest.fail(f"{model_name} absent from /model/info; the deployment did not register")


def _resolved_costs(client: EndpointsClient, model: str) -> tuple[float, float]:
    """The rate the proxy resolved from the cost map, with no override configured."""
    resolved = _entry(client.proxy.model_info(), model).model_info
    assert resolved.input_cost_per_token is not None, (
        f"{model}: /model/info resolved no input_cost_per_token, so the deployment "
        f"matched no cost-map entry at all"
    )
    assert resolved.output_cost_per_token is not None, (
        f"{model}: /model/info resolved no output_cost_per_token"
    )
    return resolved.input_cost_per_token, resolved.output_cost_per_token


class TestRegionalUpliftPricing:
    @pytest.mark.covers("mgmt.model.info.reports_regional_uplift")
    def test_regional_variants_price_above_global(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        global_model = _register(endpoints_client, resources, "global")
        global_input, global_output = _resolved_costs(endpoints_client, global_model)

        assert _matches(global_input, GLOBAL_INPUT_COST), (
            f"global variant input rate drifted from the cost map: "
            f"expected {GLOBAL_INPUT_COST}, got {global_input}"
        )
        assert _matches(global_output, GLOBAL_OUTPUT_COST), (
            f"global variant output rate drifted from the cost map: "
            f"expected {GLOBAL_OUTPUT_COST}, got {global_output}"
        )

        for prefix in ("us", "eu"):
            model = _register(endpoints_client, resources, prefix)
            input_cost, output_cost = _resolved_costs(endpoints_client, model)

            assert _matches(input_cost, global_input * REGIONAL_UPLIFT), (
                f"{prefix}. variant input rate {input_cost} is not the expected "
                f"{REGIONAL_UPLIFT}x uplift over the global rate {global_input}; a "
                f"regional deployment billed at the global rate under-charges every call"
            )
            assert _matches(output_cost, global_output * REGIONAL_UPLIFT), (
                f"{prefix}. variant output rate {output_cost} is not the expected "
                f"{REGIONAL_UPLIFT}x uplift over the global rate {global_output}"
            )

"""Live e2e: a service_tier request bills every component at the tier's own rates.

Pins the tier-billing fixes (#35923, #35925): a priority-tier call must price
input and output at the deployment's `*_priority` rates, including the reasoning
tokens inside output (the shipped bug billed reasoning at the default-tier rate),
and the spend row must record the tier the bill was computed on.

The deployment carries custom base AND priority rates, each distinct, so a bill
computed from the wrong tier (or a mix) cannot match the expected numbers. The
prompt is a fresh unique marker per run, keeping cached tokens out of the math.
The response's own `service_tier` echo is asserted first: if OpenAI ever declined
priority processing and served the default tier, the test fails there instead of
producing a vacuous rate comparison. Reasoning is requested explicitly with
`reasoning_effort`, so the reasoning-rate assertion rests on a parameter the test
sets rather than on whatever the model happens to do by default.
"""

import pytest

from cost_rows import (
    approx_equal,
    assert_fresh_tokens_billed_at,
    assert_total_is_sum_of_components,
    poll_cost_row,
    register_priced_model,
)
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

BACKEND = "openai/gpt-5.6-luna"
OPENAI_API_KEY = "os.environ/OPENAI_API_KEY"

INPUT_RATE = 4e-05
OUTPUT_RATE = 8e-05
PRIORITY_INPUT_RATE = 6e-05
PRIORITY_OUTPUT_RATE = 1.6e-04

REASONING_EFFORT = "high"


class TestServiceTierPricing:
    @pytest.mark.covers("quota_management.spend_tracking.service_tier.bills_tier_rates")
    def test_priority_tier_bills_priority_rates(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy,
            resources,
            "tier-priced",
            LiteLLMParamsBody(
                model=BACKEND,
                api_key=OPENAI_API_KEY,
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
                input_cost_per_token_priority=PRIORITY_INPUT_RATE,
                output_cost_per_token_priority=PRIORITY_OUTPUT_RATE,
            ),
        )

        chat = unwrap(
            client.proxy.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                f"{unique_marker()} Compute 47*83 - 19*7 step by step, "
                                "then reply with just the final number."
                            ),
                        )
                    ],
                    max_completion_tokens=4000,
                    service_tier="priority",
                    reasoning_effort=REASONING_EFFORT,
                ),
            )
        )
        assert chat.service_tier == "priority", (
            f"OpenAI served tier {chat.service_tier!r} instead of priority; "
            "tier billing was never exercised"
        )
        assert chat.id, f"chat response carried no id: {chat}"

        row = poll_cost_row(client.proxy, chat.id)
        assert row is not None, f"no spend row with a cost breakdown landed for {chat.id}"
        breakdown = row.breakdown

        assert breakdown.service_tier == "priority", (
            f"the bill records pricing basis {breakdown.service_tier!r}, not priority"
        )

        assert_fresh_tokens_billed_at(row, PRIORITY_INPUT_RATE)
        assert breakdown.output_cost is not None and approx_equal(
            breakdown.output_cost, (row.completion_tokens or 0) * PRIORITY_OUTPUT_RATE
        ), (
            f"output_cost {breakdown.output_cost} != {row.completion_tokens} tokens * priority rate "
            f"{PRIORITY_OUTPUT_RATE} (base rate would give {(row.completion_tokens or 0) * OUTPUT_RATE})"
        )

        usage = chat.usage
        assert usage is not None and usage.completion_tokens_details is not None, (
            f"no completion token details on the priority call: {chat}"
        )
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        assert reasoning_tokens > 0, f"the reasoning question produced no reasoning tokens: {usage}"
        assert breakdown.reasoning_cost is not None and approx_equal(
            breakdown.reasoning_cost, reasoning_tokens * PRIORITY_OUTPUT_RATE
        ), (
            f"reasoning_cost {breakdown.reasoning_cost} != {reasoning_tokens} reasoning tokens * "
            f"priority rate {PRIORITY_OUTPUT_RATE} (the default-tier rate would give "
            f"{reasoning_tokens * OUTPUT_RATE})"
        )

        assert_total_is_sum_of_components(row)

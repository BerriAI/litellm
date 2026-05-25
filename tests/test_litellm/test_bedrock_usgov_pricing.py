"""
Validate AWS GovCloud (Bedrock us-gov-*) Anthropic pricing entries.

AWS Bedrock pricing in GovCloud carries a +20% premium over the global
Anthropic prices (not the +10% commercial-US premium). Until 2026-05-22
these entries silently mirrored commercial US, undercharging customers
by ~9%.

Source: https://aws.amazon.com/bedrock/pricing/

  Sonnet 4.5 in us-gov-* (per million tokens):
    input          = $3.60
    output         = $18.00
    cache write 5m = $4.50
    cache write 1h = $7.20
    cache read     = $0.36

Reference: https://github.com/BerriAI/litellm/issues/27120
"""

import json
import os

import pytest


@pytest.fixture(scope="module")
def model_data():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        return json.load(f)


SONNET_4_5_USGOV_KEYS = [
    "bedrock/us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "bedrock/us-gov-east-1/claude-sonnet-4-5-20250929-v1:0",
    "bedrock/us-gov-west-1/claude-sonnet-4-5-20250929-v1:0",
    "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
]


@pytest.mark.parametrize("model_key", SONNET_4_5_USGOV_KEYS)
def test_usgov_sonnet_4_5_pricing(model_data, model_key):
    """Each us-gov sonnet-4-5 entry must carry the +20%-over-global rates
    that AWS publishes on the GovCloud pricing page.
    """
    assert model_key in model_data, f"Missing model entry: {model_key}"
    info = model_data[model_key]

    assert info["input_cost_per_token"] == 3.6e-06, (
        f"{model_key}: input_cost_per_token should be $3.60/MTok "
        f"(got {info['input_cost_per_token']})"
    )
    assert (
        info["output_cost_per_token"] == 1.8e-05
    ), f"{model_key}: output_cost_per_token should be $18.00/MTok"
    assert (
        info["cache_creation_input_token_cost"] == 4.5e-06
    ), f"{model_key}: 5m cache write should be $4.50/MTok"
    assert (
        info["cache_creation_input_token_cost_above_1hr"] == 7.2e-06
    ), f"{model_key}: 1h cache write should be $7.20/MTok"
    assert (
        info["cache_read_input_token_cost"] == 3.6e-07
    ), f"{model_key}: cache read should be $0.36/MTok"


def test_usgov_carries_20_percent_premium_over_global(model_data):
    """The us-gov rates must equal 1.2x the global anthropic.* rates,
    matching AWS's documented GovCloud uplift.
    """
    global_key = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    usgov_key = "bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0"
    global_info = model_data[global_key]
    usgov_info = model_data[usgov_key]
    for field in (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "cache_read_input_token_cost",
    ):
        ratio = usgov_info[field] / global_info[field]
        assert (
            abs(ratio - 1.2) < 1e-9
        ), f"{field}: us-gov / global ratio is {ratio}, expected 1.2"


# The us-gov.anthropic.* cross-region inference profile is the only us-gov
# entry that carries the 1M-context `_above_200k_tokens` pricing tier — the
# bedrock/us-gov-{east,west}-1/ entries are capped at 200k tokens.
USGOV_CROSS_REGION_KEY = "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"

EXPECTED_USGOV_ABOVE_200K = {
    "input_cost_per_token_above_200k_tokens": 7.2e-06,
    "output_cost_per_token_above_200k_tokens": 2.7e-05,
    "cache_creation_input_token_cost_above_200k_tokens": 9.0e-06,
    "cache_creation_input_token_cost_above_1hr_above_200k_tokens": 1.44e-05,
    "cache_read_input_token_cost_above_200k_tokens": 7.2e-07,
}


@pytest.mark.parametrize("field,expected", EXPECTED_USGOV_ABOVE_200K.items())
def test_usgov_cross_region_above_200k_carries_gov_premium(model_data, field, expected):
    """The `_above_200k_tokens` tier on the us-gov cross-region inference
    profile must also carry the +20% GovCloud uplift. The original PR
    corrected the base rates but left the 200k-tier fields at the +10%
    commercial-US rates, undercharging long-context requests.
    """
    info = model_data[USGOV_CROSS_REGION_KEY]
    assert field in info, f"{USGOV_CROSS_REGION_KEY}: missing field {field}"
    assert (
        info[field] == expected
    ), f"{USGOV_CROSS_REGION_KEY}: {field} should be {expected} (got {info[field]})"


def test_usgov_cross_region_above_200k_ratio_to_global(model_data):
    """Cross-check via the property-based invariant: every `_above_200k_tokens`
    field on the us-gov cross-region profile must equal 1.2x the global
    anthropic.* rate, the same GovCloud uplift the base tier carries.
    """
    global_key = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    global_info = model_data[global_key]
    usgov_info = model_data[USGOV_CROSS_REGION_KEY]
    for field in EXPECTED_USGOV_ABOVE_200K:
        ratio = usgov_info[field] / global_info[field]
        assert (
            abs(ratio - 1.2) < 1e-9
        ), f"{field}: us-gov / global ratio is {ratio}, expected 1.2"

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
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)






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
        assert abs(ratio - 1.2) < 1e-9, f"{field}: us-gov / global ratio is {ratio}, expected 1.2"


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
        assert abs(ratio - 1.2) < 1e-9, f"{field}: us-gov / global ratio is {ratio}, expected 1.2"






















def test_usgov_east_haiku_profile_mirrors_in_region_row(model_data):
    """us-gov-east-1 serves claude-3-haiku through the us-gov. inference profile
    only, so the profile row must bill exactly like the in-region gov row.
    """
    profile = model_data["us-gov.anthropic.claude-3-haiku-20240307-v1:0"]
    in_region = model_data["bedrock/us-gov-east-1/anthropic.claude-3-haiku-20240307-v1:0"]
    assert profile["litellm_provider"] == "bedrock_converse"
    assert {k: v for k, v in profile.items() if k != "litellm_provider"} == {
        k: v for k, v in in_region.items() if k != "litellm_provider"
    }
















GOV_ROW_SOURCES = {
    "us-gov.anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "bedrock/us-gov-west-1/anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "bedrock/us-gov-east-1/anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "us-gov.nvidia.nemotron-nano-9b-v2": "nvidia.nemotron-nano-9b-v2",
    "bedrock/us-gov-west-1/nvidia.nemotron-nano-9b-v2": "nvidia.nemotron-nano-9b-v2",
    "bedrock/us-gov-east-1/nvidia.nemotron-nano-9b-v2": "nvidia.nemotron-nano-9b-v2",
    "us-gov.xai.grok-4.6": "us.xai.grok-4.6",
    "bedrock_mantle/us-gov-west-1/xai.grok-4.6": "bedrock_mantle/xai.grok-4.6",
    "bedrock_mantle/us-gov-east-1/xai.grok-4.6": "bedrock_mantle/xai.grok-4.6",
    "bedrock/us-gov-west-1/amazon.nova-2-multimodal-embeddings-v1:0": "amazon.nova-2-multimodal-embeddings-v1:0",
    "bedrock/us-gov-west-1/amazon.nova-lite-v1:0": "amazon.nova-lite-v1:0",
    "bedrock/us-gov-west-1/amazon.nova-micro-v1:0": "amazon.nova-micro-v1:0",
    "bedrock_mantle/us-gov-west-1/google.gemma-4-e2b": "bedrock_mantle/google.gemma-4-e2b",
    "bedrock_mantle/us-gov-west-1/google.gemma-4-26b-a4b": "bedrock_mantle/google.gemma-4-26b-a4b",
    "bedrock_mantle/us-gov-west-1/google.gemma-4-31b": "bedrock_mantle/google.gemma-4-31b",
    "bedrock_mantle/us-gov-west-1/openai.gpt-oss-20b": "bedrock_mantle/openai.gpt-oss-20b",
    "bedrock_mantle/us-gov-east-1/openai.gpt-oss-20b": "bedrock_mantle/openai.gpt-oss-20b",
    "bedrock_mantle/us-gov-west-1/openai.gpt-oss-120b": "bedrock_mantle/openai.gpt-oss-120b",
    "bedrock_mantle/us-gov-east-1/openai.gpt-oss-120b": "bedrock_mantle/openai.gpt-oss-120b",
}


def _non_pricing_fields(info):
    return {k: v for k, v in info.items() if "cost" not in k and k not in ("litellm_provider", "source")}


@pytest.mark.parametrize("gov_key", GOV_ROW_SOURCES)
def test_usgov_rows_keep_commercial_limits_and_capabilities(model_data, gov_key):
    """A gov row differs from the commercial row it mirrors only in price and
    provider: context limits, mode, and capability flags stay identical, so a
    hand-copied row cannot silently drop tool calling or shrink the context window.
    """
    gov = model_data[gov_key]
    assert _non_pricing_fields(gov) == _non_pricing_fields(model_data[GOV_ROW_SOURCES[gov_key]])
    assert "search_context_cost_per_query" not in gov
    assert "source" not in gov

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
    """Gov rows preserve the commercial row's non-pricing fields."""
    gov = model_data[gov_key]
    assert _non_pricing_fields(gov) == _non_pricing_fields(model_data[GOV_ROW_SOURCES[gov_key]])

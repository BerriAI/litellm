"""
Validate that Bedrock-hosted Anthropic Claude 4.5/4.6/4.7 entries carry the
1-hour prompt-cache write tier (`cache_creation_input_token_cost_above_1hr`)
in `model_prices_and_context_window.json`.

AWS Bedrock pricing (https://aws.amazon.com/bedrock/pricing/) publishes a
separate 1-hour cache write column for the Claude 4.5 / 4.6 / 4.7 family.
Without these fields, cost tracking on Bedrock 1-hour-TTL prompt caching
falls back to the 5-minute write rate and undercounts spend by ~60%.

Source values (per million tokens) for the 1-hour cache write column,
as published on the AWS Bedrock pricing page:

  Global pricing:
    Opus 4.7  / Opus 4.6 / Opus 4.5             -> $10.00
    Sonnet 4.6 / Sonnet 4.5 (regular tier)      -> $6.00
    Sonnet 4.5 long-context (>200K tier)        -> $12.00
    Haiku 4.5                                   -> $2.00

  US pricing (10% premium over Global):
    Opus 4.7 / Opus 4.6 / Opus 4.5              -> $11.00
    Sonnet 4.6 / Sonnet 4.5 (regular tier)      -> $6.60
    Sonnet 4.5 long-context (>200K tier)        -> $13.20
    Haiku 4.5                                   -> $2.20
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


# (model_key, expected 1hr cache write per token, expected 1hr LC tier or None)
GLOBAL_EXPECTED = [
    # Opus 4.7 - $10.00 / MTok
    ("anthropic.claude-opus-4-7", 1e-05, None),
    ("global.anthropic.claude-opus-4-7", 1e-05, None),
    # Opus 4.6 - $10.00 / MTok
    ("anthropic.claude-opus-4-6-v1", 1e-05, None),
    ("global.anthropic.claude-opus-4-6-v1", 1e-05, None),
    # Opus 4.5 - $10.00 / MTok
    ("anthropic.claude-opus-4-5-20251101-v1:0", 1e-05, None),
    ("global.anthropic.claude-opus-4-5-20251101-v1:0", 1e-05, None),
    # Sonnet 4.6 - $6.00 / MTok (no separate LC tier per AWS)
    ("anthropic.claude-sonnet-4-6", 6e-06, None),
    ("global.anthropic.claude-sonnet-4-6", 6e-06, None),
    # Sonnet 4.5 - $6.00 / MTok regular, $12.00 / MTok long-context (>200K)
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", 6e-06, 1.2e-05),
    ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", 6e-06, 1.2e-05),
    # Haiku 4.5 - $2.00 / MTok
    ("anthropic.claude-haiku-4-5-20251001-v1:0", 2e-06, None),
    ("anthropic.claude-haiku-4-5@20251001", 2e-06, None),
    ("global.anthropic.claude-haiku-4-5-20251001-v1:0", 2e-06, None),
]

US_EXPECTED = [
    # US is +10% over Global.
    ("us.anthropic.claude-opus-4-7", 1.1e-05, None),
    ("us.anthropic.claude-opus-4-6-v1", 1.1e-05, None),
    ("us.anthropic.claude-opus-4-5-20251101-v1:0", 1.1e-05, None),
    ("us.anthropic.claude-sonnet-4-6", 6.6e-06, None),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 6.6e-06, 1.32e-05),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", 2.2e-06, None),
]


@pytest.mark.parametrize(
    "model_key, expected_1hr, expected_1hr_lc", GLOBAL_EXPECTED + US_EXPECTED
)
def test_bedrock_anthropic_1hr_cache_write_pricing(
    model_data, model_key, expected_1hr, expected_1hr_lc
):
    assert model_key in model_data, f"Missing model entry: {model_key}"
    info = model_data[model_key]

    # 1hr cache write rate must be present and exact.
    assert "cache_creation_input_token_cost_above_1hr" in info, (
        f"{model_key}: missing cache_creation_input_token_cost_above_1hr - "
        "AWS Bedrock charges a separate 1-hour cache write rate for this model"
    )
    assert info["cache_creation_input_token_cost_above_1hr"] == expected_1hr, (
        f"{model_key}: 1hr cache write rate "
        f"{info['cache_creation_input_token_cost_above_1hr']} does not match "
        f"expected {expected_1hr} from AWS Bedrock pricing"
    )

    # 1hr cache write rate must be 1.6x the 5-minute rate (AWS standard ratio).
    five_min = info["cache_creation_input_token_cost"]
    ratio = info["cache_creation_input_token_cost_above_1hr"] / five_min
    assert (
        abs(ratio - 1.6) < 1e-9
    ), f"{model_key}: 1hr/5min ratio is {ratio}, expected 1.6"

    # Long-context (>200K) tier, where AWS publishes one.
    if expected_1hr_lc is not None:
        assert (
            "cache_creation_input_token_cost_above_1hr_above_200k_tokens" in info
        ), f"{model_key}: missing 1hr cache write tier for >200K context"
        assert (
            info["cache_creation_input_token_cost_above_1hr_above_200k_tokens"]
            == expected_1hr_lc
        ), (
            f"{model_key}: long-context 1hr cache write rate "
            f"{info['cache_creation_input_token_cost_above_1hr_above_200k_tokens']} "
            f"does not match expected {expected_1hr_lc}"
        )
        five_min_lc = info["cache_creation_input_token_cost_above_200k_tokens"]
        ratio_lc = (
            info["cache_creation_input_token_cost_above_1hr_above_200k_tokens"]
            / five_min_lc
        )
        assert (
            abs(ratio_lc - 1.6) < 1e-9
        ), f"{model_key}: long-context 1hr/5min ratio is {ratio_lc}, expected 1.6"

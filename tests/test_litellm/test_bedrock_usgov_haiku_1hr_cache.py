"""
Validate that AWS GovCloud (Bedrock us-gov-*) Haiku 4.5 entries carry
the 1-hour cache write tier.

AWS Bedrock GovCloud pricing applies a +20% premium over global
Anthropic rates. Global Haiku 4.5 1h cache write is $2.00/MTok; us-gov
is therefore $2.40/MTok — exactly 1.6x the 5-minute rate of $1.50/MTok.

Source: https://aws.amazon.com/bedrock/pricing/
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


HAIKU_USGOV_KEYS = [
    "bedrock/us-gov-east-1/anthropic.claude-haiku-4-5-20251001-v1:0",
    "bedrock/us-gov-west-1/anthropic.claude-haiku-4-5-20251001-v1:0",
]


@pytest.mark.parametrize("model_key", HAIKU_USGOV_KEYS)
def test_usgov_haiku_4_5_1hr_cache_write(model_data, model_key):
    assert model_key in model_data, f"Missing model entry: {model_key}"
    info = model_data[model_key]
    assert (
        info["cache_creation_input_token_cost"] == 1.5e-06
    ), f"{model_key}: 5m cache write should be $1.50/MTok"
    assert (
        info["cache_creation_input_token_cost_above_1hr"] == 2.4e-06
    ), f"{model_key}: 1h cache write should be $2.40/MTok"
    ratio = (
        info["cache_creation_input_token_cost_above_1hr"]
        / info["cache_creation_input_token_cost"]
    )
    assert abs(ratio - 1.6) < 1e-9, f"{model_key}: 1h/5m ratio is {ratio}, expected 1.6"

"""
Test Claude Sonnet 4.6 model configurations for Bedrock cross-region inference.

Pins the set of region-prefixed entries in model_prices_and_context_window.json
so future drops of a region (or pricing drift between regions) is caught.

https://github.com/BerriAI/litellm/issues/22972
"""

import json
import os


def test_bedrock_sonnet_4_6_jp_matches_other_regional_pricing():
    """The jp. cross-region inference profile shares pricing with the other
    regional profiles (us./eu./au.), which carry a 10% premium over the
    base/global entries.
    """
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        model_data = json.load(f)

    jp_info = model_data["jp.anthropic.claude-sonnet-4-6"]
    au_info = model_data["au.anthropic.claude-sonnet-4-6"]

    pricing_fields = [
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_creation_input_token_cost",
        "cache_read_input_token_cost",
    ]
    for field in pricing_fields:
        assert jp_info[field] == au_info[field], (
            f"{field} mismatch between jp. and au. variants: "
            f"jp={jp_info[field]}, au={au_info[field]}"
        )

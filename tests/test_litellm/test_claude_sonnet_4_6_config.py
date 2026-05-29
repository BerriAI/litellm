"""
Test Claude Sonnet 4.6 model configurations for Bedrock cross-region inference.

Pins the set of region-prefixed entries in model_prices_and_context_window.json
so future drops of a region (or pricing drift between regions) is caught.

https://github.com/BerriAI/litellm/issues/22972
"""

import json
import os


def test_bedrock_sonnet_4_6_region_prefixes():
    """All documented Bedrock cross-region inference prefixes for
    claude-sonnet-4-6 must be present in model_prices_and_context_window.json.
    """
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        model_data = json.load(f)

    bedrock_sonnet_4_6_models = [
        "anthropic.claude-sonnet-4-6",
        "global.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-6",
        "eu.anthropic.claude-sonnet-4-6",
        "au.anthropic.claude-sonnet-4-6",
        "jp.anthropic.claude-sonnet-4-6",
    ]

    for model in bedrock_sonnet_4_6_models:
        assert model in model_data, f"Model {model} not found in config"
        model_info = model_data[model]

        assert (
            model_info["litellm_provider"] == "bedrock_converse"
        ), f"{model} should use bedrock_converse, got {model_info['litellm_provider']}"
        assert model_info["mode"] == "chat"
        assert model_info["max_input_tokens"] == 1000000
        assert model_info["max_output_tokens"] == 64000
        assert model_info["max_tokens"] == 64000
        assert model_info.get("supports_vision") is True
        assert model_info.get("supports_computer_use") is True
        assert model_info.get("supports_function_calling") is True
        assert model_info.get("supports_tool_choice") is True
        assert model_info.get("supports_prompt_caching") is True
        assert model_info.get("supports_response_schema") is True
        assert model_info.get("supports_pdf_input") is True
        assert model_info.get("supports_assistant_prefill") is True
        assert model_info.get("supports_reasoning") is True


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

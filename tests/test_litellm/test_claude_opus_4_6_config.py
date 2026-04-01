"""
Validate Claude Opus 4.6 model configuration entries.
"""

import json
import os

import litellm


def test_opus_4_6_australia_region_uses_au_prefix_not_apac():
    """
    Test that Australia region uses 'au.' prefix instead of incorrect 'apac.' prefix.

    AWS Bedrock cross-region inference uses specific regional prefixes:
    - 'us.' for United States
    - 'eu.' for Europe
    - 'au.' for Australia (ap-southeast-2)
    - 'apac.' for Asia-Pacific (Singapore, ap-southeast-1)

    This test ensures the Claude Opus 4.6 model correctly uses 'au.' for Australia,
    and that 'apac.' is NOT incorrectly used for Australia region.

    Related: The 'apac.' prefix is valid for Asia-Pacific (Singapore) region models,
    but should not be used for Australia which has its own 'au.' prefix.
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    # Verify au.anthropic.claude-opus-4-6-v1 exists (correct)
    assert "au.anthropic.claude-opus-4-6-v1" in model_data, \
        "Missing Australia region model: au.anthropic.claude-opus-4-6-v1"

    # Verify apac.anthropic.claude-opus-4-6-v1 does NOT exist (incorrect)
    assert "apac.anthropic.claude-opus-4-6-v1" not in model_data, \
        "Incorrect model entry exists: apac.anthropic.claude-opus-4-6-v1 should be au.anthropic.claude-opus-4-6-v1"

    # Verify the au. model is registered in bedrock_converse_models
    assert "au.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models, \
        "au.anthropic.claude-opus-4-6-v1 not registered in bedrock_converse_models"

    # Verify apac. is NOT registered for this model
    assert "apac.anthropic.claude-opus-4-6-v1" not in litellm.bedrock_converse_models, \
        "apac.anthropic.claude-opus-4-6-v1 should not be in bedrock_converse_models"


def test_opus_4_6_model_pricing_and_capabilities():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    expected_models = {
        "claude-opus-4-6": {
            "provider": "anthropic",
            "has_long_context_pricing": True,
            "tool_use_system_prompt_tokens": 346,
            "max_input_tokens": 1000000,
        },
        "claude-opus-4-6-20260205": {
            "provider": "anthropic",
            "has_long_context_pricing": True,
            "tool_use_system_prompt_tokens": 346,
            "max_input_tokens": 1000000,
        },
        "anthropic.claude-opus-4-6-v1": {
            "provider": "bedrock_converse",
            "has_long_context_pricing": True,
            "tool_use_system_prompt_tokens": 346,
            "max_input_tokens": 1000000,
        },
        "vertex_ai/claude-opus-4-6": {
            "provider": "vertex_ai-anthropic_models",
            "has_long_context_pricing": True,
            "tool_use_system_prompt_tokens": 346,
            "max_input_tokens": 1000000,
        },
        "azure_ai/claude-opus-4-6": {
            "provider": "azure_ai",
            "has_long_context_pricing": False,
            "tool_use_system_prompt_tokens": 159,
            "max_input_tokens": 200000,
        },
    }

    for model_name, config in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == config["provider"]
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == config["max_input_tokens"]
        assert info["max_output_tokens"] == 128000
        assert info["max_tokens"] == 128000

        assert info["input_cost_per_token"] == 5e-06
        assert info["output_cost_per_token"] == 2.5e-05
        assert info["cache_creation_input_token_cost"] == 6.25e-06
        assert info["cache_read_input_token_cost"] == 5e-07

        if config["has_long_context_pricing"]:
            assert info["input_cost_per_token_above_200k_tokens"] == 1e-05
            assert info["output_cost_per_token_above_200k_tokens"] == 3.75e-05
            assert info["cache_creation_input_token_cost_above_200k_tokens"] == 1.25e-05
            assert info["cache_read_input_token_cost_above_200k_tokens"] == 1e-06

        assert info["supports_assistant_prefill"] is False
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True
        assert info["tool_use_system_prompt_tokens"] == config["tool_use_system_prompt_tokens"]


def test_opus_4_6_bedrock_regional_model_pricing():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    expected_models = {
        "global.anthropic.claude-opus-4-6-v1": {
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 2.5e-05,
            "cache_creation_input_token_cost": 6.25e-06,
            "cache_read_input_token_cost": 5e-07,
            "input_cost_per_token_above_200k_tokens": 1e-05,
            "output_cost_per_token_above_200k_tokens": 3.75e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 1.25e-05,
            "cache_read_input_token_cost_above_200k_tokens": 1e-06,
        },
        "us.anthropic.claude-opus-4-6-v1": {
            "input_cost_per_token": 5.5e-06,
            "output_cost_per_token": 2.75e-05,
            "cache_creation_input_token_cost": 6.875e-06,
            "cache_read_input_token_cost": 5.5e-07,
            "input_cost_per_token_above_200k_tokens": 1.1e-05,
            "output_cost_per_token_above_200k_tokens": 4.125e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 1.375e-05,
            "cache_read_input_token_cost_above_200k_tokens": 1.1e-06,
        },
        "eu.anthropic.claude-opus-4-6-v1": {
            "input_cost_per_token": 5.5e-06,
            "output_cost_per_token": 2.75e-05,
            "cache_creation_input_token_cost": 6.875e-06,
            "cache_read_input_token_cost": 5.5e-07,
            "input_cost_per_token_above_200k_tokens": 1.1e-05,
            "output_cost_per_token_above_200k_tokens": 4.125e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 1.375e-05,
            "cache_read_input_token_cost_above_200k_tokens": 1.1e-06,
        },
        "au.anthropic.claude-opus-4-6-v1": {
            "input_cost_per_token": 5.5e-06,
            "output_cost_per_token": 2.75e-05,
            "cache_creation_input_token_cost": 6.875e-06,
            "cache_read_input_token_cost": 5.5e-07,
            "input_cost_per_token_above_200k_tokens": 1.1e-05,
            "output_cost_per_token_above_200k_tokens": 4.125e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 1.375e-05,
            "cache_read_input_token_cost_above_200k_tokens": 1.1e-06,
        },
    }

    for model_name, expected in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]
        assert info["litellm_provider"] == "bedrock_converse"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        assert info["max_tokens"] == 128000
        assert info["supports_assistant_prefill"] is False
        assert info["tool_use_system_prompt_tokens"] == 346
        for key, value in expected.items():
            assert info[key] == value


def test_opus_4_6_alias_and_dated_metadata_match():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    alias = model_data["claude-opus-4-6"]
    dated = model_data["claude-opus-4-6-20260205"]

    keys_to_match = [
        "max_input_tokens",
        "max_output_tokens",
        "max_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "cache_read_input_token_cost",
        "input_cost_per_token_above_200k_tokens",
        "output_cost_per_token_above_200k_tokens",
        "cache_creation_input_token_cost_above_200k_tokens",
        "cache_read_input_token_cost_above_200k_tokens",
        "supports_assistant_prefill",
        "tool_use_system_prompt_tokens",
    ]
    for key in keys_to_match:
        assert alias[key] == dated[key], f"Mismatch for {key}"


def test_opus_4_6_bedrock_converse_registration():
    assert "anthropic.claude-opus-4-6-v1" in litellm.BEDROCK_CONVERSE_MODELS
    assert "global.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    assert "us.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    assert "eu.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models
    assert "au.anthropic.claude-opus-4-6-v1" in litellm.bedrock_converse_models

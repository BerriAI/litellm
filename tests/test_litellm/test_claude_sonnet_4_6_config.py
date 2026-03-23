"""
Validate Claude Sonnet 4.6 model configuration entries.
"""

import json
import os

import litellm


def test_sonnet_4_6_model_pricing_and_capabilities():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    expected_models = {
        "claude-sonnet-4-6": {
            "provider": "anthropic",
            "has_long_context_pricing": True,
            "tool_use_system_prompt_tokens": 346,
            "max_input_tokens": 1000000,
            "input_cost_per_token_above_200k_tokens": 3e-06,
            "output_cost_per_token_above_200k_tokens": 1.5e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 3.75e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3e-07,
        },
        "vertex_ai/claude-sonnet-4-6": {
            "provider": "vertex_ai-anthropic_models",
            "has_long_context_pricing": True,
            "tool_use_system_prompt_tokens": 346,
            "max_input_tokens": 1000000,
            "input_cost_per_token_above_200k_tokens": 3e-06,
            "output_cost_per_token_above_200k_tokens": 1.5e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 3.75e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3e-07,
        },
        "azure_ai/claude-sonnet-4-6": {
            "provider": "azure_ai",
            "has_long_context_pricing": False,
            "tool_use_system_prompt_tokens": None,
            "max_input_tokens": 1000000,
        },
    }

    for model_name, config in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == config["provider"]
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == config["max_input_tokens"]
        assert info["max_output_tokens"] == 64000
        assert info["max_tokens"] == 64000

        assert info["input_cost_per_token"] == 3e-06
        assert info["output_cost_per_token"] == 1.5e-05
        assert info["cache_creation_input_token_cost"] == 3.75e-06
        assert info["cache_read_input_token_cost"] == 3e-07

        if config["has_long_context_pricing"]:
            assert info["input_cost_per_token_above_200k_tokens"] == config["input_cost_per_token_above_200k_tokens"]
            assert info["output_cost_per_token_above_200k_tokens"] == config["output_cost_per_token_above_200k_tokens"]
            assert info["cache_creation_input_token_cost_above_200k_tokens"] == config["cache_creation_input_token_cost_above_200k_tokens"]
            assert info["cache_read_input_token_cost_above_200k_tokens"] == config["cache_read_input_token_cost_above_200k_tokens"]

        assert info["supports_assistant_prefill"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True

        if config["tool_use_system_prompt_tokens"] is not None:
            assert info["tool_use_system_prompt_tokens"] == config["tool_use_system_prompt_tokens"]


def test_sonnet_4_6_bedrock_regional_model_pricing():
    """
    Validate that Bedrock Claude Sonnet 4.6 regional models have correct pricing.

    Bedrock does not surcharge for >200k token context windows, so
    all *_above_200k_tokens prices must equal their base counterparts.
    max_input_tokens should be 1,000,000 (1M) for the 1M context window.
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    # Bedrock does not surcharge for >200k tokens; above_200k prices equal base prices.
    expected_models = {
        "anthropic.claude-sonnet-4-6": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_creation_input_token_cost": 3.75e-06,
            "cache_read_input_token_cost": 3e-07,
            "input_cost_per_token_above_200k_tokens": 3e-06,
            "output_cost_per_token_above_200k_tokens": 1.5e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 3.75e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3e-07,
        },
        "global.anthropic.claude-sonnet-4-6": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_creation_input_token_cost": 3.75e-06,
            "cache_read_input_token_cost": 3e-07,
            "input_cost_per_token_above_200k_tokens": 3e-06,
            "output_cost_per_token_above_200k_tokens": 1.5e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 3.75e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3e-07,
        },
        "us.anthropic.claude-sonnet-4-6": {
            "input_cost_per_token": 3.3e-06,
            "output_cost_per_token": 1.65e-05,
            "cache_creation_input_token_cost": 4.125e-06,
            "cache_read_input_token_cost": 3.3e-07,
            "input_cost_per_token_above_200k_tokens": 3.3e-06,
            "output_cost_per_token_above_200k_tokens": 1.65e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 4.125e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3.3e-07,
        },
        "eu.anthropic.claude-sonnet-4-6": {
            "input_cost_per_token": 3.3e-06,
            "output_cost_per_token": 1.65e-05,
            "cache_creation_input_token_cost": 4.125e-06,
            "cache_read_input_token_cost": 3.3e-07,
            "input_cost_per_token_above_200k_tokens": 3.3e-06,
            "output_cost_per_token_above_200k_tokens": 1.65e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 4.125e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3.3e-07,
        },
        "au.anthropic.claude-sonnet-4-6": {
            "input_cost_per_token": 3.3e-06,
            "output_cost_per_token": 1.65e-05,
            "cache_creation_input_token_cost": 4.125e-06,
            "cache_read_input_token_cost": 3.3e-07,
            "input_cost_per_token_above_200k_tokens": 3.3e-06,
            "output_cost_per_token_above_200k_tokens": 1.65e-05,
            "cache_creation_input_token_cost_above_200k_tokens": 4.125e-06,
            "cache_read_input_token_cost_above_200k_tokens": 3.3e-07,
        },
    }

    for model_name, expected in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]
        assert info["litellm_provider"] == "bedrock_converse"
        assert info["max_input_tokens"] == 1000000, \
            f"{model_name}: max_input_tokens should be 1000000, got {info['max_input_tokens']}"
        assert info["max_output_tokens"] == 64000
        assert info["max_tokens"] == 64000
        for key, value in expected.items():
            assert info[key] == value, \
                f"{model_name}[{key}]: expected {value}, got {info[key]}"


def test_sonnet_4_6_bedrock_converse_registration():
    assert "anthropic.claude-sonnet-4-6" in litellm.BEDROCK_CONVERSE_MODELS
    assert "global.anthropic.claude-sonnet-4-6" in litellm.bedrock_converse_models
    assert "us.anthropic.claude-sonnet-4-6" in litellm.bedrock_converse_models
    assert "eu.anthropic.claude-sonnet-4-6" in litellm.bedrock_converse_models
    assert "au.anthropic.claude-sonnet-4-6" in litellm.bedrock_converse_models

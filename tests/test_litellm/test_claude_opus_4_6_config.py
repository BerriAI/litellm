"""
Validate Claude Opus 4.6 model configuration entries.
"""

import json
import os

import litellm


def test_opus_4_6_model_pricing_and_capabilities():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        model_data = json.load(f)

    expected_models = {
        "claude-opus-4-6": {
            "provider": "anthropic",
            "has_long_context_pricing": True,
        },
        "anthropic.claude-opus-4-6-v1:0": {
            "provider": "bedrock_converse",
            "has_long_context_pricing": True,
        },
        "vertex_ai/claude-opus-4-6": {
            "provider": "vertex_ai-anthropic_models",
            "has_long_context_pricing": True,
        },
        "azure_ai/claude-opus-4-6": {
            "provider": "azure_ai",
            "has_long_context_pricing": False,
        },
    }

    for model_name, config in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == config["provider"]
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 200000
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

        assert info["supports_assistant_prefill"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True
        if "tool_use_system_prompt_tokens" in info:
            assert info["tool_use_system_prompt_tokens"] == 159


def test_opus_4_6_bedrock_converse_registration():
    assert "anthropic.claude-opus-4-6-v1:0" in litellm.BEDROCK_CONVERSE_MODELS

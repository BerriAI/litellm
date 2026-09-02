"""Unit tests for Morph provider integration."""

import os
from unittest.mock import patch

import litellm
from litellm import MorphChatConfig, get_llm_provider

# Force model loading
litellm.add_known_models()


def test_morph_config_get_provider_info():
    """Test that MorphChatConfig returns correct provider info."""
    config = MorphChatConfig()

    # Test with environment variable
    with patch.dict(os.environ, {"MORPH_API_KEY": "test-key-from-env"}):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.morphllm.com/v1"
        assert api_key == "test-key-from-env"

    # Test with passed api_key
    api_base, api_key = config._get_openai_compatible_provider_info(None, "direct-key")
    assert api_base == "https://api.morphllm.com/v1"
    assert api_key == "direct-key"

    # Test with custom api_base
    api_base, api_key = config._get_openai_compatible_provider_info(
        "https://custom.morph.com", "key"
    )
    assert api_base == "https://custom.morph.com"
    assert api_key == "key"


def test_morph_get_llm_provider():
    """Test that get_llm_provider correctly identifies morph models."""
    # Test with morph/model format
    _, custom_llm_provider, _, _ = get_llm_provider("morph/morph-v3-large")
    assert custom_llm_provider == "morph"

    _, custom_llm_provider, _, _ = get_llm_provider("morph/morph-v3-fast")
    assert custom_llm_provider == "morph"

    _, custom_llm_provider, _, _ = get_llm_provider("morph/morph-kimik3")
    assert custom_llm_provider == "morph"


def test_morph_in_provider_lists():
    """Test that morph is included in all necessary provider lists."""
    import litellm
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
    )

    # Check morph is in openai_compatible_providers
    assert "morph" in openai_compatible_providers

    # Check morph endpoint is in openai_compatible_endpoints
    assert "https://api.morphllm.com/v1" in openai_compatible_endpoints

    # Check morph is in provider_list
    assert "morph" in litellm.provider_list

    # Check models are in model_list after initialization
    assert all(
        model in litellm.model_list
        for model in [
            "morph/morph-dsv4flash",
            "morph/morph-glm53-744b",
            "morph/morph-glm53flash",
            "morph/morph-kimik3",
            "morph/morph-kimik3-fast",
            "morph/morph-v3-large",
            "morph/morph-v3-fast",
        ]
    )


def test_morph_model_info():
    """Test that morph models have correct configuration."""
    import litellm

    model_info = litellm.get_model_info("morph/morph-v3-large")

    assert model_info["litellm_provider"] == "morph"
    assert model_info["mode"] == "chat"
    assert model_info["max_tokens"] == 262144
    assert model_info["max_input_tokens"] == 262144
    assert model_info["max_output_tokens"] == 131072
    assert model_info["input_cost_per_token"] == 9e-07  # $0.9/1M tokens
    assert model_info["output_cost_per_token"] == 1.9e-06  # $1.9/1M tokens
    assert model_info["supports_function_calling"] is False
    assert model_info["supports_vision"] is False
    assert model_info["supports_system_messages"] is True


def test_morph_open_model_info():
    model_info = litellm.get_model_info("morph/morph-glm53flash")

    assert model_info["litellm_provider"] == "morph"
    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 1048576
    assert model_info["input_cost_per_token"] == 1.5e-07
    assert model_info["cache_read_input_token_cost"] == 1e-08
    assert model_info["output_cost_per_token"] == 4.2e-07
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_prompt_caching"] is True
    assert model_info["supports_response_schema"] is True
    assert model_info["supports_vision"] is True


def test_morph_supported_params():
    """Test that MorphChatConfig returns correct supported parameters."""
    config = MorphChatConfig()
    supported_params = config.get_supported_openai_params("morph/morph-v3-large")

    expected_params = [
        "frequency_penalty",
        "max_tokens",
        "messages",
        "model",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    ]

    assert all(param in supported_params for param in expected_params)


def test_morph_maps_tool_and_response_format_params():
    config = MorphChatConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    response_format = {"type": "json_object"}

    mapped = config.map_openai_params(
        non_default_params={
            "max_completion_tokens": 64,
            "response_format": response_format,
            "tool_choice": "required",
            "tools": tools,
        },
        optional_params={},
        model="morph/morph-glm53flash",
        drop_params=False,
    )

    assert mapped["max_tokens"] == 64
    assert mapped["response_format"] == response_format
    assert mapped["tool_choice"] == "required"
    assert mapped["tools"] == tools


def test_morph_custom_llm_provider():
    """Test that morph models are correctly identified."""
    config = MorphChatConfig()
    assert config.custom_llm_provider == "morph"

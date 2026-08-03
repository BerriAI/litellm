"""Unit tests for Morph provider integration."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

import litellm
from litellm import MorphChatConfig, get_llm_provider

MORPH_MODELS = {
    "morph-v3-fast": (262144, 131072, 8e-7, 1.2e-6, 0, ["text"]),
    "morph-v3-large": (262144, 131072, 9e-7, 1.9e-6, 0, ["text"]),
    "morph-qwen35-397b": (262144, 131072, 5e-7, 3.5e-6, 3e-7, ["text", "image"]),
    "morph-minimax3-428b": (256000, 256000, 3e-7, 1.2e-6, 0, ["text", "image"]),
    "morph-glm52-744b": (1048576, 1048576, 1.1e-6, 4.1e-6, 2.2e-7, ["text"]),
    "morph-qwen36-27b": (131072, 131072, 2.89e-7, 2.4e-6, 0, ["text"]),
    "morph-dsv4flash": (1048576, 1048576, 1.39e-7, 2.78e-7, 0, ["text"]),
    "morph-gemma4-31b": (175000, 175000, 1.4e-7, 4e-7, 8e-8, ["text", "image"]),
    "morph-kimik3": (1048576, 1048576, 2.9e-6, 1.4e-5, 2.9e-7, ["text", "image"]),
    "morph-kimik3-fast": (1048576, 1048576, 6e-6, 2.25e-5, 6e-7, ["text", "image"]),
}

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
    api_base, api_key = config._get_openai_compatible_provider_info("https://custom.morph.com", "key")
    assert api_base == "https://custom.morph.com"
    assert api_key == "key"


def test_morph_get_llm_provider():
    """Test that get_llm_provider correctly identifies morph models."""
    for model in MORPH_MODELS:
        _, custom_llm_provider, _, _ = get_llm_provider(f"morph/{model}")
        assert custom_llm_provider == "morph"


def test_morph_in_provider_lists():
    """Test that morph is included in all necessary provider lists."""
    import litellm
    from litellm.constants import (
        openai_compatible_providers,
        openai_compatible_endpoints,
    )

    # Check morph is in openai_compatible_providers
    assert "morph" in openai_compatible_providers

    # Check morph endpoint is in openai_compatible_endpoints
    assert "https://api.morphllm.com/v1" in openai_compatible_endpoints

    # Check morph is in provider_list
    assert "morph" in litellm.provider_list

    # Check models are in model_list after initialization
    assert all(model in litellm.model_list for model in [f"morph/{model}" for model in MORPH_MODELS])


def test_morph_model_info():
    """Test that morph models have correct configuration."""
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as file:
        model_cost = json.load(file)

    for model, expected in MORPH_MODELS.items():
        max_input, max_output, input_cost, output_cost, cache_cost, modalities = expected
        model_info = model_cost[f"morph/{model}"]
        public_model_info = litellm.get_model_info(f"morph/{model}")
        assert model_info["litellm_provider"] == "morph"
        assert model_info["mode"] == "chat"
        assert model_info["max_input_tokens"] == max_input
        assert model_info["max_output_tokens"] == max_output
        assert model_info["max_tokens"] == max_output
        assert model_info["input_cost_per_token"] == input_cost
        assert model_info["output_cost_per_token"] == output_cost
        assert model_info.get("cache_read_input_token_cost") == cache_cost
        assert model_info["supported_modalities"] == modalities
        assert model_info["source"] == "https://www.morphllm.com/api/models/json"
        assert model_info["supports_native_streaming"] is True
        assert model_info["supports_vision"] is ("image" in modalities)
        assert public_model_info["max_input_tokens"] == max_input
        assert public_model_info["max_output_tokens"] == max_output
        assert public_model_info["input_cost_per_token"] == input_cost
        assert public_model_info["output_cost_per_token"] == output_cost

        if model.startswith("morph-v3-"):
            assert model_info["supports_function_calling"] is False
            assert model_info["supports_system_messages"] is True
        else:
            assert model_info["supports_function_calling"] is True
            assert model_info["supports_native_structured_output"] is True
            assert model_info["supports_prompt_caching"] is True
            assert model_info["supports_response_schema"] is True
            assert model_info["supports_system_messages"] is True


def test_morph_model_info_matches_backup():
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as file:
        model_cost = json.load(file)
    with open(repo_root / "litellm/model_prices_and_context_window_backup.json") as file:
        backup_model_cost = json.load(file)

    for model in MORPH_MODELS:
        model_key = f"morph/{model}"
        assert backup_model_cost[model_key] == model_cost[model_key]


def test_morph_custom_llm_provider():
    """Test that morph models are correctly identified."""
    config = MorphChatConfig()
    assert config.custom_llm_provider == "morph"

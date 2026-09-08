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
    assert all(
        model in litellm.model_list
        for model in ["morph/morph-v3-large", "morph/morph-v3-fast"]
    )




def test_morph_supported_params():
    """Test that MorphChatConfig returns correct supported parameters."""
    config = MorphChatConfig()
    supported_params = config.get_supported_openai_params("morph/morph-v3-large")

    expected_params = [
        "messages",
        "model",
        "stream",
    ]

    assert all(param in supported_params for param in expected_params)


def test_morph_custom_llm_provider():
    """Test that morph models are correctly identified."""
    config = MorphChatConfig()
    assert config.custom_llm_provider == "morph"

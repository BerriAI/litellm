import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import get_llm_provider


def test_get_llm_provider_hyperbolic():
    """Test that hyperbolic/ prefix returns the correct provider"""
    model, provider, _, _ = get_llm_provider(model="hyperbolic/deepseek-v3")
    assert provider == "hyperbolic"
    assert model == "deepseek-v3"


def test_hyperbolic_completion_call():
    """Test basic completion call structure for Hyperbolic"""
    # This is primarily a structure test since we don't have actual API keys
    try:
        litellm.set_verbose = True
        response = litellm.completion(
            model="hyperbolic/qwen-2.5-72b",
            messages=[{"role": "user", "content": "Hello!"}],
            mock_response="Hi there!",
        )
        assert response is not None
    except Exception as e:
        # Expected to fail without valid API key, but should recognize the provider
        assert "hyperbolic" in str(e).lower() or "api" in str(e).lower()


def test_hyperbolic_config_initialization():
    """Test that HyperbolicChatConfig initializes correctly"""
    from litellm.llms.hyperbolic.chat.transformation import HyperbolicChatConfig

    config = HyperbolicChatConfig()
    assert config.custom_llm_provider == "hyperbolic"


def test_hyperbolic_get_openai_compatible_provider_info():
    """Test API base and key handling"""
    from litellm.llms.hyperbolic.chat.transformation import HyperbolicChatConfig

    config = HyperbolicChatConfig()

    # Test default API base
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == "https://api.hyperbolic.xyz/v1"
    # api_key may be set from environment, so we don't test for None

    # Test custom API base
    custom_base = "https://custom.hyperbolic.com/v1"
    api_base, api_key = config._get_openai_compatible_provider_info(custom_base, "test-key")
    assert api_base == custom_base
    assert api_key == "test-key"


def test_hyperbolic_in_provider_lists():
    """Test that hyperbolic is in all relevant provider lists"""
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
        openai_text_completion_compatible_providers,
    )

    assert "hyperbolic" in openai_compatible_providers
    assert "hyperbolic" in openai_text_completion_compatible_providers
    assert "https://api.hyperbolic.xyz/v1" in openai_compatible_endpoints


def test_hyperbolic_models_configuration():
    """Test that Hyperbolic models are properly configured"""
    import json
    import os
    
    # Load model configuration directly from the JSON file
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    # Test a few key models
    test_models = [
        "hyperbolic/deepseek-ai/DeepSeek-V3",
        "hyperbolic/Qwen/Qwen2.5-Coder-32B-Instruct",
        "hyperbolic/deepseek-ai/DeepSeek-R1",
    ]

    for model in test_models:
        assert model in model_data
        model_info = model_data[model]
        assert model_info["litellm_provider"] == "hyperbolic"
        assert model_info["mode"] == "chat"
        assert "max_tokens" in model_info
        assert "input_cost_per_token" in model_info
        assert "output_cost_per_token" in model_info


def test_hyperbolic_supported_params():
    """Test that supported OpenAI parameters are correctly configured"""
    from litellm.llms.hyperbolic.chat.transformation import HyperbolicChatConfig

    config = HyperbolicChatConfig()
    supported_params = config.get_supported_openai_params("hyperbolic/deepseek-v3")

    # Check for essential parameters
    assert "messages" in supported_params
    assert "model" in supported_params
    assert "stream" in supported_params
    assert "temperature" in supported_params
    assert "max_tokens" in supported_params
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
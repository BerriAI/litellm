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


def test_get_llm_provider_inference():
    """Test that inference/ prefix returns the correct provider"""
    model, provider, _, _ = get_llm_provider(model="inference/meta-llama/llama-3.2-3b-instruct/fp-16")
    assert provider == "inference"
    assert model == "meta-llama/llama-3.2-3b-instruct/fp-16"


def test_inference_completion_call():
    """Test basic completion call structure for Inference.net"""
    # This is primarily a structure test since we don't have actual API keys
    try:
        litellm.set_verbose = True
        response = litellm.completion(
            model="inference/meta-llama/llama-3.2-3b-instruct/fp-16",
            messages=[{"role": "user", "content": "Hello!"}],
            mock_response="Hi there!",
        )
        assert response is not None
    except Exception as e:
        # Expected to fail without valid API key, but should recognize the provider
        assert "inference" in str(e).lower() or "api" in str(e).lower()


def test_inference_config_initialization():
    """Test that InferenceChatConfig initializes correctly"""
    from litellm.llms.inference.chat.transformation import InferenceChatConfig

    config = InferenceChatConfig()
    assert config.custom_llm_provider == "inference"


def test_inference_get_openai_compatible_provider_info():
    """Test API base and key handling for inference.net"""
    from litellm.llms.inference.chat.transformation import InferenceChatConfig

    config = InferenceChatConfig()

    # Test default API base
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == "https://api.inference.net/v1"
    # api_key may be set from environment, so we don't test for None

    # Test custom API base
    custom_base = "https://custom.inference.net/v1"
    api_base, api_key = config._get_openai_compatible_provider_info(custom_base, "test-key")
    assert api_base == custom_base
    assert api_key == "test-key"


def test_inference_in_provider_lists():
    """Test that inference is in all relevant provider lists"""
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
        openai_text_completion_compatible_providers,
    )

    assert "inference" in openai_compatible_providers
    assert "inference" in openai_text_completion_compatible_providers
    assert "https://api.inference.net/v1" in openai_compatible_endpoints


def test_inference_models_configuration():
    """Test that Inference.net models are properly configured"""
    import json
    import os
    
    # Load model configuration directly from the JSON file
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    # Test a key model
    test_models = [
        "inference/meta-llama/llama-3.2-3b-instruct/fp-16",
        "inference/meta-llama/llama-3.1-8b-instruct/fp-16",
        "inference/mistralai/mistral-nemo-12b-instruct/fp-8",
    ]

    for model in test_models:
        assert model in model_data
        model_info = model_data[model]
        assert model_info["litellm_provider"] == "inference"
        assert model_info["mode"] == "chat"
        assert "max_tokens" in model_info
        assert "input_cost_per_token" in model_info
        assert "output_cost_per_token" in model_info


def test_inference_supported_params():
    """Test that supported OpenAI parameters are correctly configured for inference.net"""
    from litellm.llms.inference.chat.transformation import InferenceChatConfig

    config = InferenceChatConfig()
    supported_params = config.get_supported_openai_params("inference/meta-llama/llama-3.2-3b-instruct/fp-16")

    # Check for essential parameters
    assert "messages" in supported_params
    assert "model" in supported_params
    assert "stream" in supported_params
    assert "temperature" in supported_params
    assert "max_tokens" in supported_params
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
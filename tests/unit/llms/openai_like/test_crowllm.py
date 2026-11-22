"""
Tests for the CrowLLM provider configuration.
"""

import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestCrowLLMProviderConfig:
    """Test CrowLLM provider configuration"""

    def test_crowllm_json_config_exists(self):
        """Test that crowllm is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("crowllm")

        crowllm = JSONProviderRegistry.get("crowllm")
        assert crowllm is not None
        assert crowllm.base_url == "https://crowllm.com/v1"
        assert crowllm.api_key_env == "CROWLLM_API_KEY"

    def test_crowllm_provider_resolution(self):
        """Test that provider resolution finds crowllm"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="crowllm/glm-5.3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "glm-5.3"
        assert provider == "crowllm"
        assert api_base is None

    def test_crowllm_in_provider_list(self):
        """Test that crowllm is registered as a provider"""
        assert "crowllm" in litellm.provider_list


def test_crowllm_supported_endpoints():
    """The endpoint matrix must advertise the protocols the gateway serves."""
    import json

    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"
    data = json.loads(root_path.read_text())
    endpoints = data["providers"]["crowllm"]["endpoints"]
    assert endpoints["chat_completions"] is True
    assert endpoints["responses"] is True
    assert endpoints["messages"] is True
    assert endpoints["embeddings"] is False

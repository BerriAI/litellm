"""
Unit test for LiteLLM Proxy Responses API configuration.
"""

import pytest

from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_litellm_proxy_responses_api_config():
    """Test that litellm_proxy provider returns correct Responses API config"""
    from litellm.llms.litellm_proxy.responses.transformation import (
        LiteLLMProxyResponsesAPIConfig,
    )

    config = ProviderConfigManager.get_provider_responses_api_config(
        model="litellm_proxy/gpt-4",
        provider=LlmProviders.LITELLM_PROXY,
    )
    print(f"config: {config}")
    assert config is not None, "Config should not be None for litellm_proxy provider"
    assert isinstance(
        config, LiteLLMProxyResponsesAPIConfig
    ), f"Expected LiteLLMProxyResponsesAPIConfig, got {type(config)}"
    assert (
        config.custom_llm_provider == LlmProviders.LITELLM_PROXY
    ), "custom_llm_provider should be LITELLM_PROXY"


def test_litellm_proxy_responses_api_config_get_complete_url():
    """Test that get_complete_url works correctly"""
    import os
    from litellm.llms.litellm_proxy.responses.transformation import (
        LiteLLMProxyResponsesAPIConfig,
    )

    config = LiteLLMProxyResponsesAPIConfig()

    # Test with explicit api_base
    url = config.get_complete_url(
        api_base="https://my-proxy.example.com",
        litellm_params={},
    )
    assert url == "https://my-proxy.example.com/responses"

    # Test with trailing slash
    url = config.get_complete_url(
        api_base="https://my-proxy.example.com/",
        litellm_params={},
    )
    assert url == "https://my-proxy.example.com/responses"

    # Test that it raises error when api_base is None and env var is not set
    if "LITELLM_PROXY_API_BASE" in os.environ:
        del os.environ["LITELLM_PROXY_API_BASE"]
    
    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(api_base=None, litellm_params={})


def test_litellm_proxy_responses_api_config_inherits_from_openai():
    """Test that LiteLLMProxyResponsesAPIConfig extends OpenAI config properly"""
    from litellm.llms.litellm_proxy.responses.transformation import (
        LiteLLMProxyResponsesAPIConfig,
    )
    from litellm.llms.openai.responses.transformation import (
        OpenAIResponsesAPIConfig,
    )

    config = LiteLLMProxyResponsesAPIConfig()
    
    # Should inherit from OpenAI config
    assert isinstance(config, OpenAIResponsesAPIConfig)
    
    # Should have the correct provider set
    assert config.custom_llm_provider == LlmProviders.LITELLM_PROXY


if __name__ == "__main__":
    test_litellm_proxy_responses_api_config()
    test_litellm_proxy_responses_api_config_get_complete_url()
    test_litellm_proxy_responses_api_config_inherits_from_openai()
    print("All tests passed!")

"""
Unit test for LiteLLM Proxy Responses API configuration.
"""

from litellm.types.utils import LlmProviders


def test_litellm_proxy_responses_api_config_inherits_from_openai():
    """Test that LiteLLMProxyResponsesAPIConfig extends OpenAI config properly"""
    from litellm.llms.litellm_proxy.responses.transformation import (
        LiteLLMProxyResponsesAPIConfig,
    )
    from litellm.llms.openai.responses.transformation import (
        OpenAIResponsesAPIConfig,
    )

    config = LiteLLMProxyResponsesAPIConfig()

    assert isinstance(config, OpenAIResponsesAPIConfig)

    assert config.custom_llm_provider == LlmProviders.LITELLM_PROXY

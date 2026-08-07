"""
Unit tests for DashScope Anthropic-compatible Messages support.
"""

from litellm.llms.dashscope.messages.transformation import (
    DEFAULT_DASHSCOPE_ANTHROPIC_MESSAGES_API_BASE,
    DashScopeAnthropicMessagesConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_should_map_dashscope_compatible_base_to_anthropic_messages_endpoint():
    config = DashScopeAnthropicMessagesConfig()

    assert config.custom_llm_provider == "dashscope"
    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == DEFAULT_DASHSCOPE_ANTHROPIC_MESSAGES_API_BASE
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/apps/anthropic/v1/messages/",
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/apps/anthropic/v1",
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/custom-root",
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == "https://dashscope.aliyuncs.com/custom-root/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == "https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/apps/anthropic",
            api_key=None,
            model="qwen3-max",
            optional_params={},
            litellm_params={},
        )
        == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
    )


def test_should_prepare_dashscope_anthropic_messages_headers():
    config = DashScopeAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
        model="qwen3-max",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        api_key="dashscope-key",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert headers["x-api-key"] == "dashscope-key"
    assert headers["content-type"] == "application/json"
    assert "anthropic-version" not in headers
    assert "anthropic-beta" not in headers
    assert api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_should_preserve_existing_dashscope_anthropic_messages_headers():
    config = DashScopeAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={
            "x-api-key": "existing-key",
            "content-type": "application/custom-json",
        },
        model="qwen3-max",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        api_key="dashscope-key",
        api_base=None,
    )

    assert headers["x-api-key"] == "existing-key"
    assert headers["content-type"] == "application/custom-json"
    assert api_base is None


def test_should_route_dashscope_provider_to_native_messages_config():
    qwen_max_config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="qwen3-max",
        provider=LlmProviders.DASHSCOPE,
    )
    qwen_plus_config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="qwen-plus",
        provider=LlmProviders.DASHSCOPE,
    )

    assert isinstance(qwen_max_config, DashScopeAnthropicMessagesConfig)
    assert isinstance(qwen_plus_config, DashScopeAnthropicMessagesConfig)

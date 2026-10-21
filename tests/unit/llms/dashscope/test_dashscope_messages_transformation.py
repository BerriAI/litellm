"""
Unit tests for DashScope's Anthropic-compatible Messages endpoint.
"""

import pytest

import litellm
from litellm.llms.dashscope.messages.transformation import (
    DashScopeAnthropicMessagesConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

WORKSPACE = "https://llm-abc123.cn-beijing.maas.aliyuncs.com"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            None,
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            WORKSPACE,
            f"{WORKSPACE}/apps/anthropic/v1/messages",
        ),
        (
            f"{WORKSPACE}/compatible-mode/v1",
            f"{WORKSPACE}/apps/anthropic/v1/messages",
        ),
        (
            "https://gateway.example.com/anthropic",
            "https://gateway.example.com/anthropic/v1/messages",
        ),
    ],
)
def test_get_complete_url(api_base: str | None, expected: str):
    config = DashScopeAnthropicMessagesConfig()

    url = config.get_complete_url(
        api_base=api_base,
        api_key="sk-test",
        model="qwen3.7-plus",
        optional_params={},
        litellm_params={},
    )

    assert url == expected


def test_dashscope_resolves_to_the_native_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="qwen3.7-plus",
        provider=LlmProviders.DASHSCOPE,
    )

    assert isinstance(config, DashScopeAnthropicMessagesConfig)
    assert config.custom_llm_provider == "dashscope"


def test_dashscope_prefix_reaches_the_provider():
    model, provider, _, _ = litellm.get_llm_provider(
        "dashscope/qwen3.7-plus",
        api_base=WORKSPACE,
    )

    assert provider == "dashscope"
    assert model == "qwen3.7-plus"

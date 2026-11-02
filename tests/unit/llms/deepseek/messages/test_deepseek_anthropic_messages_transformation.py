import pytest

import litellm
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.deepseek.messages.transformation import (
    DeepSeekAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager


def test_deepseek_provider_uses_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.DEEPSEEK,
    )

    assert isinstance(config, DeepSeekAnthropicMessagesConfig)
    assert config.custom_llm_provider == "deepseek"


def test_deepseek_anthropic_messages_config_defaults():
    config = DeepSeekAnthropicMessagesConfig()

    assert config.custom_llm_provider == "deepseek"
    assert config.get_api_base() == "https://api.deepseek.com/anthropic"


def test_anthropic_provider_keeps_default_config_for_deepseek_named_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, DeepSeekAnthropicMessagesConfig)


def test_deepseek_anthropic_messages_url_defaults_to_anthropic_endpoint():
    config = DeepSeekAnthropicMessagesConfig()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/anthropic/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/anthropic",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/v1/messages",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )


def test_deepseek_anthropic_messages_headers_use_deepseek_key():
    config = DeepSeekAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="deepseek-v4-pro",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-deepseek",
        api_base="https://example.test/anthropic",
    )

    assert api_base == "https://example.test/anthropic"
    assert headers["x-api-key"] == "sk-deepseek"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


def test_deepseek_anthropic_messages_preserves_thinking_and_sanitizes_custom_tools():
    config = DeepSeekAnthropicMessagesConfig()
    messages = [
        {
            "role": "user",
            "content": "Use the tool.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should call the tool.",
                    "signature": "sig",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Sao Paulo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Sunny",
                }
            ],
        },
    ]

    request = config.transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=messages,
        anthropic_messages_optional_request_params={
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {
                    "type": "custom",
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"},
                },
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 1,
                },
            ],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["messages"] == messages
    assert request["thinking"] == {"type": "enabled"}
    assert request["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object"},
    }
    assert request["tools"][1]["type"] == "web_search_20260209"


def test_deepseek_anthropic_messages_drops_budget_tokens_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DeepSeekAnthropicMessagesConfig()
    warned: list[tuple[object, str]] = []

    def mock_warn(thinking: object, model: str) -> None:
        warned.append((thinking, model))

    monkeypatch.setattr(
        "litellm.llms.deepseek.messages.transformation._warn_on_dropped_budget",
        mock_warn,
    )

    request = config.transform_anthropic_messages_request(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "solve math"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4000,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["thinking"] == {"type": "enabled"}
    assert len(warned) == 1
    assert warned[0][1] == "deepseek-reasoner"


def test_deepseek_anthropic_messages_maps_effort_and_adaptive_thinking() -> None:
    config = DeepSeekAnthropicMessagesConfig()

    # 1. output_config with effort: low maps to thinking enabled without budget_tokens
    req_effort = config.transform_anthropic_messages_request(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "solve math"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4000,
            "output_config": {"effort": "low"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert req_effort["thinking"] == {"type": "enabled"}
    assert "output_config" not in req_effort

    # 2. adaptive thinking maps to thinking enabled without budget_tokens
    req_adaptive = config.transform_anthropic_messages_request(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "solve math"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4000,
            "thinking": {"type": "adaptive"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert req_adaptive["thinking"] == {"type": "enabled"}
    assert "budget_tokens" not in req_adaptive["thinking"]

    # 3. disabled thinking is preserved as disabled
    req_disabled = config.transform_anthropic_messages_request(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "solve math"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4000,
            "thinking": {"type": "disabled"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert req_disabled["thinking"] == {"type": "disabled"}

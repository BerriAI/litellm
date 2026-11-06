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
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert request["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object"},
    }
    assert request["tools"][1]["type"] == "web_search_20260209"


def _tool_loop_history():
    return [
        {"role": "user", "content": "List the files in /tmp, then count them."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Listing them."},
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls /tmp"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt\nb.txt"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Count them.", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_2", "name": "bash", "input": {"command": "ls /tmp | wc -l"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "2"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "tool_use", "id": "toolu_4", "name": "bash", "input": {"command": "ls -a /tmp"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_4", "content": "a.txt\nb.txt"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "There are 2 files."}]},
        {"role": "user", "content": "Now delete a.txt."},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_3", "name": "bash", "input": {"command": "rm /tmp/a.txt"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_3", "content": ""}]},
    ]


def _transform(messages, thinking_params):
    return DeepSeekAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 100, **thinking_params},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


@pytest.mark.parametrize(
    "thinking_params",
    [
        {},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"thinking": {"type": "adaptive"}},
    ],
    ids=["thinking_absent", "thinking_enabled", "thinking_adaptive"],
)
def test_deepseek_anthropic_messages_fills_empty_thinking_on_tool_use_turns(thinking_params):
    messages = _tool_loop_history()

    request = _transform(messages, thinking_params)

    empty_thinking = {"type": "thinking", "thinking": ""}
    expected = _tool_loop_history()
    expected[1] = {**expected[1], "content": [empty_thinking, *expected[1]["content"]]}
    expected[9] = {**expected[9], "content": [empty_thinking, *expected[9]["content"]]}
    assert request["messages"] == expected
    assert messages == _tool_loop_history()


def test_deepseek_anthropic_messages_leaves_history_alone_when_thinking_disabled():
    request = _transform(_tool_loop_history(), {"thinking": {"type": "disabled"}})

    assert request["messages"] == _tool_loop_history()


def test_deepseek_anthropic_messages_passes_malformed_turns_through_unchanged():
    malformed_turn = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}, "oops"],
    }
    tail = _tool_loop_history()[8:]

    request = _transform(["hello", malformed_turn, *tail], {})

    filled_turn = {**tail[1], "content": [{"type": "thinking", "thinking": ""}, *tail[1]["content"]]}
    assert request["messages"] == ["hello", malformed_turn, tail[0], filled_turn, tail[2]]

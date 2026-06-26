import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
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


def test_deepseek_anthropic_messages_preserves_thinking_sanitizes_tools_and_backfills_reasoning_content():
    """
    Thinking-mode request through the DeepSeek anthropic endpoint:
      - thinking blocks in assistant history are preserved (untouched)
      - custom tool discriminator is stripped (existing behaviour)
      - assistant messages missing `reasoning_content` get a placeholder
        backfill so DeepSeek does not reject the request with
        "The reasoning_content in the thinking mode must be passed back"
    """
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

    # thinking param preserved
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 1024}

    # tools sanitized (custom stripped, hosted tool kept)
    assert request["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object"},
    }
    assert request["tools"][1]["type"] == "web_search_20260209"

    # user + tool_result messages unchanged
    assert request["messages"][0] == messages[0]
    assert request["messages"][2] == messages[2]

    # assistant message: thinking block content preserved, reasoning_content backfilled
    assistant = request["messages"][1]
    assert assistant["content"] == messages[1]["content"]
    assert assistant["reasoning_content"] == " "


# ---------------------------------------------------------------------------
# reasoning_content backfill unit tests
# ---------------------------------------------------------------------------


def test_fill_reasoning_content_injects_placeholder_when_missing():
    config = DeepSeekAnthropicMessagesConfig()
    out = config._fill_reasoning_content(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert out[1]["reasoning_content"] == " "
    assert "reasoning_content" not in out[0]


def test_fill_reasoning_content_preserves_existing_value():
    config = DeepSeekAnthropicMessagesConfig()
    out = config._fill_reasoning_content(
        [
            {
                "role": "assistant",
                "content": "x",
                "reasoning_content": "real chain",
            }
        ]
    )
    assert out[0]["reasoning_content"] == "real chain"


def test_fill_reasoning_content_promotes_from_provider_specific_fields():
    config = DeepSeekAnthropicMessagesConfig()
    out = config._fill_reasoning_content(
        [
            {
                "role": "assistant",
                "content": "x",
                "provider_specific_fields": {
                    "reasoning_content": "stored chain",
                    "other": "keep",
                },
            }
        ]
    )
    assert out[0]["reasoning_content"] == "stored chain"
    assert "reasoning_content" not in out[0]["provider_specific_fields"]
    assert out[0]["provider_specific_fields"]["other"] == "keep"


def test_fill_reasoning_content_skips_non_assistant_messages():
    config = DeepSeekAnthropicMessagesConfig()
    out = config._fill_reasoning_content(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
        ]
    )
    assert "reasoning_content" not in out[0]
    assert "reasoning_content" not in out[1]


def test_fill_reasoning_content_handles_empty_list():
    config = DeepSeekAnthropicMessagesConfig()
    assert config._fill_reasoning_content([]) == []


def test_fill_reasoning_content_does_not_mutate_input():
    config = DeepSeekAnthropicMessagesConfig()
    src = [{"role": "assistant", "content": "x"}]
    src_snapshot = dict(src[0])
    config._fill_reasoning_content(src)
    assert src[0] == src_snapshot


def test_transform_does_not_backfill_when_thinking_disabled():
    """Non-thinking requests must not get spurious reasoning_content injection."""
    config = DeepSeekAnthropicMessagesConfig()
    out = config.transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "reasoning_content" not in out["messages"][1]


def test_thinking_mode_active_returns_true_only_when_explicitly_enabled():
    """Guard must fire ONLY on `thinking: {type: enabled}`, not other shapes."""
    config = DeepSeekAnthropicMessagesConfig()
    # explicit enabled -> fill
    assert (
        config._thinking_mode_active(
            model="deepseek-v4-pro",
            optional_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        )
        is True
    )
    # type=disabled -> no fill
    assert (
        config._thinking_mode_active(
            model="deepseek-v4-pro",
            optional_params={"thinking": {"type": "disabled"}},
        )
        is False
    )
    # thinking absent -> no fill (most common non-thinking case)
    assert (
        config._thinking_mode_active(
            model="deepseek-v4-pro",
            optional_params={"max_tokens": 100},
        )
        is False
    )
    # thinking present but not a dict (defensive) -> no fill
    assert (
        config._thinking_mode_active(
            model="deepseek-v4-pro",
            optional_params={"thinking": None},
        )
        is False
    )

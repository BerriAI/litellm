import pytest

from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.openai_like.messages.transformation import (
    OpenAILikeAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture
def config() -> OpenAILikeAnthropicMessagesConfig:
    return OpenAILikeAnthropicMessagesConfig()


@pytest.mark.parametrize(
    "api_base, expected",
    [
        ("https://host/v1", "https://host/v1/messages"),
        ("https://host/v1/", "https://host/v1/messages"),
        ("https://host", "https://host/v1/messages"),
        ("https://host/v1/messages", "https://host/v1/messages"),
        ("https://api.deepseek.com/anthropic", "https://api.deepseek.com/anthropic/v1/messages"),
        ("https://api.deepseek.com/anthropic/v1", "https://api.deepseek.com/anthropic/v1/messages"),
    ],
)
def test_get_complete_url_handles_api_base_variants(config, api_base, expected):
    url = config.get_complete_url(
        api_base=api_base,
        api_key="sk-test",
        model="some-model",
        optional_params={},
        litellm_params={},
    )
    assert url == expected


def test_get_complete_url_requires_api_base(config):
    with pytest.raises(ValueError, match="api_base is required"):
        config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="some-model",
            optional_params={},
            litellm_params={},
        )


def test_request_stays_in_anthropic_shape(config):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize this",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    optional_params = {
        "max_tokens": 256,
        "system": "You are a careful assistant",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "temperature": 0.3,
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "stream": False,
    }

    payload = config.transform_anthropic_messages_request(
        model="some-model",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert payload["model"] == "some-model"
    assert payload["messages"] == messages
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["system"] == "You are a careful assistant"
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert payload["max_tokens"] == 256
    assert payload["tools"] == optional_params["tools"]

    openai_only_keys = {
        "max_completion_tokens",
        "stop",
        "n",
        "logprobs",
        "response_format",
        "frequency_penalty",
    }
    assert openai_only_keys.isdisjoint(payload.keys())


def test_request_requires_max_tokens(config):
    with pytest.raises(AnthropicError, match="max_tokens is required"):
        config.transform_anthropic_messages_request(
            model="some-model",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_optional_request_params={"system": "s"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_validate_environment_sets_bearer_and_anthropic_defaults(config):
    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="some-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base == "https://host/v1"


def test_validate_environment_does_not_overwrite_caller_headers(config):
    headers, _ = config.validate_anthropic_messages_environment(
        headers={
            "authorization": "Bearer caller-token",
            "anthropic-version": "2024-10-22",
            "content-type": "application/json",
        },
        model="some-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    assert headers["authorization"] == "Bearer caller-token"
    assert headers["anthropic-version"] == "2024-10-22"


def test_validate_environment_preserves_standard_cased_caller_headers(config):
    headers, _ = config.validate_anthropic_messages_environment(
        headers={
            "Authorization": "Bearer caller-token",
            "Anthropic-Version": "2024-10-22",
            "Content-Type": "application/json",
        },
        model="some-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    lowercased = {key.lower() for key in headers}
    assert len(lowercased) == len(headers)
    assert headers["Authorization"] == "Bearer caller-token"
    assert headers["Anthropic-Version"] == "2024-10-22"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_honors_x_api_key_when_present(config):
    headers, _ = config.validate_anthropic_messages_environment(
        headers={"X-Api-Key": "caller-key"},
        model="some-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    assert "authorization" not in {key.lower() for key in headers}
    assert headers["X-Api-Key"] == "caller-key"


def test_validate_environment_injects_anthropic_beta_for_context_management(config):
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="some-model",
        messages=[],
        optional_params={
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        },
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    assert "context-management-2025-06-27" in headers["anthropic-beta"].split(",")


def test_validate_environment_injects_anthropic_beta_for_fast_mode(config):
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="some-model",
        messages=[],
        optional_params={"speed": "fast"},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    assert "fast-mode-2026-02-01" in headers["anthropic-beta"].split(",")


def test_validate_environment_merges_existing_anthropic_beta(config):
    headers, _ = config.validate_anthropic_messages_environment(
        headers={"anthropic-beta": "caller-flag"},
        model="some-model",
        messages=[],
        optional_params={"speed": "fast"},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )
    beta_values = set(headers["anthropic-beta"].split(","))
    assert "caller-flag" in beta_values
    assert "fast-mode-2026-02-01" in beta_values


def test_request_strips_advisor_blocks_when_advisor_tool_absent(config):
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking out loud"},
                {"type": "server_tool_use", "id": "advisor_1", "name": "advisor", "input": {}},
                {"type": "advisor_tool_result", "tool_use_id": "advisor_1", "content": "stale"},
            ],
        },
    ]

    payload = config.transform_anthropic_messages_request(
        model="some-model",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 64},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    flattened_types = [
        block.get("type")
        for message in payload["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
    ]
    assert "advisor_tool_result" not in flattened_types
    assert "server_tool_use" not in flattened_types


def test_request_maps_reasoning_effort_to_thinking(config):
    max_tokens = 8192
    payload = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": max_tokens,
            "reasoning_effort": "medium",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "reasoning_effort" not in payload
    assert isinstance(payload.get("thinking"), dict)
    assert payload["thinking"].get("type") == "enabled"
    assert payload["thinking"]["budget_tokens"] < max_tokens


def test_passthrough_disables_anthropic_beta_filtering(config):
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    assert config.should_filter_anthropic_beta_headers() is False
    assert AnthropicMessagesConfig().should_filter_anthropic_beta_headers() is True


def test_anthropic_beta_survives_provider_filter_on_passthrough_path(config):
    from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"Anthropic-Beta": "caller-flag"},
        model="some-model",
        messages=[],
        optional_params={"speed": "fast"},
        litellm_params={},
        api_key="sk-test",
        api_base="https://host/v1",
    )

    # The deployment routes as provider "openai", which has no beta mapping, so an
    # unconditional filter would drop every anthropic-beta value. The handler must
    # skip filtering for this config so the native upstream still receives them.
    if config.should_filter_anthropic_beta_headers():
        headers = update_headers_with_filtered_beta(headers=dict(headers), provider="openai")

    survived = set(headers.get("anthropic-beta", "").split(","))
    assert {"caller-flag", "fast-mode-2026-02-01"} <= survived

    stripped = update_headers_with_filtered_beta(headers=dict(headers), provider="openai")
    assert "anthropic-beta" not in stripped


def test_json_provider_messages_config_probes_capabilities_under_provider_slug():
    """Capability probes in the shared pass-through helpers read
    ``self.custom_llm_provider``. The JSON-provider config knows its slug, so it
    must expose it; the generic OpenAI-like config has no class-level namespace
    and keeps the inherited ``anthropic`` default."""
    from litellm.llms.openai_like.json_loader import SimpleProviderConfig
    from litellm.llms.openai_like.messages.transformation import (
        JSONProviderAnthropicMessagesConfig,
    )

    provider = SimpleProviderConfig(
        slug="exampleprovider",
        data={"base_url": "https://api.example.com/v1", "api_key_env": "EXAMPLE_API_KEY"},
    )
    assert JSONProviderAnthropicMessagesConfig(provider).custom_llm_provider == "exampleprovider"
    assert OpenAILikeAnthropicMessagesConfig().custom_llm_provider == "anthropic"

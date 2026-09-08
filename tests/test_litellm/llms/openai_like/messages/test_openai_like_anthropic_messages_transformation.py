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
    payload = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 8192,
            "reasoning_effort": "medium",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert "reasoning_effort" not in payload
    assert isinstance(payload.get("thinking"), dict)
    assert payload["thinking"].get("type") == "enabled"
    assert payload["thinking"]["budget_tokens"] < payload["max_tokens"]


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


def _cache_control_request_params() -> tuple[list, dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "write a regex for a US phone number",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        }
    ]
    optional_params = {
        "max_tokens": 256,
        "system": [
            {
                "type": "text",
                "text": "You are Claude Code.",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }
        ],
        "tools": [
            {
                "name": "lookup",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }
    return messages, optional_params


def test_request_strips_cache_control_ttl_everywhere(config):
    """Regression: Claude Code always sends ``cache_control: {type: ephemeral,
    ttl: 1h}``, and strict non-Anthropic /v1/messages validators 400 the whole
    request on the ttl extension (``cache_control.ttl: 1h is not supported``)."""
    messages, optional_params = _cache_control_request_params()

    payload = config.transform_anthropic_messages_request(
        model="some-model",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_request_defaults_missing_cache_control_type_and_drops_non_dict(config):
    payload = config.transform_anthropic_messages_request(
        model="some-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a", "cache_control": {"ttl": "1h"}},
                    {"type": "text", "text": "b", "cache_control": None},
                ],
            }
        ],
        anthropic_messages_optional_request_params={"max_tokens": 64},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    blocks = payload["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]


def test_native_anthropic_config_keeps_cache_control_ttl():
    """Anthropic itself accepts ttl, so the normalization must stay scoped to
    the OpenAI-like passthrough and never reach the native Anthropic path."""
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    messages, optional_params = _cache_control_request_params()
    payload = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_deployment_opt_in_keeps_cache_control_ttl():
    config = OpenAILikeAnthropicMessagesConfig(cache_control_ttl=True)
    payload = config.transform_anthropic_messages_request(
        model="some-model",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            }
        ],
        anthropic_messages_optional_request_params={"max_tokens": 16},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_json_provider_constraint_opts_into_cache_control_ttl():
    from litellm.llms.openai_like.json_loader import SimpleProviderConfig
    from litellm.llms.openai_like.messages.transformation import (
        JSONProviderAnthropicMessagesConfig,
    )

    base_data = {"base_url": "https://api.example.com/v1", "api_key_env": "EXAMPLE_API_KEY"}
    strict = JSONProviderAnthropicMessagesConfig(SimpleProviderConfig(slug="strictprov", data=base_data))
    lenient = JSONProviderAnthropicMessagesConfig(
        SimpleProviderConfig(slug="lenientprov", data={**base_data, "constraints": {"cache_control_ttl": True}})
    )

    def transform(provider_config):
        messages, optional_params = _cache_control_request_params()
        return provider_config.transform_anthropic_messages_request(
            model="some-model",
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert transform(strict)["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert transform(lenient)["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_request_strips_ttl_only_where_the_messages_api_defines_cache_control(config):
    """Regression: the sanitizer must only touch ``cache_control`` where the
    Messages API defines it (request, system, tools, content blocks, tool_result
    content), never application data such as ``tool_use.input`` or a tool's
    ``input_schema`` that happens to contain a ``cache_control`` key."""
    tool_input = {"cache_control": {"type": "ephemeral", "ttl": "1h"}, "query": "x"}
    input_schema = {
        "type": "object",
        "properties": {"cache_control": {"type": "string", "ttl": "1h"}},
    }
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": tool_input}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    "content": [
                        {"type": "text", "text": "result", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                    ],
                },
                {"type": "text", "text": "plain string content stays", "cache_control": {"ttl": "1h"}},
            ],
        },
        {"role": "user", "content": "a plain string message"},
    ]
    optional_params = {
        "max_tokens": 64,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "tools": [
            {
                "name": "lookup",
                "input_schema": input_schema,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }

    payload = config.transform_anthropic_messages_request(
        model="some-model",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert payload["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["input_schema"] == input_schema
    assert payload["messages"][0]["content"][0]["input"] == tool_input
    tool_result = payload["messages"][1]["content"][0]
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert tool_result["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][1]["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][2] == {"role": "user", "content": "a plain string message"}

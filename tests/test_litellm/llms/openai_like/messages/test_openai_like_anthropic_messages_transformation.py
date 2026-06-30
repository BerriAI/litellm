import pytest

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
    with pytest.raises(ValueError, match="max_tokens is required"):
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

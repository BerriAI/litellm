import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


@pytest.fixture
def config() -> AnthropicMessagesConfig:
    return AnthropicMessagesConfig()


@pytest.mark.parametrize(
    "api_base, expected",
    [
        # default when no api_base is configured
        (None, "https://api.anthropic.com/v1/messages"),
        # bases without a version suffix
        ("https://host", "https://host/v1/messages"),
        ("https://host/", "https://host/v1/messages"),
        # bases that already carry /v1 must not become /v1/v1/messages
        ("https://host/v1", "https://host/v1/messages"),
        ("https://host/v1/", "https://host/v1/messages"),
        ("https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/messages"),
        # full endpoint passes through untouched
        ("https://host/v1/messages", "https://host/v1/messages"),
        # provider-prefixed path segments are preserved
        ("https://gateway.example.com/anthropic", "https://gateway.example.com/anthropic/v1/messages"),
        ("https://gateway.example.com/anthropic/v1", "https://gateway.example.com/anthropic/v1/messages"),
    ],
)
def test_get_complete_url_deduplicates_trailing_v1(config, api_base, expected, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    url = config.get_complete_url(
        api_base=api_base,
        api_key="sk-test",
        model="claude-sonnet-4-5",
        optional_params={},
        litellm_params={},
    )
    assert url == expected

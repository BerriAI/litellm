from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)


def test_transform_basic_request():
    """Test basic request with only model and messages."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result == {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_transform_includes_system():
    """Test that system prompt is included when provided."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful assistant.",
    )

    assert result["system"] == "You are a helpful assistant."
    assert result["model"] == "claude-3-5-sonnet"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


def test_transform_includes_tools():
    """Test that tools are included when provided."""
    config = AnthropicCountTokensConfig()

    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        tools=tools,
    )

    assert result["tools"] == tools


def test_transform_includes_system_and_tools():
    """Test that both system and tools are included together."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system="Be helpful",
        tools=[{"name": "my_tool", "input_schema": {"type": "object"}}],
    )

    assert "system" in result
    assert "tools" in result
    assert "messages" in result
    assert "model" in result


def test_transform_no_system_no_tools():
    """Test that None system/tools are not included."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system=None,
        tools=None,
    )

    assert "system" not in result
    assert "tools" not in result


def test_get_endpoint_no_api_base_returns_anthropic_default():
    """#29764 baseline: with no api_base, the endpoint is api.anthropic.com."""
    config = AnthropicCountTokensConfig()
    assert config.get_anthropic_count_tokens_endpoint() == "https://api.anthropic.com/v1/messages/count_tokens"


def test_get_endpoint_with_api_base_only_appends_full_path():
    """#29764: a bare api_base (e.g. http://vllm-host:8000) must have the
    full /v1/messages/count_tokens path appended."""
    config = AnthropicCountTokensConfig()
    assert (
        config.get_anthropic_count_tokens_endpoint(api_base="http://vllm-host:8000")
        == "http://vllm-host:8000/v1/messages/count_tokens"
    )


def test_get_endpoint_with_api_base_ending_in_v1_appends_messages_count_tokens():
    """#29764 main scenario: vLLM-style configs typically pass
    `http://host:port/v1` as api_base — append only the messages path so
    we don't double up the /v1."""
    config = AnthropicCountTokensConfig()
    assert (
        config.get_anthropic_count_tokens_endpoint(api_base="http://vllm-host:8000/v1")
        == "http://vllm-host:8000/v1/messages/count_tokens"
    )


def test_get_endpoint_with_api_base_ending_in_messages_appends_count_tokens():
    """If a caller already terminated api_base with /v1/messages, just
    append /count_tokens — don't repeat /messages."""
    config = AnthropicCountTokensConfig()
    assert (
        config.get_anthropic_count_tokens_endpoint(api_base="https://example.com/v1/messages")
        == "https://example.com/v1/messages/count_tokens"
    )


def test_get_endpoint_with_full_count_tokens_url_returned_verbatim():
    """If the caller explicitly passes the full count_tokens URL, hand it
    back unchanged — no idempotency footgun."""
    config = AnthropicCountTokensConfig()
    url = "https://example.com/v1/messages/count_tokens"
    assert config.get_anthropic_count_tokens_endpoint(api_base=url) == url


def test_get_endpoint_strips_trailing_slash_on_api_base():
    """A trailing slash on api_base must not produce a double slash in the
    constructed URL."""
    config = AnthropicCountTokensConfig()
    assert (
        config.get_anthropic_count_tokens_endpoint(api_base="http://vllm-host:8000/v1/")
        == "http://vllm-host:8000/v1/messages/count_tokens"
    )


# --- token_counter api_base env-fallback resolution (#29765) ----------------
import os
from unittest.mock import AsyncMock, patch

import pytest

from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter

_HANDLER = (
    "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler.handle_count_tokens_request"
)


@pytest.mark.asyncio
async def test_count_tokens_env_fallback_prefers_anthropic_api_base(monkeypatch):
    """ANTHROPIC_API_BASE wins over ANTHROPIC_BASE_URL in the env fallback."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "http://from-api-base:8000")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://from-base-url:9000")

    with patch(_HANDLER, new=AsyncMock(return_value={"input_tokens": 5})) as m:
        await AnthropicTokenCounter().count_tokens(
            model_to_use="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
        )

    assert m.call_args.kwargs["api_base"] == "http://from-api-base:8000"


@pytest.mark.asyncio
async def test_count_tokens_falls_back_to_anthropic_base_url(monkeypatch):
    """#29765 review (willcai1984): a deployment configured only via
    ANTHROPIC_BASE_URL must still reach its backend — main.py / common_utils.py
    already honor it, so token counting has to as well."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://from-base-url:9000")

    with patch(_HANDLER, new=AsyncMock(return_value={"input_tokens": 5})) as m:
        await AnthropicTokenCounter().count_tokens(
            model_to_use="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
        )

    assert m.call_args.kwargs["api_base"] == "http://from-base-url:9000"


@pytest.mark.asyncio
async def test_count_tokens_litellm_params_api_base_wins(monkeypatch):
    """Explicit litellm_params.api_base beats both env vars."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "http://from-api-base:8000")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://from-base-url:9000")

    with patch(_HANDLER, new=AsyncMock(return_value={"input_tokens": 5})) as m:
        await AnthropicTokenCounter().count_tokens(
            model_to_use="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_base": "http://explicit:7000"}},
        )

    assert m.call_args.kwargs["api_base"] == "http://explicit:7000"

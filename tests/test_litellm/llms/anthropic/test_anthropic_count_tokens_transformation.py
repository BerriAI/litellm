import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path
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

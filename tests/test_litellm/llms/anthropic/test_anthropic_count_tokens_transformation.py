import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
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


def test_build_count_tokens_url_no_base():
    """No custom base -> default Anthropic endpoint."""
    config = AnthropicCountTokensConfig()
    assert (
        config._build_count_tokens_url(None)
        == "https://api.anthropic.com/v1/messages/count_tokens"
    )


def test_build_count_tokens_url_bare_base():
    """Plain domain base gets full path appended."""
    config = AnthropicCountTokensConfig()
    assert (
        config._build_count_tokens_url("https://my-proxy.example.com")
        == "https://my-proxy.example.com/v1/messages/count_tokens"
    )


def test_build_count_tokens_url_base_with_v1_messages():
    """Base already ending in /v1/messages gets only /count_tokens."""
    config = AnthropicCountTokensConfig()
    assert (
        config._build_count_tokens_url("https://my-proxy.example.com/v1/messages")
        == "https://my-proxy.example.com/v1/messages/count_tokens"
    )


def test_build_count_tokens_url_already_full():
    """Base already containing the full path is returned as-is."""
    config = AnthropicCountTokensConfig()
    assert (
        config._build_count_tokens_url(
            "https://my-proxy.example.com/v1/messages/count_tokens"
        )
        == "https://my-proxy.example.com/v1/messages/count_tokens"
    )


def test_build_count_tokens_url_trailing_slash_stripped():
    """Trailing slash on the base is ignored."""
    config = AnthropicCountTokensConfig()
    assert (
        config._build_count_tokens_url("https://my-proxy.example.com/")
        == "https://my-proxy.example.com/v1/messages/count_tokens"
    )

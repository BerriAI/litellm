import httpx
import pytest
import respx

import litellm
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
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


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, "https://api.anthropic.com/v1/messages/count_tokens"),
        ("", "https://api.anthropic.com/v1/messages/count_tokens"),
        ("https://gateway.example", "https://gateway.example/v1/messages/count_tokens"),
        ("https://gateway.example/", "https://gateway.example/v1/messages/count_tokens"),
        ("https://gateway.example/v1", "https://gateway.example/v1/messages/count_tokens"),
        ("https://gateway.example/anthropic/v1/messages", "https://gateway.example/anthropic/v1/messages/count_tokens"),
    ],
)
def test_endpoint_appends_count_tokens_path_to_deployment_api_base(api_base, expected, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert AnthropicCountTokensConfig().get_anthropic_count_tokens_endpoint(api_base) == expected


@pytest.mark.parametrize("env_name", ["ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"])
@pytest.mark.parametrize("api_base", [None, ""])
def test_endpoint_without_deployment_api_base_follows_env_base(env_name, api_base, monkeypatch):
    """Chat and the federated exchange resolve an unset deployment base through the environment,
    so an env-only gateway must receive the count too, never Anthropic's public host."""
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv(env_name, "https://env-gateway.example/v1/messages/")
    assert (
        AnthropicCountTokensConfig().get_anthropic_count_tokens_endpoint(api_base)
        == "https://env-gateway.example/v1/messages/count_tokens"
    )


def test_endpoint_prefers_deployment_api_base_over_env_base(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://env-gateway.example")
    assert (
        AnthropicCountTokensConfig().get_anthropic_count_tokens_endpoint("https://gateway.example/v1")
        == "https://gateway.example/v1/messages/count_tokens"
    )


@pytest.fixture
def httpx_transport_clients(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    client_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if client_cache is not None:
        client_cache.flush_cache()
    yield
    if client_cache is not None:
        client_cache.flush_cache()


@pytest.mark.asyncio
async def test_handler_posts_to_count_tokens_path_under_deployment_api_base(httpx_transport_clients):
    """A deployment api_base names the chat host, so a handler that posts to it verbatim lands on
    the host root, gets a 404, and the official count silently degrades to the local tokenizer."""
    with respx.mock:
        route = respx.post("https://gateway.example/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 7})
        )
        result = await AnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-ant-api03-test-key",
            api_base="https://gateway.example",
        )

    assert route.called
    assert result == {"input_tokens": 7}

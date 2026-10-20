"""
Test MiniMax Anthropic-compatible API support
"""

import json
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.minimax.messages.transformation import MinimaxMessagesConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", [None, "https://api.minimaxi.com/anthropic", "https://custom.example/anthropic"])
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("effort_source", ["native", "alias"])
async def test_flash_messages_effort_reaches_transport(
    api_base: str | None, effort: str | None, effort_source: str
) -> None:
    native_options: Final = {} if effort is None else {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    options: Final = {"reasoning_effort": effort} if effort is not None and effort_source == "alias" else native_options

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert str(request.url) == f"{api_base or 'https://api.minimax.io/anthropic'}/v1/messages"
        assert body["model"] == "MiniMax-M3.1-Flash"
        assert body.get("output_config") == native_options.get("output_config")
        assert body.get("thinking") == native_options.get("thinking")
        assert "reasoning_effort" not in body
        assert body["temperature"] == 0.7
        assert body["messages"][0]["content"][1]["source"]["data"] == "aGVsbG8="
        assert body["messages"][1]["content"] == [
            {"type": "thinking", "thinking": "I should inspect the image.", "signature": "test-signature"},
            {"type": "tool_use", "id": "call_inspect", "name": "inspect", "input": {}},
        ]
        assert body["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "call_inspect", "content": "A cat"}
        assert body["tools"][0]["name"] == "inspect"
        return httpx.Response(200, json={
            "id": "test-message", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with handler.client:
        response: Final = await anthropic_messages(
            model="minimax/MiniMax-M3.1-Flash", max_tokens=128,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
            ]}, {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should inspect the image.", "signature": "test-signature"},
                {"type": "tool_use", "id": "call_inspect", "name": "inspect", "input": {}},
            ]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_inspect", "content": "A cat"}]}],
            tools=[{"name": "inspect", "input_schema": {"type": "object", "properties": {}}}],
            temperature=0.7, api_key="test-key", api_base=api_base,
            client=handler, **options,
        )
    assert response["content"] == [{"type": "text", "text": "ok"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model,thinking", [
    ("MiniMax-M3", None),
    ("MiniMax-M3", {"type": "disabled"}),
    ("MiniMax-M3", {"type": "adaptive"}),
    ("MiniMax-M2.1", None),
    ("my-saved-model", None),
])
async def test_minimax_saved_messages_reach_transport(model: str, thinking: dict[str, str] | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert body["model"] == model
        assert body.get("thinking") == thinking
        assert "output_config" not in body
        return httpx.Response(200, json={
            "id": "test-message", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with handler.client:
        response: Final = await anthropic_messages(
            model=f"minimax/{model}", max_tokens=128,
            messages=[{"role": "user", "content": "Hi"}],
            api_key="test-key", thinking=thinking, client=handler,
            reasoning_effort="max" if model == "MiniMax-M3" else None,
        )
    assert response["content"] == [{"type": "text", "text": "ok"}]


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_messages_alias_respects_provider_effort_contract(model: str) -> None:
    from litellm.types.router import GenericLiteLLMParams

    config: Final = MinimaxMessagesConfig()
    result: Final = config.transform_anthropic_messages_request(
        model=model, messages=[{"role": "user", "content": "Hi"}],
        anthropic_messages_optional_request_params={"max_tokens": 128, "reasoning_effort": "max"},
        litellm_params=GenericLiteLLMParams(), headers={},
    )
    assert "reasoning_effort" not in result
    if model == "MiniMax-M3.1-Flash":
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "max"}
    else:
        assert "thinking" not in result
        assert "output_config" not in result


def test_flash_messages_none_effort_is_rejected() -> None:
    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.types.router import GenericLiteLLMParams

    with pytest.raises(AnthropicError, match="requires low, medium, high, xhigh or max effort"):
        MinimaxMessagesConfig().transform_anthropic_messages_request(
            model="MiniMax-M3.1-Flash", messages=[{"role": "user", "content": "Hi"}],
            anthropic_messages_optional_request_params={"max_tokens": 128, "reasoning_effort": "none"},
            litellm_params=GenericLiteLLMParams(), headers={},
        )


def test_minimax_anthropic_config():
    """Test that MinimaxMessagesConfig is properly configured"""
    config = MinimaxMessagesConfig()

    # Test custom_llm_provider
    assert config.custom_llm_provider == "minimax"

    # Test get_api_base default
    api_base = config.get_api_base()
    assert api_base == "https://api.minimax.io/anthropic/v1/messages"

    # Test get_api_base with custom value
    custom_base = config.get_api_base(
        api_base="https://api.minimaxi.com/anthropic/v1/messages"
    )
    assert custom_base == "https://api.minimaxi.com/anthropic/v1/messages"


def test_minimax_provider_routing():
    """Test that minimax provider is properly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with minimax/ prefix
    model, provider, api_key, api_base = get_llm_provider(
        model="minimax/MiniMax-M2.1",
        api_base="https://api.minimax.io/anthropic/v1/messages",
    )
    assert provider == "minimax"
    assert model == "MiniMax-M2.1"


def test_minimax_provider_config_manager():
    """Test that ProviderConfigManager returns MinimaxMessagesConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="MiniMax-M2.1", provider=LlmProviders.MINIMAX
    )

    assert config is not None
    assert isinstance(config, MinimaxMessagesConfig)
    assert config.custom_llm_provider == "minimax"


if __name__ == "__main__":
    # Run basic tests that don't require API key
    print("Testing MiniMax Anthropic Config...")
    test_minimax_anthropic_config()
    print("✓ Config test passed")

    print("\nTesting MiniMax Provider Routing...")
    test_minimax_provider_routing()
    print("✓ Routing test passed")

    print("\nTesting MiniMax Provider Config Manager...")
    test_minimax_provider_config_manager()
    print("✓ Provider config manager test passed")

    print("\n✅ All basic tests passed!")


def test_minimax_messages_env_key_attached(monkeypatch):
    """Regression: an env-only MINIMAX_API_KEY must be attached on /v1/messages validation"""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-env-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    config = MinimaxMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="MiniMax-M2.1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
    )
    assert headers["x-api-key"] == "test-minimax-env-key"


def test_minimax_messages_explicit_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    config = MinimaxMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="MiniMax-M2.1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key="param-key",
    )
    assert headers["x-api-key"] == "param-key"

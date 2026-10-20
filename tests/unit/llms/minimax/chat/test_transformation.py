"""
Test MiniMax OpenAI-compatible API support
"""

import json
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig


@pytest.mark.parametrize("api_base", [None, "https://api.minimaxi.com/v1", "https://custom.example/v1"])
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
def test_flash_chat_effort_reaches_transport(api_base: str | None, effort: str | None) -> None:
    options: Final = {} if effort is None else {"reasoning_effort": effort}

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert str(request.url) == f"{api_base or 'https://api.minimax.io/v1'}/chat/completions"
        assert body["model"] == "MiniMax-M3.1-Flash"
        assert body.get("reasoning_effort") == effort
        assert "thinking" not in body
        assert "service_tier" not in body
        assert "reasoning_split" not in body
        assert body["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
        assert body["messages"][1]["reasoning_content"] == "I should inspect the image."
        assert body["messages"][1]["tool_calls"][0]["id"] == "call_inspect"
        assert body["messages"][2] == {"role": "tool", "tool_call_id": "call_inspect", "content": "A cat"}
        assert body["tools"][0]["function"]["name"] == "inspect"
        return httpx.Response(200, json={
            "id": "test-chat", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        response: Final = litellm.completion(
            model="minimax/MiniMax-M3.1-Flash",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ]}, {
                "role": "assistant", "content": None,
                "reasoning_content": "I should inspect the image.",
                "tool_calls": [{"id": "call_inspect", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}],
            }, {"role": "tool", "tool_call_id": "call_inspect", "content": "A cat"}],
            tools=[{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object", "properties": {}}}}],
            api_key="test-key", api_base=api_base, client=HTTPHandler(client=client), **options,
        )
    assert response.choices[0].message.content == "ok"


@pytest.mark.parametrize("model,thinking", [
    ("MiniMax-M3", None), ("MiniMax-M3", {"type": "adaptive"}), ("MiniMax-M3", {"type": "disabled"}),
    ("MiniMax-M2.1", None), ("my-saved-model", None),
])
def test_minimax_saved_chat_model_reaches_transport(model: str, thinking: dict[str, str] | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert body["model"] == model
        assert body.get("thinking") == thinking
        assert "reasoning_effort" not in body
        return httpx.Response(200, json={
            "id": "test-chat", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        response: Final = litellm.completion(
            model=f"minimax/{model}", messages=[{"role": "user", "content": "Hi"}],
            api_key="test-key", thinking=thinking, client=HTTPHandler(client=client),
        )
    assert response.choices[0].message.content == "ok"


def test_flash_model_info_keeps_legacy_lookup_available() -> None:
    info: Final = litellm.get_model_info("minimax/MiniMax-M3.1-Flash")
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 524288
    assert info["input_cost_per_token"] == 1.5e-7
    assert info["output_cost_per_token"] == 6e-7
    assert info["cache_read_input_token_cost"] == 1.5e-8
    assert info["supports_vision"] is True
    assert info["supports_video_input"] is True
    assert info["supports_max_reasoning_effort"] is True
    assert info["reasoning_effort_levels"] == ["low", "medium", "high", "xhigh", "max"]
    assert litellm.get_model_info("minimax/MiniMax-M2.1")["litellm_provider"] == "minimax"


def test_minimax_chat_config():
    """Test that MinimaxChatConfig is properly configured"""
    config = MinimaxChatConfig()

    # Test get_api_base default
    api_base = config.get_api_base()
    assert api_base == "https://api.minimax.io/v1"

    # Test get_api_base with custom value
    custom_base = config.get_api_base(api_base="https://api.minimaxi.com/v1")
    assert custom_base == "https://api.minimaxi.com/v1"

    # Test get_complete_url
    complete_url = config.get_complete_url(
        api_base="https://api.minimax.io/v1",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert complete_url == "https://api.minimax.io/v1/chat/completions"


def test_minimax_chat_config_url_variations():
    """Test URL handling with different base URL formats"""
    config = MinimaxChatConfig()

    # Test with /v1 ending
    url1 = config.get_complete_url(
        api_base="https://api.minimax.io/v1",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url1 == "https://api.minimax.io/v1/chat/completions"

    # Test with trailing slash
    url2 = config.get_complete_url(
        api_base="https://api.minimax.io/",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url2 == "https://api.minimax.io/v1/chat/completions"

    # Test without trailing slash
    url3 = config.get_complete_url(
        api_base="https://api.minimax.io",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url3 == "https://api.minimax.io/v1/chat/completions"

    # Test with full path already
    url4 = config.get_complete_url(
        api_base="https://api.minimax.io/v1/chat/completions",
        api_key=None,
        model="MiniMax-M2.1",
        optional_params={},
        litellm_params={},
    )
    assert url4 == "https://api.minimax.io/v1/chat/completions"


def test_minimax_provider_routing():
    """Test that minimax provider is properly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with minimax/ prefix
    model, provider, api_key, api_base = get_llm_provider(
        model="minimax/MiniMax-M2.1", api_base="https://api.minimax.io/v1"
    )
    assert provider == "minimax"
    assert model == "MiniMax-M2.1"


def test_minimax_provider_config_manager():
    """Test that ProviderConfigManager returns MinimaxChatConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="MiniMax-M2.1", provider=LlmProviders.MINIMAX
    )

    assert config is not None
    assert isinstance(config, MinimaxChatConfig)


if __name__ == "__main__":
    # Run basic tests that don't require API key
    print("Testing MiniMax Chat Config...")
    test_minimax_chat_config()
    print("✓ Config test passed")

    print("\nTesting MiniMax Chat Config URL Variations...")
    test_minimax_chat_config_url_variations()
    print("✓ URL variations test passed")

    print("\nTesting MiniMax Provider Routing...")
    test_minimax_provider_routing()
    print("✓ Routing test passed")

    print("\nTesting MiniMax Provider Config Manager...")
    test_minimax_provider_config_manager()
    print("✓ Provider config manager test passed")

    print("\n✅ All basic tests passed!")

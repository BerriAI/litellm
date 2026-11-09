"""
Test MiniMax OpenAI-compatible API support
"""

import json
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
import respx

import litellm
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig


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


@pytest.fixture
def minimax_request_mock(local_model_cost_map, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return respx_mock.post("https://api.minimax.io/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "MiniMax-M3.1-Flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )


def _sent_body(model: str, request_mock: respx.Route, effort: str, drop_params: bool = True) -> dict:
    litellm.completion(
        model=f"minimax/{model}",
        messages=[{"role": "user", "content": "What is the weather in Oslo?"}],
        api_key="fake-minimax-key",
        reasoning_effort=effort,
        drop_params=drop_params,
    )
    assert request_mock.call_count == 1
    return json.loads(request_mock.calls[0].request.content)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_minimax_flash_forwards_every_declared_reasoning_effort(minimax_request_mock: respx.Route, effort: str):
    """Issue #49574: reasoning_effort sent to MiniMax-M3.1-Flash was stripped before the upstream call."""
    assert _sent_body("MiniMax-M3.1-Flash", minimax_request_mock, effort)["reasoning_effort"] == effort


def test_minimax_flash_cost_map_entry_declares_its_effort_levels(local_model_cost_map):
    info: Final = litellm.get_model_info("MiniMax-M3.1-Flash", custom_llm_provider="minimax")
    assert info["supports_reasoning"] is True
    assert info["reasoning_effort_levels"] == ["low", "medium", "high", "xhigh", "max"]
    assert info["input_cost_per_token"] == 1.5e-07
    assert info["output_cost_per_token"] == 6e-07


@pytest.mark.parametrize("model", ["MiniMax-M3.2", "a-deployment-name-the-map-does-not-know"])
def test_minimax_forwards_reasoning_effort_for_unmapped_models(minimax_request_mock: respx.Route, model: str):
    """A MiniMax model newer than the cost map keeps reasoning_effort instead of losing it."""
    assert f"minimax/{model}" not in litellm.model_cost
    assert _sent_body(model, minimax_request_mock, "low", drop_params=False)["reasoning_effort"] == "low"


def test_minimax_unmapped_m_series_model_is_reasoning_capable(local_model_cost_map):
    """The minimax-reasoning-baseline rule marks new M-series ids as reasoning, so thinking is kept too."""
    assert litellm.supports_reasoning(model="MiniMax-M3.2", custom_llm_provider="minimax") is True
    assert "thinking" in MinimaxChatConfig().get_supported_openai_params(model="MiniMax-M3.2")


@pytest.mark.respx(assert_all_called=False)
def test_minimax_drops_reasoning_effort_for_a_known_non_reasoning_model(
    minimax_request_mock: respx.Route, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(litellm.model_cost["minimax/MiniMax-M3"], "supports_reasoning", False)
    litellm.get_model_info.cache_clear()

    assert "reasoning_effort" not in MinimaxChatConfig().get_supported_openai_params(model="MiniMax-M3")
    assert "reasoning_effort" not in _sent_body("MiniMax-M3", minimax_request_mock, "low")

    with pytest.raises(litellm.UnsupportedParamsError, match="reasoning_effort"):
        _sent_body("MiniMax-M3", minimax_request_mock, "low", drop_params=False)

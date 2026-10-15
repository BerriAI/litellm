"""
Unit tests for the WebWright provider configuration.

WebWright exposes an OpenAI-compatible chat completions API at
https://webwright.ai/v1 and serves the `wren-r1` model, so the provider is a
thin configuration over the shared OpenAI-like transformation.
"""

import os
from unittest import mock

import litellm

from litellm.llms.webwright.chat.transformation import WebwrightChatConfig


def test_get_llm_provider_webwright() -> None:
    """Test that get_llm_provider correctly identifies WebWright"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with webwright/model-name format
    model, provider, api_key, api_base = get_llm_provider("webwright/wren-r1")
    assert model == "wren-r1"
    assert provider == "webwright"

    # Test with api_base containing the WebWright endpoint
    model, provider, api_key, api_base = get_llm_provider("wren-r1", api_base="https://webwright.ai/v1")
    assert model == "wren-r1"
    assert provider == "webwright"
    assert api_base == "https://webwright.ai/v1"


def test_webwright_config_initialization() -> None:
    """Test WebwrightChatConfig initializes correctly"""
    config = WebwrightChatConfig()
    assert config.custom_llm_provider == "webwright"


def test_webwright_get_openai_compatible_provider_info() -> None:
    """Test WebWright API base and key handling"""
    config = WebwrightChatConfig()

    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://webwright.ai/v1"
        assert api_key is None

    with mock.patch.dict(
        os.environ,
        {
            "WEBWRIGHT_API_KEY": "env-key",
            "WEBWRIGHT_API_BASE": "https://env.webwright.ai/v1",
        },
    ):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.webwright.ai/v1"
        assert api_key == "env-key"

    # Explicit arguments take precedence over the environment
    api_base, api_key = config.get_openai_compatible_provider_info("https://custom.webwright.ai/v1", "param-key")
    assert api_base == "https://custom.webwright.ai/v1"
    assert api_key == "param-key"


def test_webwright_in_provider_lists() -> None:
    """Test that webwright is registered in all necessary provider lists"""
    assert "webwright" in litellm.openai_compatible_providers
    assert "webwright" in litellm.openai_text_completion_compatible_providers
    assert "webwright" in litellm.provider_list
    assert "https://webwright.ai/v1" in litellm.openai_compatible_endpoints


def test_webwright_supported_params() -> None:
    """Test that supported OpenAI parameters are correctly configured"""
    config = WebwrightChatConfig()
    supported_params = config.get_supported_openai_params("webwright/wren-r1")

    for param in (
        "messages",
        "model",
        "stream",
        "temperature",
        "max_tokens",
        "tools",
        "tool_choice",
    ):
        assert param in supported_params


def test_webwright_completion_call() -> None:
    """Test the completion call structure for WebWright with a mocked response"""
    response = litellm.completion(
        model="webwright/wren-r1",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="test-key",
        mock_response="Hi there!",
    )
    assert response is not None
    assert response.choices[0].message.content == "Hi there!"

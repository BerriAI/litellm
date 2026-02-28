"""
Test MiniMax OpenAI-compatible API support
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

import litellm
from litellm import completion
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
        stream=False
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
        model="minimax/MiniMax-M2.1",
        api_base="https://api.minimax.io/v1"
    )
    assert provider == "minimax"
    assert model == "MiniMax-M2.1"


def test_minimax_provider_config_manager():
    """Test that ProviderConfigManager returns MinimaxChatConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager
    
    config = ProviderConfigManager.get_provider_chat_config(
        model="MiniMax-M2.1",
        provider=LlmProviders.MINIMAX
    )
    
    assert config is not None
    assert isinstance(config, MinimaxChatConfig)


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_basic():
    """Test basic chat completion with MiniMax OpenAI-compatible API"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1"
    )
    
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_with_reasoning_split():
    """Test completion with reasoning_split parameter (MiniMax M2.1 feature)"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Solve this problem: 2+2=?"}
        ],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
        extra_body={"reasoning_split": True}
    )
    
    assert response is not None
    # Check if reasoning_details is present in response
    if hasattr(response.choices[0].message, "reasoning_details"):
        assert response.choices[0].message.reasoning_details is not None


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_with_tools():
    """Test completion with tool calling (function calling)"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools,
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1"
    )
    
    assert response is not None
    assert hasattr(response, "choices")


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_streaming():
    """Test streaming completion"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1"
    )
    
    chunks = []
    for chunk in response:
        chunks.append(chunk)
    
    assert len(chunks) > 0


def test_minimax_map_reasoning_details_to_reasoning_content():
    """Test that reasoning_details is mapped to reasoning_content in streaming chunks."""
    config = MinimaxChatConfig()

    # Simulate MiniMax streaming chunk with reasoning_details
    choices = [
        {"delta": {"reasoning_details": "Let me think step by step..."}}
    ]
    result = config._map_reasoning_to_reasoning_content(choices)

    assert "reasoning_content" in result[0]["delta"]
    assert result[0]["delta"]["reasoning_content"] == "Let me think step by step..."
    assert "reasoning_details" not in result[0]["delta"]


def test_minimax_map_reasoning_details_preserves_standard_reasoning():
    """Test that standard reasoning field still works for MiniMax."""
    config = MinimaxChatConfig()

    # Simulate chunk with standard 'reasoning' field
    choices = [
        {"delta": {"reasoning": "Standard reasoning content"}}
    ]
    result = config._map_reasoning_to_reasoning_content(choices)

    assert "reasoning_content" in result[0]["delta"]
    assert result[0]["delta"]["reasoning_content"] == "Standard reasoning content"


def test_minimax_map_reasoning_details_no_reasoning():
    """Test that chunks without reasoning fields are unchanged."""
    config = MinimaxChatConfig()

    choices = [
        {"delta": {"content": "Normal content"}}
    ]
    result = config._map_reasoning_to_reasoning_content(choices)

    assert result[0]["delta"] == {"content": "Normal content"}


def test_extract_reasoning_content_with_reasoning_details():
    """Test that _extract_reasoning_content handles reasoning_details field."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        _extract_reasoning_content,
    )

    # MiniMax non-streaming response with reasoning_details
    message = {
        "content": "The answer is 4.",
        "reasoning_details": "2+2=4 because addition..."
    }
    reasoning, content = _extract_reasoning_content(message)

    assert reasoning == "2+2=4 because addition..."
    assert content == "The answer is 4."


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

    print("\nTesting MiniMax reasoning_details mapping...")
    test_minimax_map_reasoning_details_to_reasoning_content()
    print("✓ reasoning_details mapping test passed")

    test_minimax_map_reasoning_details_preserves_standard_reasoning()
    print("✓ standard reasoning mapping test passed")

    test_minimax_map_reasoning_details_no_reasoning()
    print("✓ no reasoning test passed")

    test_extract_reasoning_content_with_reasoning_details()
    print("✓ extract reasoning content test passed")
    
    print("\n✅ All basic tests passed!")


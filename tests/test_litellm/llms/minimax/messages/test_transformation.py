"""
Test MiniMax Anthropic-compatible API support
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
from litellm.llms.minimax.messages.transformation import MinimaxMessagesConfig


def test_minimax_anthropic_config():
    """Test that MinimaxMessagesConfig is properly configured"""
    config = MinimaxMessagesConfig()
    
    # Test custom_llm_provider
    assert config.custom_llm_provider == "minimax"
    
    # Test get_api_base default
    api_base = config.get_api_base()
    assert api_base == "https://api.minimax.io/anthropic/v1/messages"
    
    # Test get_api_base with custom value
    custom_base = config.get_api_base(api_base="https://api.minimaxi.com/anthropic/v1/messages")
    assert custom_base == "https://api.minimaxi.com/anthropic/v1/messages"


def test_minimax_provider_routing():
    """Test that minimax provider is properly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with minimax/ prefix
    model, provider, api_key, api_base = get_llm_provider(
        model="minimax/MiniMax-M2.1",
        api_base="https://api.minimax.io/anthropic/v1/messages"
    )
    assert provider == "minimax"
    assert model == "MiniMax-M2.1"


def test_minimax_provider_config_manager():
    """Test that ProviderConfigManager returns MinimaxMessagesConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager
    
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="MiniMax-M2.1",
        provider=LlmProviders.MINIMAX
    )
    
    assert config is not None
    assert isinstance(config, MinimaxMessagesConfig)
    assert config.custom_llm_provider == "minimax"


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_completion_basic():
    """Test basic completion with MiniMax Anthropic-compatible API"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/anthropic/v1/messages"
    )
    
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_completion_with_thinking():
    """Test completion with thinking parameter (MiniMax M2.1 feature)"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[{"role": "user", "content": "Solve this problem: 2+2=?"}],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/anthropic/v1/messages",
        thinking={"type": "enabled", "budget_tokens": 1000}
    )
    
    assert response is not None
    # Check if thinking content is present in response
    for choice in response.choices:
        if hasattr(choice.message, "content"):
            # MiniMax returns thinking blocks similar to Anthropic
            assert choice.message.content is not None


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_completion_with_tools():
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
        api_base="https://api.minimax.io/anthropic/v1/messages"
    )
    
    assert response is not None
    assert hasattr(response, "choices")


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


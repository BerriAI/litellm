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


class TestMinimaxReasoningDetails:
    """Test reasoning_details handling in streaming and non-streaming responses."""

    def test_streaming_reasoning_details_mapped_to_reasoning_content(self):
        """reasoning_details array in delta should be concatenated into reasoning_content."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [
            {
                "delta": {
                    "reasoning_details": [
                        {"text": "Step 1: "},
                        {"text": "analyze the problem."},
                    ]
                }
            }
        ]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert result[0]["delta"]["reasoning_content"] == "Step 1: analyze the problem."
        assert "reasoning_details" not in result[0]["delta"]

    def test_streaming_empty_reasoning_details_not_mapped(self):
        """Empty reasoning_details array should not produce reasoning_content."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [{"delta": {"reasoning_details": []}}]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert "reasoning_content" not in result[0]["delta"]
        assert "reasoning_details" not in result[0]["delta"]

    def test_streaming_reasoning_details_with_empty_text(self):
        """reasoning_details with empty text fields should not produce reasoning_content."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [{"delta": {"reasoning_details": [{"text": ""}, {"text": ""}]}}]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert "reasoning_content" not in result[0]["delta"]

    def test_streaming_reasoning_field_still_mapped(self):
        """Parent class mapping of reasoning → reasoning_content should still work."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [{"delta": {"reasoning": "thinking..."}}]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert result[0]["delta"]["reasoning_content"] == "thinking..."
        assert "reasoning" not in result[0]["delta"]

    def test_streaming_content_not_affected(self):
        """Regular content in delta should not be touched."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [
            {
                "delta": {
                    "content": "The answer is 4.",
                    "reasoning_details": [{"text": "2+2=4"}],
                }
            }
        ]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert result[0]["delta"]["content"] == "The answer is 4."
        assert result[0]["delta"]["reasoning_content"] == "2+2=4"

    def test_streaming_reasoning_content_takes_precedence_over_details(self):
        """If reasoning_content already set, reasoning_details should not overwrite it."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [
            {
                "delta": {
                    "reasoning_content": "already set",
                    "reasoning_details": [{"text": "should not overwrite"}],
                }
            }
        ]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert result[0]["delta"]["reasoning_content"] == "already set"
        assert "reasoning_details" not in result[0]["delta"]

    def test_streaming_reasoning_details_not_a_list(self):
        """Non-list reasoning_details should be popped without setting reasoning_content."""
        from litellm.llms.minimax.chat.transformation import MinimaxStreamingHandler

        handler = MinimaxStreamingHandler.__new__(MinimaxStreamingHandler)
        choices = [{"delta": {"reasoning_details": "not a list"}}]
        result = handler._map_reasoning_to_reasoning_content(choices)
        assert "reasoning_content" not in result[0]["delta"]
        assert "reasoning_details" not in result[0]["delta"]

    def test_nonstreaming_reasoning_details_extracted(self):
        """Non-streaming: reasoning_details should be extracted as reasoning_content."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            _extract_reasoning_content,
        )

        message = {
            "content": "The answer is 4.",
            "reasoning_details": [
                {"text": "Let me think: "},
                {"text": "2+2=4."},
            ],
        }
        reasoning, content = _extract_reasoning_content(message)
        assert reasoning == "Let me think: 2+2=4."
        assert content == "The answer is 4."

    def test_nonstreaming_reasoning_content_takes_precedence(self):
        """reasoning_content field should take precedence over reasoning_details."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            _extract_reasoning_content,
        )

        message = {
            "content": "answer",
            "reasoning_content": "direct reasoning",
            "reasoning_details": [{"text": "detail reasoning"}],
        }
        reasoning, content = _extract_reasoning_content(message)
        assert reasoning == "direct reasoning"

    def test_nonstreaming_empty_reasoning_details(self):
        """Empty reasoning_details should return None reasoning."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            _extract_reasoning_content,
        )

        message = {"content": "answer", "reasoning_details": []}
        reasoning, content = _extract_reasoning_content(message)
        assert reasoning is None
        assert content == "answer"

    def test_nonstreaming_reasoning_details_not_a_list(self):
        """Non-list reasoning_details in non-streaming should return None."""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            _extract_reasoning_content,
        )

        message = {"content": "answer", "reasoning_details": "not a list"}
        reasoning, content = _extract_reasoning_content(message)
        assert reasoning is None
        assert content == "answer"

    def test_get_model_response_iterator_returns_minimax_handler(self):
        """MinimaxChatConfig should return MinimaxStreamingHandler."""
        from litellm.llms.minimax.chat.transformation import (
            MinimaxStreamingHandler,
        )

        config = MinimaxChatConfig()
        handler = config.get_model_response_iterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=False,
        )
        assert isinstance(handler, MinimaxStreamingHandler)


@pytest.mark.skip(reason="Requires actual MiniMax API key")
def test_minimax_chat_completion_basic():
    """Test basic chat completion with MiniMax OpenAI-compatible API"""
    response = completion(
        model="minimax/MiniMax-M2.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
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
            {"role": "user", "content": "Solve this problem: 2+2=?"},
        ],
        api_key=os.getenv("MINIMAX_API_KEY"),
        api_base="https://api.minimax.io/v1",
        extra_body={"reasoning_split": True},
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
        api_base="https://api.minimax.io/v1",
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
        api_base="https://api.minimax.io/v1",
    )

    chunks = []
    for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0


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

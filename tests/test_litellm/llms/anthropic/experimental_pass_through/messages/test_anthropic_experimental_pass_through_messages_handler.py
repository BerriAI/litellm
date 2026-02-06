import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from unittest.mock import AsyncMock, MagicMock, patch

from litellm.anthropic_interface import messages
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import Delta, ModelResponse, StreamingChoices


def test_anthropic_experimental_pass_through_messages_handler():
    """
    Test that api key is passed to litellm.completion
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="openai/claude-3-5-sonnet-20240620",
                api_key="test-api-key",
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_completion.assert_called_once()
        mock_completion.call_args.kwargs["api_key"] == "test-api-key"


def test_anthropic_experimental_pass_through_messages_handler_dynamic_api_key_and_api_base_and_custom_values():
    """
    Test that api key is passed to litellm.completion
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="azure/o1",
                api_key="test-api-key",
                api_base="test-api-base",
                custom_key="custom_value",
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_completion.assert_called_once()
        mock_completion.call_args.kwargs["api_key"] == "test-api-key"
        mock_completion.call_args.kwargs["api_base"] == "test-api-base"
        mock_completion.call_args.kwargs["custom_key"] == "custom_value"


def test_anthropic_experimental_pass_through_messages_handler_custom_llm_provider():
    """
    Test that litellm.completion is called when a custom LLM provider is given
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="my-custom-model",
                custom_llm_provider="my-custom-llm",
                api_key="test-api-key",
            )
        except Exception as e:
            print(f"Error: {e}")

        # Assert that litellm.completion was called when using a custom LLM provider
        mock_completion.assert_called_once()

        # Verify that the custom provider was passed through
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["custom_llm_provider"] == "my-custom-llm"
        assert call_kwargs["model"] == "my-custom-llm/my-custom-model"
        assert call_kwargs["api_key"] == "test-api-key"


@pytest.mark.asyncio
async def test_bedrock_converse_budget_tokens_preserved():
    """
    Test that budget_tokens value in thinking parameter is correctly passed to Bedrock Converse API
    when using messages.acreate with bedrock/converse model.
    
    The bug was that the messages -> completion adapter was converting thinking to reasoning_effort
    and losing the original budget_tokens value, causing it to use the default (128) instead.
    """
    client = AsyncHTTPHandler()
    
    with patch.object(client, "post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = "mock response"
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "4"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 15
            }
        }
        mock_post.return_value = mock_response
        
        try:
            await messages.acreate(
                client=client,
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                thinking={
                    "budget_tokens": 1024,
                    "type": "enabled"
                },
            )
        except Exception:
            pass  # Expected due to mock response format
        
        mock_post.assert_called_once()
        
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs.get("json") or json.loads(call_kwargs.get("data", "{}"))
        print("Request json: ", json.dumps(json_data, indent=4, default=str))
        
        additional_fields = json_data.get("additionalModelRequestFields", {})
        thinking_config = additional_fields.get("thinking", {})
        
        assert "thinking" in additional_fields, "thinking parameter should be in additionalModelRequestFields"
        assert thinking_config.get("type") == "enabled", "thinking.type should be 'enabled'"
        assert thinking_config.get("budget_tokens") == 1024, f"thinking.budget_tokens should be 1024, but got {thinking_config.get('budget_tokens')}"


def test_openai_model_with_thinking_converts_to_reasoning_effort():
    """
    Test that when using a non-Anthropic model (like OpenAI gpt-5.2) with thinking parameter,
    the thinking is converted to reasoning_effort and NOT passed as thinking.
    
    This ensures we don't regress on issue #16052 where non-Anthropic models would fail
    with UnsupportedParamsError when thinking was passed directly.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="openai/gpt-5.2",
                api_key="test-api-key",
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024
                },
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_completion.assert_called_once()
        
        call_kwargs = mock_completion.call_args.kwargs
        
        # Verify reasoning_effort is set (converted from thinking)
        assert "reasoning_effort" in call_kwargs, "reasoning_effort should be passed to completion"
        assert call_kwargs["reasoning_effort"] == "minimal", f"reasoning_effort should be 'minimal' for budget_tokens=1024, got {call_kwargs.get('reasoning_effort')}"
        
        # Verify thinking is NOT passed (non-Claude model)
        assert "thinking" not in call_kwargs, "thinking should NOT be passed for non-Claude models"


class TestThinkingParameterTransformation:
    """Core tests for thinking parameter transformation logic."""

    def test_claude_model_preserves_thinking_with_budget_tokens(self):
        """Test that Claude models get thinking parameter passed through with exact budget_tokens."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        
        thinking = {"type": "enabled", "budget_tokens": 5000}
        result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
            thinking=thinking,
            model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        )
        
        assert result == {"thinking": thinking}
        assert result["thinking"]["budget_tokens"] == 5000

    def test_non_claude_model_converts_thinking_to_reasoning_effort(self):
        """Test that non-Claude models convert thinking to reasoning_effort."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        
        thinking = {"type": "enabled", "budget_tokens": 1024}
        result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
            thinking=thinking,
            model="openai/gpt-5.2",
        )
        
        assert result == {"reasoning_effort": "minimal"}
        assert "thinking" not in result

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
    Test that api key is passed to litellm.responses for OpenAI models.
    OpenAI and Azure models are routed directly to the Responses API.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.responses", return_value="test-response") as mock_responses:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="openai/claude-3-5-sonnet-20240620",
                api_key="test-api-key",
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_responses.assert_called_once()
        assert mock_responses.call_args.kwargs["api_key"] == "test-api-key"


def test_anthropic_experimental_pass_through_messages_handler_dynamic_api_key_and_api_base_and_custom_values():
    """
    Test that api key, api base, and extra kwargs are forwarded to litellm.responses for Azure models.
    Azure models are routed directly to the Responses API.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.responses", return_value="test-response") as mock_responses:
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
        mock_responses.assert_called_once()
        assert mock_responses.call_args.kwargs["api_key"] == "test-api-key"
        assert mock_responses.call_args.kwargs["api_base"] == "test-api-base"
        assert mock_responses.call_args.kwargs["custom_key"] == "custom_value"


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
    # Mock litellm.acompletion which is called internally by anthropic_messages_handler
    mock_response = ModelResponse(
        id="test-id",
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "4"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        try:
            await messages.acreate(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                thinking={
                    "budget_tokens": 1024,
                    "type": "enabled"
                },
            )
        except Exception:
            pass  # Expected due to response format conversion

        mock_acompletion.assert_called_once()

        call_kwargs = mock_acompletion.call_args.kwargs
        print("acompletion call kwargs: ", json.dumps(call_kwargs, indent=4, default=str))

        # Verify thinking parameter is passed through with budget_tokens preserved
        thinking_param = call_kwargs.get("thinking")
        assert thinking_param is not None, "thinking parameter should be passed to acompletion"
        assert thinking_param.get("type") == "enabled", "thinking.type should be 'enabled'"
        assert thinking_param.get("budget_tokens") == 1024, f"thinking.budget_tokens should be 1024, but got {thinking_param.get('budget_tokens')}"


def test_openai_model_with_thinking_converts_to_reasoning():
    """
    Test that when using an OpenAI model with thinking parameter, the thinking is
    converted to a Responses API `reasoning` param (NOT passed as thinking).

    OpenAI models are routed directly to the Responses API, so we verify that
    litellm.responses() is called with `reasoning` properly set.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.responses", return_value="test-response") as mock_responses:
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

        mock_responses.assert_called_once()

        call_kwargs = mock_responses.call_args.kwargs

        # Verify reasoning is set (converted from thinking)
        assert "reasoning" in call_kwargs, "reasoning should be passed to litellm.responses"

        # budget_tokens=1024 -> effort="minimal" (< 2000 threshold)
        expected_reasoning = {"effort": "minimal", "summary": "detailed"}
        assert call_kwargs["reasoning"] == expected_reasoning, (
            f"reasoning should be {expected_reasoning} for budget_tokens=1024, "
            f"got {call_kwargs.get('reasoning')}"
        )

        # Verify thinking is NOT passed directly to the Responses API
        assert "thinking" not in call_kwargs, "thinking should NOT be passed directly to litellm.responses"


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

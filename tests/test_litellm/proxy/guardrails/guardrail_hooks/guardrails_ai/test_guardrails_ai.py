from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.guardrails_ai.guardrails_ai import (
    GuardrailsAI,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_guardrails_ai_process_input():
    """Test the process_input method of GuardrailsAI with various scenarios"""

    # Initialize the GuardrailsAI instance
    guardrails_ai_guardrail = GuardrailsAI(
        guardrail_name="test_guard",
        api_base="http://test.example.com",
        guard_name="gibberish-guard",
    )

    # Test case 1: Valid completion call with messages
    with patch.object(
        guardrails_ai_guardrail,
        "make_guardrails_ai_api_request_pre_call_request",
        return_value="processed text",
    ) as mock_api_request:

        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, how are you?"},
            ]
        }

        result = await guardrails_ai_guardrail.process_input(data, "completion")

        # Verify the API was called with the user message
        mock_api_request.assert_called_once_with(
            text_input="Hello, how are you?", request_data=data
        )

        # Verify the message was updated
        assert result["messages"][1]["content"] == "processed text"
        # System message should remain unchanged
        assert result["messages"][0]["content"] == "You are a helpful assistant"

    # Test case 2: Valid acompletion call with messages
    with patch.object(
        guardrails_ai_guardrail,
        "make_guardrails_ai_api_request_pre_call_request",
        return_value="async processed text",
    ) as mock_api_request:

        data = {"messages": [{"role": "user", "content": "What is the weather?"}]}

        result = await guardrails_ai_guardrail.process_input(data, "acompletion")

        mock_api_request.assert_called_once_with(
            text_input="What is the weather?", request_data=data
        )

        assert result["messages"][0]["content"] == "async processed text"

    # Test case 3: Invalid request without messages
    data_no_messages = {"model": "gpt-3.5-turbo"}

    result = await guardrails_ai_guardrail.process_input(data_no_messages, "completion")

    # Should return data unchanged
    assert result == data_no_messages

    # Test case 4: Messages with no user text (get_last_user_message returns None)
    with patch(
        "litellm.litellm_core_utils.prompt_templates.common_utils.get_last_user_message",
        return_value=None,
    ):
        data = {
            "messages": [{"role": "system", "content": "You are a helpful assistant"}]
        }

        result = await guardrails_ai_guardrail.process_input(data, "completion")

        # Should return data unchanged when no user message found
        assert result == data

    # Test case 5: Different call_type that should not be processed
    data = {"messages": [{"role": "user", "content": "Hello"}]}

    result = await guardrails_ai_guardrail.process_input(data, "embeddings")

    # Should return data unchanged for non-completion call types
    assert result == data

    # Test case 6: Complex conversation with multiple messages
    with patch.object(
        guardrails_ai_guardrail,
        "make_guardrails_ai_api_request_pre_call_request",
        return_value="sanitized message",
    ) as mock_api_request:

        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ]
        }

        result = await guardrails_ai_guardrail.process_input(data, "completion")

        # Should process the last user message
        mock_api_request.assert_called_once_with(
            text_input="Second question", request_data=data
        )

        # Only the last user message should be updated
        assert result["messages"][0]["content"] == "You are a helpful assistant"
        assert result["messages"][1]["content"] == "First question"
        assert result["messages"][2]["content"] == "First answer"
        assert result["messages"][3]["content"] == "sanitized message"

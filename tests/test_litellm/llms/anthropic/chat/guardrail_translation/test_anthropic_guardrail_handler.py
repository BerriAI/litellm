"""
Unit tests for Anthropic Messages Guardrail Translation Handler

Tests the handler's ability to process streaming output for Anthropic Messages API
with guardrail transformations, specifically testing edge cases with empty choices.
"""

import os
import sys
from typing import Any, List, Literal, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.anthropic.chat.guardrail_translation.handler import (
    AnthropicMessagesHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs


class MockPassThroughGuardrail(CustomGuardrail):
    """Mock guardrail that passes through without blocking - for testing streaming fallback behavior"""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Simply return inputs unchanged"""
        return inputs


class TestAnthropicMessagesHandlerStreamingOutputProcessing:
    """Test streaming output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_model_response(self):
        """Test that streaming response with None model_response doesn't raise error

        This test verifies the fix for the bug where accessing model_response.choices[0]
        would raise an error when _build_complete_streaming_response returns None.
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return None
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=True
        ), patch(
            "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
            return_value=None,
        ):
            responses_so_far = [b"data: some chunk"]

            # This should not raise an error
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses unchanged
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_choices(self):
        """Test that streaming response with empty choices doesn't raise IndexError

        This test verifies the fix for the bug where accessing model_response.choices[0]
        would raise IndexError when the response has an empty choices list.
        """
        from litellm.types.utils import ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a mock response with empty choices
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-3",
            object="chat.completion",
            choices=[],  # Empty choices
        )

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return the mock response
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=True
        ), patch(
            "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
            return_value=mock_response,
        ):
            responses_so_far = [b"data: some chunk"]

            # This should not raise IndexError
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses unchanged
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_with_valid_choices(self):
        """Test that streaming response with valid choices still works correctly"""
        from litellm.types.utils import Choices, Message, ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a mock response with valid choices
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-3",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Hello world",
                        role="assistant",
                    ),
                )
            ],
        )

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return the mock response
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=True
        ), patch(
            "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
            return_value=mock_response,
        ):
            responses_so_far = [b"data: some chunk"]

            # This should process successfully
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_stream_not_ended(self):
        """Test that streaming response falls back to text processing when stream hasn't ended"""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mock _check_streaming_has_ended to return False (stream not ended)
        with patch.object(
            handler, "_check_streaming_has_ended", return_value=False
        ), patch.object(
            handler, "get_streaming_string_so_far", return_value="partial text"
        ):
            responses_so_far = [b"data: some chunk"]

            # This should process successfully using text-based guardrail
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses
            assert result == responses_so_far


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

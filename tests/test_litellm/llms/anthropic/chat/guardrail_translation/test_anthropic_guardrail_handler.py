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

sys.path.insert(0, os.path.abspath("../../../../../../.."))  # Adds the parent directory to the system path

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


class MockDynamicGuardrail(CustomGuardrail):
    """Mock guardrail that records dynamic params from request metadata."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.dynamic_params: Optional[dict] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.dynamic_params = self.get_guardrail_dynamic_request_body_params(request_data)
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
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=None,
            ),
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


class TestAnthropicMessagesHandlerInputProcessing:
    """Test input processing preserves litellm_metadata for dynamic guardrails."""

    @pytest.mark.asyncio
    async def test_process_input_messages_preserves_litellm_metadata_guardrails(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockDynamicGuardrail(guardrail_name="cygnal-monitor")

        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
            "litellm_metadata": {"guardrails": [{"cygnal-monitor": {"extra_body": {"policy_id": "policy-123"}}}]},
        }

        with patch("litellm.proxy.proxy_server.premium_user", True):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data.get("litellm_metadata", {}).get("guardrails")
        assert guardrail.dynamic_params == {"policy_id": "policy-123"}

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
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=mock_response,
            ),
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
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=mock_response,
            ),
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
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=False),
            patch.object(handler, "get_streaming_string_so_far", return_value="partial text"),
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

    @pytest.mark.asyncio
    async def test_process_input_messages_with_anthropic_native_tools(self):
        """Test that Anthropic native tools (tool_search_tool_regex) are preserved correctly

        This test verifies the fix for the bug where Anthropic native tools like
        tool_search_tool_regex_20251119 were being converted to OpenAI format and then
        not properly converted back, causing API errors.

        The guardrail converts tools to OpenAI format for processing, then they need to be
        converted back to Anthropic format. Native Anthropic tools should be preserved as-is,
        while regular tools should be converted to type="custom".
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
            "tools": [
                {
                    "type": "tool_search_tool_regex_20251119",
                    "name": "tool_search_tool_regex",
                },
                {
                    "name": "get_weather",
                    "description": "Get the weather at a specific location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                    "defer_loading": True,
                },
            ],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        # Verify tools are in correct Anthropic format
        tools = result["tools"]
        assert len(tools) == 2

        # First tool should be preserved as Anthropic native tool
        assert tools[0]["type"] == "tool_search_tool_regex_20251119"
        assert tools[0]["name"] == "tool_search_tool_regex"

        # Second tool should be converted to Anthropic custom tool format
        assert tools[1]["type"] == "custom"
        assert tools[1]["name"] == "get_weather"
        assert tools[1]["description"] == "Get the weather at a specific location"
        assert "input_schema" in tools[1]


class MockStructuredMessagesGuardrail(CustomGuardrail):
    """Mock guardrail that returns compressed structured_messages."""

    def __init__(self, guardrail_name: str, compressed_messages: List[Any]):
        super().__init__(guardrail_name=guardrail_name)
        self.compressed_messages = compressed_messages

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        result = dict(inputs)
        result["structured_messages"] = self.compressed_messages
        return GenericGuardrailAPIInputs(**result)


class TestAnthropicStructuredMessagesApplied:
    """Test that structured_messages from guardrail response are applied back in Anthropic format."""

    @pytest.mark.asyncio
    async def test_structured_messages_replaced_in_anthropic_format(self):
        """When guardrail returns structured_messages (OpenAI format), they must be
        converted back to Anthropic format and written to data['messages']."""
        handler = AnthropicMessagesHandler()

        compressed_openai_messages = [
            {"role": "user", "content": "compressed content"},
        ]
        guardrail = MockStructuredMessagesGuardrail(
            guardrail_name="test-compressor",
            compressed_messages=compressed_openai_messages,
        )

        data: dict = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "original long content that gets compressed"},
            ],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        if isinstance(content, str):
            assert content == "compressed content"
        elif isinstance(content, list):
            assert any(block.get("text") == "compressed content" for block in content if isinstance(block, dict))

    @pytest.mark.asyncio
    async def test_no_structured_messages_falls_back_to_text_patching(self):
        """When guardrail returns no structured_messages, the original text-patch path runs."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        original_content = "original content"
        data: dict = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": original_content}],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        messages = result["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        if isinstance(content, str):
            assert content == original_content


class TestFunctionSetupLitellmMetadataReference:
    """Regression test: guardrail info written to litellm_metadata must appear in
    standard logging payload for endpoints that use litellm_metadata (e.g. /v1/messages)."""

    def test_litellm_metadata_reference_not_copy(self):
        """litellm_params['metadata'] must be the same object as kwargs['litellm_metadata']
        so mutations by guardrails after function_setup are visible at logging time."""
        import litellm
        from datetime import datetime

        litellm_metadata_dict: dict = {"user_api_key": "test-key"}
        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
            "litellm_call_id": "test-call-id",
            "litellm_metadata": litellm_metadata_dict,
        }

        logging_obj, _ = litellm.utils.function_setup(
            original_function="anthropic_messages",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.now(),
            **kwargs,
        )

        litellm_metadata_dict["standard_logging_guardrail_information"] = [
            {"guardrail_name": "test-guardrail", "guardrail_status": "success"}
        ]

        metadata_in_logging = logging_obj.litellm_params.get("metadata", {})
        assert "standard_logging_guardrail_information" in metadata_in_logging, (
            "guardrail info written to litellm_metadata after function_setup must be "
            "visible in litellm_params['metadata'] — check function_setup uses a "
            "reference not a copy for litellm_metadata endpoints"
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

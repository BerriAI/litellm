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
        self.dynamic_params = self.get_guardrail_dynamic_request_body_params(
            request_data
        )
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


class TestAnthropicMessagesHandlerInputProcessing:
    """Test input processing preserves litellm_metadata for dynamic guardrails."""

    @pytest.mark.asyncio
    async def test_process_input_messages_preserves_litellm_metadata_guardrails(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockDynamicGuardrail(guardrail_name="cygnal-monitor")

        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
            "litellm_metadata": {
                "guardrails": [
                    {
                        "cygnal-monitor": {
                            "extra_body": {"policy_id": "policy-123"}
                        }
                    }
                ]
            },
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
                    "name": "tool_search_tool_regex"
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
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    },
                    "defer_loading": True
                }
            ]
        }

        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=MagicMock()
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


class MockToolCallDeletionGuardrail(CustomGuardrail):
    """Mock guardrail that marks specific tool calls for deletion via guardrail_deleted flag."""

    def __init__(self, guardrail_name: str, indices_to_delete: Optional[List[int]] = None):
        super().__init__(guardrail_name=guardrail_name)
        self.indices_to_delete = indices_to_delete or []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        tool_calls = inputs.get("tool_calls", [])
        for idx in self.indices_to_delete:
            if idx < len(tool_calls):
                if isinstance(tool_calls[idx], dict):
                    tool_calls[idx]["guardrail_deleted"] = True
        return inputs


class TestAnthropicOutputToolCallDeletion:
    """Test guardrail_deleted support for Anthropic output tool calls."""

    @pytest.mark.asyncio
    async def test_partial_tool_call_deletion(self):
        """One tool_use block deleted, text and other tool_use remain."""
        handler = AnthropicMessagesHandler()
        guardrail = MockToolCallDeletionGuardrail(
            guardrail_name="test", indices_to_delete=[0]
        )

        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Let me help you."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "dangerous_tool",
                    "input": {"action": "delete"},
                },
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "safe_tool",
                    "input": {"query": "hello"},
                },
            ],
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
        )

        # Text block should remain, first tool_use deleted, second remains
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Let me help you."
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["name"] == "safe_tool"
        # stop_reason should stay tool_use since one tool_use remains
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_full_tool_call_deletion(self):
        """All tool_use blocks deleted, stop_reason changes to end_turn."""
        handler = AnthropicMessagesHandler()
        guardrail = MockToolCallDeletionGuardrail(
            guardrail_name="test", indices_to_delete=[0, 1]
        )

        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "I will run two tools."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "tool_a",
                    "input": {"x": 1},
                },
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "tool_b",
                    "input": {"y": 2},
                },
            ],
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
        )

        # Only text block should remain
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "I will run two tools."
        # stop_reason should change since no tool_use blocks remain
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_no_deletion_regression(self):
        """No deletion flag set — tool calls should pass through unchanged."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Running tool."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "my_tool",
                    "input": {"a": "b"},
                },
            ],
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
        )

        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["name"] == "my_tool"
        assert result["stop_reason"] == "tool_use"


class MockDeletionWithReplacementGuardrail(CustomGuardrail):
    """Mock guardrail that deletes all tool calls and adds replacement text."""

    def __init__(self, guardrail_name: str, replacement_text: str):
        super().__init__(guardrail_name=guardrail_name)
        self.replacement_text = replacement_text

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        tool_calls = inputs.get("tool_calls", [])
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc["guardrail_deleted"] = True

        texts = list(inputs.get("texts", []))
        texts.append(self.replacement_text)

        result: GenericGuardrailAPIInputs = {"texts": texts}
        if tool_calls:
            result["tool_calls"] = tool_calls  # type: ignore
        return result


class TestAnthropicReplacementText:
    """Test adding replacement text when deleting tool calls."""

    @pytest.mark.asyncio
    async def test_tool_only_response_deletion_with_replacement_text(self):
        """Tool-call-only response: delete all and inject replacement text."""
        handler = AnthropicMessagesHandler()
        guardrail = MockDeletionWithReplacementGuardrail(
            guardrail_name="test",
            replacement_text="Tool call blocked by policy.",
        )

        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "dangerous_tool",
                    "input": {"action": "delete"},
                },
            ],
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
        )

        # Tool use deleted, replacement text injected
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Tool call blocked by policy."
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_only_response_no_deletion_no_replacement(self):
        """Tool-call-only response with passthrough: no text added."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "my_tool",
                    "input": {"a": "b"},
                },
            ],
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
        )

        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["stop_reason"] == "tool_use"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

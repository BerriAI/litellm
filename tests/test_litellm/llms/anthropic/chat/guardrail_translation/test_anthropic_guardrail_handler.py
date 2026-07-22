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
            "litellm_metadata": {
                "guardrails": [
                    {"cygnal-monitor": {"extra_body": {"policy_id": "policy-123"}}}
                ]
            },
        }

        with patch("litellm.proxy.proxy_server.premium_user", True):
            await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

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
            patch.object(
                handler, "get_streaming_string_so_far", return_value="partial text"
            ),
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
            "messages": [
                {"role": "user", "content": "What is the weather in San Francisco?"}
            ],
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


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])


class TestAnthropicMessagesIncrementalScan:
    """PR #33278: only_scan_new_messages through the real /v1/messages translation
    handler (the path Claude Code uses). Encodes the wire payloads observed in the
    live validation against a real Bedrock guardrail.
    """

    def _bedrock_guardrail(self):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail

        return BedrockGuardrail(
            guardrail_name="bedrock-incremental-anthropic",
            guardrailIdentifier="test-guardrail",
            guardrailVersion="DRAFT",
            default_on=True,
            only_scan_new_messages=True,
        )

    def _data(self, messages, session_id):
        return {
            "model": "claude-sonnet-4-5",
            "messages": messages,
            "system": "You are a helpful geography assistant.",
            "litellm_session_id": session_id,
        }

    @pytest.mark.asyncio
    async def test_first_turn_scans_all_eligible_then_second_turn_scans_only_diff(self):
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-diff"
        turn1 = [{"role": "user", "content": "What is the capital of France?"}]
        turn2 = turn1 + [
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(
                data=self._data(turn1, sid), guardrail_to_apply=guardrail
            )
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "What is the capital of France?"
            ]
            mock_api.reset_mock()
            await handler.process_input_messages(
                data=self._data(turn2, sid), guardrail_to_apply=guardrail
            )
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "Paris.",
                "What is the capital of Germany?",
            ]

    @pytest.mark.asyncio
    async def test_identical_resend_makes_no_guardrail_call(self):
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-resend"
        msgs = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            assert mock_api.call_count == 1
            mock_api.reset_mock()
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            mock_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_edited_history_message_is_rescanned(self):
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-edit"
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        edited = [{"role": "user", "content": "What is the capital and population of France?"}]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            mock_api.reset_mock()
            await handler.process_input_messages(data=self._data(edited, sid), guardrail_to_apply=guardrail)
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "What is the capital and population of France?"
            ]

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_use_keeps_text_segments(self):
        """A message carrying both text and a tool_use block must not lose its text.
        (tool_use inputs and tool_result content are dropped from texts on the
        anthropic input path today; that is pre-existing baseline behavior.)"""
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-tools"
        msgs = [
            {"role": "user", "content": "Search for the weather in Paris"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look that up for you."},
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"query": "canary-args"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "canary-result"}],
            },
            {"role": "user", "content": "Thanks, summarize the result."},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            scanned = [m["content"] for m in mock_api.call_args.kwargs["messages"]]
            assert "Let me look that up for you." in scanned, "text beside a tool_use must be scanned"
            assert "Search for the weather in Paris" in scanned
            assert "Thanks, summarize the result." in scanned

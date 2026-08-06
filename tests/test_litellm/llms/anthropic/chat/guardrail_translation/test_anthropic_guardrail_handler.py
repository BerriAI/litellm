"""
Unit tests for Anthropic Messages Guardrail Translation Handler

Tests the handler's ability to process streaming output for Anthropic Messages API
with guardrail transformations, specifically testing edge cases with empty choices.
"""

import json
import os
import sys
from typing import Any, Literal, Optional
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


class MockRecordingGuardrail(CustomGuardrail):
    """Mock guardrail that records the request_data it was handed."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.request_data: Optional[dict] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.request_data = request_data
        return inputs


class TestAnthropicMessagesHandlerStreamingRequestData:
    """Post-call guardrails on streaming /v1/messages receive the response and identity metadata"""

    @pytest.mark.asyncio
    async def test_terminal_chunk_passes_assembled_response_and_metadata(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.utils import Choices, Message, ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="test")
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-sonnet-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello world", role="assistant"),
                )
            ],
        )

        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=mock_response,
            ),
        ):
            await handler.process_output_streaming_response(
                responses_so_far=[b"data: some chunk"],
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
                user_api_key_dict=UserAPIKeyAuth(user_id="u-1", team_id="t-1"),
                request_data={"model": "claude-sonnet-4-5"},
            )

        assert guardrail.request_data is not None
        assert guardrail.request_data["response"] is mock_response
        assert (
            guardrail.request_data["litellm_metadata"]["user_api_key_user_id"] == "u-1"
        )

    @pytest.mark.asyncio
    async def test_mid_stream_chunk_passes_responses_so_far_and_metadata(self):
        from litellm.proxy._types import UserAPIKeyAuth

        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="test")
        responses_so_far = [b"data: some chunk"]

        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=False),
            patch.object(
                handler, "get_streaming_string_so_far", return_value="partial text"
            ),
        ):
            await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
                user_api_key_dict=UserAPIKeyAuth(user_id="u-1", team_id="t-1"),
                request_data={"model": "claude-sonnet-4-5"},
            )

        assert guardrail.request_data is not None
        assert guardrail.request_data["responses"] is responses_so_far
        assert (
            guardrail.request_data["litellm_metadata"]["user_api_key_user_id"] == "u-1"
        )


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


class ToolAppendingGuardrail(CustomGuardrail):
    """Guardrail that appends a new OpenAI-format function tool, mimicking a
    guardrail that injects a retrieval/recovery tool the model can later call."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        tools = list(inputs.get("tools") or [])
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "injected_tool",
                    "description": "injected by guardrail",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
        inputs["tools"] = tools
        return inputs


class TestAnthropicMessagesHandlerToolInjection:
    """A tool a guardrail injects in OpenAI format must survive the write-back
    to Anthropic format alongside the request's original tools."""

    @pytest.mark.asyncio
    async def test_injected_tool_survives_when_request_already_has_tools(self):
        handler = AnthropicMessagesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="test")

        data = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather at a specific location",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        names = [t.get("name") for t in result["tools"]]
        assert "get_weather" in names
        assert "injected_tool" in names

    @pytest.mark.asyncio
    async def test_injected_tool_survives_when_request_has_no_tools(self):
        handler = AnthropicMessagesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="test")

        data = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        assert [t.get("name") for t in result["tools"]] == ["injected_tool"]


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
        (tool_use inputs are still dropped from texts on the anthropic input path;
        tool_result content is scanned, see TestAnthropicMessagesToolResultScanning.)"""
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


class MockMaskingGuardrail(CustomGuardrail):
    """Records every text handed to it and masks a canary token in place."""

    def __init__(self, guardrail_name: str = "mask-canary"):
        super().__init__(guardrail_name=guardrail_name)
        self.seen_texts: list[str] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = list(inputs.get("texts") or [])
        self.seen_texts.extend(texts)
        inputs["texts"] = [t.replace("POISON", "[BLOCKED]") for t in texts]
        return inputs


class TestAnthropicMessagesToolResultScanning:
    """LIT-5251: tool_result blocks carry whatever a client's local tool fetched, so
    they are the request-path payload an indirect prompt injection actually arrives in.
    Both wire shapes Anthropic accepts must be scanned and rewritten in place.
    """

    def _data(self, messages):
        return {"model": "claude-sonnet-4-5", "messages": messages}

    @pytest.mark.asyncio
    async def test_string_form_tool_result_is_scanned_and_written_back(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        messages = [
            {"role": "user", "content": "fetch the page"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "curl"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "page says POISON here"}],
            },
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "page says POISON here" in guardrail.seen_texts, "string-form tool_result must reach the guardrail"
        assert messages[2]["content"][0]["content"] == "page says [BLOCKED] here", (
            "masked text must be written back into the tool_result, not dropped"
        )

    @pytest.mark.asyncio
    async def test_list_form_tool_result_is_scanned_and_written_back(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        messages = [
            {"role": "user", "content": "fetch the page"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [
                            {"type": "text", "text": "first POISON block"},
                            {"type": "text", "text": "second POISON block"},
                        ],
                    }
                ],
            },
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "first POISON block" in guardrail.seen_texts
        assert "second POISON block" in guardrail.seen_texts
        blocks = messages[1]["content"][0]["content"]
        assert blocks[0]["text"] == "first [BLOCKED] block"
        assert blocks[1]["text"] == "second [BLOCKED] block"

    @pytest.mark.asyncio
    async def test_write_back_targets_stay_aligned_across_mixed_shapes(self):
        """The write-back is positional, so a single mis-indexed target silently
        writes one message's masked text over another's."""
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        messages = [
            {"role": "user", "content": "plain POISON string"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "sibling POISON text"},
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "string POISON result"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu2",
                        "content": [{"type": "text", "text": "nested POISON result"}],
                    },
                ],
            },
            {"role": "user", "content": "trailing POISON string"},
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert messages[0]["content"] == "plain [BLOCKED] string"
        assert messages[1]["content"][0]["text"] == "sibling [BLOCKED] text"
        assert messages[1]["content"][1]["content"] == "string [BLOCKED] result"
        assert messages[1]["content"][2]["content"][0]["text"] == "nested [BLOCKED] result"
        assert messages[2]["content"] == "trailing [BLOCKED] string"

    @pytest.mark.asyncio
    async def test_image_inside_tool_result_is_collected(self):
        handler = AnthropicMessagesHandler()

        class ImageRecordingGuardrail(MockMaskingGuardrail):
            def __init__(self):
                super().__init__()
                self.seen_images: list[str] = []

            async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
                self.seen_images.extend(inputs.get("images") or [])
                return await super().apply_guardrail(inputs, request_data, input_type, logging_obj)

        guardrail = ImageRecordingGuardrail()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [
                            {"type": "text", "text": "screenshot POISON"},
                            {"type": "image", "source": {"type": "base64", "data": "SCREENSHOT_BYTES"}},
                        ],
                    }
                ],
            }
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "SCREENSHOT_BYTES" in guardrail.seen_images, "images nested in a tool_result must be scanned too"

    @pytest.mark.asyncio
    async def test_tool_result_is_skipped_when_guardrail_skips_tool_messages(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        messages = [
            {"role": "user", "content": "keep me POISON"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "skip me POISON"}],
            },
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "skip me POISON" not in guardrail.seen_texts
        assert messages[1]["content"][0]["content"] == "skip me POISON"
        assert messages[0]["content"] == "keep me [BLOCKED]"


class InputsRecordingGuardrail(MockMaskingGuardrail):
    def __init__(self):
        super().__init__(guardrail_name="scan-only-capture")
        self.captured_inputs: Optional[GenericGuardrailAPIInputs] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.captured_inputs = inputs
        return await super().apply_guardrail(inputs, request_data, input_type, logging_obj)


class StructuredMessagesRewritingGuardrail(CustomGuardrail):
    """Returns a new structured_messages list with a canary redacted, like redaction guardrails do."""

    def __init__(self):
        super().__init__(guardrail_name="structured-rewrite")

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        structured = inputs.get("structured_messages") or []
        inputs["structured_messages"] = [
            json.loads(json.dumps(message).replace("POISON", "[BLOCKED]")) for message in structured
        ]
        return inputs


class TestAnthropicMessagesScanOnlyToolResults:
    def _guardrail(self):
        guardrail = InputsRecordingGuardrail()
        guardrail.scan_only_tool_results = True
        return guardrail

    @pytest.mark.asyncio
    async def test_structured_write_back_merges_into_the_full_conversation(self):
        handler = AnthropicMessagesHandler()
        guardrail = StructuredMessagesRewritingGuardrail()
        guardrail.scan_only_tool_results = True
        data = {
            "model": "claude-sonnet-4-5",
            "system": "You are a careful agent harness.",
            "messages": [
                {"role": "user", "content": "fetch the page"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "curl"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "fetched POISON page"}],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["system"] == "You are a careful agent harness."
        assert [m["role"] for m in data["messages"]] == ["user", "assistant", "user"], (
            "a redacting guardrail must not strip out-of-scope turns from the request"
        )
        serialized = json.dumps(data["messages"])
        assert "fetch the page" in serialized
        assert "tool_use" in serialized
        assert "fetched [BLOCKED] page" in serialized
        assert "POISON" not in serialized

    @pytest.mark.asyncio
    async def test_scan_narrows_to_tool_results_and_write_back_stays_aligned(self):
        handler = AnthropicMessagesHandler()
        guardrail = self._guardrail()
        data = {
            "model": "claude-sonnet-4-5",
            "system": "You are a trusted agent harness with POISON heuristics.",
            "tools": [
                {
                    "name": "Bash",
                    "description": "run a command",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "messages": [
                {"role": "user", "content": "scaffolding POISON prompt"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "curl"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "sibling POISON text"},
                        {"type": "tool_result", "tool_use_id": "tu1", "content": "fetched POISON page"},
                    ],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.seen_texts == ["fetched POISON page"], (
            "only the tool_result payload may reach the guardrail"
        )
        assert guardrail.captured_inputs is not None
        assert guardrail.captured_inputs.get("tools") is None
        assert [m["role"] for m in guardrail.captured_inputs["structured_messages"]] == ["tool"]
        assert data["messages"][2]["content"][1]["content"] == "fetched [BLOCKED] page"
        assert data["messages"][0]["content"] == "scaffolding POISON prompt", (
            "out-of-scope content must come back untouched, not masked or dropped"
        )
        assert data["messages"][2]["content"][0]["text"] == "sibling POISON text"

    @pytest.mark.asyncio
    async def test_guardrail_synthesized_tools_are_appended_without_replacing_request_tools(self):
        handler = AnthropicMessagesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="tool-appending")
        guardrail.scan_only_tool_results = True
        original_tools = [
            {
                "name": "get_weather",
                "description": "Get the weather at a specific location",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
            }
        ]
        data = {
            "model": "claude-sonnet-4-5",
            "tools": original_tools,
            "messages": [
                {"role": "user", "content": "what's the weather?"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "sunny"}],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [t["name"] for t in data["tools"]] == ["get_weather", "injected_tool"], (
            "a tool the guardrail synthesized must reach the model, converted to Anthropic format, "
            "without the request's own tools being replaced or dropped"
        )
        assert data["tools"][0] == original_tools[0]

    @pytest.mark.asyncio
    async def test_guardrail_is_not_called_when_the_request_has_no_tool_results(self):
        handler = AnthropicMessagesHandler()
        guardrail = self._guardrail()
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "What is 2 plus 2?"}],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.captured_inputs is None
        assert guardrail.seen_texts == []

    @pytest.mark.asyncio
    async def test_images_are_scoped_the_same_way_as_texts(self):
        handler = AnthropicMessagesHandler()
        guardrail = self._guardrail()
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "USER_IMG"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": [
                                {"type": "text", "text": "screenshot POISON"},
                                {"type": "image", "source": {"type": "base64", "data": "TOOL_IMG"}},
                            ],
                        }
                    ],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.captured_inputs is not None
        assert guardrail.captured_inputs.get("images") == ["TOOL_IMG"]

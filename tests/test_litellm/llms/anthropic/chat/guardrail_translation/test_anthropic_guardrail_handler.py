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


class MockToolRemovingGuardrail(CustomGuardrail):
    """Mock guardrail that removes tools by name - for testing tool reconciliation."""

    def __init__(self, guardrail_name: str, tools_to_remove: List[str]):
        super().__init__(guardrail_name=guardrail_name)
        self.tools_to_remove = tools_to_remove

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Remove specified tools from the input."""
        tools = inputs.get("tools")
        if tools is not None:
            inputs["tools"] = [
                t
                for t in tools
                if isinstance(t, dict)
                and t.get("function", {}).get("name") not in self.tools_to_remove
            ]
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


class TestReconcileGuardrailedTools:
    """Test _reconcile_guardrailed_tools preserves Anthropic-native tool format."""

    def test_tools_unchanged_returns_original(self):
        """When guardrail returns tools unchanged, original Anthropic tools are preserved."""
        original_anthropic_tools = [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "text_editor"},
        ]
        openai_tools_before = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {"type": "function", "function": {"name": "text_editor", "parameters": {}}},
        ]
        # Guardrail returns tools unchanged
        openai_tools_after = list(openai_tools_before)

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert result == original_anthropic_tools
        # Verify the native types are preserved
        assert result[0]["type"] == "bash_20250124"
        assert result[1]["type"] == "text_editor_20250124"

    def test_tool_removed_by_guardrail(self):
        """When guardrail removes a tool, corresponding Anthropic tool is removed."""
        original_anthropic_tools = [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "text_editor"},
            {"type": "web_search_20260209", "name": "web_search"},
        ]
        openai_tools_before = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {"type": "function", "function": {"name": "text_editor", "parameters": {}}},
            {"type": "function", "function": {"name": "web_search", "parameters": {}}},
        ]
        # Guardrail removed "bash" tool
        openai_tools_after = [
            {"type": "function", "function": {"name": "text_editor", "parameters": {}}},
            {"type": "function", "function": {"name": "web_search", "parameters": {}}},
        ]

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert len(result) == 2
        assert result[0]["name"] == "text_editor"
        assert result[0]["type"] == "text_editor_20250124"
        assert result[1]["name"] == "web_search"
        assert result[1]["type"] == "web_search_20260209"

    def test_no_original_tools_returns_guardrail_output(self):
        """When there are no original Anthropic tools, return guardrail output as-is."""
        openai_tools_after = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
        ]

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=None,
            openai_tools_before=[],
            openai_tools_after=openai_tools_after,
        )

        assert result == openai_tools_after

    def test_all_tools_removed_by_guardrail(self):
        """When guardrail removes all tools, return empty list."""
        original_anthropic_tools = [
            {"type": "bash_20250124", "name": "bash"},
        ]
        openai_tools_before = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
        ]
        openai_tools_after: list = []

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert result == []

    def test_standard_anthropic_tools_preserved(self):
        """Standard Anthropic tools (without special types) are also preserved."""
        original_anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ]
        openai_tools_before = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            },
        ]
        openai_tools_after = list(openai_tools_before)

        result = AnthropicMessagesHandler._reconcile_guardrailed_tools(
            original_anthropic_tools=original_anthropic_tools,
            openai_tools_before=openai_tools_before,
            openai_tools_after=openai_tools_after,
        )

        assert result == original_anthropic_tools
        # Verify original structure is preserved (input_schema, not parameters)
        assert "input_schema" in result[0]


def _mock_proxy_server():
    """Create a mock for litellm.proxy.proxy_server when proxy deps aren't installed."""
    import types

    mock_module = types.ModuleType("litellm.proxy.proxy_server")
    mock_module.premium_user = True  # type: ignore[attr-defined]
    return mock_module


class TestAnthropicToolFormatPreservation:
    """Integration tests for Anthropic-native tool format preservation through guardrails."""

    @pytest.mark.asyncio
    async def test_native_anthropic_tools_preserved_through_passthrough_guardrail(self):
        """Anthropic-native tools (bash_20250124, etc.) survive guardrail processing unchanged.

        This is the core regression test for the bug where guardrail processing
        converted Anthropic tools to OpenAI format (type: "function"), causing
        the Anthropic API to reject the request.
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "List files in current directory"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "text_editor"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Tools must remain in Anthropic-native format
        tools = result["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "bash_20250124"
        assert tools[0]["name"] == "bash"
        assert tools[1]["type"] == "text_editor_20250124"
        assert tools[1]["name"] == "text_editor"

    @pytest.mark.asyncio
    async def test_native_anthropic_tools_with_removal_guardrail(self):
        """When a guardrail removes a tool, the remaining tools stay in Anthropic format."""
        handler = AnthropicMessagesHandler()
        guardrail = MockToolRemovingGuardrail(
            guardrail_name="test", tools_to_remove=["bash"]
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "text_editor"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        # Only text_editor should remain, in Anthropic format
        tools = result["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "text_editor_20250124"
        assert tools[0]["name"] == "text_editor"

    @pytest.mark.asyncio
    async def test_mixed_tools_preserved_through_guardrail(self):
        """Mixed Anthropic-native and standard tools are all preserved."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        tools = result["tools"]
        assert len(tools) == 2
        assert tools[0]["type"] == "bash_20250124"
        assert tools[0]["name"] == "bash"
        # Standard tool preserves its original structure
        assert tools[1]["name"] == "get_weather"
        assert "input_schema" in tools[1]

    @pytest.mark.asyncio
    async def test_web_search_tool_preserved_through_guardrail(self):
        """Web search tool type is preserved through guardrail processing."""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        }

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_server()}):
            result = await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        tools = result["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search_20250305"
        assert tools[0]["name"] == "web_search"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

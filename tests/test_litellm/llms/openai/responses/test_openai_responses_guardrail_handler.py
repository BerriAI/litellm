"""
Unit tests for OpenAI Responses API Guardrail Translation Handler

Tests the handler's ability to process input/output for the Responses API
with guardrail transformations.
"""

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.responses.main import GenericResponseOutputItem, OutputText
from litellm.types.utils import CallTypes


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing that transforms text"""

    async def apply_guardrail(self, text: str) -> str:
        """Append [GUARDRAILED] to text"""
        return f"{text} [GUARDRAILED]"


class TestOpenAIResponsesHandlerDiscovery:
    """Test that the handler is properly discovered by the guardrail system"""

    def test_handler_discovered_for_responses(self):
        """Test that handler is discovered for CallTypes.responses"""
        handler_class = get_guardrail_translation_mapping(CallTypes.responses)
        assert handler_class == OpenAIResponsesHandler

    def test_handler_discovered_for_aresponses(self):
        """Test that handler is discovered for CallTypes.aresponses"""
        handler_class = get_guardrail_translation_mapping(CallTypes.aresponses)
        assert handler_class == OpenAIResponsesHandler

    def test_handler_has_required_methods(self):
        """Test that handler has required methods"""
        handler = OpenAIResponsesHandler()
        assert hasattr(handler, "process_input_messages")
        assert hasattr(handler, "process_output_response")
        assert callable(handler.process_input_messages)
        assert callable(handler.process_output_response)


class TestOpenAIResponsesHandlerInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_process_input_string(self):
        """Test processing simple string input"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"input": "Hello world", "model": "gpt-4"}

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"] == "Hello world [GUARDRAILED]"
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_process_input_none(self):
        """Test processing when input is None"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "gpt-4"}

        result = await handler.process_input_messages(data, guardrail)

        assert "input" not in result
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_process_input_list_with_string_content(self):
        """Test processing list input with string content"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "input": [
                {"role": "user", "content": "Hello", "type": "message"},
                {"role": "user", "content": "World", "type": "message"},
            ],
            "model": "gpt-4",
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"][0]["content"] == "Hello [GUARDRAILED]"
        assert result["input"][1]["content"] == "World [GUARDRAILED]"
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_process_input_list_with_multimodal_content(self):
        """Test processing list input with multimodal content"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                    ],
                    "type": "message",
                }
            ],
            "model": "gpt-4",
        }

        result = await handler.process_input_messages(data, guardrail)

        assert (
            result["input"][0]["content"][0]["text"]
            == "Describe this image [GUARDRAILED]"
        )
        # Image URL should remain unchanged
        assert (
            result["input"][0]["content"][1]["image_url"]["url"]
            == "https://example.com/image.jpg"
        )

    @pytest.mark.asyncio
    async def test_process_input_with_empty_content(self):
        """Test processing input with empty or None content"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "input": [
                {"role": "user", "content": None, "type": "message"},
                {"role": "user", "content": "", "type": "message"},
            ],
            "model": "gpt-4",
        }

        result = await handler.process_input_messages(data, guardrail)

        # None content should remain None
        assert result["input"][0]["content"] is None
        # Empty string should be processed
        assert result["input"][1]["content"] == " [GUARDRAILED]"


class TestOpenAIResponsesHandlerOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_response_simple(self):
        """Test processing simple output response"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a mock response
        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                GenericResponseOutputItem(
                    type="message",
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(
                            type="output_text", text="Hello user", annotations=None
                        ),
                    ],
                )
            ],
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.output[0].content[0].text == "Hello user [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_output_response_multiple_items(self):
        """Test processing output response with multiple output items"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                GenericResponseOutputItem(
                    type="message",
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(
                            type="output_text",
                            text="First message",
                            annotations=None,
                        ),
                    ],
                ),
                GenericResponseOutputItem(
                    type="message",
                    id="msg_124",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(
                            type="output_text",
                            text="Second message",
                            annotations=None,
                        ),
                    ],
                ),
            ],
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.output[0].content[0].text == "First message [GUARDRAILED]"
        assert result.output[1].content[0].text == "Second message [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_output_response_multiple_content_items(self):
        """Test processing output response with multiple content items in one output"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                GenericResponseOutputItem(
                    type="message",
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(type="output_text", text="Part 1", annotations=None),
                        OutputText(type="output_text", text="Part 2", annotations=None),
                    ],
                )
            ],
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.output[0].content[0].text == "Part 1 [GUARDRAILED]"
        assert result.output[0].content[1].text == "Part 2 [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_output_response_with_dict_format(self):
        """Test processing output response where content items are dicts instead of OutputText objects"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Simulate response with dict content (which can happen in some cases)
        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                {
                    "type": "message",
                    "id": "msg_123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello from dict"},
                    ],
                }
            ],
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.output[0]["content"][0]["text"] == "Hello from dict [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_output_response_no_text_content(self):
        """Test that handler skips processing when there's no text content"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[],
        )

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged response
        assert result == response


class TestOpenAIResponsesHandlerHelperMethods:
    """Test helper methods"""

    def test_has_text_content_with_text(self):
        """Test _has_text_content returns True when text exists"""
        handler = OpenAIResponsesHandler()

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                GenericResponseOutputItem(
                    type="message",
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(type="output_text", text="Hello", annotations=None),
                    ],
                )
            ],
        )

        assert handler._has_text_content(response) is True

    def test_has_text_content_without_text(self):
        """Test _has_text_content returns False when no text exists"""
        handler = OpenAIResponsesHandler()

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[],
        )

        assert handler._has_text_content(response) is False

    def test_has_text_content_with_empty_text(self):
        """Test _has_text_content with empty text values"""
        handler = OpenAIResponsesHandler()

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                GenericResponseOutputItem(
                    type="message",
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(type="output_text", text="", annotations=None),
                    ],
                )
            ],
        )

        # Empty string should still return False
        assert handler._has_text_content(response) is False

    def test_has_text_content_with_dict_format(self):
        """Test _has_text_content with dict-based output items"""
        handler = OpenAIResponsesHandler()

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                {
                    "type": "message",
                    "id": "msg_123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello"},
                    ],
                }
            ],
        )

        assert handler._has_text_content(response) is True


class TestOpenAIResponsesHandlerEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_process_input_with_non_list_non_string(self):
        """Test processing when input is neither string nor list"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"input": 123, "model": "gpt-4"}  # Invalid type

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged
        assert result["input"] == 123

    @pytest.mark.asyncio
    async def test_process_input_mixed_content_types(self):
        """Test processing with mixed content types in list"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "input": [
                {"role": "user", "content": "String content", "type": "message"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List content"},
                        {"type": "image_url", "image_url": {"url": "http://example.com"}},
                    ],
                    "type": "message",
                },
            ],
            "model": "gpt-4",
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"][0]["content"] == "String content [GUARDRAILED]"
        assert result["input"][1]["content"][0]["text"] == "List content [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_output_with_none_text(self):
        """Test processing output when text field is None"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            model="gpt-4",
            object="response",
            status="completed",
            output=[
                GenericResponseOutputItem(
                    type="message",
                    id="msg_123",
                    status="completed",
                    role="assistant",
                    content=[
                        OutputText(type="output_text", text=None, annotations=None),
                    ],
                )
            ],
        )

        result = await handler.process_output_response(response, guardrail)

        # Should skip processing and return unchanged
        assert result == response


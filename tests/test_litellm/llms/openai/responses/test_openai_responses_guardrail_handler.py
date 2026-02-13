"""
Unit tests for OpenAI Responses API Guardrail Translation Handler

Tests the handler's ability to process input/output for the Responses API
with guardrail transformations.
"""

import os
import sys
from typing import Any, List, Literal, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from fastapi import HTTPException
from openai.types.responses import ResponseFunctionToolCall

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.responses.main import GenericResponseOutputItem, OutputText
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing that transforms text for requests and blocks responses"""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        For requests: Append [GUARDRAILED] to text
        For responses: Block by raising HTTPException (masking responses is no longer supported)
        """
        texts = inputs.get("texts", [])
        if input_type == "response":
            # Responses should be blocked, not masked
            raise HTTPException(
                status_code=400,
                detail={"error": "Response blocked by guardrail", "texts": texts},
            )
        # For requests, we can still mask/transform
        inputs["texts"] = [f"{text} [GUARDRAILED]" for text in texts]
        return inputs


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
        """Test processing simple output response - should block, not mask

        After unified_guardrail.py changes, responses can only be blocked/rejected, not masked.
        This test verifies that the guardrail properly blocks responses.
        """
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a mock response with dict format (works with current handler)
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
                        {"type": "output_text", "text": "Hello user"},
                    ],
                }
            ],
        )

        # Response should be blocked, not masked
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(response, guardrail)

        assert exc_info.value.status_code == 400
        assert "Response blocked by guardrail" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_process_output_response_multiple_items(self):
        """Test processing output response with multiple output items - should block, not mask

        After unified_guardrail.py changes, responses can only be blocked/rejected, not masked.
        This test verifies that the guardrail properly blocks responses with multiple items.
        """
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Use dict format (works with current handler)
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
                        {"type": "output_text", "text": "First message"},
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_124",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Second message"},
                    ],
                },
            ],
        )

        # Response should be blocked, not masked
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(response, guardrail)

        assert exc_info.value.status_code == 400
        assert "Response blocked by guardrail" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_process_output_response_multiple_content_items(self):
        """Test processing output response with multiple content items - should block, not mask

        After unified_guardrail.py changes, responses can only be blocked/rejected, not masked.
        This test verifies that the guardrail properly blocks responses with multiple content items.
        """
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Use dict format (works with current handler)
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
                        {"type": "output_text", "text": "Part 1"},
                        {"type": "output_text", "text": "Part 2"},
                    ],
                }
            ],
        )

        # Response should be blocked, not masked
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(response, guardrail)

        assert exc_info.value.status_code == 400
        assert "Response blocked by guardrail" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_process_output_response_with_dict_format(self):
        """Test processing output response with dict format - should block, not mask

        After unified_guardrail.py changes, responses can only be blocked/rejected, not masked.
        This test verifies blocking works even when content items are dicts instead of OutputText objects.
        """
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

        # Response should be blocked, not masked
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(response, guardrail)

        assert exc_info.value.status_code == 400
        assert "Response blocked by guardrail" in str(exc_info.value.detail)

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
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com"},
                        },
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


class TestOpenAIResponsesHandlerToolCallExtraction:
    """Test tool call extraction functionality"""

    def test_extract_tool_call_from_function_call_output(self):
        """Test extracting tool calls from ResponseFunctionToolCall in response output"""
        handler = OpenAIResponsesHandler()

        # Create output item matching the user's provided response structure
        output_item = ResponseFunctionToolCall(
            arguments='{"location":"Boston, MA","unit":"celsius"}',
            call_id="call_4SjsMeA6DUHwGKaE87ZojgOF",
            name="get_current_weather",
            type="function_call",
            id="fc_0a8bd293ceb771ca00693240cb185c8196b4b4d23948c6ac88",
            status="completed",
        )

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[Any] = []
        task_mappings: List[Tuple[int, int]] = []

        # Extract tool calls
        handler._extract_output_text_and_images(
            output_item=output_item,
            output_idx=0,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            task_mappings=task_mappings,
            tool_calls_to_check=tool_calls_to_check,
        )

        # Verify tool call was extracted
        assert len(tool_calls_to_check) == 1
        assert len(texts_to_check) == 0  # No text content in tool call

        # Verify tool call structure
        tool_call = tool_calls_to_check[0]
        assert tool_call["id"] == "call_4SjsMeA6DUHwGKaE87ZojgOF"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_current_weather"
        assert (
            tool_call["function"]["arguments"]
            == '{"location":"Boston, MA","unit":"celsius"}'
        )
        assert tool_call["index"] == 0

    def test_extract_tool_call_from_dict_format(self):
        """Test extracting tool calls from dict representation of function call"""
        handler = OpenAIResponsesHandler()

        # Create output item as dict (another format that may be encountered)
        output_item = {
            "arguments": '{"location":"Boston, MA","unit":"celsius"}',
            "call_id": "call_4SjsMeA6DUHwGKaE87ZojgOF",
            "name": "get_current_weather",
            "type": "function_call",
            "id": "fc_0a8bd293ceb771ca00693240cb185c8196b4b4d23948c6ac88",
            "status": "completed",
        }

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[Any] = []
        task_mappings: List[Tuple[int, int]] = []

        # Extract tool calls
        handler._extract_output_text_and_images(
            output_item=output_item,
            output_idx=0,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            task_mappings=task_mappings,
            tool_calls_to_check=tool_calls_to_check,
        )

        # Verify tool call was extracted
        assert len(tool_calls_to_check) == 1
        assert len(texts_to_check) == 0  # No text content in tool call

        # Verify tool call structure
        tool_call = tool_calls_to_check[0]
        assert tool_call["id"] == "call_4SjsMeA6DUHwGKaE87ZojgOF"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_current_weather"
        assert (
            tool_call["function"]["arguments"]
            == '{"location":"Boston, MA","unit":"celsius"}'
        )

    @pytest.mark.asyncio
    async def test_process_output_response_with_tool_calls(self):
        """Test processing output response containing function tool calls"""
        handler = OpenAIResponsesHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a full response matching user's provided structure
        response = ResponsesAPIResponse(
            id="resp_zlasw86v56zobnneYprKIagz33tpQeh7arqL9mrI1oec47HNQLGz0VL0PpM9z67EADHExs7UjtyGqpoBKcM9oR6icMGx826UsXnlvu3ZvIyrVA1CaMgeaMo9H5DdQMhvmXtriqXpikuyYbIsko97x8GvtBIoSCcovM9s5KCwJ4eWSjfr51d6-GwLIMkCNbQI6AN11uYyIKrIfCt_9j7FZdBnRHhZ0_zE7E1LYWQPm9G9_nPmTyh9FXNLUZ9Uib1SejrCetPargnpQeBibaXqPoj_pXFKvgc-_-znG5IWEsM8WH9Pjbm6uWEwpUiCxt8yfjQGEADqaluLAts1mnzQVEhCtZbU67QG3ebSG-rXtBw511f2pJPzZ8kI4hPISmZL8Co3LmIrdpmzzb02sQRoH3v4HCwzVGXgtRwRYkdpffebYElQWzvYDhqIHFHKNavfF8mC5AVPvPRA5h1Pf3utTf26",
            created_at=1764901066,
            model="gpt-4.1-mini-2025-04-14",
            object="response",
            status="completed",
            output=[
                ResponseFunctionToolCall(
                    arguments='{"location":"Boston, MA","unit":"celsius"}',
                    call_id="call_4SjsMeA6DUHwGKaE87ZojgOF",
                    name="get_current_weather",
                    type="function_call",
                    id="fc_0a8bd293ceb771ca00693240cb185c8196b4b4d23948c6ac88",
                    status="completed",
                )
            ],
        )

        # Response should be blocked since MockGuardrail blocks responses
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(response, guardrail)

        assert exc_info.value.status_code == 400
        assert "Response blocked by guardrail" in str(exc_info.value.detail)

    def test_extract_mixed_content_with_text_and_tool_calls(self):
        """Test extracting both text and tool calls from response"""
        handler = OpenAIResponsesHandler()

        # Create a response with both text and tool call outputs
        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[Any] = []
        task_mappings: List[Tuple[int, int]] = []

        # First extract from a message output
        text_output = {
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "I'll check the weather for you"},
            ],
        }

        handler._extract_output_text_and_images(
            output_item=text_output,
            output_idx=0,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            task_mappings=task_mappings,
            tool_calls_to_check=tool_calls_to_check,
        )

        # Then extract from a tool call output
        tool_call_output = ResponseFunctionToolCall(
            arguments='{"location":"Boston, MA","unit":"celsius"}',
            call_id="call_4SjsMeA6DUHwGKaE87ZojgOF",
            name="get_current_weather",
            type="function_call",
            id="fc_0a8bd293ceb771ca00693240cb185c8196b4b4d23948c6ac88",
            status="completed",
        )

        handler._extract_output_text_and_images(
            output_item=tool_call_output,
            output_idx=1,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            task_mappings=task_mappings,
            tool_calls_to_check=tool_calls_to_check,
        )

        # Verify both were extracted
        assert len(texts_to_check) == 1
        assert texts_to_check[0] == "I'll check the weather for you"
        assert len(tool_calls_to_check) == 1
        assert tool_calls_to_check[0]["function"]["name"] == "get_current_weather"

    def test_extract_text_from_basemodel_instance(self):
        """Test extracting text from GenericResponseOutputItem as BaseModel instance

        This test verifies that _extract_output_text_and_images correctly handles
        GenericResponseOutputItem when passed as a Pydantic BaseModel instance
        (not as a dict). This addresses the issue where isinstance(output_item, BaseModel)
        was failing because the handler was importing BaseModel from openai instead of pydantic.
        """
        handler = OpenAIResponsesHandler()

        # Create a proper GenericResponseOutputItem instance (Pydantic BaseModel)
        output_item = GenericResponseOutputItem(
            type="message",
            id="msg_123",
            status="completed",
            role="assistant",
            content=[
                OutputText(
                    type="output_text",
                    text="Hi! My name is Ishaan.",
                    annotations=[],
                )
            ],
        )

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[Any] = []
        task_mappings: List[Tuple[int, int]] = []

        # Extract text from the BaseModel instance
        handler._extract_output_text_and_images(
            output_item=output_item,
            output_idx=0,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            task_mappings=task_mappings,
            tool_calls_to_check=tool_calls_to_check,
        )

        # Verify text was extracted correctly
        assert len(texts_to_check) == 1
        assert texts_to_check[0] == "Hi! My name is Ishaan."
        assert len(task_mappings) == 1
        assert task_mappings[0] == (0, 0)  # (output_idx, content_idx)
        assert len(tool_calls_to_check) == 0  # No tool calls in this output

    def test_extract_text_from_basemodel_with_multiple_content_items(self):
        """Test extracting multiple text items from GenericResponseOutputItem BaseModel

        This test verifies that the handler correctly processes a BaseModel instance
        with multiple content items in the content array.
        """
        handler = OpenAIResponsesHandler()

        # Create GenericResponseOutputItem with multiple content items
        output_item = GenericResponseOutputItem(
            type="message",
            id="msg_456",
            status="completed",
            role="assistant",
            content=[
                OutputText(
                    type="output_text",
                    text="First paragraph.",
                    annotations=[],
                ),
                OutputText(
                    type="output_text",
                    text="Second paragraph.",
                    annotations=[],
                ),
                OutputText(
                    type="output_text",
                    text="Third paragraph.",
                    annotations=[],
                ),
            ],
        )

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[Any] = []
        task_mappings: List[Tuple[int, int]] = []

        # Extract all text items
        handler._extract_output_text_and_images(
            output_item=output_item,
            output_idx=0,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            task_mappings=task_mappings,
            tool_calls_to_check=tool_calls_to_check,
        )

        # Verify all text items were extracted
        assert len(texts_to_check) == 3
        assert texts_to_check[0] == "First paragraph."
        assert texts_to_check[1] == "Second paragraph."
        assert texts_to_check[2] == "Third paragraph."
        assert len(task_mappings) == 3
        assert task_mappings[0] == (0, 0)
        assert task_mappings[1] == (0, 1)
        assert task_mappings[2] == (0, 2)


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


class TestOpenAIResponsesHandlerStreamingOutputProcessing:
    """Test streaming output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_output(self):
        """Test that streaming response with empty output doesn't raise IndexError

        This test verifies the fix for the bug where accessing model_response_choices[0]
        would raise IndexError when the response.completed event has an empty output array.
        """
        handler = OpenAIResponsesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Simulate a response.completed streaming event with empty output
        responses_so_far = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "output": [],  # Empty output - this was causing the IndexError
                    "status": "completed",
                },
            }
        ]

        # This should not raise IndexError
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses unchanged
        assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_missing_output_key(self):
        """Test that streaming response with missing output key doesn't raise IndexError

        This test verifies the handler gracefully handles when the response dict
        doesn't contain an 'output' key at all.
        """
        handler = OpenAIResponsesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Simulate a response.completed streaming event with missing output key
        responses_so_far = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "status": "completed",
                    # No 'output' key - get() will return []
                },
            }
        ]

        # This should not raise IndexError
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses unchanged
        assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_unrecognized_output_type(self):
        """Test that streaming response with unrecognized output types doesn't raise IndexError

        This test verifies the handler gracefully handles when output items are of
        unrecognized types that _convert_response_output_to_choices skips over.
        """
        handler = OpenAIResponsesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Simulate a response.completed streaming event with unrecognized output type
        responses_so_far = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "output": [
                        {
                            "type": "unknown_type",  # Unrecognized type
                            "id": "item_123",
                            "data": "some data",
                        }
                    ],
                    "status": "completed",
                },
            }
        ]

        # This should not raise IndexError
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses unchanged
        assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_with_valid_output(self):
        """Test that streaming response with valid output still works correctly"""
        handler = OpenAIResponsesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Simulate a response.completed streaming event with valid message output
        responses_so_far = [
            {
                "type": "response.created",
                "response": {"id": "resp_123"},
            },
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "id": "msg_123"},
            },
            {
                "type": "response.content_part.added",
                "part": {"type": "output_text", "text": ""},
            },
            {
                "type": "response.output_text.delta",
                "delta": "Hello",
            },
            {
                "type": "response.output_text.delta",
                "delta": " world",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "Hello world"},
                            ],
                        }
                    ],
                    "status": "completed",
                },
            },
        ]

        # This should process successfully
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses
        assert result == responses_so_far

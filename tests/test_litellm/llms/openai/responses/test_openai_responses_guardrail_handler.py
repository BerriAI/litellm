"""
Unit tests for OpenAI Responses API Guardrail Translation Handler

Tests the handler's ability to process input/output for the Responses API
with guardrail transformations.
"""

import copy
from collections.abc import Callable
from typing import Any, List, Literal, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest


from fastapi import HTTPException
from openai.types.responses import ResponseFunctionToolCall

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
)
from litellm.llms.openai.responses.guardrail_translation.tool_merge import merge_guardrailed_tools
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
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


class MockRecordingGuardrail(MockPassThroughGuardrail):
    """Pass-through guardrail that records every apply_guardrail inputs payload"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_inputs: List[GenericGuardrailAPIInputs] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.seen_inputs.append(inputs)
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
    async def test_process_output_streaming_response_null_response(self):
        handler = OpenAIResponsesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")
        responses_so_far = [{"type": "response.completed", "response": None}]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

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

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_writes_back_guardrailed_text(self):
        """Guardrailed text must be written back into the response.completed chunk in-place."""

        class RewriteGuardrail(CustomGuardrail):
            """Replaces '<TOKEN_1>' with 'john@example.com' to simulate PII unmasking."""

            async def apply_guardrail(
                self,
                inputs: GenericGuardrailAPIInputs,
                request_data: dict,
                input_type: Literal["request", "response"],
                logging_obj: Optional[Any] = None,
            ) -> GenericGuardrailAPIInputs:
                texts = inputs.get("texts", [])
                inputs["texts"] = [
                    t.replace("<TOKEN_1>", "john@example.com") for t in texts
                ]
                return inputs

        handler = OpenAIResponsesHandler()
        guardrail = RewriteGuardrail(guardrail_name="test-rewrite")

        responses_so_far = [
            {"type": "response.output_text.delta", "delta": "send to "},
            {"type": "response.output_text.delta", "delta": "<TOKEN_1>"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_123",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "send to <TOKEN_1>"},
                            ],
                        }
                    ],
                    "status": "completed",
                },
            },
        ]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        completed_chunk = next(
            c
            for c in result
            if isinstance(c, dict) and c.get("type") == "response.completed"
        )
        output_text = completed_chunk["response"]["output"][0]["content"][0]["text"]
        assert (
            output_text == "send to john@example.com"
        ), f"Expected PII token to be unmasked in response.completed output, got: {output_text!r}"

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_pass_through_unchanged(self):
        """A pass-through guardrail must not modify the output text."""
        handler = OpenAIResponsesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="pass-through")

        original_text = "No PII here, just normal text."
        responses_so_far = [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_456",
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_456",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": original_text}],
                        }
                    ],
                    "status": "completed",
                },
            }
        ]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        output_text = result[-1]["response"]["output"][0]["content"][0]["text"]
        assert output_text == original_text

    @pytest.mark.asyncio
    async def test_failed_stream_scans_delta_text(self):
        """A stream ending in response.failed has text only in delta events; the
        fallback scan must assemble and scan it instead of skipping on an empty string."""
        handler = OpenAIResponsesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="test")

        responses_so_far = [
            {"type": "response.created", "response": {"id": "resp_123"}},
            {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_123"}},
            {
                "type": "response.output_text.delta",
                "item_id": "msg_123",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_123",
                "output_index": 0,
                "content_index": 0,
                "delta": " world",
            },
            {"type": "response.failed", "response": {"id": "resp_123", "status": "failed"}},
        ]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        assert result == responses_so_far
        assert [inputs.get("texts") for inputs in guardrail.seen_inputs] == [["Hello world"]]

    def test_get_streaming_string_so_far_prefers_done_text_over_deltas(self):
        """The done event repeats the whole part, so deltas must not be double counted;
        a part with no done event yet still contributes its joined deltas."""
        handler = OpenAIResponsesHandler()

        events = [
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": " world",
            },
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Hello world",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_2",
                "output_index": 1,
                "content_index": 0,
                "delta": "; unfinished",
            },
        ]

        assert handler.get_streaming_string_so_far(events) == "Hello world; unfinished"


class TestGetStructuredMessages:
    """Test the get_structured_messages method for Responses API handler."""

    def test_should_convert_string_input_to_messages(self):
        """Test that a simple string input is converted to OpenAI messages."""
        handler = OpenAIResponsesHandler()
        data = {"input": "What is the capital of France?"}
        result = handler.get_structured_messages(data)
        assert result is not None
        assert len(result) >= 1
        found_user = False
        for msg in result:
            if isinstance(msg, dict) and msg.get("role") == "user":
                found_user = True
                break
        assert found_user, f"Expected a user message, got: {result}"

    def test_should_convert_list_input_to_messages(self):
        """Test that list input (ResponseInputParam) is converted to OpenAI messages."""
        handler = OpenAIResponsesHandler()
        data = {
            "input": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        result = handler.get_structured_messages(data)
        assert result is not None
        assert len(result) >= 3

    def test_should_include_instructions_as_system_message(self):
        """Test that instructions are included as a system message."""
        handler = OpenAIResponsesHandler()
        data = {
            "input": "Roll a d20",
            "instructions": "You are a helpful dungeon master.",
        }
        result = handler.get_structured_messages(data)
        assert result is not None
        has_system = any(
            isinstance(msg, dict) and msg.get("role") == "system" for msg in result
        )
        assert has_system, f"Expected system message from instructions, got: {result}"

    def test_should_return_none_when_no_input(self):
        """Test that None is returned when input key is missing."""
        handler = OpenAIResponsesHandler()
        data = {"model": "gpt-4o"}
        result = handler.get_structured_messages(data)
        assert result is None

    def test_should_return_none_for_none_input(self):
        """Test that None is returned when input is explicitly None."""
        handler = OpenAIResponsesHandler()
        data = {"input": None}
        result = handler.get_structured_messages(data)
        assert result is None


class ToolAppendingGuardrail(CustomGuardrail):
    """Guardrail that appends a new function tool, mimicking a guardrail that
    injects a retrieval/recovery tool the model can later call."""

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


class TestOpenAIResponsesHandlerToolInjection:
    """A tool a guardrail injects must survive the write-back to Responses format."""

    def test_merge_keeps_guardrail_appended_tool(self):
        """merge_guardrailed_tools must not drop the extra appended tool."""
        original = [{"type": "function", "name": "a"}]
        groups = [form.chat_tools for form in LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(original)]
        guardrailed = [
            *groups[0],
            {"type": "function", "function": {"name": "b", "description": "", "parameters": {"type": "object"}}},
        ]
        merged = merge_guardrailed_tools(original, groups, guardrailed)
        assert [t["name"] for t in merged] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_injected_tool_survives_when_request_already_has_tools(self):
        """Regression: the merge dropped the injected tool whenever the request
        already carried tools, so the model never saw it."""
        handler = OpenAIResponsesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="test")

        data = {
            "input": [{"role": "user", "content": "hi", "type": "message"}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "model": "gpt-4",
        }

        result = await handler.process_input_messages(data, guardrail)

        names = [t.get("name") for t in result["tools"]]
        assert "get_weather" in names
        assert "injected_tool" in names


COMPRESSED_MARKER = "[compressed document; retrieve the full text with hash=b573993006976af767214fac]"


class StructuredRewriteGuardrail(CustomGuardrail):
    """Guardrail that rewrites whole messages via structured_messages and leaves
    texts untouched, the way message-compressing guardrails do."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        messages = list(inputs.get("structured_messages") or [])
        first_user = next(i for i, m in enumerate(messages) if m.get("role") == "user")
        rewritten = [
            {**m, "content": COMPRESSED_MARKER} if i == first_user else m for i, m in enumerate(messages)
        ]
        return {**inputs, "structured_messages": rewritten}


class ToolOutputRewriteGuardrail(CustomGuardrail):
    """Guardrail that compresses the first tool-result row, the way Headroom does."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        messages = list(inputs.get("structured_messages") or [])
        first_tool = next(i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "tool")
        rewritten = [
            {**m, "content": COMPRESSED_MARKER} if i == first_tool else m for i, m in enumerate(messages)
        ]
        return {**inputs, "structured_messages": rewritten}


class DroppingRewriteGuardrail(CustomGuardrail):
    """Guardrail that rewrites the first user row and drops the last row, so the
    rewrite can only land through the full-conversion fallback."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        messages = list(inputs.get("structured_messages") or [])
        first_user = next(i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user")
        rewritten = [
            {**m, "content": COMPRESSED_MARKER} if i == first_user else m for i, m in enumerate(messages)
        ]
        return {**inputs, "structured_messages": rewritten[:-1]}


def _texts(item: dict) -> list[str]:
    content = item.get("content")
    if isinstance(content, str):
        return [content]
    return [part["text"] for part in content]


class TestStructuredMessagesWriteBack:
    """A guardrail's structured_messages rewrite must land in the Responses request,
    not only the per-text mapping the chat handler shares with it."""

    @pytest.mark.asyncio
    async def test_list_input_gets_rewritten_messages_and_keeps_instructions(self):
        handler = OpenAIResponsesHandler()
        data = {
            "model": "gpt-5.6",
            "instructions": "Answer from the memo only.",
            "input": [
                {"role": "user", "content": "memo " * 400},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, StructuredRewriteGuardrail())

        assert result["instructions"] == "Answer from the memo only."
        user_items = [item for item in result["input"] if item.get("role") == "user"]
        assert [_texts(item) for item in user_items] == [[COMPRESSED_MARKER], ["What is the codename?"]]
        assert not any(item.get("role") == "system" for item in result["input"])
        assert _texts(next(item for item in result["input"] if item.get("role") == "assistant")) == ["Understood."]

    @pytest.mark.asyncio
    async def test_string_input_becomes_rewritten_message_list(self):
        handler = OpenAIResponsesHandler()
        data = {"model": "gpt-5.6", "input": "memo " * 400}

        result = await handler.process_input_messages(data, StructuredRewriteGuardrail())

        assert [_texts(item) for item in result["input"]] == [[COMPRESSED_MARKER]]
        assert "instructions" not in result

    @pytest.mark.asyncio
    async def test_developer_item_preserved_verbatim_by_row_patch(self):
        handler = OpenAIResponsesHandler()
        developer_item = {"role": "developer", "content": "Always answer in French."}
        data = {
            "model": "gpt-5.6",
            "input": [
                developer_item,
                {"role": "user", "content": "memo " * 400},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, StructuredRewriteGuardrail())

        assert result["input"][0] is developer_item
        assert developer_item["content"] == "Always answer in French."
        assert _texts(result["input"][1]) == [COMPRESSED_MARKER]
        assert _texts(result["input"][2]) == ["What is the codename?"]

    @pytest.mark.asyncio
    async def test_reasoning_and_function_call_items_survive_tool_output_compression(self):
        handler = OpenAIResponsesHandler()
        reasoning_item = {
            "id": "rs_123",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "gAAAAA-signed-reasoning",
        }
        function_call_item = {
            "id": "fc_123",
            "type": "function_call",
            "call_id": "call_abc",
            "name": "read_document",
            "arguments": '{"path": "memo.txt"}',
            "status": "completed",
        }
        data = {
            "model": "gpt-5.6",
            "instructions": "Answer from the memo only.",
            "input": [
                reasoning_item,
                function_call_item,
                {"type": "function_call_output", "call_id": "call_abc", "output": "memo " * 400},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, ToolOutputRewriteGuardrail())

        assert result["instructions"] == "Answer from the memo only."
        assert result["input"][0] is reasoning_item
        assert reasoning_item["encrypted_content"] == "gAAAAA-signed-reasoning"
        assert result["input"][1] is function_call_item
        assert function_call_item["id"] == "fc_123"
        assert result["input"][2] == {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": COMPRESSED_MARKER,
        }
        assert result["input"][3] == {"role": "user", "content": "What is the codename?"}

    @pytest.mark.asyncio
    async def test_web_search_call_item_preserved_verbatim(self):
        handler = OpenAIResponsesHandler()
        web_search_item = {
            "id": "ws_123",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "codename memo"},
        }
        data = {
            "model": "gpt-5.6",
            "input": [
                web_search_item,
                {"role": "user", "content": "memo " * 400},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, StructuredRewriteGuardrail())

        assert result["input"][0] is web_search_item
        assert _texts(result["input"][1]) == [COMPRESSED_MARKER]
        assert _texts(result["input"][2]) == ["What is the codename?"]

    @pytest.mark.asyncio
    async def test_row_count_change_falls_back_to_full_conversion(self):
        handler = OpenAIResponsesHandler()
        data = {
            "model": "gpt-5.6",
            "input": [
                {"role": "developer", "content": "Always answer in French."},
                {"role": "user", "content": "memo " * 400},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, DroppingRewriteGuardrail())

        assert len(result["input"]) == 2
        developer = next(item for item in result["input"] if item.get("role") == "developer")
        assert developer["content"] == [{"type": "input_text", "text": "Always answer in French."}]
        assert _texts(next(item for item in result["input"] if item.get("role") == "user")) == [COMPRESSED_MARKER]

    @pytest.mark.asyncio
    async def test_same_inputs_object_back_keeps_the_text_mapping(self):
        handler = OpenAIResponsesHandler()
        original_input = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": [{"type": "input_text", "text": "Again"}]},
        ]
        data = {"model": "gpt-5.6", "input": original_input}

        result = await handler.process_input_messages(data, MockGuardrail())

        assert result["input"] is original_input
        assert [_texts(item) for item in result["input"]] == [["Hello [GUARDRAILED]"], ["Again [GUARDRAILED]"]]


class AllToolOutputsRewriteGuardrail(CustomGuardrail):
    """Guardrail that compresses every tool-result row."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        messages = list(inputs.get("structured_messages") or [])
        rewritten = [
            {**m, "content": COMPRESSED_MARKER} if isinstance(m, dict) and m.get("role") == "tool" else m
            for m in messages
        ]
        return {**inputs, "structured_messages": rewritten}


class AssistantRewriteGuardrail(CustomGuardrail):
    """Guardrail that rewrites the first assistant row's content."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        messages = list(inputs.get("structured_messages") or [])
        first = next(i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant")
        rewritten = [{**m, "content": COMPRESSED_MARKER} if i == first else m for i, m in enumerate(messages)]
        return {**inputs, "structured_messages": rewritten}


class DictStructuredMessagesGuardrail(CustomGuardrail):
    """Guardrail that hands back a raw evaluation dict instead of a message list,
    the way HiddenLayer v2 does."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        return {**inputs, "structured_messages": {"evaluation": "allowed", "messages": []}}


def _parallel_tool_call_input() -> list:
    return [
        {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read_a", "arguments": "{}"},
        {"id": "fc_2", "type": "function_call", "call_id": "call_2", "name": "read_b", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "memo " * 400},
        {"type": "function_call_output", "call_id": "call_2", "output": "note " * 400},
        {"role": "user", "content": "What is the codename?"},
    ]


class TestProvenancePatching:
    """The O(n) provenance pass must keep patching rewritten rows in place for the
    shapes real agent loops produce, and fall back safely everywhere else."""

    @pytest.mark.asyncio
    async def test_parallel_tool_call_outputs_both_patched(self):
        handler = OpenAIResponsesHandler()
        raw_input = _parallel_tool_call_input()
        fc_1, fc_2 = raw_input[0], raw_input[1]
        data = {"model": "gpt-5.6", "input": raw_input}

        result = await handler.process_input_messages(data, AllToolOutputsRewriteGuardrail())

        assert result["input"][0] is fc_1
        assert result["input"][1] is fc_2
        assert result["input"][2] == {"type": "function_call_output", "call_id": "call_1", "output": COMPRESSED_MARKER}
        assert result["input"][3] == {"type": "function_call_output", "call_id": "call_2", "output": COMPRESSED_MARKER}
        assert result["input"][4] == {"role": "user", "content": "What is the codename?"}

    @pytest.mark.asyncio
    async def test_assistant_turn_with_tool_call_keeps_items_verbatim(self):
        handler = OpenAIResponsesHandler()
        assistant_item = {"role": "assistant", "content": "Let me read the memo."}
        function_call_item = {
            "id": "fc_9",
            "type": "function_call",
            "call_id": "call_9",
            "name": "read_document",
            "arguments": '{"path": "memo.txt"}',
        }
        data = {
            "model": "gpt-5.6",
            "input": [
                assistant_item,
                function_call_item,
                {"type": "function_call_output", "call_id": "call_9", "output": "memo " * 400},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, ToolOutputRewriteGuardrail())

        assert result["input"][0] is assistant_item
        assert result["input"][1] is function_call_item
        assert result["input"][2] == {"type": "function_call_output", "call_id": "call_9", "output": COMPRESSED_MARKER}

    @pytest.mark.asyncio
    async def test_rewrite_of_merged_tool_call_message_falls_back(self):
        handler = OpenAIResponsesHandler()
        raw_input = _parallel_tool_call_input()
        data = {"model": "gpt-5.6", "input": raw_input}

        result = await handler.process_input_messages(data, AssistantRewriteGuardrail())

        assert not any(item is original for item in result["input"] for original in raw_input)
        assistant_items = [item for item in result["input"] if item.get("role") == "assistant"]
        assert [_texts(item) for item in assistant_items] == [[COMPRESSED_MARKER]]

    @pytest.mark.asyncio
    async def test_rewrite_of_lone_function_call_message_falls_back(self):
        handler = OpenAIResponsesHandler()
        data = {
            "model": "gpt-5.6",
            "input": [
                {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read_a", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "memo memo"},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        raw_input = data["input"]
        result = await handler.process_input_messages(data, AssistantRewriteGuardrail())

        assert not any(item is original for item in result["input"] for original in raw_input)
        assistant_items = [item for item in result["input"] if item.get("role") == "assistant"]
        assert [_texts(item) for item in assistant_items] == [[COMPRESSED_MARKER]]

    def test_provenance_bails_on_non_mapping_item(self):
        from litellm.llms.openai.responses.guardrail_translation.handler import _input_item_provenance

        assert _input_item_provenance(["not a mapping"], []) is None

    def test_provenance_bails_when_expected_messages_disagree(self):
        from litellm.llms.openai.responses.guardrail_translation.handler import _input_item_provenance

        assert _input_item_provenance([{"role": "user", "content": "hi"}], [{"role": "user", "content": "bye"}]) is None

    def test_provenance_bails_on_unpredicted_merge(self):
        from litellm.llms.openai.responses.guardrail_translation.handler import _input_item_provenance
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        raw_input = [
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read_a", "arguments": "{}"},
            {"role": "assistant", "content": "Reading the memo now."},
        ]
        expected = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=raw_input, responses_api_request={}
        )
        assert len(expected) == 1
        assert _input_item_provenance(raw_input, expected) is None

    def test_provenance_maps_and_taints_parallel_tool_calls(self):
        from litellm.llms.openai.responses.guardrail_translation.handler import _input_item_provenance
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        raw_input = _parallel_tool_call_input()
        expected = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=raw_input, responses_api_request={}
        )
        provenance = _input_item_provenance(raw_input, expected)
        assert provenance is not None
        item_for_message, tainted = provenance
        assert tainted == {0}
        assert dict(item_for_message) == {1: 2, 2: 3, 3: 4}


class TestDictStructuredMessagesGuard:
    """A guardrail handing back a non-list structured_messages payload must not
    blow up the request; the write-back is skipped instead."""

    @pytest.mark.asyncio
    async def test_list_input_survives_dict_structured_messages(self):
        handler = OpenAIResponsesHandler()
        original_input = [{"role": "user", "content": "Hello"}]
        data = {"model": "gpt-5.6", "input": original_input}

        result = await handler.process_input_messages(data, DictStructuredMessagesGuardrail())

        assert result["input"] is original_input
        assert result["input"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_string_input_survives_dict_structured_messages(self):
        handler = OpenAIResponsesHandler()
        data = {"model": "gpt-5.6", "input": "Hello there"}

        result = await handler.process_input_messages(data, DictStructuredMessagesGuardrail())

        assert result["input"] == "Hello there"


class SystemRewriteGuardrail(CustomGuardrail):
    """Guardrail that rewrites the system row, the way prompt-hardening guardrails do."""

    def __init__(self, rewritten_content: Any = COMPRESSED_MARKER):
        super().__init__()
        self.rewritten_content = rewritten_content

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        messages = list(inputs.get("structured_messages") or [])
        first = next(i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system")
        rewritten = [
            {**m, "content": self.rewritten_content} if i == first else m for i, m in enumerate(messages)
        ]
        return {**inputs, "structured_messages": rewritten}


class TestPatchEdgeBranches:
    @pytest.mark.asyncio
    async def test_multimodal_user_item_rewritten_through_conversion(self):
        handler = OpenAIResponsesHandler()
        data = {
            "model": "gpt-5.6",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "memo " * 400}]},
                {"role": "user", "content": "What is the codename?"},
            ],
        }

        result = await handler.process_input_messages(data, StructuredRewriteGuardrail())

        assert _texts(result["input"][0]) == [COMPRESSED_MARKER]
        assert result["input"][1] == {"role": "user", "content": "What is the codename?"}

    @pytest.mark.asyncio
    async def test_instructions_rewrite_lands_in_instructions_field(self):
        handler = OpenAIResponsesHandler()
        user_item = {"role": "user", "content": "What is the codename?"}
        data = {
            "model": "gpt-5.6",
            "instructions": "Answer from the memo only.",
            "input": [user_item],
        }

        result = await handler.process_input_messages(data, SystemRewriteGuardrail())

        assert result["instructions"] == COMPRESSED_MARKER
        assert result["input"][0] is user_item

    @pytest.mark.asyncio
    async def test_non_string_instructions_rewrite_falls_back(self):
        handler = OpenAIResponsesHandler()
        user_item = {"role": "user", "content": "What is the codename?"}
        data = {
            "model": "gpt-5.6",
            "instructions": "Answer from the memo only.",
            "input": [user_item],
        }

        result = await handler.process_input_messages(
            data, SystemRewriteGuardrail(rewritten_content=[{"type": "text", "text": COMPRESSED_MARKER}])
        )

        assert result["input"][0] is not user_item

    @pytest.mark.asyncio
    async def test_unpredicted_merge_falls_back_through_patch(self):
        handler = OpenAIResponsesHandler()
        raw_input = [
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read_a", "arguments": "{}"},
            {"role": "assistant", "content": "Reading the memo now."},
            {"type": "function_call_output", "call_id": "call_1", "output": "memo memo"},
            {"role": "user", "content": "memo " * 400},
        ]
        data = {"model": "gpt-5.6", "input": raw_input}

        result = await handler.process_input_messages(data, StructuredRewriteGuardrail())

        assert not any(item is original for item in result["input"] for original in raw_input)
        user_items = [item for item in result["input"] if item.get("role") == "user"]
        assert _texts(user_items[0]) == [COMPRESSED_MARKER]

    def test_item_rewrite_field_ignores_non_string_type(self):
        from litellm.llms.openai.responses.guardrail_translation.handler import _item_rewrite_field

        assert _item_rewrite_field({"type": 123, "content": "hello"}) is None


class ToolEditingGuardrail(CustomGuardrail):
    """Guardrail that rewrites the flattened chat tools it was handed through ``edit``"""

    def __init__(self, edit: Callable[[list[dict]], list[dict]], **kwargs):
        super().__init__(**kwargs)
        self.edit = edit

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        inputs["tools"] = self.edit(list(inputs.get("tools") or []))
        return inputs


def _codex_request(input_value):
    """A Responses API request shaped like what the Codex CLI sends when an MCP server is configured"""
    return {
        "model": "gpt-5.3-codex",
        "input": input_value,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "strict": False,
            },
            {
                "type": "namespace",
                "name": "mcp__confluence",
                "description": "Confluence tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "confluence_get_page",
                        "description": "Get a page",
                        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                        "strict": False,
                    },
                    {
                        "type": "function",
                        "name": "confluence_search",
                        "description": "Search pages",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                        "strict": False,
                    },
                ],
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
                "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "x"'},
            },
            {"type": "web_search"},
        ],
    }


def _tool_named(tools, name):
    return next(tool for tool in tools if tool.get("name") == name)


class TestOpenAIResponsesHandlerNamespaceTools:
    """Codex sends MCP tools as ``namespace`` tools; a guardrail must never flatten them (GH #39183)"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_value",
        ["hi", [{"role": "user", "content": "hi", "type": "message"}]],
        ids=["string_input", "list_input"],
    )
    async def test_pass_through_guardrail_leaves_tools_untouched(self, input_value):
        data = _codex_request(input_value)
        expected_tools = copy.deepcopy(data["tools"])

        result = await OpenAIResponsesHandler().process_input_messages(
            data, MockPassThroughGuardrail(guardrail_name="test")
        )

        assert result["tools"] == expected_tools

    @pytest.mark.asyncio
    async def test_appending_guardrail_keeps_namespace_and_adds_tool(self):
        data = _codex_request("hi")
        expected_tools = copy.deepcopy(data["tools"])

        result = await OpenAIResponsesHandler().process_input_messages(
            data, ToolAppendingGuardrail(guardrail_name="test")
        )

        assert result["tools"][:-1] == expected_tools
        assert result["tools"][-1]["type"] == "function"
        assert result["tools"][-1]["name"] == "injected_tool"

    @pytest.mark.asyncio
    async def test_dropping_one_member_prunes_only_that_member(self):
        data = _codex_request("hi")
        expected_tools = copy.deepcopy(data["tools"])
        guardrail = ToolEditingGuardrail(
            edit=lambda tools: [t for t in tools if t["function"]["name"] != "mcp__confluence__confluence_search"],
            guardrail_name="test",
        )

        result = await OpenAIResponsesHandler().process_input_messages(data, guardrail)

        namespace = _tool_named(result["tools"], "mcp__confluence")
        assert [member["name"] for member in namespace["tools"]] == ["confluence_get_page"]
        assert namespace["tools"][0] == expected_tools[1]["tools"][0]
        assert [t for t in result["tools"] if t is not namespace] == [expected_tools[0], *expected_tools[2:]]

    @pytest.mark.asyncio
    async def test_editing_a_member_lands_on_that_member_without_the_namespace_prefix(self):
        data = _codex_request("hi")
        expected_tools = copy.deepcopy(data["tools"])

        def redact_search(tools):
            for tool in tools:
                if tool["function"]["name"] == "mcp__confluence__confluence_search":
                    tool["function"]["description"] = "Confluence tools\n\nREDACTED"
            return tools

        result = await OpenAIResponsesHandler().process_input_messages(
            data, ToolEditingGuardrail(edit=redact_search, guardrail_name="test")
        )

        namespace = _tool_named(result["tools"], "mcp__confluence")
        assert namespace["tools"][0] == expected_tools[1]["tools"][0]
        assert namespace["tools"][1] == {**expected_tools[1]["tools"][1], "description": "REDACTED"}
        assert {k: v for k, v in namespace.items() if k != "tools"} == {
            k: v for k, v in expected_tools[1].items() if k != "tools"
        }

    @pytest.mark.asyncio
    async def test_dropping_every_member_drops_the_namespace(self):
        data = _codex_request("hi")
        expected_tools = copy.deepcopy(data["tools"])
        guardrail = ToolEditingGuardrail(
            edit=lambda tools: [t for t in tools if not t["function"]["name"].startswith("mcp__confluence__")],
            guardrail_name="test",
        )

        result = await OpenAIResponsesHandler().process_input_messages(data, guardrail)

        assert result["tools"] == [expected_tools[0], *expected_tools[2:]]

    @pytest.mark.asyncio
    async def test_edited_top_level_function_is_rewritten_in_place(self):
        data = _codex_request("hi")
        expected_tools = copy.deepcopy(data["tools"])

        def rename_weather(tools):
            for tool in tools:
                if tool["function"]["name"] == "get_weather":
                    tool["function"]["description"] = "Weather lookup (guarded)"
            return tools

        result = await OpenAIResponsesHandler().process_input_messages(
            data, ToolEditingGuardrail(edit=rename_weather, guardrail_name="test")
        )

        assert result["tools"][0] == {**expected_tools[0], "description": "Weather lookup (guarded)"}
        assert result["tools"][1:] == expected_tools[1:]


class TestOpenAIResponsesHandlerMalformedTools:
    @pytest.mark.asyncio
    async def test_request_tools_that_are_not_a_list_never_reach_the_guardrail(self):
        handler = OpenAIResponsesHandler()
        seen: list[list[dict]] = []

        def record(tools):
            seen.append(tools)
            return tools

        guardrail = ToolEditingGuardrail(edit=record, guardrail_name="test")
        data = {"input": "hi", "tools": {"type": "function", "name": "get_weather"}}

        result = await handler.process_input_messages(data, guardrail)

        assert seen == [[]]
        assert result["input"] == "hi"


class TestBuildBlockSseChunks:
    """build_block_sse_chunks turns a streaming ModifyResponseException into 200 SSE events"""

    def _exc(self, original_response=None):
        from litellm.exceptions import ModifyResponseException

        return ModifyResponseException(
            message="Blocked by policy.",
            model="gpt-5.4-mini",
            request_data={},
            guardrail_name="test",
            original_response=original_response,
        )

    def _payloads(self, chunks):
        import json

        return [json.loads(chunk.decode().removeprefix("data: ").strip()) for chunk in chunks]

    def test_standalone_block_emits_complete_synthetic_stream(self):
        handler = OpenAIResponsesHandler()
        payloads = self._payloads(handler.build_block_sse_chunks(self._exc(), stream_started=False))
        types = [payload["type"] for payload in payloads]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        completed = payloads[-1]["response"]
        assert completed["id"].startswith("resp_")
        assert completed["model"] == "gpt-5.4-mini"
        assert completed["output"][0]["content"][0]["text"] == "Blocked by policy."

    def test_continuation_appends_item_at_next_output_index_with_real_usage(self):
        handler = OpenAIResponsesHandler()
        yielded = [
            {"type": "response.created", "response": {"id": "resp_live", "model": "gpt-5.4-mini-2026-01-01"}},
            {"type": "response.output_item.added", "output_index": 2, "item": {"id": "msg_orig"}},
        ]
        original = yielded + [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_live",
                    "model": "gpt-5.4-mini-2026-01-01",
                    "output": [],
                    "usage": {"input_tokens": 7, "output_tokens": 21, "total_tokens": 28},
                },
            }
        ]
        payloads = self._payloads(
            handler.build_block_sse_chunks(
                self._exc(original_response=original), stream_started=True, responses_so_far=yielded
            )
        )
        types = [payload["type"] for payload in payloads]
        assert "response.created" not in types
        assert types[0] == "response.output_item.done"
        assert payloads[0]["output_index"] == 2
        assert payloads[0]["item"]["id"] == "msg_orig"
        assert payloads[0]["item"]["status"] == "completed"
        assert types[1] == "response.output_item.added"
        assert payloads[1]["output_index"] == 3
        completed = payloads[-1]["response"]
        assert completed["id"] == "resp_live"
        assert completed["model"] == "gpt-5.4-mini-2026-01-01"
        assert completed["output"][0]["content"][0]["text"] == "Blocked by policy."
        assert completed["usage"] == {"input_tokens": 7, "output_tokens": 21, "total_tokens": 28}

    def test_continuation_reads_usage_from_typed_completed_event(self):
        from litellm.types.llms.openai import (
            ResponseCompletedEvent,
            ResponsesAPIResponse,
            ResponsesAPIStreamEvents,
        )

        handler = OpenAIResponsesHandler()
        original = [
            ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=ResponsesAPIResponse.model_validate(
                    {
                        "id": "resp_live",
                        "created_at": 1,
                        "model": "gpt-5.4-mini",
                        "output": [],
                        "usage": {"input_tokens": 7, "output_tokens": 21, "total_tokens": 28},
                    }
                ),
            )
        ]
        payloads = self._payloads(
            handler.build_block_sse_chunks(
                self._exc(original_response=original), stream_started=True, responses_so_far=[]
            )
        )
        completed = payloads[-1]["response"]
        assert completed["usage"]["input_tokens"] == 7
        assert completed["usage"]["output_tokens"] == 21
        assert completed["usage"]["total_tokens"] == 28

    def test_continuation_closes_open_item_given_pydantic_events_with_enum_types(self):
        from litellm.types.llms.openai import (
            BaseLiteLLMOpenAIResponseObject,
            ContentPartAddedEvent,
            OutputItemAddedEvent,
            OutputTextDeltaEvent,
            ResponsesAPIStreamEvents,
        )

        handler = OpenAIResponsesHandler()
        open_item = GenericResponseOutputItem.model_validate(
            {"type": "message", "id": "msg_live", "status": "in_progress", "role": "assistant", "content": []}
        )
        yielded = [
            OutputItemAddedEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, output_index=0, item=open_item
            ),
            ContentPartAddedEvent(
                type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
                item_id="msg_live",
                output_index=0,
                content_index=0,
                part=BaseLiteLLMOpenAIResponseObject.model_validate(
                    {"type": "output_text", "text": "", "annotations": []}
                ),
            ),
            OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                item_id="msg_live",
                output_index=0,
                content_index=0,
                delta="partial ",
            ),
            OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                item_id="msg_live",
                output_index=0,
                content_index=0,
                delta="text",
            ),
        ]
        payloads = self._payloads(
            handler.build_block_sse_chunks(
                self._exc(original_response=yielded), stream_started=True, responses_so_far=yielded
            )
        )
        types = [payload["type"] for payload in payloads]
        assert types[:3] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert payloads[0]["text"] == "partial text"
        assert payloads[2]["item"]["id"] == "msg_live"
        assert payloads[2]["item"]["status"] == "completed"
        assert payloads[2]["item"]["content"][0]["text"] == "partial text"
        assert types[3] == "response.output_item.added"
        assert payloads[3]["output_index"] == 1

    def test_continuation_closes_open_function_call_as_incomplete(self):
        handler = OpenAIResponsesHandler()
        yielded = [
            {"type": "response.created", "response": {"id": "resp_live", "model": "gpt-5.4-mini"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_live",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_1",
                    "name": "run_payment",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_live",
                "output_index": 0,
                "delta": '{"amount": 100}',
            },
        ]
        payloads = self._payloads(
            handler.build_block_sse_chunks(
                self._exc(original_response=yielded), stream_started=True, responses_so_far=yielded
            )
        )
        types = [payload["type"] for payload in payloads]
        assert types[0] == "response.output_item.done"
        closed = payloads[0]["item"]
        assert closed["id"] == "fc_live"
        assert closed["type"] == "function_call"
        assert closed["status"] == "incomplete"
        assert closed["name"] == "run_payment"
        assert "content" not in closed
        assert types[1] == "response.output_item.added"
        assert payloads[1]["output_index"] == 1
        assert types[-1] == "response.completed"

    def test_continuation_without_open_item_emits_no_closing_events(self):
        handler = OpenAIResponsesHandler()
        yielded = [
            {"type": "response.created", "response": {"id": "resp_live", "model": "gpt-5.4-mini"}},
            {"type": "response.in_progress", "response": {"id": "resp_live"}},
        ]
        payloads = self._payloads(
            handler.build_block_sse_chunks(
                self._exc(original_response=yielded), stream_started=True, responses_so_far=yielded
            )
        )
        types = [payload["type"] for payload in payloads]
        assert types[0] == "response.output_item.added"
        assert types[-1] == "response.completed"
        dones = [payload for payload in payloads if payload["type"] == "response.output_item.done"]
        assert len(dones) == 1
        assert dones[0]["item"]["content"][0]["text"] == "Blocked by policy."

"""
Unit tests for OpenAI Chat Completions Guardrail Translation Handler

Tests the handler's ability to process input/output for Chat Completions API
with guardrail transformations, including tool calls.
"""

import json
import os
import sys
from typing import Any, List, Literal, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    GenericGuardrailAPIInputs,
    Message,
    ModelResponse,
)


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing that transforms text and tool calls"""

    def __init__(self, guardrail_name: str = "test"):
        super().__init__(guardrail_name=guardrail_name)
        self.last_inputs = None
        self.last_request_data = None
        self.tool_calls_modified = False

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Mock apply_guardrail that uppercases text and modifies tool calls"""
        self.last_inputs = inputs
        self.last_request_data = request_data

        # Return modified inputs (uppercase texts for testing)
        texts = inputs.get("texts", [])
        modified_texts = [text.upper() for text in texts]

        # Modify tool calls in place if present
        tool_calls = inputs.get("tool_calls", [])
        if tool_calls:
            self.tool_calls_modified = True
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "function" in tool_call:
                    function = tool_call["function"]
                    if "arguments" in function:
                        # Modify arguments to uppercase JSON string
                        try:
                            args_dict = json.loads(function["arguments"])
                            # Uppercase all string values
                            for key, value in args_dict.items():
                                if isinstance(value, str):
                                    args_dict[key] = value.upper()
                            function["arguments"] = json.dumps(args_dict)
                        except json.JSONDecodeError:
                            # If not JSON, just uppercase the string
                            function["arguments"] = function["arguments"].upper()

        # Return modified inputs as GenericGuardrailAPIInputs
        result: GenericGuardrailAPIInputs = {"texts": modified_texts}
        if tool_calls:
            result["tool_calls"] = tool_calls  # type: ignore
        if "images" in inputs:
            result["images"] = inputs["images"]  # type: ignore
        return result


class TestOpenAIChatCompletionsHandlerToolsInput:
    """Test input processing with tools (function definitions)"""

    @pytest.mark.asyncio
    async def test_tools_passed_to_guardrail(self):
        """Test that tools (function definitions) are passed to the guardrail"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create input data with tools (function definitions)
        data = {
            "messages": [
                {"role": "user", "content": "What's the weather in Boston?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify tools were passed to guardrail
        assert guardrail.last_inputs is not None
        assert "tools" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tools"]) == 1

        tool = guardrail.last_inputs["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get the current weather in a location"
        assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_multiple_tools_passed_to_guardrail(self):
        """Test that multiple tools are passed to the guardrail"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        data = {
            "messages": [
                {"role": "user", "content": "What's the weather and time?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await handler.process_input_messages(data, guardrail)

        assert guardrail.last_inputs is not None
        assert "tools" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tools"]) == 2
        assert guardrail.last_inputs["tools"][0]["function"]["name"] == "get_weather"
        assert guardrail.last_inputs["tools"][1]["function"]["name"] == "get_time"

    @pytest.mark.asyncio
    async def test_no_tools_in_request(self):
        """Test that requests without tools work correctly"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }

        await handler.process_input_messages(data, guardrail)

        assert guardrail.last_inputs is not None
        # tools should not be in inputs if not provided
        assert "tools" not in guardrail.last_inputs or guardrail.last_inputs.get("tools") is None

    @pytest.mark.asyncio
    async def test_tools_and_tool_calls_both_passed(self):
        """Test that both tools (definitions) and tool_calls (invocations) are passed"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        data = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Boston"}',
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
        }

        await handler.process_input_messages(data, guardrail)

        assert guardrail.last_inputs is not None
        # Both should be present
        assert "tools" in guardrail.last_inputs
        assert "tool_calls" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tools"]) == 1
        assert len(guardrail.last_inputs["tool_calls"]) == 1


class TestOpenAIChatCompletionsHandlerToolCallsInput:
    """Test input processing with tool calls"""

    @pytest.mark.asyncio
    async def test_extract_tool_calls_from_input_messages(self):
        """Test that tool calls are extracted from input messages"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create input data with tool calls (assistant message)
        data = {
            "messages": [
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps(
                                    {"location": "San Francisco", "unit": "celsius"}
                                ),
                            },
                        }
                    ],
                },
            ]
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify tool calls were extracted and passed to guardrail
        assert guardrail.last_inputs is not None
        assert "tool_calls" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tool_calls"]) == 1

        tool_call = guardrail.last_inputs["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["function"]["name"] == "get_weather"
        # Note: tool call arguments may already be modified by guardrail
        # Check that it contains location parameter
        assert "location" in tool_call["function"]["arguments"]

        # Verify tool call was modified by guardrail
        assert guardrail.tool_calls_modified is True

        # Verify the message was updated with modified tool call
        modified_tool_call = data["messages"][1]["tool_calls"][0]
        args = json.loads(modified_tool_call["function"]["arguments"])
        assert args["location"] == "SAN FRANCISCO"  # Should be uppercased
        assert args["unit"] == "CELSIUS"  # Should be uppercased

    @pytest.mark.asyncio
    async def test_extract_tool_calls_and_text_from_input_messages(self):
        """Test that both tool calls and text content are extracted"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create input data with both text and tool calls
        data = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": json.dumps({"location": "Boston"}),
                            },
                        }
                    ],
                },
            ]
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify both texts and tool calls were extracted
        assert guardrail.last_inputs is not None
        assert "texts" in guardrail.last_inputs
        assert "tool_calls" in guardrail.last_inputs

        # Should have 2 texts (user message + assistant message)
        assert len(guardrail.last_inputs["texts"]) == 2
        assert "What's the weather?" in guardrail.last_inputs["texts"]
        assert "Let me check that for you." in guardrail.last_inputs["texts"]

        # Should have 1 tool call
        assert len(guardrail.last_inputs["tool_calls"]) == 1
        assert (
            guardrail.last_inputs["tool_calls"][0]["function"]["name"]
            == "get_current_weather"
        )

        # Verify text content was modified
        assert data["messages"][0]["content"] == "WHAT'S THE WEATHER?"
        assert data["messages"][1]["content"] == "LET ME CHECK THAT FOR YOU."

        # Verify tool call was modified
        modified_tool_call = data["messages"][1]["tool_calls"][0]
        args = json.loads(modified_tool_call["function"]["arguments"])
        assert args["location"] == "BOSTON"

    @pytest.mark.asyncio
    async def test_extract_multiple_tool_calls_from_input(self):
        """Test extraction of multiple tool calls"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create input data with multiple tool calls
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "NYC"}),
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": json.dumps({"timezone": "EST"}),
                            },
                        },
                    ],
                }
            ]
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify multiple tool calls were extracted
        assert guardrail.last_inputs is not None
        assert "tool_calls" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tool_calls"]) == 2

        # Verify both tool calls
        tool_calls = guardrail.last_inputs["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"

        # Verify both were modified
        modified_tool_calls = data["messages"][0]["tool_calls"]
        args1 = json.loads(modified_tool_calls[0]["function"]["arguments"])
        args2 = json.loads(modified_tool_calls[1]["function"]["arguments"])
        assert args1["location"] == "NYC"
        assert args2["timezone"] == "EST"

    @pytest.mark.asyncio
    async def test_tool_calls_separate_from_texts(self):
        """Test that tool calls are passed as a separate parameter, not mixed with texts"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        data = {
            "messages": [
                {"role": "user", "content": "Get weather for LA"},
                {
                    "role": "assistant",
                    "content": "Sure!",
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Los Angeles"}),
                            },
                        }
                    ],
                },
            ]
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify tool calls and texts are separate
        assert guardrail.last_inputs is not None
        texts = guardrail.last_inputs.get("texts", [])
        tool_calls = guardrail.last_inputs.get("tool_calls", [])

        # Texts should only contain the content strings
        assert len(texts) == 2
        assert "Get weather for LA" in texts
        assert "Sure!" in texts

        # Tool call arguments should NOT be in texts
        assert not any("Los Angeles" in text for text in texts)

        # Tool calls should be separate
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        # Check that it contains city parameter (may be modified by guardrail)
        assert "city" in tool_calls[0]["function"]["arguments"]

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_input(self):
        """Test that messages without tool calls work correctly"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create input data without tool calls
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify no tool calls were passed to guardrail
        assert guardrail.last_inputs is not None
        tool_calls = guardrail.last_inputs.get("tool_calls", [])
        assert len(tool_calls) == 0

        # Verify text was still processed
        assert len(guardrail.last_inputs["texts"]) == 2
        assert data["messages"][0]["content"] == "HELLO"
        assert data["messages"][1]["content"] == "HI THERE!"

    @pytest.mark.asyncio
    async def test_empty_tool_calls_list(self):
        """Test that empty tool_calls list is handled correctly"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        data = {
            "messages": [
                {"role": "assistant", "content": "Hello", "tool_calls": []},
            ]
        }

        # Process the input
        await handler.process_input_messages(data, guardrail)

        # Verify empty tool_calls doesn't cause issues
        assert guardrail.last_inputs is not None
        tool_calls = guardrail.last_inputs.get("tool_calls", [])
        assert len(tool_calls) == 0


class TestOpenAIChatCompletionsHandlerToolCallsOutput:
    """Test output processing with tool calls"""

    @pytest.mark.asyncio
    async def test_extract_tool_calls_from_output_response(self):
        """Test that tool calls are extracted from output responses"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create a mock response with tool calls
        response = ModelResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_789",
                                type="function",
                                function=Function(
                                    name="search_database",
                                    arguments=json.dumps({"query": "python tutorials"}),
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        # Process the output
        await handler.process_output_response(response, guardrail)

        # Verify tool calls were extracted and passed to guardrail
        assert guardrail.last_inputs is not None
        assert "tool_calls" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tool_calls"]) == 1

        tool_call = guardrail.last_inputs["tool_calls"][0]
        assert tool_call["function"]["name"] == "search_database"
        # Check that it contains query parameter (may be modified by guardrail)
        assert "query" in tool_call["function"]["arguments"]

        # Verify tool call was modified in response
        response_tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(response_tool_call.function.arguments)
        assert args["query"] == "PYTHON TUTORIALS"  # Should be uppercased

    @pytest.mark.asyncio
    async def test_extract_tool_calls_and_content_from_output(self):
        """Test extraction of both content and tool calls from output"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        response = ModelResponse(
            id="chatcmpl-456",
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content="I'll search for that information.",
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_999",
                                type="function",
                                function=Function(
                                    name="web_search",
                                    arguments=json.dumps(
                                        {"keywords": "litellm documentation"}
                                    ),
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        # Process the output
        await handler.process_output_response(response, guardrail)

        # Verify both texts and tool calls were extracted
        assert guardrail.last_inputs is not None
        assert "texts" in guardrail.last_inputs
        assert "tool_calls" in guardrail.last_inputs

        assert len(guardrail.last_inputs["texts"]) == 1
        assert "I'll search for that information." in guardrail.last_inputs["texts"]

        assert len(guardrail.last_inputs["tool_calls"]) == 1

        # Verify both were modified
        assert (
            response.choices[0].message.content == "I'LL SEARCH FOR THAT INFORMATION."
        )
        response_tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(response_tool_call.function.arguments)
        assert args["keywords"] == "LITELLM DOCUMENTATION"

    @pytest.mark.asyncio
    async def test_extract_multiple_tool_calls_from_output(self):
        """Test extraction of multiple tool calls from output"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        response = ModelResponse(
            id="chatcmpl-789",
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_1",
                                type="function",
                                function=Function(
                                    name="get_weather",
                                    arguments=json.dumps({"location": "Tokyo"}),
                                ),
                            ),
                            ChatCompletionMessageToolCall(
                                id="call_2",
                                type="function",
                                function=Function(
                                    name="get_news",
                                    arguments=json.dumps({"topic": "technology"}),
                                ),
                            ),
                        ],
                    ),
                )
            ],
        )

        # Process the output
        await handler.process_output_response(response, guardrail)

        # Verify multiple tool calls were extracted
        assert guardrail.last_inputs is not None
        assert "tool_calls" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tool_calls"]) == 2

        # Verify both tool calls
        tool_calls = guardrail.last_inputs["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_news"

        # Verify both were modified
        response_tool_calls = response.choices[0].message.tool_calls
        args1 = json.loads(response_tool_calls[0].function.arguments)
        args2 = json.loads(response_tool_calls[1].function.arguments)
        assert args1["location"] == "TOKYO"
        assert args2["topic"] == "TECHNOLOGY"

    @pytest.mark.asyncio
    async def test_extract_tool_calls_from_real_openai_response(self):
        """Test extraction of tool calls from a real OpenAI API response structure"""
        handler = OpenAIChatCompletionsHandler()
        guardrail = MockGuardrail()

        # Create a response matching the exact structure from OpenAI API
        response = ModelResponse(
            id="chatcmpl-abc123",
            created=1699896916,
            model="gpt-4o-mini",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_abc123",
                                type="function",
                                function=Function(
                                    name="get_current_weather",
                                    arguments='{\n"location": "Boston, MA"\n}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        # Process the output
        await handler.process_output_response(response, guardrail)

        # Verify tool calls were extracted and passed to guardrail
        assert guardrail.last_inputs is not None
        assert "tool_calls" in guardrail.last_inputs
        assert len(guardrail.last_inputs["tool_calls"]) == 1

        # Verify the tool call details
        tool_call = guardrail.last_inputs["tool_calls"][0]
        assert tool_call["id"] == "call_abc123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_current_weather"

        # Verify arguments can be parsed
        args = json.loads(tool_call["function"]["arguments"])
        assert "location" in args

        # Verify tool call was modified by guardrail (location should be uppercased)
        response_tool_call = response.choices[0].message.tool_calls[0]
        modified_args = json.loads(response_tool_call.function.arguments)
        assert modified_args["location"] == "BOSTON, MA"  # Should be uppercased

        # Verify response metadata
        assert response.id == "chatcmpl-abc123"
        assert response.model == "gpt-4o-mini"
        assert response.choices[0].finish_reason == "tool_calls"


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


class TestOpenAIChatCompletionsHandlerStreamingOutput:
    """Test streaming output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_choices(self):
        """Test that streaming response with empty choices doesn't raise IndexError

        This test verifies the fix for the bug where accessing chunk.choices[0]
        would raise IndexError when a streaming chunk has an empty choices list.
        """
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        handler = OpenAIChatCompletionsHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a streaming chunk with empty choices
        chunk_with_empty_choices = ModelResponseStream(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[],  # Empty choices - this was causing the IndexError
        )

        responses_so_far = [chunk_with_empty_choices]

        # This should not raise IndexError
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses unchanged
        assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_with_valid_choices(self):
        """Test that streaming response with valid choices still works correctly"""
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        handler = OpenAIChatCompletionsHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create streaming chunks with valid choices
        chunk1 = ModelResponseStream(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Hello"),
                    finish_reason=None,
                )
            ],
        )

        chunk2 = ModelResponseStream(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=" world"),
                    finish_reason="stop",
                )
            ],
        )

        responses_so_far = [chunk1, chunk2]

        # This should process successfully
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses
        assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_mixed_empty_and_valid_choices_no_finish(self):
        """Test streaming response with mix of empty and valid choices chunks (stream not finished)

        This tests the has_stream_ended check when iterating through chunks with mixed choices.
        The stream hasn't finished yet (no finish_reason), so it won't trigger stream_chunk_builder.
        """
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        handler = OpenAIChatCompletionsHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mix of chunks - some with empty choices, some with valid choices
        # Stream hasn't finished (no finish_reason)
        chunk_empty = ModelResponseStream(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[],
        )

        chunk_valid = ModelResponseStream(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Hello"),
                    finish_reason=None,  # Stream not finished
                )
            ],
        )

        responses_so_far = [chunk_empty, chunk_valid]

        # This should not raise IndexError when checking has_stream_ended
        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=guardrail,
            litellm_logging_obj=None,
        )

        # Should return the responses
        assert result == responses_so_far


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

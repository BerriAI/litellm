#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""
import json
import os
import sys
import unittest

import pytest

from litellm.google_genai.main import agenerate_content

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import json
import os
import sys

import pytest

import litellm


def test_adapter_import():
    """Test that the adapter can be imported successfully"""
    from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

    # Should not raise any exceptions
    assert GoogleGenAIAdapter is not None
    assert GenerateContentToCompletionHandler is not None

def test_single_content_transformation():
    """Test the transformation from generate_content to completion format with single content"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    # Test input
    model = "gpt-3.5-turbo"
    contents = {
        "role": "user",
        "parts": [{"text": "Hello, how are you?"}]
    }
    config = {
        "temperature": 0.7,
        "maxOutputTokens": 100
    }
    
    # Transform to completion format
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents,
        config=config
    )
    
    # Verify the transformation
    assert completion_request["model"] == "gpt-3.5-turbo"
    assert len(completion_request["messages"]) == 1
    assert completion_request["messages"][0]["role"] == "user"
    assert completion_request["messages"][0]["content"] == "Hello, how are you?"
    assert completion_request["temperature"] == 0.7
    assert completion_request["max_tokens"] == 100

def test_list_contents_transformation():
    """Test transformation with list of contents (conversation history)"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    # Test input with conversation history
    model = "gpt-3.5-turbo"
    contents = [
        {
            "role": "user",
            "parts": [{"text": "Hello, how are you?"}]
        },
        {
            "role": "model", 
            "parts": [{"text": "I'm doing well, thank you!"}]
        },
        {
            "role": "user",
            "parts": [{"text": "What's the weather like?"}]
        }
    ]
    
    # Transform to completion format
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents
    )
    
    # Verify the transformation
    assert completion_request["model"] == "gpt-3.5-turbo"
    assert len(completion_request["messages"]) == 3
    
    # Check first message
    assert completion_request["messages"][0]["role"] == "user"
    assert completion_request["messages"][0]["content"] == "Hello, how are you?"
    
    # Check second message
    assert completion_request["messages"][1]["role"] == "assistant"
    assert completion_request["messages"][1]["content"] == "I'm doing well, thank you!"
    
    # Check third message
    assert completion_request["messages"][2]["role"] == "user"
    assert completion_request["messages"][2]["content"] == "What's the weather like?"

def test_config_parameter_mapping():
    """Test that config parameters are correctly mapped"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = {"role": "user", "parts": [{"text": "Test"}]}
    config = {
        "temperature": 0.8,
        "maxOutputTokens": 150,
        "topP": 0.9,
        "stopSequences": ["END", "STOP"]
    }
    
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents,
        config=config
    )
    
    # Verify parameter mapping
    assert completion_request["temperature"] == 0.8
    assert completion_request["max_tokens"] == 150
    assert completion_request["top_p"] == 0.9
    assert completion_request["stop"] == ["END", "STOP"]

def test_tools_transformation():
    """Test transformation of Google GenAI tools to OpenAI tools format"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = {"role": "user", "parts": [{"text": "What's the weather?"}]}
    
    # Google GenAI tools format
    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get current weather information",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["location"]
                    }
                },
                {
                    "name": "get_forecast",
                    "description": "Get weather forecast",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "days": {"type": "integer"}
                        }
                    }
                }
            ]
        }
    ]
    
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents,
        tools=tools
    )
    
    # Verify tools transformation
    assert "tools" in completion_request
    openai_tools = completion_request["tools"]
    assert len(openai_tools) == 2
    
    # Check first tool
    tool1 = openai_tools[0]
    assert tool1["type"] == "function"
    assert tool1["function"]["name"] == "get_weather"
    assert tool1["function"]["description"] == "Get current weather information"
    assert "parameters" in tool1["function"]
    assert tool1["function"]["parameters"]["properties"]["location"]["type"] == "string"
    
    # Check second tool
    tool2 = openai_tools[1]
    assert tool2["type"] == "function"
    assert tool2["function"]["name"] == "get_forecast"
    assert tool2["function"]["description"] == "Get weather forecast"

def test_tool_config_transformation():
    """Test transformation of Google GenAI tool_config to OpenAI tool_choice"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = {"role": "user", "parts": [{"text": "Test"}]}
    
    # Test different tool config modes
    test_cases = [
        ({"functionCallingConfig": {"mode": "AUTO"}}, "auto"),
        ({"functionCallingConfig": {"mode": "ANY"}}, "required"),
        ({"functionCallingConfig": {"mode": "NONE"}}, "none"),
        ({"functionCallingConfig": {"mode": "UNKNOWN"}}, "auto"),  # Default case
    ]
    
    for tool_config, expected_choice in test_cases:
        completion_request = adapter.translate_generate_content_to_completion(
            model=model,
            contents=contents,
            tool_config=tool_config
        )
        
        assert "tool_choice" in completion_request
        assert completion_request["tool_choice"] == expected_choice

def test_function_call_message_transformation():
    """Test transformation of messages with function calls"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = [
        {
            "role": "user",
            "parts": [{"text": "What's the weather in San Francisco?"}]
        },
        {
            "role": "model",
            "parts": [
                {"text": "I'll check the weather for you."},
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "San Francisco"}
                    }
                }
            ]
        }
    ]
    
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents
    )
    
    # Verify the transformation
    messages = completion_request["messages"]
    assert len(messages) == 2
    
    # Check user message
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What's the weather in San Francisco?"
    
    # Check assistant message with tool call
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "I'll check the weather for you."
    assert "tool_calls" in assistant_msg
    assert len(assistant_msg["tool_calls"]) == 1
    
    tool_call = assistant_msg["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    
    # Verify arguments are properly JSON encoded
    args = json.loads(tool_call["function"]["arguments"])
    assert args["location"] == "San Francisco"

def test_function_response_message_transformation():
    """Test transformation of messages with function responses"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Here's the weather data:"},
                {
                    "functionResponse": {
                        "name": "get_weather",
                        "response": {
                            "temperature": "72F",
                            "condition": "sunny",
                            "humidity": "45%"
                        }
                    }
                }
            ]
        }
    ]
    
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents
    )
    
    # Verify the transformation
    messages = completion_request["messages"]
    assert len(messages) == 2  # User message + tool message
    
    # Check user message
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "Here's the weather data:"
    
    # Check tool message
    tool_msg = messages[1]
    assert tool_msg["role"] == "tool"
    assert "call_get_weather" in tool_msg["tool_call_id"]
    
    # Verify function response content
    response_content = json.loads(tool_msg["content"])
    assert response_content["temperature"] == "72F"
    assert response_content["condition"] == "sunny"
    assert response_content["humidity"] == "45%"

def test_completion_to_generate_content_with_tool_calls():
    """Test transforming completion response with tool calls back to generate_content format"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    from litellm.types.llms.openai import (
        ChatCompletionAssistantMessage,
        ChatCompletionAssistantToolCall,
        ChatCompletionToolCallFunctionChunk,
    )
    from litellm.types.utils import Choices, ModelResponse, Usage
    
    adapter = GoogleGenAIAdapter()
    
    # Create mock tool call
    mock_tool_call = ChatCompletionAssistantToolCall(
        id="call_123",
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name="get_weather",
            arguments='{"location": "San Francisco"}'
        )
    )
    
    # Create mock assistant message with tool call
    mock_message = ChatCompletionAssistantMessage(
        role="assistant",
        content="I'll check the weather for you.",
        tool_calls=[mock_tool_call]
    )
    
    mock_choice = Choices(
        finish_reason="tool_calls",
        index=0,
        message=mock_message
    )
    
    mock_usage = Usage(
        prompt_tokens=15,
        completion_tokens=25,
        total_tokens=40
    )
    
    mock_response = ModelResponse(
        id="test-123",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion",
        usage=mock_usage
    )
    
    # Transform back to generate_content format
    generate_content_response = adapter.translate_completion_to_generate_content(mock_response)
    
    # Verify the transformation
    assert "candidates" in generate_content_response
    candidate = generate_content_response["candidates"][0]
    assert candidate["finishReason"] == "STOP"  # tool_calls maps to STOP
    
    # Check content parts
    parts = candidate["content"]["parts"]
    assert len(parts) == 2
    
    # Check text part
    text_part = parts[0]
    assert text_part["text"] == "I'll check the weather for you."
    
    # Check function call part
    function_part = parts[1]
    assert "functionCall" in function_part
    function_call = function_part["functionCall"]
    assert function_call["name"] == "get_weather"
    assert function_call["args"]["location"] == "San Francisco"
    
    # Check text field
    assert generate_content_response["text"] == "I'll check the weather for you."

def test_streaming_tool_calls_transformation():
    """Test streaming transformation with tool calls"""
    from litellm.google_genai.adapters.transformation import (
        GoogleGenAIAdapter,
        GoogleGenAIStreamWrapper,
    )
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponse,
        StreamingChoices,
    )
    
    adapter = GoogleGenAIAdapter()
    
    # Create mock function for tool call
    mock_function = Function(
        name="get_weather",
        arguments='{"location": "SF"}'
    )
    
    # Create mock streaming tool call delta
    mock_tool_call_delta = ChatCompletionDeltaToolCall(
        id="call_123",
        type="function",
        function=mock_function,
        index=0
    )
    
    # Create mock delta with tool call
    mock_delta = Delta(
        content=None,
        tool_calls=[mock_tool_call_delta]
    )
    
    mock_choice = StreamingChoices(
        finish_reason=None,
        index=0,
        delta=mock_delta
    )
    
    mock_response = ModelResponse(
        id="test-streaming",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion.chunk"
    )
    
    # Create mock wrapper for accumulation state
    mock_wrapper = GoogleGenAIStreamWrapper(completion_stream=None)
    
    # Transform streaming chunk
    streaming_chunk = adapter.translate_streaming_completion_to_generate_content(mock_response, mock_wrapper)
    
    # Verify the transformation
    assert "candidates" in streaming_chunk
    candidate = streaming_chunk["candidates"][0]
    
    # Check parts
    parts = candidate["content"]["parts"]
    assert len(parts) == 1
    
    # Check function call part
    function_part = parts[0]
    assert "functionCall" in function_part
    function_call = function_part["functionCall"]
    assert function_call["name"] == "get_weather"
    assert function_call["args"]["location"] == "SF"

def test_streaming_partial_tool_calls_accumulation():
    """Test accumulation of partial tool call arguments across streaming chunks"""
    from litellm.google_genai.adapters.transformation import (
        GoogleGenAIAdapter,
        GoogleGenAIStreamWrapper,
    )
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponse,
        StreamingChoices,
    )
    
    adapter = GoogleGenAIAdapter()
    mock_wrapper = GoogleGenAIStreamWrapper(completion_stream=None)
    
    # Simulate partial chunks that create valid JSON when accumulated
    partial_chunks = [
        ('read_file', '{"path"'),     # First chunk: {"path"
        (None, ': "/Users'),          # Second chunk: : "/Users  
        (None, '/is'),                # Third chunk: /is
        (None, 'haanjaffe'),          # Fourth chunk: haanjaffe
        (None, 'r/Github/li'),        # Fifth chunk: r/Github/li
        (None, 'tellm'),              # Sixth chunk: tellm
        (None, '/README.md'),         # Seventh chunk: /README.md
        (None, '"}')                  # Final chunk: "}
    ]
    
    # Process each partial chunk
    accumulated_results = []
    tool_call_id = "call_read_file_123"  # Same ID for all chunks
    
    for function_name, chunk_args in partial_chunks:
        # Create mock function for tool call with partial arguments
        mock_function = Function(
            name=function_name,  # Only set in first chunk
            arguments=chunk_args
        )
        
        # Create mock streaming tool call delta
        mock_tool_call_delta = ChatCompletionDeltaToolCall(
            id=tool_call_id,  # Same ID across all chunks
            type="function",
            function=mock_function,
            index=0
        )
        
        # Create mock delta with tool call
        mock_delta = Delta(
            content=None,
            tool_calls=[mock_tool_call_delta]
        )
        
        mock_choice = StreamingChoices(
            finish_reason=None,
            index=0,
            delta=mock_delta
        )
        
        mock_response = ModelResponse(
            id="test-streaming",
            choices=[mock_choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion.chunk"
        )
        
        # Transform streaming chunk with accumulation
        streaming_chunk = adapter.translate_streaming_completion_to_generate_content(mock_response, mock_wrapper)
        accumulated_results.append(streaming_chunk)
    
    # Verify accumulation behavior
    # Most chunks should be empty (because JSON is incomplete)
    empty_chunks = [chunk for chunk in accumulated_results if not chunk]
    non_empty_chunks = [chunk for chunk in accumulated_results if chunk]
    
    # Should have several empty chunks while accumulating
    assert len(empty_chunks) > 0, "Should have empty chunks while accumulating partial JSON"
    
    # Should have exactly one non-empty chunk when JSON becomes complete
    assert len(non_empty_chunks) == 1, f"Should have exactly one complete chunk, got {len(non_empty_chunks)}"
    
    # Verify the final complete chunk
    final_chunk = non_empty_chunks[0]
    assert "candidates" in final_chunk
    candidate = final_chunk["candidates"][0]
    
    # Check parts
    parts = candidate["content"]["parts"]
    assert len(parts) == 1, f"Expected 1 part, got {len(parts)}"
    
    # Check function call part
    function_part = parts[0]
    assert "functionCall" in function_part, "Should have functionCall in the final chunk"
    function_call = function_part["functionCall"]
    assert function_call["name"] == "read_file", f"Expected function name 'read_file', got {function_call['name']}"
    assert function_call["args"]["path"] == "/Users/ishaanjaffer/Github/litellm/README.md", f"Expected complete path, got {function_call['args']}"
    
    # Verify that accumulated_tool_calls is cleaned up after completion
    assert len(mock_wrapper.accumulated_tool_calls) == 0, "Should clean up completed tool calls from accumulator"

def test_streaming_multiple_partial_tool_calls():
    """Test accumulation of multiple partial tool calls simultaneously"""
    from litellm.google_genai.adapters.transformation import (
        GoogleGenAIAdapter,
        GoogleGenAIStreamWrapper,
    )
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponse,
        StreamingChoices,
    )
    
    adapter = GoogleGenAIAdapter()
    mock_wrapper = GoogleGenAIStreamWrapper(completion_stream=None)
    
    # Test data for two tool calls being accumulated simultaneously
    # Format: (tool_call_id, function_name, args_chunk, index)
    test_chunks = [
        ("call_1", "read_file", '{"file1"', 0),    # {"file1"
        ("call_2", "write_file", '{"file2"', 1),   # {"file2"
        ("call_1", None, ': "test1.txt"', 0),      # : "test1.txt"
        ("call_2", None, ': "test2.txt"', 1),      # : "test2.txt"
        ("call_1", None, '}', 0),                  # }
        ("call_2", None, '}', 1),                  # }
    ]
    
    completed_chunks = []
    
    for call_id, function_name, args_chunk, index in test_chunks:
        # Create mock function for tool call
        mock_function = Function(
            name=function_name,
            arguments=args_chunk
        )
        
        # Create mock streaming tool call delta
        mock_tool_call_delta = ChatCompletionDeltaToolCall(
            id=call_id,
            type="function",
            function=mock_function,
            index=index
        )
        
        # Create mock delta with tool call
        mock_delta = Delta(
            content=None,
            tool_calls=[mock_tool_call_delta]
        )
        
        mock_choice = StreamingChoices(
            finish_reason=None,
            index=0,
            delta=mock_delta
        )
        
        mock_response = ModelResponse(
            id="test-streaming",
            choices=[mock_choice],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion.chunk"
        )
        
        # Transform streaming chunk with accumulation
        streaming_chunk = adapter.translate_streaming_completion_to_generate_content(mock_response, mock_wrapper)
        if streaming_chunk:  # Only collect non-empty chunks
            completed_chunks.append(streaming_chunk)
    
    # Should have exactly 2 completed chunks (one for each tool call)
    assert len(completed_chunks) == 2, f"Expected 2 completed chunks, got {len(completed_chunks)}"
    
    # Extract function calls from completed chunks
    function_calls = []
    for chunk in completed_chunks:
        parts = chunk["candidates"][0]["content"]["parts"]
        for part in parts:
            if "functionCall" in part:
                function_calls.append(part["functionCall"])
    
    # Should have 2 function calls
    assert len(function_calls) == 2, f"Expected 2 function calls, got {len(function_calls)}"
    
    # Verify both function calls are complete and correct
    function_names = [fc["name"] for fc in function_calls]
    assert "read_file" in function_names, "Should have read_file function call"
    assert "write_file" in function_names, "Should have write_file function call"
    
    # Verify arguments are correctly assembled
    for fc in function_calls:
        if fc["name"] == "read_file":
            assert fc["args"]["file1"] == "test1.txt", f"Expected file1: test1.txt, got {fc['args']}"
        elif fc["name"] == "write_file":
            assert fc["args"]["file2"] == "test2.txt", f"Expected file2: test2.txt, got {fc['args']}"
    
    # Verify cleanup
    assert len(mock_wrapper.accumulated_tool_calls) == 0, "Should clean up all completed tool calls"

def test_mixed_content_transformation():
    """Test transformation of mixed content (text + function calls)"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = [
        {
            "role": "model",
            "parts": [
                {"text": "I'll help you with that. Let me check the weather and also get the forecast."},
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "San Francisco"}
                    }
                },
                {
                    "functionCall": {
                        "name": "get_forecast", 
                        "args": {"location": "San Francisco", "days": 3}
                    }
                }
            ]
        }
    ]
    
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents
    )
    
    # Verify the transformation
    messages = completion_request["messages"]
    assert len(messages) == 1
    
    assistant_msg = messages[0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "I'll help you with that. Let me check the weather and also get the forecast."
    assert "tool_calls" in assistant_msg
    assert len(assistant_msg["tool_calls"]) == 2
    
    # Check first tool call
    tool_call1 = assistant_msg["tool_calls"][0]
    assert tool_call1["function"]["name"] == "get_weather"
    args1 = json.loads(tool_call1["function"]["arguments"])
    assert args1["location"] == "San Francisco"
    
    # Check second tool call
    tool_call2 = assistant_msg["tool_calls"][1]
    assert tool_call2["function"]["name"] == "get_forecast"
    args2 = json.loads(tool_call2["function"]["arguments"])
    assert args2["location"] == "San Francisco"
    assert args2["days"] == 3

def test_completion_to_generate_content_transformation():
    """Test transforming a completion response back to generate_content format"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    from litellm.types.llms.openai import ChatCompletionAssistantMessage
    from litellm.types.utils import Choices, ModelResponse, Usage
    
    adapter = GoogleGenAIAdapter()
    
    # Create proper mock response using actual types
    mock_message = ChatCompletionAssistantMessage(
        role="assistant",
        content="Hello! I'm doing well, thank you for asking."
    )
    
    mock_choice = Choices(
        finish_reason="stop",
        index=0,
        message=mock_message
    )
    
    mock_usage = Usage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    
    # Create mock completion response
    mock_response = ModelResponse(
        id="test-123",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion",
        usage=mock_usage
    )
    
    # Transform back to generate_content format
    generate_content_response = adapter.translate_completion_to_generate_content(mock_response)
    
    # Verify the transformation
    assert "text" in generate_content_response
    assert generate_content_response["text"] == "Hello! I'm doing well, thank you for asking."
    
    assert "candidates" in generate_content_response
    assert len(generate_content_response["candidates"]) == 1
    
    candidate = generate_content_response["candidates"][0]
    assert candidate["finishReason"] == "STOP"
    assert candidate["index"] == 0
    assert candidate["content"]["role"] == "model"
    assert len(candidate["content"]["parts"]) == 1
    assert candidate["content"]["parts"][0]["text"] == "Hello! I'm doing well, thank you for asking."
    
    assert "usageMetadata" in generate_content_response
    usage = generate_content_response["usageMetadata"]
    assert usage["promptTokenCount"] == 10
    assert usage["candidatesTokenCount"] == 20
    assert usage["totalTokenCount"] == 30

def test_finish_reason_mapping():
    """Test that finish reasons are correctly mapped"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    # Test different finish reason mappings
    test_cases = [
        ("stop", "STOP"),
        ("length", "MAX_TOKENS"),
        ("content_filter", "SAFETY"),
        ("tool_calls", "STOP"),
        ("unknown_reason", "STOP"),  # Default case
        (None, "STOP")  # None case
    ]
    
    for openai_reason, expected_google_reason in test_cases:
        result = adapter._map_finish_reason(openai_reason)
        assert result == expected_google_reason

def test_empty_content_handling():
    """Test handling of empty or missing content"""
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    
    adapter = GoogleGenAIAdapter()
    
    # Test with empty parts
    model = "gpt-3.5-turbo"
    contents = {
        "role": "user",
        "parts": []
    }
    
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents
    )
    
    # Should still create a valid request but with empty messages
    assert completion_request["model"] == "gpt-3.5-turbo"
    assert "messages" in completion_request
    assert len(completion_request["messages"]) == 0

def test_handler_parameter_exclusion():
    """Test that the handler properly excludes Google GenAI-specific parameters"""
    from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler

    # Test parameters that should be excluded
    model = "gpt-3.5-turbo"
    contents = {"role": "user", "parts": [{"text": "Test"}]}
    config = {"temperature": 0.7}
    
    extra_kwargs = {
        "agenerate_content_stream": True,  # Should be excluded
        "generate_content_stream": True,   # Should be excluded
    }
    
    completion_kwargs = GenerateContentToCompletionHandler._prepare_completion_kwargs(
        model=model,
        contents=contents,
        config=config,
        stream=False,
        extra_kwargs=extra_kwargs
    )
    
    # Verify Google GenAI-specific parameters are excluded
    assert "agenerate_content_stream" not in completion_kwargs
    assert "generate_content_stream" not in completion_kwargs
    
    # Verify valid OpenAI parameters are present
    assert "model" in completion_kwargs
    assert completion_kwargs["model"] == "gpt-3.5-turbo"
    assert "temperature" in completion_kwargs
    assert completion_kwargs["temperature"] == 0.7

@pytest.mark.parametrize("function_name,is_async,is_stream", [
    ("generate_content", False, False),
    ("agenerate_content", True, False),
    ("generate_content_stream", False, True),
    ("agenerate_content_stream", True, True),
])
def test_api_base_and_api_key_passthrough(function_name, is_async, is_stream):
    """Test that api_base and api_key parameters are passed through to litellm.completion/acompletion when using generate_content"""
    import asyncio
    import unittest.mock
    
    litellm._turn_on_debug()

    # Import the specific function being tested
    if function_name == "generate_content":
        from litellm.google_genai.main import generate_content as test_function
    elif function_name == "agenerate_content":
        from litellm.google_genai.main import agenerate_content as test_function
    elif function_name == "generate_content_stream":
        from litellm.google_genai.main import generate_content_stream as test_function
    elif function_name == "agenerate_content_stream":
        from litellm.google_genai.main import agenerate_content_stream as test_function

    # Test input parameters
    model = "gpt-3.5-turbo"
    test_api_base = "https://test-api.example.com"
    test_api_key = "test-api-key-123"
    
    # Mock the appropriate litellm function (completion vs acompletion)
    mock_target = 'litellm.acompletion' if is_async else 'litellm.completion'
    
    with unittest.mock.patch(mock_target) as mock_completion:
        # Mock return value
        mock_return = unittest.mock.MagicMock()
        if is_async:
            # For async functions, return a coroutine that resolves to the mock
            async def mock_async_return():
                return mock_return
            mock_completion.return_value = mock_async_return()
        else:
            mock_completion.return_value = mock_return
        
        # Define the test call
        def make_test_call():
            return test_function(
                model=model,
                contents={
                    "role": "user",
                    "parts": [{"text": "Hello, world!"}]
                },
                config={
                    "temperature": 0.7,
                },
                api_base=test_api_base,
                api_key=test_api_key
            )
        
        # Call the handler with api_base and api_key
        try:
            if is_async:
                # Run the async function
                async def run_async_test():
                    return await make_test_call()
                
                asyncio.run(run_async_test())
            else:
                make_test_call()
        except Exception:
            # Ignore any errors from the mock response processing
            pass
        
        # Verify that the appropriate litellm function was called
        mock_completion.assert_called_once()
        
        # Get the arguments passed to litellm.completion/acompletion
        call_args, call_kwargs = mock_completion.call_args
        
        # Verify that api_base and api_key were passed through
        assert "api_base" in call_kwargs, f"api_base not found in completion kwargs: {call_kwargs.keys()}"
        assert call_kwargs["api_base"] == test_api_base, f"Expected api_base {test_api_base}, got {call_kwargs['api_base']}"
        
        assert "api_key" in call_kwargs, f"api_key not found in completion kwargs: {call_kwargs.keys()}"
        assert call_kwargs["api_key"] == test_api_key, f"Expected api_key {test_api_key}, got {call_kwargs['api_key']}"
        
        # Verify other expected parameters
        assert call_kwargs["model"] == model
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Hello, world!"
        assert call_kwargs["temperature"] == 0.7
        
        # Verify stream parameter for streaming functions
        if is_stream:
            pass
        else:
            # For non-streaming, stream should be False or not present
            assert call_kwargs.get("stream") is not True, f"Expected stream not True for {function_name}"

def test_shared_schema_normalization_utilities():
    """Test the shared schema normalization utility functions work correctly"""
    from litellm.litellm_core_utils.json_validation_rule import (
        normalize_json_schema_types,
        normalize_tool_schema,
    )

    # Test normalize_json_schema_types with nested structures
    schema_with_uppercase_types = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "age": {"type": "INTEGER"},
            "active": {"type": "BOOLEAN"},
            "scores": {
                "type": "ARRAY",
                "items": {"type": "NUMBER"}
            },
            "metadata": {
                "type": "OBJECT",
                "properties": {
                    "nested_field": {"type": "STRING"}
                }
            }
        },
        "required": ["name", "age"]
    }
    
    normalized_schema = normalize_json_schema_types(schema_with_uppercase_types)
    
    # Check top-level type normalization
    assert normalized_schema["type"] == "object"
    
    # Check properties normalization
    props = normalized_schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["age"]["type"] == "integer"
    assert props["active"]["type"] == "boolean"
    assert props["scores"]["type"] == "array"
    assert props["scores"]["items"]["type"] == "number"
    assert props["metadata"]["type"] == "object"
    assert props["metadata"]["properties"]["nested_field"]["type"] == "string"
    
    # Check non-type fields are preserved
    assert normalized_schema["required"] == ["name", "age"]
    
    # Test normalize_tool_schema
    tool_with_uppercase_types = {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "param1": {"type": "STRING"},
                    "param2": {"type": "BOOLEAN"}
                }
            }
        }
    }
    
    normalized_tool = normalize_tool_schema(tool_with_uppercase_types)
    
    # Check that function info is preserved
    assert normalized_tool["type"] == "function"
    assert normalized_tool["function"]["name"] == "test_function"
    assert normalized_tool["function"]["description"] == "A test function"
    
    # Check that parameters are normalized
    params = normalized_tool["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["param1"]["type"] == "string"
    assert params["properties"]["param2"]["type"] == "boolean"
    
    # Test edge cases
    assert normalize_json_schema_types("not_a_dict") == "not_a_dict"
    assert normalize_json_schema_types([{"type": "STRING"}]) == [{"type": "string"}]
    assert normalize_tool_schema("not_a_dict") == "not_a_dict"

@pytest.mark.asyncio
async def test_google_generate_content_with_openai():
    """
    
    """
    import unittest.mock

    from litellm.types.llms.openai import ChatCompletionAssistantMessage
    from litellm.types.router import GenericLiteLLMParams
    from litellm.types.utils import Choices, ModelResponse, Usage

    # Create a proper mock response object with expected attributes
    mock_message = ChatCompletionAssistantMessage(
        role="assistant",
        content="Hello! How can I help you today?"
    )

    mock_choice = Choices(
        finish_reason="stop",
        index=0,
        message=mock_message
    )
    
    mock_usage = Usage(
        prompt_tokens=10,
        completion_tokens=20, 
        total_tokens=30
    )
    
    mock_response = ModelResponse(
        id="test-123",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=mock_usage
    )
    
    # Use AsyncMock for proper async function mocking
    with unittest.mock.patch("litellm.acompletion", new_callable=unittest.mock.AsyncMock) as mock_completion:
        # Set the return value directly on the MagicMock
        mock_completion.return_value = mock_response
        
        response = await agenerate_content(
            model="openai/gpt-4o-mini",
            contents=[
                {"role": "user", "parts": [{"text": "Hello, world!"}]}
            ],
                systemInstruction={"parts": [{"text": "You are a helpful assistant."}]},
            safetySettings=[
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "OFF"
                }
            ]
        )
        
        # Print the request args sent to litellm.completion
        call_args, call_kwargs = mock_completion.call_args
        print("Arguments sent to litellm.completion:")
        print(f"Args: {call_args}")
        print(f"Kwargs: {call_kwargs}")
        
        # Verify the mock was called
        mock_completion.assert_called_once()
        
        # Print the response for verification
        print(f"Response: {response}")
        ######################################################### 
        # validate only expected fields were sent to litellm.completion
        passed_fields = set(call_kwargs.keys())
        # remove any GenericLiteLLMParams fields
        passed_fields = passed_fields - set(GenericLiteLLMParams.model_fields.keys())
        assert passed_fields == set(["model", "messages"]), f"Expected only model and messages to be passed through, got {passed_fields}"
@pytest.mark.asyncio
async def test_agenerate_content_x_goog_api_key_header():
    """
    Test that agenerate_content passes x-goog-api-key header correctly.
    
    This test verifies that when calling agenerate_content with a Google GenAI model,
    the HTTP request includes the x-goog-api-key header with the correct API key value.
    """
    import os
    import unittest.mock

    import httpx
    
    test_api_key = "test-gemini-api-key-123"
    
    # Mock environment to ensure we use our test API key
    with unittest.mock.patch.dict(os.environ, {"GEMINI_API_KEY": test_api_key}, clear=False):
        # Mock the AsyncHTTPHandler's post method to capture headers
        with unittest.mock.patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=unittest.mock.AsyncMock) as mock_post:
            # Mock a successful response
            mock_response = unittest.mock.MagicMock()
            mock_response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello! How can I help you today?"}],
                            "role": "model"
                        },
                        "finishReason": "STOP",
                        "index": 0
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 15
                }
            }
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_post.return_value = mock_response
            
            # Call agenerate_content with Google AI Studio model
            try:
                response = await agenerate_content(
                    model="gemini/gemini-1.5-flash",
                    contents=[
                        {"role": "user", "parts": [{"text": "Hello, world!"}]}
                    ],
                    api_key=test_api_key
                )
            except Exception:
                # Ignore any response processing errors, we just want to check the headers
                pass
            
            # Verify that AsyncHTTPHandler.post was called
            mock_post.assert_called_once()
            
            # Get the arguments passed to the post call
            call_args, call_kwargs = mock_post.call_args
            
            # Verify that headers contain x-goog-api-key
            headers = call_kwargs.get("headers", {})
            assert "x-goog-api-key" in headers, f"x-goog-api-key header not found in headers: {list(headers.keys())}"
            
            # Verify the API key is set (could be our test key or from api_key parameter)
            api_key_value = headers["x-goog-api-key"]
            assert api_key_value == test_api_key, f"Expected x-goog-api-key to be {test_api_key}, got {api_key_value}"
            
            # Verify other expected headers
            assert headers.get("Content-Type") == "application/json", f"Expected Content-Type application/json, got {headers.get('Content-Type')}"
            
            print(f"✓ Test passed: x-goog-api-key header correctly set to {api_key_value}")
            print(f"✓ All headers: {list(headers.keys())}")

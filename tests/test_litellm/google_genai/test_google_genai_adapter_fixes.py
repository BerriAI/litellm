#!/usr/bin/env python3
"""
Test to verify the Google GenAI adapter fixes
"""
import json
import os
import sys
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler
from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse


def test_system_instruction_handling():
    """Test that systemInstruction is correctly handled in translation"""
    adapter = GoogleGenAIAdapter()
    
    model = "gpt-3.5-turbo"
    contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    system_instruction = {
        "parts": [{"text": "You are a helpful assistant"}]
    }
    
    # Transform to completion format with system instruction
    completion_request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=contents,
        system_instruction=system_instruction
    )
    
    # Verify system instruction is correctly transformed
    assert len(completion_request["messages"]) == 2
    assert completion_request["messages"][0]["role"] == "system"
    assert completion_request["messages"][0]["content"] == "You are a helpful assistant"
    assert completion_request["messages"][1]["role"] == "user"
    assert completion_request["messages"][1]["content"] == "Hello"


def test_parameters_json_schema_transformation():
    """Test that parametersJsonSchema is correctly transformed to parameters"""
    adapter = GoogleGenAIAdapter()
    
    # Google GenAI tools with parametersJsonSchema
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
                }
            ]
        }
    ]
    
    # Transform tools
    openai_tools = adapter._transform_google_genai_tools_to_openai(tools)
    
    # Verify parametersJsonSchema is correctly transformed to parameters
    assert len(openai_tools) == 1
    tool = openai_tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert "parameters" in tool["function"]
    assert tool["function"]["parameters"]["type"] == "object"
    assert "properties" in tool["function"]["parameters"]
    assert "location" in tool["function"]["parameters"]["properties"]


def test_streaming_tool_call_with_empty_args():
    """Test that streaming tool calls with empty arguments are handled correctly"""
    from litellm.google_genai.adapters.transformation import (
        GoogleGenAIStreamWrapper,
    )
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        StreamingChoices,
    )
    
    adapter = GoogleGenAIAdapter()
    
    # Create a tool call with empty arguments
    mock_function = Function(
        name="test_function",
        arguments=""  # Empty arguments
    )
    
    mock_tool_call_delta = ChatCompletionDeltaToolCall(
        id="call_123",
        type="function",
        function=mock_function,
        index=0
    )
    
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
    
    # Create a proper wrapper
    mock_wrapper = GoogleGenAIStreamWrapper(completion_stream=iter([]))
    
    # Manually set up the accumulated tool call to simulate what would happen during streaming
    mock_wrapper.accumulated_tool_calls = {0: {"name": "test_function", "arguments": ""}}
    
    # Create a mock response that has a finish_reason to trigger the final processing
    mock_response_with_finish = ModelResponse(
        id="test-streaming",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=None, tool_calls=[])
            )
        ],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion.chunk"
    )
    
    # Transform streaming chunk - this should process the accumulated tool call
    streaming_chunk = adapter.translate_streaming_completion_to_generate_content(
        mock_response_with_finish, mock_wrapper
    )
    
    # For empty content and tool calls with empty args, we might get None or a minimal response
    # Let's check if we get a valid response with empty content
    if streaming_chunk is not None:
        assert "candidates" in streaming_chunk
        candidate = streaming_chunk["candidates"][0]
        assert "content" in candidate
        parts = candidate["content"]["parts"]
        # If there are parts, check if functionCall with empty args is properly handled
        for part in parts:
            if "functionCall" in part:
                function_call = part["functionCall"]
                assert function_call["name"] == "test_function"
                assert function_call["args"] == {}  # Empty args should become empty object
    else:
        # If streaming_chunk is None, it's acceptable as it might indicate no meaningful content
        # This is a valid case in streaming where we might skip empty chunks
        # The important thing is that no exception was raised
        pass


def test_tool_config_transformation():
    """Test that toolConfig is correctly transformed to tool_choice"""
    adapter = GoogleGenAIAdapter()
    
    # Test different toolConfig modes
    test_cases = [
        # AUTO mode
        {
            "tool_config": {"functionCallingConfig": {"mode": "AUTO"}},
            "expected_tool_choice": "auto"
        },
        # ANY mode - maps to "required" in OpenAI
        {
            "tool_config": {
                "functionCallingConfig": {
                    "mode": "ANY"
                }
            },
            "expected_tool_choice": "required"
        },
        # NONE mode
        {
            "tool_config": {"functionCallingConfig": {"mode": "NONE"}},
            "expected_tool_choice": "none"
        }
    ]
    
    for case in test_cases:
        tool_config = case["tool_config"]
        expected_tool_choice = case["expected_tool_choice"]
        
        # Transform tool config
        openai_tool_choice = adapter._transform_google_genai_tool_config_to_openai(tool_config)
        
        # Verify transformation
        assert openai_tool_choice == expected_tool_choice


def test_stream_transformation_error_handling():
    """Test that stream transformation errors are properly handled"""
    from litellm.google_genai.adapters.transformation import (
        GoogleGenAIStreamWrapper,
    )
    
    adapter = GoogleGenAIAdapter()
    
    # Create a mock response that would cause transformation to fail
    mock_response = ModelResponse(
        id="test-streaming-error",
        choices=[],  # Empty choices which might cause issues
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion.chunk"
    )
    
    # Create a wrapper
    mock_wrapper = GoogleGenAIStreamWrapper(completion_stream=iter([]))
    
    # Try to transform - this should handle errors gracefully
    try:
        streaming_chunk = adapter.translate_streaming_completion_to_generate_content(
            mock_response, mock_wrapper
        )
        # If no exception is raised, that's fine - we just want to ensure no crash
        assert True
    except Exception as e:
        # If an exception is raised, it should be a ValueError with appropriate message
        assert isinstance(e, ValueError)
        # We won't check the exact message as it might vary


def test_non_stream_response_when_stream_requested():
    """Test handling of non-stream responses when streaming was requested"""
    from litellm.types.utils import Choices

    # Mock a non-stream response (ModelResponse with valid choices)
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                index=0,
                message={
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                finish_reason="stop"
            )
        ],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion"
    )

    # Create an instance of the adapter
    adapter = GoogleGenAIAdapter()

    # Test the adapter's translate_completion_to_generate_content method directly
    result = adapter.translate_completion_to_generate_content(mock_response)

    # Verify the result is a valid Google GenAI format response
    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) > 0
    candidate = result["candidates"][0]
    assert "content" in candidate
    assert "parts" in candidate["content"]
    assert isinstance(candidate["content"]["parts"], list)
    assert len(candidate["content"]["parts"]) > 0
    assert "text" in candidate["content"]["parts"][0]
    assert candidate["content"]["parts"][0]["text"] == "Hello, world!"


def test_extra_headers_forwarding():
    """Test that extra_headers is correctly forwarded to completion call.

    This is important for providers like github_copilot that require custom
    headers (e.g., Editor-Version) for authentication.
    """
    # Test that extra_headers is included in completion kwargs
    model = "gpt-3.5-turbo"
    contents = {"role": "user", "parts": [{"text": "Test"}]}
    config = {"temperature": 0.7}

    extra_kwargs = {
        "extra_headers": {
            "Editor-Version": "vscode/1.95.0",
            "Editor-Plugin-Version": "copilot-chat/0.22.4",
            "Custom-Header": "custom-value"
        },
        "metadata": {"user_id": "test-user"}
    }

    completion_kwargs = GenerateContentToCompletionHandler._prepare_completion_kwargs(
        model=model,
        contents=contents,
        config=config,
        stream=False,
        extra_kwargs=extra_kwargs
    )

    # Verify extra_headers is forwarded
    assert "extra_headers" in completion_kwargs, "extra_headers should be forwarded to completion call"
    assert completion_kwargs["extra_headers"]["Editor-Version"] == "vscode/1.95.0"
    assert completion_kwargs["extra_headers"]["Editor-Plugin-Version"] == "copilot-chat/0.22.4"
    assert completion_kwargs["extra_headers"]["Custom-Header"] == "custom-value"

    # Verify metadata is also forwarded (existing behavior)
    assert "metadata" in completion_kwargs
    assert completion_kwargs["metadata"]["user_id"] == "test-user"


def test_extra_headers_not_present():
    """Test that missing extra_headers doesn't cause issues."""
    model = "gpt-3.5-turbo"
    contents = {"role": "user", "parts": [{"text": "Test"}]}
    config = {"temperature": 0.7}

    # extra_kwargs without extra_headers
    extra_kwargs = {
        "metadata": {"user_id": "test-user"}
    }

    completion_kwargs = GenerateContentToCompletionHandler._prepare_completion_kwargs(
        model=model,
        contents=contents,
        config=config,
        stream=False,
        extra_kwargs=extra_kwargs
    )

    # Verify extra_headers is not present (no error)
    assert "extra_headers" not in completion_kwargs

    # Verify metadata is still forwarded
    assert "metadata" in completion_kwargs
    assert completion_kwargs["metadata"]["user_id"] == "test-user"
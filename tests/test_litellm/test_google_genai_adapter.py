#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import os
import sys

import pytest


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

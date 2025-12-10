import asyncio
import json
import os
import sys
from unittest.mock import Mock

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.bedrock.chat.invoke_transformations.amazon_qwen2_transformation import (
    AmazonQwen2Config,
)
from litellm.types.utils import ModelResponse


def test_qwen2_get_supported_params():
    """Test that Qwen2 config returns correct supported parameters"""
    config = AmazonQwen2Config()
    params = config.get_supported_openai_params(model="qwen2/test-model")
    
    expected_params = ["max_tokens", "temperature", "top_p", "top_k", "stop", "stream"]
    for param in expected_params:
        assert param in params


def test_qwen2_map_openai_params():
    """Test that OpenAI parameters are correctly mapped to Qwen2 format"""
    config = AmazonQwen2Config()
    non_default_params = {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["</s>", "<|im_end|>"],
        "stream": True
    }
    optional_params = {}
    
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="qwen2/test-model",
        drop_params=False
    )
    
    assert result["max_tokens"] == 100
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
    assert result["top_k"] == 40
    assert result["stop"] == ["</s>", "<|im_end|>"]
    assert result["stream"] is True


def test_qwen2_convert_messages_to_prompt():
    """Test that messages are correctly converted to Qwen2 prompt format"""
    config = AmazonQwen2Config()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"}
    ]
    
    prompt = config._convert_messages_to_prompt(messages)
    
    expected_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you!<|im_end|>
<|im_start|>user
What's the weather like?<|im_end|>
<|im_start|>assistant
"""
    
    assert prompt == expected_prompt


def test_qwen2_transform_request():
    """Test that the request is correctly transformed to Qwen2 format"""
    config = AmazonQwen2Config()
    
    messages = [
        {"role": "user", "content": "Hello, world!"}
    ]
    
    optional_params = {
        "max_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.9
    }
    
    request_body = config.transform_request(
        model="qwen2/test-model",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    assert "prompt" in request_body
    assert request_body["max_gen_len"] == 50
    assert request_body["temperature"] == 0.8
    assert request_body["top_p"] == 0.9
    
    # Check that the prompt contains the expected format
    assert "<|im_start|>user" in request_body["prompt"]
    assert "Hello, world!" in request_body["prompt"]
    assert "<|im_end|>" in request_body["prompt"]


def test_qwen2_transform_response_with_text_field():
    """Test that Qwen2 response with 'text' field is correctly transformed to OpenAI format"""
    config = AmazonQwen2Config()
    
    # Mock response data with 'text' field (Qwen2 format)
    mock_response_data = {
        "text": "<|im_start|>assistant\nHello! How can I help you today?<|im_end|>",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    
    # Mock the raw response
    mock_raw_response = Mock()
    mock_raw_response.json.return_value = mock_response_data
    
    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello!"}]
    
    result = config.transform_response(
        model="qwen2/test-model",
        messages=messages,
        raw_response=mock_raw_response,
        model_response=model_response,
        logging_obj=Mock(),
        optional_params={},
        litellm_params={},
        api_key="test-key",
        request_data={},
        encoding=None
    )
    
    # Check that the response is correctly formatted
    assert len(result.choices) == 1
    assert result.choices[0]["message"]["role"] == "assistant"
    assert result.choices[0]["message"]["content"] == "Hello! How can I help you today?"
    assert result.choices[0]["finish_reason"] == "stop"
    
    # Check usage information
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 15
    assert result.usage["total_tokens"] == 25


def test_qwen2_transform_response_with_generation_field():
    """Test that Qwen2 response also supports 'generation' field for compatibility"""
    config = AmazonQwen2Config()
    
    # Mock response data with 'generation' field (Qwen3 format, but Qwen2 should handle it)
    mock_response_data = {
        "generation": "<|im_start|>assistant\nHello! How can I help you today?<|im_end|>",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    
    # Mock the raw response
    mock_raw_response = Mock()
    mock_raw_response.json.return_value = mock_response_data
    
    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello!"}]
    
    result = config.transform_response(
        model="qwen2/test-model",
        messages=messages,
        raw_response=mock_raw_response,
        model_response=model_response,
        logging_obj=Mock(),
        optional_params={},
        litellm_params={},
        api_key="test-key",
        request_data={},
        encoding=None
    )
    
    # Check that the response is correctly formatted
    assert len(result.choices) == 1
    assert result.choices[0]["message"]["role"] == "assistant"
    assert result.choices[0]["message"]["content"] == "Hello! How can I help you today?"
    assert result.choices[0]["finish_reason"] == "stop"


def test_qwen2_transform_response_prefers_generation_over_text():
    """Test that Qwen2 prefers 'generation' field over 'text' when both are present"""
    config = AmazonQwen2Config()
    
    # Mock response data with both fields
    mock_response_data = {
        "generation": "This is from generation field",
        "text": "This is from text field",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    
    # Mock the raw response
    mock_raw_response = Mock()
    mock_raw_response.json.return_value = mock_response_data
    
    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello!"}]
    
    result = config.transform_response(
        model="qwen2/test-model",
        messages=messages,
        raw_response=mock_raw_response,
        model_response=model_response,
        logging_obj=Mock(),
        optional_params={},
        litellm_params={},
        api_key="test-key",
        request_data={},
        encoding=None
    )
    
    # Should prefer 'generation' field
    assert result.choices[0]["message"]["content"] == "This is from generation field"


def test_qwen2_transform_response_without_usage():
    """Test response transformation when usage information is not provided"""
    config = AmazonQwen2Config()
    
    # Mock response data without usage
    mock_response_data = {
        "text": "Hello! How can I help you today?"
    }
    
    # Mock the raw response
    mock_raw_response = Mock()
    mock_raw_response.json.return_value = mock_response_data
    
    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello!"}]
    
    result = config.transform_response(
        model="qwen2/test-model",
        messages=messages,
        raw_response=mock_raw_response,
        model_response=model_response,
        logging_obj=Mock(),
        optional_params={},
        litellm_params={},
        api_key="test-key",
        request_data={},
        encoding=None
    )
    
    # Check that the response is correctly formatted
    assert len(result.choices) == 1
    assert result.choices[0]["message"]["role"] == "assistant"
    assert result.choices[0]["message"]["content"] == "Hello! How can I help you today?"
    assert result.choices[0]["finish_reason"] == "stop"


def test_qwen2_provider_detection():
    """Test that Qwen2 provider is correctly detected from model names"""
    from litellm.utils import ProviderConfigManager
    from litellm.types.utils import LlmProviders
    
    # Test with qwen2/ prefix
    config = ProviderConfigManager.get_provider_chat_config(
        model="qwen2/arn:aws:bedrock:us-east-1:123456789012:imported-model/test-qwen2",
        provider=LlmProviders.BEDROCK
    )
    
    assert config is not None
    assert isinstance(config, AmazonQwen2Config)


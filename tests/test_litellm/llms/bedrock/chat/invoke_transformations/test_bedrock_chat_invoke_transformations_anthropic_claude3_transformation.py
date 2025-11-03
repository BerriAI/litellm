import asyncio
import json
import os
import sys

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)


def test_get_supported_params_thinking():
    config = AmazonAnthropicClaudeConfig()
    params = config.get_supported_openai_params(
        model="anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert "thinking" in params


def test_prompt_cache_key_removed_from_bedrock_request():
    """
    Test that prompt_cache_key is removed from Bedrock Anthropic requests.
    
    This prevents the error: "prompt_cache_key: Extra inputs are not permitted"
    when using Claude models on AWS Bedrock.
    
    Related issue: Bedrock doesn't support prompt_cache_key parameter
    """
    config = AmazonAnthropicClaudeConfig()
    
    # Simulate a request with prompt_cache_key
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    optional_params = {
        "max_tokens": 100,
        "temperature": 0.7,
        "prompt_cache_key": "test-cache-key-12345",  # This should be removed
        "stream": True,  # This should also be removed for Bedrock
    }
    
    litellm_params = {}
    headers = {}
    
    # Transform the request
    result = config.transform_request(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )
    
    # Verify prompt_cache_key is NOT in the result
    assert "prompt_cache_key" not in result, "prompt_cache_key should be removed from Bedrock request"
    
    # Verify model is also removed (Bedrock specific)
    assert "model" not in result, "model should be removed from Bedrock request"
    
    # Verify stream is also removed (Bedrock specific)
    assert "stream" not in result, "stream should be removed from Bedrock request"
    
    # Verify anthropic_version is added (Bedrock specific)
    assert "anthropic_version" in result
    
    # Verify other parameters are still present
    assert "max_tokens" in result
    assert result["max_tokens"] == 100
    assert "temperature" in result
    assert result["temperature"] == 0.7
    assert "messages" in result
    
    print("✅ Test passed: prompt_cache_key successfully removed from Bedrock Anthropic request")

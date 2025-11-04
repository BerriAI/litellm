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
    
    # Verify that prompt_cache_key is NOT in Bedrock Anthropic's supported params
    supported_params = config.get_supported_openai_params("anthropic.claude-3-7-sonnet-20250219-v1:0")
    assert "prompt_cache_key" not in supported_params, (
        "prompt_cache_key should not be in Bedrock Anthropic's supported params list"
    )
    
    # Verify other common parameters are present
    assert "max_tokens" in supported_params or "max_completion_tokens" in supported_params
    assert "temperature" in supported_params
    assert "tools" in supported_params
    
    print("✅ Test passed: prompt_cache_key is not in Bedrock Anthropic's supported params")

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
    VertexAIAnthropicConfig,
)


@pytest.mark.parametrize(
    "model, expected_thinking",
    [
        ("claude-sonnet-4@20250514", True),
    ],
)
def test_vertex_ai_anthropic_thinking_param(model, expected_thinking):
    supported_openai_params = VertexAIAnthropicConfig().get_supported_openai_params(
        model=model
    )

    if expected_thinking:
        assert "thinking" in supported_openai_params
    else:
        assert "thinking" not in supported_openai_params


def test_get_supported_params_thinking():
    config = VertexAIAnthropicConfig()
    params = config.get_supported_openai_params(model="claude-sonnet-4")
    assert "thinking" in params


def test_prompt_cache_key_removed_from_vertex_ai_request():
    """
    Test that prompt_cache_key is removed from Vertex AI Anthropic requests.
    
    This prevents the error: "prompt_cache_key: Extra inputs are not permitted"
    when using Claude models on Vertex AI.
    
    Related issue: Vertex AI doesn't support prompt_cache_key parameter
    """
    config = VertexAIAnthropicConfig()
    
    # Simulate a request with prompt_cache_key
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    optional_params = {
        "max_tokens": 100,
        "temperature": 0.7,
        "prompt_cache_key": "test-cache-key-12345",  # This should be removed
    }
    
    litellm_params = {}
    headers = {"anthropic-version": "vertex-2023-10-16"}
    
    # Transform the request
    result = config.transform_request(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )
    
    # Verify prompt_cache_key is NOT in the result
    assert "prompt_cache_key" not in result, "prompt_cache_key should be removed from Vertex AI request"
    
    # Verify model is also removed (Vertex AI specific)
    assert "model" not in result, "model should be removed from Vertex AI request"
    
    # Verify other parameters are still present
    assert "max_tokens" in result
    assert result["max_tokens"] == 100
    assert "temperature" in result
    assert result["temperature"] == 0.7
    assert "messages" in result
    
    print("✅ Test passed: prompt_cache_key successfully removed from Vertex AI Anthropic request")

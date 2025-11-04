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
    
    # Verify that prompt_cache_key is NOT in Vertex AI Anthropic's supported params
    supported_params = config.get_supported_openai_params("claude-3-7-sonnet-20250219")
    assert "prompt_cache_key" not in supported_params, (
        "prompt_cache_key should not be in Vertex AI Anthropic's supported params list"
    )
    
    # Verify other common parameters are present
    assert "max_tokens" in supported_params or "max_completion_tokens" in supported_params
    assert "temperature" in supported_params
    assert "tools" in supported_params
    
    print("✅ Test passed: prompt_cache_key is not in Vertex AI Anthropic's supported params")

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


def test_vertex_ai_anthropic_web_search_header_in_completion():
    """Test that web search tool adds the required beta header for Vertex AI completion requests"""
    from unittest.mock import MagicMock, patch
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    # Create the config instance
    model_info = AnthropicModelInfo()
    
    # Test the header generation directly
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    
    # Check if web search tool is detected
    web_search_detected = model_info.is_web_search_tool_used(tools=tools)
    assert web_search_detected is True, "Web search tool should be detected"
    
    # Generate headers with is_vertex_request=True
    headers = model_info.get_anthropic_headers(
        api_key="test-key",
        web_search_tool_used=web_search_detected,
        is_vertex_request=True,
    )
    
    # Assert that the anthropic-beta header with web-search is present
    assert "anthropic-beta" in headers, "anthropic-beta header should be present"
    assert headers["anthropic-beta"] == "web-search-2025-03-05", \
        f"anthropic-beta should be 'web-search-2025-03-05', got: {headers['anthropic-beta']}"
    
    # Test that header is NOT added for non-Vertex requests
    headers_non_vertex = model_info.get_anthropic_headers(
        api_key="test-key",
        web_search_tool_used=web_search_detected,
        is_vertex_request=False,
    )
    
    # For non-Vertex (Anthropic-hosted), the web search header should NOT be in anthropic-beta
    # because Anthropic doesn't require it
    assert "anthropic-beta" not in headers_non_vertex or "web-search" not in headers_non_vertex.get("anthropic-beta", ""), \
        "anthropic-beta with web-search should not be present for non-Vertex requests"


def test_vertex_ai_anthropic_structured_output_header_not_added():
    """Test that structured output beta headers are NOT added for Vertex AI requests"""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    
    config = AnthropicConfig()
    
    # Test case 1: Vertex request with output_format should NOT add beta header
    headers_vertex = {}
    optional_params_vertex = {
        'output_format': {
            'type': 'json_schema',
            'json_schema': {
                'name': 'MathResult',
                'schema': {'properties': {'result': {'type': 'integer'}}}
            }
        },
        'is_vertex_request': True
    }
    result_vertex = config.update_headers_with_optional_anthropic_beta(headers_vertex, optional_params_vertex)
    
    assert "anthropic-beta" not in result_vertex, \
        f"Vertex request should NOT have anthropic-beta header for structured output, got: {result_vertex.get('anthropic-beta')}"
    
    # Test case 2: Non-Vertex request with output_format SHOULD add beta header
    headers_non_vertex = {}
    optional_params_non_vertex = {
        'output_format': {
            'type': 'json_schema',
            'json_schema': {
                'name': 'MathResult',
                'schema': {'properties': {'result': {'type': 'integer'}}}
            }
        },
        'is_vertex_request': False
    }
    result_non_vertex = config.update_headers_with_optional_anthropic_beta(headers_non_vertex, optional_params_non_vertex)
    
    assert "anthropic-beta" in result_non_vertex, \
        "Non-Vertex request SHOULD have anthropic-beta header for structured output"
    assert result_non_vertex["anthropic-beta"] == "structured-outputs-2025-11-13", \
        f"Expected 'structured-outputs-2025-11-13', got: {result_non_vertex.get('anthropic-beta')}"

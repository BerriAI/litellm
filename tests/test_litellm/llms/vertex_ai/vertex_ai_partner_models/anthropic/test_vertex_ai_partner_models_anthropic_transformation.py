import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path
from litellm.anthropic_beta_headers_manager import (
    update_headers_with_filtered_beta,
)
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
    assert (
        headers["anthropic-beta"] == "web-search-2025-03-05"
    ), f"anthropic-beta should be 'web-search-2025-03-05', got: {headers['anthropic-beta']}"

    # Test that header is NOT added for non-Vertex requests
    headers_non_vertex = model_info.get_anthropic_headers(
        api_key="test-key",
        web_search_tool_used=web_search_detected,
        is_vertex_request=False,
    )

    # For non-Vertex (Anthropic-hosted), the web search header should NOT be in anthropic-beta
    # because Anthropic doesn't require it
    assert (
        "anthropic-beta" not in headers_non_vertex
        or "web-search" not in headers_non_vertex.get("anthropic-beta", "")
    ), "anthropic-beta with web-search should not be present for non-Vertex requests"


def test_vertex_ai_anthropic_context_management_compact_beta_header():
    """Test that context_management with compact adds the correct beta header for Vertex AI"""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "context_management": {"edits": [{"type": "compact_20260112"}]},
        "max_tokens": 100,
        "is_vertex_request": True,
    }

    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Verify context_management is included
    assert "context_management" in result
    assert result["context_management"]["edits"][0]["type"] == "compact_20260112"

    # Verify compact beta header is in anthropic_beta field
    assert "anthropic_beta" in result
    assert "compact-2026-01-12" in result["anthropic_beta"]


def test_vertex_ai_anthropic_context_management_mixed_edits():
    """Test that context_management with both compact and other edits adds both beta headers"""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "context_management": {
            "edits": [
                {"type": "compact_20260112"},
                {"type": "replace", "message_id": "msg_123", "content": "new content"},
            ]
        },
        "max_tokens": 100,
        "is_vertex_request": True,
    }

    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Verify both beta headers are present
    assert "anthropic_beta" in result
    assert "compact-2026-01-12" in result["anthropic_beta"]
    assert "context-management-2025-06-27" in result["anthropic_beta"]


def test_vertex_ai_anthropic_structured_output_header_not_added():
    """Test that structured output beta headers are NOT added for Vertex AI requests"""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    config = AnthropicConfig()

    # Test case 1: Vertex request with output_format should NOT add beta header
    headers_vertex = {}
    optional_params_vertex = {
        "output_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "MathResult",
                "schema": {"properties": {"result": {"type": "integer"}}},
            },
        },
        "is_vertex_request": True,
    }
    result_vertex = config.update_headers_with_optional_anthropic_beta(
        headers_vertex, optional_params_vertex
    )

    assert (
        "anthropic-beta" not in result_vertex
    ), f"Vertex request should NOT have anthropic-beta header for structured output, got: {result_vertex.get('anthropic-beta')}"

    # Test case 2: Non-Vertex request with output_format SHOULD add beta header
    headers_non_vertex = {}
    optional_params_non_vertex = {
        "output_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "MathResult",
                "schema": {"properties": {"result": {"type": "integer"}}},
            },
        },
        "is_vertex_request": False,
    }
    result_non_vertex = config.update_headers_with_optional_anthropic_beta(
        headers_non_vertex, optional_params_non_vertex
    )

    assert (
        "anthropic-beta" in result_non_vertex
    ), "Non-Vertex request SHOULD have anthropic-beta header for structured output"
    assert (
        result_non_vertex["anthropic-beta"] == "structured-outputs-2025-11-13"
    ), f"Expected 'structured-outputs-2025-11-13', got: {result_non_vertex.get('anthropic-beta')}"


def test_vertex_ai_claude_sonnet_4_5_structured_output_fix():
    """
    Test fix for issue #18625: Claude Sonnet 4.5 on VertexAI should use tool-based
    structured outputs instead of output_format parameter.

    This test verifies that:
    1. Claude Sonnet 4.5 uses tool-based structured outputs on VertexAI
    2. output_format parameter is removed from the final request
    3. The fix prevents "Extra inputs are not permitted" error
    """
    config = VertexAIAnthropicConfig()

    # Test data matching the issue report
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "questions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["question", "response"],
                "additionalProperties": False,
            },
        },
    }

    messages = [{"role": "user", "content": "Generate a question and answer about AI."}]

    # Test parameters that would trigger the issue
    non_default_params = {
        "response_format": response_format,
        "max_tokens": 1000,
    }

    # Test 1: Verify map_openai_params forces tool-based approach for Claude Sonnet 4.5
    optional_params = {}
    result_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-3-5-sonnet-20241022",  # Claude Sonnet 4.5 model
        drop_params=False,
    )

    # Should have tools and tool_choice (tool-based approach)
    assert "tools" in result_params, "Tools should be present for structured output"
    assert (
        "tool_choice" in result_params
    ), "Tool choice should be present for structured output"
    assert "json_mode" in result_params, "JSON mode should be enabled"

    # Verify the tool is the response format tool
    tools = result_params["tools"]
    assert len(tools) == 1, "Should have exactly one tool for response format"
    assert tools[0]["name"] == "json_tool_call", "Tool should be named json_tool_call"

    # Test 2: Verify transform_request removes output_format parameter
    # Simulate what would happen if parent class added output_format
    test_data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": messages,
        "max_tokens": 1000,
        "tools": tools,
        "tool_choice": result_params["tool_choice"],
        "output_format": {  # This would be added by parent class for Sonnet 4.5
            "type": "json_schema",
            "schema": response_format["json_schema"]["schema"],
        },
    }

    # Mock the parent transform_request to return data with output_format
    original_transform = config.__class__.__bases__[0].transform_request

    def mock_transform_request(
        self, model, messages, optional_params, litellm_params, headers
    ):
        # Return test data that includes output_format
        return test_data.copy()

    # Temporarily replace parent method
    config.__class__.__bases__[0].transform_request = mock_transform_request

    try:
        final_data = config.transform_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            optional_params=result_params,
            litellm_params={},
            headers={},
        )

        # Verify that output_format was removed (fixes the "Extra inputs are not permitted" error)
        assert (
            "output_format" not in final_data
        ), "output_format should be removed for VertexAI"
        assert "model" not in final_data, "model should be removed for VertexAI"
        assert "tools" in final_data, "tools should still be present"
        assert "tool_choice" in final_data, "tool_choice should still be present"

    finally:
        # Restore original method
        config.__class__.__bases__[0].transform_request = original_transform


def test_vertex_ai_anthropic_other_models_still_use_tools():
    """
    Test that other Anthropic models (non-Sonnet 4.5) on VertexAI also use tool-based
    structured outputs, ensuring consistency across all models.
    """
    config = VertexAIAnthropicConfig()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
    }

    # Test with Claude 3 Sonnet (not 4.5)
    non_default_params = {"response_format": response_format}
    optional_params = {}

    result_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-3-sonnet-20240229",
        drop_params=False,
    )

    # Should still use tool-based approach
    assert (
        "tools" in result_params
    ), "Claude 3 Sonnet should also use tool-based structured output"
    assert "tool_choice" in result_params, "Tool choice should be present"
    assert "json_mode" in result_params, "JSON mode should be enabled"


def test_vertex_ai_anthropic_extra_headers_beta_propagation():
    """Test that anthropic-beta values from extra_headers are propagated to the
    anthropic_beta request body field for Vertex AI requests.

    Vertex AI requires beta flags in the request body (anthropic_beta array),
    not as HTTP headers. This mirrors the Bedrock handler's behavior of
    extracting user-specified beta headers.
    """
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "max_tokens": 100,
        "is_vertex_request": True,
        "extra_headers": {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
        },
    }

    result = config.transform_request(
        model="claude-sonnet-4-20250514",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "anthropic_beta" in result
    assert "interleaved-thinking-2025-05-14" in result["anthropic_beta"]
    assert "extra_headers" not in result


def test_vertex_ai_anthropic_extra_headers_beta_merged_with_auto_betas():
    """Test that extra_headers betas are merged with auto-detected betas
    rather than replacing them."""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "max_tokens": 100,
        "is_vertex_request": True,
        "extra_headers": {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
        },
        "context_management": {"edits": [{"type": "compact_20260112"}]},
    }

    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "anthropic_beta" in result
    assert "interleaved-thinking-2025-05-14" in result["anthropic_beta"]
    assert "compact-2026-01-12" in result["anthropic_beta"]


def test_vertex_ai_anthropic_extra_headers_comma_separated_betas():
    """Test that comma-separated beta values in extra_headers are all extracted."""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "max_tokens": 100,
        "is_vertex_request": True,
        "extra_headers": {
            "anthropic-beta": "interleaved-thinking-2025-05-14,dev-full-thinking-2025-05-14",
        },
    }

    result = config.transform_request(
        model="claude-sonnet-4-20250514",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "anthropic_beta" in result
    assert "interleaved-thinking-2025-05-14" in result["anthropic_beta"]
    assert "dev-full-thinking-2025-05-14" in result["anthropic_beta"]


def test_vertex_ai_anthropic_no_extra_headers_unchanged():
    """Test that requests without extra_headers still work normally."""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "max_tokens": 100,
        "is_vertex_request": True,
    }

    result = config.transform_request(
        model="claude-sonnet-4-20250514",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "anthropic_beta" not in result
    assert "extra_headers" not in result


def test_vertex_ai_partner_models_anthropic_remove_prompt_caching_scope_beta_header():
    """
    Test that remove_unsupported_beta correctly filters out prompt-caching-scope-2026-01-05
    from the anthropic-beta headers.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
        VertexAIPartnerModelsAnthropicMessagesConfig,
    )

    # This beta header should be removed
    PROMPT_CACHING_BETA_HEADER = "prompt-caching-scope-2026-01-05"
    headers = {
        "anthropic-beta": f"other-feature,{PROMPT_CACHING_BETA_HEADER},web-search-2025-03-05"
    }

    headers = update_headers_with_filtered_beta(headers, "vertex_ai")

    beta_header = headers.get("anthropic-beta")
    assert PROMPT_CACHING_BETA_HEADER not in (
        beta_header or ""
    ), f"{PROMPT_CACHING_BETA_HEADER} should be filtered out"
    assert "other-feature" not in (
        beta_header or ""
    ), "Other non-excluded beta headers should remain"
    assert "web-search-2025-03-05" in (
        beta_header or ""
    ), "Other non-excluded beta headers should remain"
    # If prompt-caching was the only value, header should be removed completely
    headers2 = {"anthropic-beta": PROMPT_CACHING_BETA_HEADER}
    headers2 = update_headers_with_filtered_beta(headers2, "vertex_ai")
    assert (
        "anthropic-beta" not in headers2
    ), "Header should be removed if no supported values remain"

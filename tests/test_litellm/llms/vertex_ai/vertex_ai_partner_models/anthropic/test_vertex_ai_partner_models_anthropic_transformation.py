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
    headers = {}
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
        headers=headers,
    )

    # Verify context_management is included
    assert "context_management" in result
    assert result["context_management"]["edits"][0]["type"] == "compact_20260112"

    # Verify compact beta header is in anthropic_beta body field
    assert "anthropic_beta" in result
    assert "compact-2026-01-12" in result["anthropic_beta"]

    # Verify compact beta header is also set as HTTP header
    assert "anthropic-beta" in headers
    assert "compact-2026-01-12" in headers["anthropic-beta"]


def test_vertex_ai_anthropic_context_management_mixed_edits():
    """Test that context_management with both compact and other edits adds both beta headers"""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    headers = {}
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
        headers=headers,
    )

    # Verify both beta headers are present in body field
    assert "anthropic_beta" in result
    assert "compact-2026-01-12" in result["anthropic_beta"]
    assert "context-management-2025-06-27" in result["anthropic_beta"]

    # Verify both beta headers are also set as HTTP header
    assert "anthropic-beta" in headers
    assert "compact-2026-01-12" in headers["anthropic-beta"]
    assert "context-management-2025-06-27" in headers["anthropic-beta"]


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
    structured outputs when ``response_format`` is supplied via the OpenAI-compat
    interface (``map_openai_params``).

    This test verifies that:
    1. Claude Sonnet 4.5 uses tool-based structured outputs when ``response_format``
       is given to the OpenAI-compat path (the path that triggered #18625).
    2. ``output_format`` is forwarded to Vertex AI when present — Vertex now
       accepts the field; the prior blanket-strip behavior was the silent drop
       of Anthropic Structured Outputs that this PR fixes.
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

        # output_format is now forwarded to Vertex (Vertex parity has shifted —
        # it accepts the field and uses it to enforce the JSON schema). The
        # prior behavior silently stripped it, hiding Structured Outputs from
        # callers who explicitly requested them.
        assert "output_format" in final_data
        assert final_data["output_format"]["type"] == "json_schema"
        assert (
            "model" not in final_data
        ), "model is still stripped (Vertex routes by URL)"
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
    """Test that anthropic-beta values from extra_headers are propagated to both
    the anthropic_beta request body field and the anthropic-beta HTTP header
    for Vertex AI requests.

    Vertex AI rawPredict requires beta flags as HTTP headers. The body field
    anthropic_beta alone is insufficient for features like context_management.
    """
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    headers = {}
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
        headers=headers,
    )

    assert "anthropic_beta" in result
    assert "interleaved-thinking-2025-05-14" in result["anthropic_beta"]
    assert "extra_headers" not in result

    # Verify HTTP header is also set
    assert "anthropic-beta" in headers
    assert "interleaved-thinking-2025-05-14" in headers["anthropic-beta"]


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
    headers = {}
    optional_params = {
        "max_tokens": 100,
        "is_vertex_request": True,
    }

    result = config.transform_request(
        model="claude-sonnet-4-20250514",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers=headers,
    )

    assert "anthropic_beta" not in result
    assert "extra_headers" not in result
    assert "anthropic-beta" not in headers


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


def test_vertex_ai_anthropic_output_config_effort_only_forwarded():
    """Vertex AI Claude 4.6/4.7 accept ``output_config.effort`` on rawPredict."""
    config = VertexAIAnthropicConfig()

    messages = [{"role": "user", "content": "What is 2+2?"}]
    headers: dict = {}

    optional_params = {
        "max_tokens": 1024,
        "output_config": {"effort": "high"},
    }

    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers=headers,
    )

    assert result["output_config"] == {"effort": "high"}
    assert result["max_tokens"] == 1024
    assert "messages" in result


def test_vertex_ai_anthropic_output_config_format_passes_through():
    """
    ``output_config`` containing structured-output ``format`` is FORWARDED to
    Vertex AI Claude — Vertex now accepts it and uses it for JSON Schema
    enforcement. Previously the entire field was being silently stripped, so
    Anthropic Structured Outputs never engaged on Vertex even when callers
    requested it.
    """
    config = VertexAIAnthropicConfig()
    messages = [{"role": "user", "content": "Return a person object."}]

    output_config = {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }
    }
    optional_params = {"max_tokens": 1024, "output_config": output_config}

    result = config.transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result["output_config"] == output_config


def test_vertex_ai_anthropic_output_config_format_plus_effort_preserved():
    """Both ``format`` and ``effort`` ride along on Vertex Claude 4.6/4.7."""
    config = VertexAIAnthropicConfig()
    messages = [{"role": "user", "content": "Return a person object."}]

    output_config = {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}},
            },
        },
        "effort": "high",
    }
    optional_params = {"max_tokens": 1024, "output_config": output_config}

    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "output_config" in result
    assert result["output_config"]["effort"] == "high"
    assert result["output_config"]["format"] == output_config["format"]


def test_vertex_ai_anthropic_output_config_non_dict_dropped():
    """Defensive: if ``output_config`` is somehow not a dict, drop it rather
    than forwarding malformed data downstream."""
    config = VertexAIAnthropicConfig()
    messages = [{"role": "user", "content": "hi"}]
    optional_params = {"max_tokens": 64, "output_config": "not-a-dict"}

    result = config.transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "output_config" not in result


def test_vertex_ai_anthropic_output_format_and_output_config_effort_preserved():
    """Both ``output_format`` and ``output_config.effort`` are forwarded on Vertex 4.6/4.7."""
    config = VertexAIAnthropicConfig()
    messages = [{"role": "user", "content": "Extract structured data"}]

    output_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "data",
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        },
    }

    optional_params = {
        "max_tokens": 2048,
        "output_format": output_format,
        "output_config": {"effort": "high"},
    }

    test_data = {
        "model": "claude-opus-4-6",
        "messages": messages,
        "max_tokens": 2048,
        "output_format": output_format,
        "output_config": {"effort": "high"},
    }

    original_transform = config.__class__.__bases__[0].transform_request

    def mock_transform_request(
        self, model, messages, optional_params, litellm_params, headers
    ):
        return test_data.copy()

    config.__class__.__bases__[0].transform_request = mock_transform_request

    try:
        result = config.transform_request(
            model="claude-opus-4-6",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # output_format flows through unchanged — Vertex AI Claude accepts it.
        assert result["output_format"] == output_format
        # output_config.effort now flows through (Vertex accepts it on 4.6/4.7).
        assert result["output_config"] == {"effort": "high"}
        assert result["max_tokens"] == 2048
        assert "model" not in result, "model is still stripped (Vertex routes by URL)"
    finally:
        config.__class__.__bases__[0].transform_request = original_transform


def test_sanitize_vertex_anthropic_output_params_unit():
    """Direct unit coverage for the helper itself (used by both Vertex
    Anthropic transformation paths). Mirrors the integration assertions
    above without going through the full ``transform_request`` stack."""
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.output_params_utils import (
        sanitize_vertex_anthropic_output_params,
    )

    # No-op when output_config absent.
    data: dict = {"max_tokens": 8}
    sanitize_vertex_anthropic_output_params(data)
    assert data == {"max_tokens": 8}

    # Effort-only → preserved (Vertex 4.6/4.7 accept it on rawPredict).
    data = {"output_config": {"effort": "high"}}
    sanitize_vertex_anthropic_output_params(data)
    assert data["output_config"] == {"effort": "high"}

    # Format-only → preserved unchanged.
    fmt = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    data = {"output_config": dict(fmt)}
    sanitize_vertex_anthropic_output_params(data)
    assert data["output_config"] == fmt

    # Mixed → both effort and format kept (no current Vertex-unsupported keys).
    data = {"output_config": {"format": fmt["format"], "effort": "high"}}
    sanitize_vertex_anthropic_output_params(data)
    assert data["output_config"] == {"format": fmt["format"], "effort": "high"}

    # Non-dict → dropped defensively.
    data = {"output_config": "garbage"}
    sanitize_vertex_anthropic_output_params(data)
    assert "output_config" not in data

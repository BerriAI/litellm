from unittest.mock import patch

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)


def test_validate_environment_uses_vertex_ai_location():
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "europe-west1",
        "vertex_credentials": "{}",
    }
    optional_params = {}

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ) as mock_get_url:
        config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-sonnet",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        assert mock_get_url.call_args.kwargs["vertex_location"] == "europe-west1"


def test_web_search_header_added_for_messages_endpoint():
    """Test that web search tool adds the required beta header for Vertex AI /v1/messages requests"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include web search tool in optional_params
    optional_params = {
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
        ]
    }

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Assert that the anthropic-beta header with web-search is present
        assert "anthropic-beta" in updated_headers, "anthropic-beta header should be present"
        assert updated_headers["anthropic-beta"] == "web-search-2025-03-05", \
            f"anthropic-beta should be 'web-search-2025-03-05', got: {updated_headers['anthropic-beta']}"


def test_web_search_header_not_added_without_tool():
    """Test that beta header is NOT added when web search tool is not present"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # No web search tool
    optional_params = {}

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Assert that the anthropic-beta header is NOT present when no web search tool
        assert "anthropic-beta" not in updated_headers, \
            "anthropic-beta header should not be present without web search tool"


def test_compact_context_management_header_added():
    """Test that compact-2026-01-12 beta header is added when context_management with compact_20260112 is used"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include context_management with compact_20260112
    optional_params = {
        "context_management": {
            "edits": [
                {"type": "compact_20260112"}
            ]
        }
    }

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-vertex-ai-opus-4-6",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Assert that the anthropic-beta header with compact-2026-01-12 is present
        assert "anthropic-beta" in updated_headers, "anthropic-beta header should be present"
        assert "compact-2026-01-12" in updated_headers["anthropic-beta"], \
            f"anthropic-beta should contain 'compact-2026-01-12', got: {updated_headers['anthropic-beta']}"


def test_context_management_header_added_for_other_edits():
    """Test that context-management-2025-06-27 beta header is added for non-compact edits"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include context_management with other edit types
    optional_params = {
        "context_management": {
            "edits": [
                {"type": "some_other_type"}
            ]
        }
    }

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-vertex-ai-opus-4-6",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Assert that the anthropic-beta header with context-management-2025-06-27 is present
        assert "anthropic-beta" in updated_headers, "anthropic-beta header should be present"
        assert "context-management-2025-06-27" in updated_headers["anthropic-beta"], \
            f"anthropic-beta should contain 'context-management-2025-06-27', got: {updated_headers['anthropic-beta']}"


def test_both_compact_and_context_management_headers_added():
    """Test that both compact and context-management beta headers are added when both edit types are present"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include context_management with both compact and other edit types
    optional_params = {
        "context_management": {
            "edits": [
                {"type": "compact_20260112"},
                {"type": "some_other_type"}
            ]
        }
    }

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-vertex-ai-opus-4-6",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Assert that both beta headers are present
        assert "anthropic-beta" in updated_headers, "anthropic-beta header should be present"
        assert "compact-2026-01-12" in updated_headers["anthropic-beta"], \
            f"anthropic-beta should contain 'compact-2026-01-12', got: {updated_headers['anthropic-beta']}"
        assert "context-management-2025-06-27" in updated_headers["anthropic-beta"], \
            f"anthropic-beta should contain 'context-management-2025-06-27', got: {updated_headers['anthropic-beta']}"

def test_validate_environment_with_authorization_header_calculates_api_base():
    """Test that api_base is calculated even when Authorization header is already present"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    # Simulate scenario where Authorization is already in headers (e.g., from cached extra_headers)
    headers = {"Authorization": "Bearer existing-token"}
    litellm_params = {
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
        "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
    }
    optional_params = {}

    with patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-vertex-url"
    ) as mock_get_url:
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Verify that api_base was calculated even though Authorization was already present
        assert api_base == "https://mock-vertex-url", \
            f"api_base should be calculated even with Authorization header. Got: {api_base}"
        assert mock_get_url.called, "get_complete_vertex_url should be called"
        
        # Verify Authorization header is still present
        assert "Authorization" in updated_headers, \
            "Authorization header should be preserved"


def test_vertex_ai_experimental_output_config_format_removed():
    """
    Regression test for #21407: output_config.format (structured output schema)
    must be stripped in the experimental pass-through path.
    When only format is present, the entire output_config key is removed.
    """
    from litellm.types.router import GenericLiteLLMParams

    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 50,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"greeting": {"type": "string"}},
                    "required": ["greeting"],
                },
            }
        },
    }
    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "output_config" not in result, (
        "output_config should be removed when it only contains format"
    )


def test_vertex_ai_experimental_output_config_effort_preserved():
    """
    Regression test: output_config.effort (reasoning effort for Claude 4.6)
    must be preserved in the experimental path.
    """
    from litellm.types.router import GenericLiteLLMParams

    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 50,
        "output_config": {"effort": "high"},
    }
    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "think carefully"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "output_config" in result, (
        "output_config with effort must be preserved for Claude 4.6"
    )
    assert result["output_config"]["effort"] == "high"


def test_vertex_ai_experimental_output_config_mixed():
    """
    When output_config has both format and effort, only format is stripped.
    """
    from litellm.types.router import GenericLiteLLMParams

    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 50,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {}},
            },
            "effort": "medium",
        },
    }
    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "extract carefully"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "output_config" in result
    assert "format" not in result["output_config"]
    assert result["output_config"]["effort"] == "medium"

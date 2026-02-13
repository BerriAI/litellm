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

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


class TestExcludeAnthropicBetaValuesPassThrough:
    """Tests for exclude_anthropic_beta_values in pass-through handler"""

    def test_exclude_filters_existing_header_values(self):
        """Test that exclude_anthropic_beta_values filters values from existing anthropic-beta header"""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        headers = {"anthropic-beta": "prompt-caching-2024-07-31,context-1m-2025-08-07"}
        litellm_params = {
            "vertex_ai_project": "test-project",
            "vertex_ai_location": "us-central1",
            "vertex_credentials": "{}",
            "exclude_anthropic_beta_values": ["context-1m-2025-08-07"],
        }
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

            # context-1m-2025-08-07 should be filtered out
            assert "anthropic-beta" in updated_headers
            assert "context-1m-2025-08-07" not in updated_headers["anthropic-beta"]
            assert "prompt-caching-2024-07-31" in updated_headers["anthropic-beta"]

    def test_exclude_removes_header_when_all_values_filtered(self):
        """Test that anthropic-beta header is removed when all values are excluded"""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        litellm_params = {
            "vertex_ai_project": "test-project",
            "vertex_ai_location": "us-central1",
            "vertex_credentials": "{}",
            "exclude_anthropic_beta_values": ["context-1m-2025-08-07"],
        }
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

            # anthropic-beta header should be removed entirely
            assert "anthropic-beta" not in updated_headers

    def test_exclude_prevents_auto_added_web_search_header(self):
        """Test that exclude_anthropic_beta_values can prevent auto-added web search header"""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        headers = {}
        litellm_params = {
            "vertex_ai_project": "test-project",
            "vertex_ai_location": "us-central1",
            "vertex_credentials": "{}",
            "exclude_anthropic_beta_values": ["web-search-2025-03-05"],
        }
        # Include web search tool that would normally add the header
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

            # web-search header should NOT be added because it's in the exclude list
            assert "anthropic-beta" not in updated_headers

    def test_empty_exclude_list_preserves_headers(self):
        """Test that empty exclude list doesn't filter any headers"""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        headers = {"anthropic-beta": "prompt-caching-2024-07-31,context-1m-2025-08-07"}
        litellm_params = {
            "vertex_ai_project": "test-project",
            "vertex_ai_location": "us-central1",
            "vertex_credentials": "{}",
            "exclude_anthropic_beta_values": [],
        }
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

            # All values should be preserved
            assert "anthropic-beta" in updated_headers
            assert "prompt-caching-2024-07-31" in updated_headers["anthropic-beta"]
            assert "context-1m-2025-08-07" in updated_headers["anthropic-beta"]


from litellm.litellm_core_utils.get_provider_specific_headers import (
    ProviderSpecificHeaderUtils,
)
from litellm.types.utils import ProviderSpecificHeader


class TestProviderSpecificHeaderUtils:
    def test_get_provider_specific_headers_matching_provider(self):
        """Test that the method returns extra_headers when custom_llm_provider matches."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "openai",
            "extra_headers": {
                "Authorization": "Bearer token123",
                "Custom-Header": "value",
            },
        }
        custom_llm_provider = "openai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        expected = {"Authorization": "Bearer token123", "Custom-Header": "value"}
        assert result == expected

    def test_get_provider_specific_headers_no_match_or_none(self):
        """Test that the method returns empty dict when provider doesn't match or is None."""
        # Test case 1: Provider doesn't match
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"Authorization": "Bearer token123"},
        }
        custom_llm_provider = "openai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )
        assert result == {}

        # Test case 2: provider_specific_header is None
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            None, "openai"
        )
        assert result == {}

    def test_get_provider_specific_headers_anthropic_via_vertex_ai(self):
        """Test that Anthropic headers are forwarded to Vertex AI."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {
                "anthropic-beta": "context-management-2025-06-27",
                "anthropic-version": "2023-06-01",
            },
        }
        custom_llm_provider = "vertex_ai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        expected = {
            "anthropic-beta": "context-management-2025-06-27",
            "anthropic-version": "2023-06-01",
        }
        assert result == expected

    def test_get_provider_specific_headers_anthropic_via_vertex_ai_beta(self):
        """Test that Anthropic headers are forwarded to Vertex AI Beta."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            },
        }
        custom_llm_provider = "vertex_ai_beta"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        expected = {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
        }
        assert result == expected

    def test_get_provider_specific_headers_anthropic_via_bedrock(self):
        """Test that Anthropic headers are forwarded to Bedrock."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {
                "anthropic-beta": "computer-use-2024-10-22",
            },
        }
        custom_llm_provider = "bedrock"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        expected = {
            "anthropic-beta": "computer-use-2024-10-22",
        }
        assert result == expected

    def test_get_provider_specific_headers_non_anthropic_via_vertex_ai(self):
        """Test that non-Anthropic headers are NOT forwarded to Vertex AI."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "openai",
            "extra_headers": {"X-OpenAI-Custom": "value"},
        }
        custom_llm_provider = "vertex_ai"

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )

        # Should return empty dict since openai headers shouldn't be forwarded to vertex_ai
        assert result == {}

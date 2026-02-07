import pytest

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

    def test_get_provider_specific_headers_multi_provider_anthropic_to_bedrock(self):
        """Test that anthropic headers work with bedrock provider (multi-provider support)."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic,bedrock,bedrock_converse,vertex_ai",
            "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
        }

        # Test bedrock provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test anthropic provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "anthropic"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test bedrock_converse provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock_converse"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

        # Test vertex_ai provider
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "vertex_ai"
        )
        assert result == {"anthropic-beta": "context-1m-2025-08-07"}

    def test_get_provider_specific_headers_multi_provider_no_match(self):
        """Test that non-listed providers return empty dict with multi-provider list."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic,bedrock,vertex_ai",
            "extra_headers": {"anthropic-beta": "test"},
        }

        # Test provider not in list
        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "openai"
        )
        assert result == {}

    def test_get_provider_specific_headers_with_spaces(self):
        """Test that comma-separated list with spaces is handled correctly."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic, bedrock, vertex_ai",
            "extra_headers": {"anthropic-beta": "test"},
        }

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, "bedrock"
        )
        assert result == {"anthropic-beta": "test"}

    def test_get_provider_specific_headers_none_custom_llm_provider(self):
        """Test that None custom_llm_provider returns empty dict."""
        provider_specific_header: ProviderSpecificHeader = {
            "custom_llm_provider": "anthropic",
            "extra_headers": {"anthropic-beta": "test"},
        }

        result = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header, None
        )
        assert result == {}

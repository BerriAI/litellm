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
            "extra_headers": {"Authorization": "Bearer token123", "Custom-Header": "value"}
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
            "extra_headers": {"Authorization": "Bearer token123"}
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

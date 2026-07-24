from typing import Dict, Optional

from litellm.types.utils import ProviderSpecificHeader


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: Optional[ProviderSpecificHeader],
        custom_llm_provider: Optional[str],
    ) -> Dict:
        """
        Get the provider specific headers for the given custom llm provider.

        Supports comma-separated provider lists for headers that work across multiple providers.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        stored_providers = provider_specific_header.get("custom_llm_provider", "")
        provider_list = [p.strip() for p in stored_providers.split(",")]

        if custom_llm_provider in provider_list:
            return provider_specific_header.get("extra_headers", {})

        return {}

from typing import Final

from litellm.types.utils import ProviderSpecificHeader


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | None,
        custom_llm_provider: str | None,
    ) -> dict:
        """
        Get the provider specific headers for the given custom llm provider.

        Supports comma-separated provider lists for headers that work across multiple providers.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        stored_providers: Final = provider_specific_header.get("custom_llm_provider", "")
        provider_list: Final = [p.strip() for p in stored_providers.split(",")]

        if custom_llm_provider in provider_list:
            return provider_specific_header.get("extra_headers", {})

        return {}

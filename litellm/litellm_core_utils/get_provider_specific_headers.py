from typing import Dict, Optional

from litellm.types.utils import ProviderSpecificHeader


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: Optional[ProviderSpecificHeader],
        custom_llm_provider: Optional[str],
    ) -> Dict:
        """
        Get the provider specific headers for the given custom llm provider

        Returns:
            Optional[Dict]: The provider specific headers for the given custom llm provider
        """
        if (
            provider_specific_header is not None
            and provider_specific_header.get("custom_llm_provider") == custom_llm_provider
        ):
            return provider_specific_header.get("extra_headers", {})
        return {}
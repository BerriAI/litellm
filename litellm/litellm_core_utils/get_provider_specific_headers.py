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

        This function handles both direct provider matches and cross-provider header forwarding.
        Cross-provider forwarding is supported when a hosting provider accepts headers from the
        model family it hosts (e.g., Vertex AI and Bedrock hosting Anthropic models).

        Args:
            provider_specific_header: Header configuration with provider tag and extra headers
            custom_llm_provider: The target LLM provider handling the request

        Returns:
            Dict: The provider specific headers to forward, or empty dict if no match
        """
        if provider_specific_header is None:
            return {}

        header_provider = provider_specific_header.get("custom_llm_provider")

        # Direct match (e.g., anthropic → anthropic, bedrock → bedrock)
        if header_provider == custom_llm_provider:
            return provider_specific_header.get("extra_headers", {})

        # Cross-provider forwarding: Anthropic headers → Vertex AI/Bedrock
        # These providers host Anthropic Claude models and accept Anthropic-specific headers
        # like 'anthropic-beta' (enables features like extended thinking, prompt caching)
        # Fixes: https://github.com/BerriAI/litellm/issues/15299
        if header_provider == "anthropic" and custom_llm_provider in [
            "vertex_ai",
            "vertex_ai_beta",
            "bedrock",
        ]:
            return provider_specific_header.get("extra_headers", {})

        return {}

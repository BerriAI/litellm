from typing import Dict, Optional

from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key
from litellm.types.utils import ProviderSpecificHeader

_ANTHROPIC_PROVIDER = "anthropic"


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: Optional[ProviderSpecificHeader],
        custom_llm_provider: Optional[str],
    ) -> Dict:
        """
        Get the provider specific headers for the given custom llm provider.

        Supports comma-separated provider lists for headers that work across multiple providers.

        Anthropic OAuth tokens (sk-ant-oat*) in the Authorization header are stripped for
        non-Anthropic providers to prevent them from overriding provider-specific auth
        (e.g. AWS SigV4 for Bedrock, service account credentials for Vertex AI).

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        stored_providers = provider_specific_header.get("custom_llm_provider", "")
        provider_list = [p.strip() for p in stored_providers.split(",")]

        if custom_llm_provider in provider_list:
            headers = provider_specific_header.get("extra_headers", {})
            # Anthropic OAuth tokens must not be forwarded to non-Anthropic providers.
            # Forwarding them would overwrite provider-specific auth headers
            # (e.g. Bedrock's SigV4 Authorization, Vertex AI service-account auth).
            if custom_llm_provider != _ANTHROPIC_PROVIDER:
                headers = {
                    k: v
                    for k, v in headers.items()
                    if not (k.lower() == "authorization" and is_anthropic_oauth_key(v))
                }
            return headers

        return {}

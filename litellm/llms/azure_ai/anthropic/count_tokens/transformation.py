"""
Azure AI Anthropic CountTokens API transformation logic.

Extends the base Anthropic CountTokens transformation with Azure authentication.
"""

from typing import Any, Dict, Optional

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.types.router import GenericLiteLLMParams


class AzureAIAnthropicCountTokensConfig(AnthropicCountTokensConfig):
    """
    Configuration and transformation logic for Azure AI Anthropic CountTokens API.

    Extends AnthropicCountTokensConfig with Azure authentication.
    Azure AI Anthropic uses the same endpoint format but with Azure auth headers.
    """

    def get_required_headers(
        self,
        api_key: str,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Get the required headers for the Azure AI Anthropic CountTokens API.

        Uses Azure authentication (api-key header) instead of Anthropic's x-api-key.

        Args:
            api_key: The Azure AI API key
            litellm_params: Optional LiteLLM parameters for additional auth config

        Returns:
            Dictionary of required headers with Azure authentication
        """
        # Start with base headers
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
        }

        # Use Azure authentication
        litellm_params = litellm_params or {}
        if "api_key" not in litellm_params:
            litellm_params["api_key"] = api_key

        litellm_params_obj = GenericLiteLLMParams(**litellm_params)

        # Get Azure auth headers
        azure_headers = BaseAzureLLM._base_validate_azure_environment(
            headers={}, litellm_params=litellm_params_obj
        )

        # Merge Azure auth headers
        headers.update(azure_headers)

        return headers

    def get_count_tokens_endpoint(self, api_base: str) -> str:
        """
        Get the Azure AI Anthropic CountTokens API endpoint.

        Args:
            api_base: The Azure AI API base URL 
                      (e.g., https://my-resource.services.ai.azure.com or
                       https://my-resource.services.ai.azure.com/anthropic)

        Returns:
            The endpoint URL for the CountTokens API
        """
        # Azure AI Anthropic endpoint format:
        # https://<resource>.services.ai.azure.com/anthropic/v1/messages/count_tokens
        api_base = api_base.rstrip("/")

        # Ensure the URL has /anthropic path
        if not api_base.endswith("/anthropic"):
            if "/anthropic" not in api_base:
                api_base = f"{api_base}/anthropic"

        # Add the count_tokens path
        return f"{api_base}/v1/messages/count_tokens"

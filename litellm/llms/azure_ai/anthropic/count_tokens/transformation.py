"""
Azure AI Anthropic CountTokens API transformation logic.

Extends the base Anthropic CountTokens transformation with Azure authentication.
"""

from typing import Final
from urllib.parse import urlsplit

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.litellm_core_utils.url_utils import join_api_base
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
        api_key: str | None,
        litellm_params: dict[str, object] | None = None,
    ) -> dict[str, str]:
        """
        Get the required headers for the Azure AI Anthropic CountTokens API.

        Azure AI Anthropic uses Anthropic's native API format, which requires the
        x-api-key header for authentication (in addition to Azure's api-key header).

        Args:
            api_key: The Azure AI API key
            litellm_params: Optional LiteLLM parameters for additional auth config

        Returns:
            Dictionary of required headers with both x-api-key and Azure authentication
        """
        # Start with base headers including x-api-key for Anthropic API compatibility
        headers: Final = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
            **({"x-api-key": api_key} if api_key else {}),
        }

        # Also set up Azure auth headers for flexibility
        litellm_params = litellm_params or {}
        if api_key and "api_key" not in litellm_params:
            litellm_params["api_key"] = api_key

        litellm_params_obj: Final = GenericLiteLLMParams.model_validate(litellm_params)

        # Get Azure auth headers (api-key or Authorization)
        azure_headers = BaseAzureLLM.base_validate_azure_environment(headers={}, litellm_params=litellm_params_obj)

        headers.update(azure_headers)
        if "api-key" in azure_headers:
            headers.setdefault("x-api-key", azure_headers["api-key"])

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
        base: Final = api_base.rstrip("/")
        if base.endswith("/v1/messages"):
            return join_api_base(base, "count_tokens")
        anthropic_base: Final = base if "/anthropic" in urlsplit(base).path else join_api_base(base, "anthropic")
        return join_api_base(anthropic_base, "v1/messages/count_tokens")

from typing import Optional
from urllib.parse import urlparse, urlunparse

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.types.router import GenericLiteLLMParams

# Endpoint-specific path suffixes that may appear in a deployment's api_base
# (e.g. the responses endpoint URL is stored as api_base for Azure models).
# Strip these before building the containers URL so we always start from the
# resource root (https://resource.cognitiveservices.azure.com).
_AZURE_ENDPOINT_PATHS = ("/openai/responses",)


class AzureContainerConfig(OpenAIContainerConfig):
    """
    Configuration class for Azure OpenAI container API.

    Inherits request/response transformations from OpenAIContainerConfig since
    Azure's container API is wire-compatible with OpenAI's. Only overrides
    authentication (api-key header) and URL construction (openai/v1/containers path).

    Azure container API reference:
    https://learn.microsoft.com/en-us/azure/foundry/openai/latest#containers
    """

    def validate_environment(
        self,
        headers: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers,
            litellm_params=GenericLiteLLMParams(api_key=api_key),
        )

    @staticmethod
    def _normalize_api_base(api_base: Optional[str]) -> Optional[str]:
        """Strip endpoint-specific path suffixes from api_base to get the resource root."""
        if not api_base:
            return api_base
        parsed = urlparse(api_base)
        path = parsed.path
        for ep in _AZURE_ENDPOINT_PATHS:
            idx = path.find(ep)
            if idx >= 0:
                return urlunparse(
                    (parsed.scheme, parsed.netloc, path[:idx], "", "", "")
                )
        return api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Build the Azure container endpoint URL.

        Azure container API uses the path:
          {endpoint}/openai/v1/containers
        when api_version is 'v1', 'latest', or 'preview'; otherwise:
          {endpoint}/openai/containers
        """
        return BaseAzureLLM._get_base_azure_url(
            api_base=self._normalize_api_base(api_base),
            litellm_params=litellm_params,
            route="/openai/containers",
            default_api_version="v1",
        )

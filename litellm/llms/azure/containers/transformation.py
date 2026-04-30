from typing import Optional

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.types.router import GenericLiteLLMParams


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
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/containers",
            default_api_version="v1",
        )

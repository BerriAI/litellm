from typing import Optional

from litellm.constants import AZURE_DEFAULT_CONTAINERS_API_VERSION
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.types.router import GenericLiteLLMParams


class AzureOpenAIContainerConfig(OpenAIContainerConfig):
    """Azure OpenAI Container Config.

    Inherits from OpenAIContainerConfig and overrides only Azure-specific methods.
    Request/response transformations are identical to OpenAI.
    """

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Get the complete URL for Azure container API.

        Constructs Azure-specific URLs like:
        https://{resource}.openai.azure.com/openai/v1/containers?api-version=xxx
        """
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/v1/containers",
            default_api_version=AZURE_DEFAULT_CONTAINERS_API_VERSION,
        )

    def validate_environment(
        self,
        headers: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        """Validate and set up Azure authentication headers.

        Uses Azure api-key header (not Bearer token like OpenAI).
        """
        # Create a GenericLiteLLMParams with the api_key if provided
        litellm_params = GenericLiteLLMParams(api_key=api_key) if api_key else None

        # Azure uses BaseAzureLLM's validation which handles api-key header
        return BaseAzureLLM._base_validate_azure_environment(
                headers=headers, litellm_params=litellm_params
        )

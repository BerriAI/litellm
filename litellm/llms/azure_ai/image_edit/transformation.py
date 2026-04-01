from typing import Optional

import httpx

import litellm
from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.utils import _add_path_to_api_base


class AzureFoundryFluxImageEditConfig(OpenAIImageEditConfig):
    """
    Azure AI Foundry FLUX image edit config

    Supports FLUX models including FLUX-1-kontext-pro for image editing.

    Azure AI Foundry FLUX models handle image editing through the /images/edits endpoint,
    same as standard Azure OpenAI models. The request format uses multipart/form-data
    with image files and prompt.
    """

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Validate Azure AI Foundry environment and set up authentication
        Uses Api-Key header format
        """
        api_key = AzureFoundryModelInfo.get_api_key(api_key)

        if not api_key:
            raise ValueError(
                f"Azure AI API key is required for model {model}. Set AZURE_AI_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Api-Key": api_key,  # Azure AI Foundry uses Api-Key header format
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Constructs a complete URL for Azure AI Foundry image edits API request.

        Azure AI Foundry FLUX models handle image editing through the /images/edits
        endpoint.

        Args:
        - model: Model name (deployment name for Azure AI Foundry)
        - api_base: Base URL for Azure AI endpoint
        - litellm_params: Additional parameters including api_version

        Returns:
        - Complete URL for the image edits endpoint
        """
        api_base = AzureFoundryModelInfo.get_api_base(api_base)

        if api_base is None:
            raise ValueError(
                "Azure AI API base is required. Set AZURE_AI_API_BASE environment variable or pass api_base parameter."
            )

        api_version = (
            litellm_params.get("api_version")
            or litellm.api_version
            or get_secret_str("AZURE_AI_API_VERSION")
        )
        if api_version is None:
            # API version is mandatory for Azure AI Foundry
            raise ValueError(
                "Azure API version is required. Set AZURE_AI_API_VERSION environment variable or pass api_version parameter."
            )

        # Add the path to the base URL using the model as deployment name
        # Azure AI Foundry FLUX models use /images/edits for editing
        if "/openai/deployments/" in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base,
                ending_path="/images/edits",
            )
        else:
            new_url = _add_path_to_api_base(
                api_base=api_base,
                ending_path=f"/openai/deployments/{model}/images/edits",
            )

        # Use the new query_params dictionary
        final_url = httpx.URL(new_url).copy_with(params={"api-version": api_version})

        return str(final_url)

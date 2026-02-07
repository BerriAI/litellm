from typing import Optional, cast

import httpx

import litellm
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.utils import _add_path_to_api_base


class AzureImageEditConfig(OpenAIImageEditConfig):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
        )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
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
        Constructs a complete URL for the API request.

        Args:
        - api_base: Base URL, e.g.,
            "https://litellm8397336933.openai.azure.com"
            OR
            "https://litellm8397336933.openai.azure.com/openai/deployments/<deployment_name>/images/edits?api-version=2024-05-01-preview"
        - model: Model name (deployment name).
        - litellm_params: Additional query parameters, including "api_version".

        Returns:
        - A complete URL string, e.g.,
        "https://litellm8397336933.openai.azure.com/openai/deployments/<deployment_name>/images/edits?api-version=2024-05-01-preview"
        """
        api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")
        if api_base is None:
            raise ValueError(
                f"api_base is required for Azure AI Studio. Please set the api_base parameter. Passed `api_base={api_base}`"
            )
        original_url = httpx.URL(api_base)

        # Extract api_version or use default
        api_version = cast(Optional[str], litellm_params.get("api_version"))

        # Create a new dictionary with existing params
        query_params = dict(original_url.params)

        # Add api_version if needed
        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version

        # Add the path to the base URL using the model as deployment name
        if "/openai/deployments/" not in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base,
                ending_path=f"/openai/deployments/{model}/images/edits",
            )
        else:
            new_url = api_base

        # Use the new query_params dictionary
        final_url = httpx.URL(new_url).copy_with(params=query_params)

        return str(final_url)

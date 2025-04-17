from typing import TYPE_CHECKING, Any, Optional, cast

import httpx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import *
from litellm.utils import _add_path_to_api_base

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AzureOpenAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
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
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Constructs a complete URL for the API request.

        Args:
        - api_base: Base URL, e.g.,
            "https://litellm8397336933.openai.azure.com"
            OR
            "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
        - model: Model name.
        - optional_params: Additional query parameters, including "api_version".
        - stream: If streaming is required (optional).

        Returns:
        - A complete URL string, e.g.,
        "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
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

        # Add the path to the base URL
        if "/openai/responses" not in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/openai/responses"
            )
        else:
            new_url = api_base

        # Use the new query_params dictionary
        final_url = httpx.URL(new_url).copy_with(params=query_params)

        return str(final_url)

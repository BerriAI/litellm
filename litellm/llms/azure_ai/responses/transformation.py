"""
Responses API transformation for Azure AI provider.

When api_base includes /projects/, route to the real upstream /responses
endpoint instead of falling back to the completions-style bridge.

Ref: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/generate-responses
"""

from typing import Optional
from urllib.parse import urlparse

import httpx

import litellm
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import _add_path_to_api_base


class AzureAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Azure AI Foundry Responses API.

    Extends OpenAI's responses config because Azure AI Foundry project-based
    endpoints follow the OpenAI /responses spec. Uses Azure-specific auth
    (api-key header for *.services.ai.azure.com hosts) and constructs the
    correct URL path for project-based endpoints.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE_AI

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = AzureFoundryModelInfo.get_api_key(
            api_key=litellm_params.api_key,
        )
        api_base = AzureFoundryModelInfo.get_api_base(
            api_base=litellm_params.api_base,
        )

        if api_key:
            if api_base and self._should_use_api_key_header(api_base):
                headers["api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Fall back to Azure AD token-based auth
            headers = BaseAzureLLM._base_validate_azure_environment(
                headers=headers, litellm_params=litellm_params
            )

        headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _should_use_api_key_header(api_base: str) -> bool:
        """
        Returns True if the request should use the ``api-key`` header.

        Azure AI Foundry endpoints under *.services.ai.azure.com and
        *.openai.azure.com expect the ``api-key`` header instead of
        ``Authorization: Bearer ...``.
        """
        parsed_url = urlparse(api_base)
        host = parsed_url.hostname
        if host and (
            host.endswith(".services.ai.azure.com")
            or host.endswith(".openai.azure.com")
        ):
            return True
        return False

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Build the full URL for the Azure AI Foundry Responses API.

        For project-based endpoints (api_base contains ``/projects/``),
        appends ``/openai/v1/responses`` to the base.

        Example:
            api_base = "https://<resource>.services.ai.azure.com/api/projects/<project>"
            -> "https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/responses"
        """
        api_base = AzureFoundryModelInfo.get_api_base(api_base=api_base)

        if api_base is None:
            raise ValueError(
                "api_base is required for Azure AI Responses API. "
                "Set via api_base parameter or AZURE_AI_API_BASE environment variable."
            )

        # Extract api_version
        api_version = litellm_params.get("api_version")

        # Parse query params from existing URL
        original_url = httpx.URL(api_base)
        query_params = dict(original_url.params)

        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version

        # Build the responses endpoint path
        if "/projects/" in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/openai/v1/responses"
            )
        elif "services.ai.azure.com" in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/models/responses"
            )
        else:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/v1/responses"
            )

        final_url = httpx.URL(new_url).copy_with(params=query_params)
        return str(final_url)

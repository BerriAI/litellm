"""
Responses API transformation for the Azure AI (Azure AI Foundry) provider.

Azure AI Foundry exposes a native `/responses` endpoint for models that support
it. Without this config, `azure_ai` has no Responses API implementation, so
`litellm.responses(...)` (and `/chat/completions` requests bridged to the
Responses API via ``model_info: {mode: responses}``) never reach a real upstream
`/responses` endpoint and fall back to the completions-style bridge.

This inherits from ``AzureOpenAIResponsesAPIConfig`` so it reuses Azure's
Responses request handling (flattening function tools to the top level, filtering
the ``status`` field from reasoning input items, etc.) and overrides only the
Azure AI Foundry-specific URL construction and auth. Both Azure commercial
(`*.azure.com`) and Azure Government (`*.azure.us`) hosts are supported.

Refs:
- https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/generate-responses
- Azure Government domains: https://learn.microsoft.com/en-us/azure/azure-government/compare-azure-government-global-azure
"""

from typing import Optional
from urllib.parse import urlparse

import httpx

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import _add_path_to_api_base

# Azure AI Foundry / Azure OpenAI resource host suffixes across clouds. Commercial
# (.azure.com) and US Government (.azure.us) are both included so the same auth +
# URL logic applies regardless of sovereign cloud. These hosts serve the Foundry
# ``/models/responses`` route and expect the ``api-key`` header.
_AZURE_FOUNDRY_HOST_SUFFIXES = (
    # Azure commercial
    ".services.ai.azure.com",
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    # Azure US Government
    ".services.ai.azure.us",
    ".openai.azure.us",
    ".cognitiveservices.azure.us",
)


def _azure_host(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    return urlparse(api_base).hostname


def _is_azure_foundry_host(api_base: Optional[str]) -> bool:
    """
    Return True when ``api_base`` points at an Azure Foundry / Azure OpenAI
    resource (commercial or government), which expect the ``api-key`` header.
    """
    host = _azure_host(api_base)
    return bool(host) and host.endswith(_AZURE_FOUNDRY_HOST_SUFFIXES)


def _is_project_endpoint(api_base: str) -> bool:
    """
    True for Azure AI Foundry project-based endpoints, e.g.
    ``https://<res>.services.ai.azure.com/api/projects/<proj>``.

    Matches on the URL *path* only so a ``/projects/`` substring in a query
    string does not trigger a false positive.
    """
    return "/projects/" in urlparse(api_base).path


class AzureAIResponsesAPIConfig(AzureOpenAIResponsesAPIConfig):
    """
    Configuration for the Azure AI Foundry Responses API.

    Reuses ``AzureOpenAIResponsesAPIConfig`` request handling (tool flattening,
    reasoning-item ``status`` filtering) and overrides only the Foundry-specific
    URL construction and auth so that both commercial (`*.azure.com`) and
    government (`*.azure.us`) hosts route to the correct `/responses` endpoint.
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
        api_key = AzureFoundryModelInfo.get_api_key(api_key=litellm_params.api_key)
        api_base = AzureFoundryModelInfo.get_api_base(api_base=litellm_params.api_base)

        if api_key:
            if api_base and _is_azure_foundry_host(api_base):
                headers["api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Fall back to Azure AD token-based auth (entra id / managed identity).
            headers = BaseAzureLLM._base_validate_azure_environment(headers=headers, litellm_params=litellm_params)

        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Build the full URL for the Azure AI Foundry Responses API.

        Path selection (commercial and government hosts are treated identically):

        - Project-based endpoints (path contains ``/projects/``) append
          ``/openai/v1/responses``.
          e.g. ``https://<res>.services.ai.azure.us/api/projects/<proj>``
               -> ``.../api/projects/<proj>/openai/v1/responses``
        - Azure Foundry / Azure OpenAI hosts append ``/models/responses``.
        - Any other (generic OpenAI-compatible) base appends ``/v1/responses``.
        """
        api_base = AzureFoundryModelInfo.get_api_base(api_base=api_base)

        if api_base is None:
            raise ValueError(
                "api_base is required for the Azure AI Responses API. "
                "Set it via the api_base parameter or the AZURE_AI_API_BASE "
                "environment variable."
            )

        # Preserve any query params already on the base and propagate api-version.
        original_url = httpx.URL(api_base)
        query_params = dict(original_url.params)
        api_version = litellm_params.get("api_version")
        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version

        # IMPORTANT: the project check MUST come before the host check, because
        # project URLs are also on Azure Foundry hosts. Checking the host first
        # would send project endpoints to /models/responses instead of
        # /openai/v1/responses.
        if _is_project_endpoint(api_base):
            new_url = _add_path_to_api_base(api_base=api_base, ending_path="/openai/v1/responses")
        elif _is_azure_foundry_host(api_base):
            new_url = _add_path_to_api_base(api_base=api_base, ending_path="/models/responses")
        else:
            new_url = _add_path_to_api_base(api_base=api_base, ending_path="/v1/responses")

        final_url = httpx.URL(new_url).copy_with(params=query_params)
        return str(final_url)

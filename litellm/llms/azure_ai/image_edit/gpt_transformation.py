from collections.abc import Mapping
from typing import Final

import litellm
from litellm.llms.azure.image_edit.transformation import AzureImageEditConfig
from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    get_azure_ai_auth_headers,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

AZURE_AI_GPT_IMAGE_EDIT_DEFAULT_API_VERSION: Final = "2025-04-01-preview"


class AzureFoundryGPTImageEditConfig(AzureImageEditConfig):
    """
    Azure AI Foundry gpt-image edit config.

    Foundry serves gpt-image deployments on the same
    ``/openai/deployments/{deployment}/images/edits`` route as Azure OpenAI, so the
    request and URL handling is shared with ``AzureImageEditConfig``. Only the
    credential source and the default api-version are Foundry-specific.
    """

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: llm_http_handler adds extra_headers to the returned dict in place
        params: Final = GenericLiteLLMParams(**(litellm_params or {}))
        return {
            **headers,
            **get_azure_ai_auth_headers(
                api_key=AzureFoundryModelInfo.get_api_key(params.api_key or api_key),
                litellm_params=litellm_params,
                api_key_header="api-key",
            ),
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        foundry_api_base: Final = AzureFoundryModelInfo.get_api_base(api_base)
        if foundry_api_base is None:
            raise ValueError(
                "Azure AI API base is required. Set AZURE_AI_API_BASE environment variable or pass api_base parameter."
            )
        api_version: Final = (
            litellm_params.get("api_version")
            or get_secret_str("AZURE_AI_API_VERSION")
            or litellm.api_version
            or AZURE_AI_GPT_IMAGE_EDIT_DEFAULT_API_VERSION
        )
        return super().get_complete_url(  # pyright: ignore[reportUnknownMemberType]  # bare dict in base
            model=model,
            api_base=foundry_api_base,
            litellm_params={**litellm_params, "api_version": api_version},
        )

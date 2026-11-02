from typing import Final

from litellm.llms.azure.image_edit.transformation import AzureImageEditConfig
from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    get_azure_ai_auth_headers,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams


class AzureFoundryGPTImageEditConfig(AzureImageEditConfig):
    """
    Azure AI Foundry gpt-image edit config.

    Foundry serves gpt-image deployments on the same
    ``/openai/deployments/{deployment}/images/edits`` route as Azure OpenAI, so the
    request and URL handling is shared with ``AzureImageEditConfig``. Only the
    credential source is Foundry-specific.
    """

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: dict[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        params: Final = GenericLiteLLMParams(**(litellm_params or {}))
        resolved_api_key: Final = params.api_key or api_key
        headers.update(
            get_azure_ai_auth_headers(
                api_key=AzureFoundryModelInfo.get_api_key(resolved_api_key),
                litellm_params=litellm_params,
                api_key_header="api-key",
            )
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        azure_ai_api_version: Final = litellm_params.get("api_version") or get_secret_str("AZURE_AI_API_VERSION")
        resolved_params: Final = (
            {**litellm_params, "api_version": azure_ai_api_version}
            if azure_ai_api_version is not None
            else litellm_params
        )
        return super().get_complete_url(  # pyright: ignore[reportUnknownMemberType]  # bare dict in base
            model=model,
            api_base=AzureFoundryModelInfo.get_api_base(api_base),
            litellm_params=resolved_params,
        )

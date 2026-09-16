from typing import Final

import httpx

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    api_key_header_for_base,
    get_azure_ai_auth_headers,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

_PROJECT_PATH_PREFIX: Final = ("api", "projects")
_RESPONSES_PATH: Final = ("openai", "v1", "responses")


def _responses_url(api_base: str) -> str:
    base_url: Final = httpx.URL(api_base)
    segments: Final = tuple(segment for segment in base_url.path.split("/") if segment)
    project_root: Final = segments[:3] if segments[:2] == _PROJECT_PATH_PREFIX else ()
    return str(base_url.copy_with(path="/" + "/".join((*project_root, *_RESPONSES_PATH)), query=None))


class AzureAIResponsesAPIConfig(AzureOpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE_AI

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        params: Final = litellm_params or GenericLiteLLMParams()
        auth_headers: Final = get_azure_ai_auth_headers(
            api_key=AzureFoundryModelInfo.get_api_key(params.api_key),
            litellm_params=params.model_dump(),
            api_key_header=api_key_header_for_base(AzureFoundryModelInfo.get_api_base(params.api_base)),
        )
        return {**headers, **auth_headers, "Content-Type": "application/json"}

    def supports_native_websocket(self) -> bool:
        return False

    def get_complete_url(self, api_base: str | None, litellm_params: dict) -> str:
        resolved_base: Final = AzureFoundryModelInfo.get_api_base(api_base)
        if resolved_base is None:
            raise ValueError(
                "api_base is required for the Azure AI Foundry Responses API. "
                "Set the api_base parameter or the AZURE_AI_API_BASE environment variable."
            )
        return _responses_url(resolved_base)

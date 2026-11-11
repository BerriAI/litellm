from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.hidden_params import set_hidden_param
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.chat.transformation import AzureAIGPT5Config
from litellm.llms.azure_ai.common_utils import (
    AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY,
    AzureFoundryModelInfo,
    api_key_header_for_base,
    foundry_project_root,
    get_azure_ai_auth_headers,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_RESPONSES_PATH: Final = ("openai", "v1", "responses")


def _responses_url(api_base: str) -> str:
    base_url: Final = httpx.URL(api_base)
    return str(base_url.copy_with(path="/" + "/".join((*foundry_project_root(base_url), *_RESPONSES_PATH)), query=None))


class AzureAIResponsesAPIConfig(AzureOpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE_AI

    @staticmethod
    def _supports_reasoning_effort_none(model: str) -> bool:
        return AzureAIGPT5Config.supports_reasoning_effort_level(model, "none")

    @staticmethod
    def _effort_level_is_disabled(model: str, level: str) -> bool:
        return AzureAIGPT5Config.row_disables_reasoning_effort_level(model, level)

    @staticmethod
    def _effort_resolves_to_none(model: str, effort: str | None) -> bool:
        return AzureAIGPT5Config.effort_resolves_to_none(model, effort)

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        params: Final = litellm_params or GenericLiteLLMParams()
        auth_headers: Final = get_azure_ai_auth_headers(
            api_key=AzureFoundryModelInfo.get_api_key(params.api_key),
            litellm_params=params.model_dump(),
            api_key_header=api_key_header_for_base(AzureFoundryModelInfo.get_api_base(params.api_base)),
        )
        return {
            **headers,
            **auth_headers,
            "Content-Type": "application/json",
        }

    def get_stripped_model_name(self, model: str) -> str:
        return AzureFoundryModelInfo.get_base_model(super().get_stripped_model_name(model))

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        if model is not None and AzureFoundryModelInfo.get_azure_ai_route(model) == "model_router":
            return False
        return super().should_fake_stream(model, stream, custom_llm_provider)

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

    def transform_response_api_response(
        self, model: str, raw_response: httpx.Response, logging_obj: "LiteLLMLoggingObj"
    ) -> ResponsesAPIResponse:
        response: Final = super().transform_response_api_response(model, raw_response, logging_obj)
        if response.model and AzureFoundryModelInfo.get_azure_ai_route(model) == "model_router":
            set_hidden_param(response, AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY, f"azure_ai/{response.model}")
        return response

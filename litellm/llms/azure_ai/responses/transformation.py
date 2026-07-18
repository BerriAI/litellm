import httpx

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    azure_ai_use_api_key_header,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import _add_path_to_api_base


class AzureAIResponsesAPIConfig(AzureOpenAIResponsesAPIConfig):
    """Native Responses API config for Azure AI Foundry Models.

    Foundry Models such as the GPT-5 family expose an OpenAI-compatible Responses
    endpoint at `<endpoint>/openai/v1/responses`. Routing here (instead of the
    chat-completions bridge) keeps `reasoning_effort` alongside function tools,
    which Azure rejects on `/chat/completions`.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE_AI

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = AzureFoundryModelInfo.get_api_key(litellm_params.api_key)
        api_base = AzureFoundryModelInfo.get_api_base(litellm_params.api_base)

        if api_key:
            if api_base and azure_ai_use_api_key_header(api_base):
                headers["api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers = BaseAzureLLM._base_validate_azure_environment(headers=headers, litellm_params=litellm_params)

        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = AzureFoundryModelInfo.get_api_base(api_base)
        if api_base is None:
            raise ValueError(
                "api_base is required for Azure AI Foundry Responses API. "
                "Set the api_base parameter or the AZURE_AI_API_BASE environment variable."
            )

        original_url = httpx.URL(api_base)
        query_params = dict(original_url.params)
        api_version = litellm_params.get("api_version")
        if "api-version" not in query_params and isinstance(api_version, str):
            query_params["api-version"] = api_version

        new_url = _add_path_to_api_base(api_base=api_base, ending_path="/openai/v1/responses")
        return str(httpx.URL(new_url).copy_with(params=query_params))

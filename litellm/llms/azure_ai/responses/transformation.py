"""
Azure AI Foundry Agents v2 Responses API configuration.

Uses the project-level Responses API with agent_reference for Foundry Agents v2.
Model format: azure_ai/agents/<agent_name>:<version>
"""

from typing import Final
from urllib.parse import urlencode

from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    is_agents_v2_model,
    parse_agent_reference,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class AzureAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    DEFAULT_API_VERSION: Final = "2025-05-01"

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE_AI

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key: Final = AzureFoundryModelInfo.get_api_key(litellm_params.api_key)

        headers["Content-Type"] = "application/json"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            raise ValueError(
                "api_base is required for Azure AI Foundry Agents v2. "
                "Set AZURE_AI_API_BASE or pass api_base (project endpoint)."
            )

        api_version: Final = litellm_params.get("api_version", self.DEFAULT_API_VERSION)
        normalized_api_base: Final = api_base.rstrip("/")
        query: Final = urlencode({"api-version": api_version})
        return f"{normalized_api_base}/openai/responses?{query}"

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        if not is_agents_v2_model(model):
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

        agent_name, agent_version = parse_agent_reference(model)
        request = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        request.pop("model", None)
        request["agent_reference"] = {
            "name": agent_name,
            "version": agent_version,
            "type": "agent_reference",
        }
        return request

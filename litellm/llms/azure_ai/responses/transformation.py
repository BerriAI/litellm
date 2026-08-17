"""
Azure AI Foundry Agents v2 Responses API configuration.

Uses the project-level Responses API with agent_reference for Foundry Agents v2.
Model format: azure_ai/agents/<agent_name>:<version>
"""

from collections.abc import Mapping
from typing import Final

from litellm.llms.azure.common_utils import BaseAzureLLM
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
        headers: dict,  # mutable-ok: signature fixed by BaseResponsesAPIConfig
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: outbound HTTP headers
        resolved_params: Final = litellm_params or GenericLiteLLMParams()
        api_key: Final = AzureFoundryModelInfo.get_api_key(resolved_params.api_key)

        return {
            **headers,
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: signature fixed by BaseResponsesAPIConfig
    ) -> str:
        resolved_api_base: Final = AzureFoundryModelInfo.get_api_base(api_base)
        if resolved_api_base is None:
            raise ValueError(
                "api_base is required for Azure AI Foundry Agents v2. "
                "Set AZURE_AI_API_BASE or pass api_base (project endpoint)."
            )

        return BaseAzureLLM._get_base_azure_url(  # pyright: ignore[reportPrivateUsage] # shared azure url builder, called this way by every azure config
            api_base=resolved_api_base,
            litellm_params=litellm_params,
            route="/openai/responses",
            default_api_version=self.DEFAULT_API_VERSION,
        )

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,  # mutable-ok: signature fixed by BaseResponsesAPIConfig
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: signature fixed by BaseResponsesAPIConfig
    ) -> dict:  # mutable-ok: JSON request body
        request: Final = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if not is_agents_v2_model(model):
            return request

        agent_name, agent_version = parse_agent_reference(model)
        return {
            **{key: value for key, value in request.items() if key != "model"},
            "agent_reference": {
                "name": agent_name,
                "version": agent_version,
                "type": "agent_reference",
            },
        }

    def merge_extra_body(
        self,
        data: Mapping[str, object],
        extra_body: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: JSON request body
        """Keep the model-derived agent, so `extra_body` cannot invoke a different one."""
        merged: Final = super().merge_extra_body(data=data, extra_body=extra_body)
        agent_reference: Final = data.get("agent_reference")
        if agent_reference is None:
            return merged
        return {**merged, "agent_reference": agent_reference}


def get_azure_ai_responses_api_config(model: str | None) -> AzureAIResponsesAPIConfig | None:
    """Azure AI serves the Responses API for Foundry Agents v2 references only."""
    if model is None or not is_agents_v2_model(model):
        return None
    return AzureAIResponsesAPIConfig()

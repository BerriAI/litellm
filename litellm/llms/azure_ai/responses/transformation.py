from collections.abc import Collection, Mapping
from typing import Final

import httpx

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.chat.transformation import AzureAIGPT5Config
from litellm.llms.azure_ai.common_utils import (
    AZURE_AI_AGENTS_SCOPE,
    AzureFoundryModelInfo,
    api_key_header_for_base,
    get_azure_ai_auth_headers,
)
from litellm.types.llms.azure_ai import FoundryAgentReference
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

_PROJECT_PATH_PREFIX: Final = ("api", "projects")
_RESPONSES_PATH: Final = ("openai", "v1", "responses")
_AGENT_OWNED_PARAMS: Final = frozenset({"instructions", "temperature", "top_p", "tools", "text", "reasoning"})


def _responses_url(api_base: str) -> str:
    base_url: Final = httpx.URL(api_base)
    segments: Final = tuple(segment for segment in base_url.path.split("/") if segment)
    project_root: Final = segments[:3] if segments[:2] == _PROJECT_PATH_PREFIX else ()
    return str(base_url.copy_with(path="/" + "/".join((*project_root, *_RESPONSES_PATH)), query=None))


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


def _agent_reference(model: str) -> FoundryAgentReference:
    name, _, version = model.split("agents/", 1)[1].partition(":")
    if version:
        return FoundryAgentReference(type="agent_reference", name=name, version=version)
    return FoundryAgentReference(type="agent_reference", name=name)


class AzureAIAgentResponsesAPIConfig(AzureAIResponsesAPIConfig):
    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: signature fixed by the override contract
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:  # mutable-ok: same
        params: Final = litellm_params or GenericLiteLLMParams()
        return super().validate_environment(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent returns a bare dict
            headers, model, params.model_copy(update={"azure_scope": params.azure_scope or AZURE_AI_AGENTS_SCOPE})
        )

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: override contract
        inherited: Final[list[str]] = super().get_supported_openai_params(model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent returns a bare list  # mutable-ok: same
        return [param for param in inherited if param not in _AGENT_OWNED_PARAMS]

    def map_openai_params_for_deployment(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
        *,
        declared_reasoning_efforts: Collection[str],
    ) -> Mapping[str, object]:
        mapped: Final = super().map_openai_params_for_deployment(
            response_api_optional_params, model, drop_params, declared_reasoning_efforts=declared_reasoning_efforts
        )
        return {key: value for key, value in mapped.items() if key not in _AGENT_OWNED_PARAMS}

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],  # mutable-ok: signature fixed by the override contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: same
    ) -> dict[str, object]:  # mutable-ok: same
        request: Final[dict[str, object]] = super().transform_responses_api_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent returns a bare dict  # mutable-ok: same
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return {
            **{key: value for key, value in request.items() if key != "model"},
            "agent_reference": _agent_reference(model),
        }

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        return {**extra_body, "agent_reference": _agent_reference(model)}

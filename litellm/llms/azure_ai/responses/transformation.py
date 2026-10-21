import os
from collections.abc import Collection, Mapping, Sequence
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.exceptions import UnsupportedParamsError
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
_AGENT_OWNED_PARAMS: Final = frozenset({"temperature", "top_p", "tools", "text", "reasoning"})
_CALLER_AGENT_KEYS: Final = ("agent", "agent_reference")


def _project_root(base_url: httpx.URL) -> tuple[str, ...]:
    segments: Final = tuple(segment for segment in base_url.path.split("/") if segment)
    return segments[:3] if segments[:2] == _PROJECT_PATH_PREFIX else ()


def _responses_url(api_base: str) -> str:
    base_url: Final = httpx.URL(api_base)
    return str(base_url.copy_with(path="/" + "/".join((*_project_root(base_url), *_RESPONSES_PATH)), query=None))


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
        api_base: Final = AzureFoundryModelInfo.get_api_base(params.api_base)
        on_project: Final = api_base is not None and bool(_project_root(httpx.URL(api_base)))
        auth_params: Final = (
            params.model_copy(
                update={"azure_scope": params.azure_scope or os.getenv("AZURE_SCOPE") or AZURE_AI_AGENTS_SCOPE}
            )
            if on_project
            else params
        )
        auth_headers: Final = get_azure_ai_auth_headers(
            api_key=AzureFoundryModelInfo.get_api_key(params.api_key),
            litellm_params=auth_params.model_dump(),
            api_key_header="Authorization" if on_project else api_key_header_for_base(api_base),
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


_INPUT_ITEMS: Final = TypeAdapter(list[object])


def _with_system_prompt(instructions: str, request_input: object) -> Sequence[object]:
    conversation: Final = (
        [{"type": "message", "role": "user", "content": request_input}]
        if isinstance(request_input, str)
        else _INPUT_ITEMS.validate_python(request_input)
    )
    return [{"type": "message", "role": "system", "content": instructions}, *conversation]


class AzureAIAgentResponsesAPIConfig(AzureAIResponsesAPIConfig):
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
        resends_instructions: Final = bool(mapped.get("instructions")) and bool(mapped.get("previous_response_id"))
        if resends_instructions and not drop_params:
            raise UnsupportedParamsError(
                message=(
                    "azure_ai Foundry agents store a turn's system message with the conversation, so `instructions` "
                    "with `previous_response_id` would add another copy each turn. Send `instructions` on the first "
                    "turn only, or set `drop_params: true` to drop it"
                ),
                llm_provider=LlmProviders.AZURE_AI.value,
                model=model,
            )
        dropped: Final = _AGENT_OWNED_PARAMS | (
            frozenset({"instructions"}) if resends_instructions else frozenset[str]()
        )
        return {key: value for key, value in mapped.items() if key not in dropped}

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
        instructions: Final = request.get("instructions")
        return {
            **{key: value for key, value in request.items() if key not in ("model", "instructions")},
            **(
                {"input": _with_system_prompt(instructions, request.get("input"))}
                if isinstance(instructions, str) and instructions
                else {}
            ),
            "agent_reference": _agent_reference(model),
        }

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        deployment_agent: Final = _agent_reference(model)
        other_agents: Final = tuple(
            key for key in _CALLER_AGENT_KEYS if key in extra_body and extra_body[key] != deployment_agent
        )
        if other_agents and not (litellm.drop_params is True or litellm_params.drop_params is True):
            raise UnsupportedParamsError(
                message=(
                    f"This model entry always calls Foundry agent {deployment_agent['name']!r}, so `{other_agents[0]}` "
                    "in the request cannot pick another agent or version. Remove it, call the model entry for that "
                    "agent, or set `drop_params: true` to drop it"
                ),
                llm_provider=LlmProviders.AZURE_AI.value,
                model=model,
            )
        return {
            **{key: value for key, value in extra_body.items() if key not in _CALLER_AGENT_KEYS},
            "agent_reference": deployment_agent,
        }

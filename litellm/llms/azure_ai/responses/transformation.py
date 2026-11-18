import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import TypeAdapter

from litellm.exceptions import UnsupportedParamsError
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
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_RESPONSES_PATH: Final = ("openai", "v1", "responses")
_RESPONSE_FIELDS: Final = TypeAdapter(dict[str, object])
_REQUEST_TOOLS: Final = TypeAdapter(tuple[object, ...])
_TOOL_PARAMS: Final = frozenset({"tools", "tool_choice"})
_INPUT_ITEMS: Final = TypeAdapter(list[object])


def _responses_url(api_base: str) -> str:
    base_url: Final = httpx.URL(api_base)
    return str(base_url.copy_with(path="/" + "/".join((*foundry_project_root(base_url), *_RESPONSES_PATH)), query=None))


def _is_custom_tool(tool: object) -> bool:
    match tool:
        case {"type": "custom"}:
            return True
        case _:
            return False


def _without_custom_members(tool: object) -> object | None:
    match tool:
        case {"type": "custom"}:
            return None
        case {"type": "namespace" | "allowed_tools", "tools": list()}:
            fields: Final = _RESPONSE_FIELDS.validate_python(tool)
            members: Final = _REQUEST_TOOLS.validate_python(fields["tools"])
            kept: Final = [member for member in members if not _is_custom_tool(member)]
            if not kept:
                return None
            if len(kept) == len(members):
                return tool
            return {**fields, "tools": kept}
        case _:
            return tool


def _without_custom_tools(mapped: Mapping[str, object], model: str, drop_params: bool) -> Mapping[str, object]:
    tools: Final = _REQUEST_TOOLS.validate_python(mapped.get("tools") or ())
    stripped: Final = tuple(_without_custom_members(tool) for tool in tools)
    if all(kept is tool for kept, tool in zip(stripped, tools)):
        return mapped
    if not drop_params:
        raise UnsupportedParamsError(
            message=(
                "azure_ai Model Router rejects `custom` tools on the Responses API. Send them as function tools, or "
                "set `drop_params: true` to drop them"
            ),
            llm_provider=LlmProviders.AZURE_AI.value,
            model=model,
        )
    kept_tools: Final = [tool for tool in stripped if tool is not None]
    tool_choice: Final = _without_custom_members(mapped.get("tool_choice"))
    rest: Final = {key: value for key, value in mapped.items() if key not in _TOOL_PARAMS}
    if not kept_tools:
        return rest
    return {**rest, "tools": kept_tools, **({} if tool_choice is None else {"tool_choice": tool_choice})}


def _function_call_item(item: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "type": "function_call",
        "call_id": item.get("call_id"),
        "name": item.get("name"),
        "arguments": json.dumps({"content": item.get("input") or ""}),
    }


def _function_call_output_item(item: Mapping[str, object]) -> Mapping[str, object]:
    return {"type": "function_call_output", "call_id": item.get("call_id"), "output": item.get("output")}


def _router_input_item(item: object) -> object:
    match item:
        case {"type": "custom_tool_call"}:
            return _function_call_item(_RESPONSE_FIELDS.validate_python(item))
        case {"type": "custom_tool_call_output"}:
            return _function_call_output_item(_RESPONSE_FIELDS.validate_python(item))
        case _:
            return item


def _with_router_input(model: str, request: Mapping[str, object]) -> Mapping[str, object]:
    request_input: Final = request.get("input")
    if not isinstance(request_input, list) or AzureFoundryModelInfo.get_azure_ai_route(model) != "model_router":
        return request
    return {**request, "input": [_router_input_item(item) for item in _INPUT_ITEMS.validate_python(request_input)]}


def _stamp_routed_model(model: str, response: ResponsesAPIResponse) -> None:
    if response.model and AzureFoundryModelInfo.get_azure_ai_route(model) == "model_router":
        set_hidden_param(response, AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY, f"azure_ai/{response.model}")


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

    def transform_compact_response_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],  # mutable-ok: signature fixed by the override contract
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: same
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: return type fixed by the override contract
        compact: Final[tuple[str, dict[str, object]]] = super().transform_compact_response_api_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent types the body as a bare dict  # mutable-ok: same
            model=self.get_stripped_model_name(model),
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        url, body = compact
        return url, dict(_with_router_input(model, body))

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
        return dict(_with_router_input(model, request))

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        if model is not None and AzureFoundryModelInfo.get_azure_ai_route(model) == "model_router":
            return False
        return super().should_fake_stream(model, stream, custom_llm_provider)

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: return type fixed by the override contract
        mapped: Final[dict[str, object]] = super().map_openai_params(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent types the mapped params as a bare dict  # mutable-ok: same
            response_api_optional_params, model, drop_params
        )
        if AzureFoundryModelInfo.get_azure_ai_route(model) != "model_router":
            return mapped
        return dict(_without_custom_tools(mapped, model, drop_params))

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
        _stamp_routed_model(model, response)
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: "LiteLLMLoggingObj",
    ) -> ResponsesAPIStreamingResponse:
        event: Final = super().transform_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # parent types parsed_chunk as a bare dict
            model=model, parsed_chunk=dict(parsed_chunk), logging_obj=logging_obj
        )
        response: Final = getattr(event, "response", None)
        if isinstance(response, ResponsesAPIResponse):
            _stamp_routed_model(model, response)
        return event

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.hidden_params import set_hidden_param
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.chat.transformation import AzureAIGPT5Config
from litellm.llms.azure_ai.common_utils import (
    AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY,
    AzureFoundryModelInfo,
    api_key_header_for_base,
    foundry_project_root,
    get_azure_ai_auth_headers,
    with_foundry_project_scope,
    without_api_version,
)
from litellm.responses.litellm_completion_transformation.custom_tools import (
    CustomToolKey,
    convert_custom_tool_to_function_tool,
    extract_custom_tool_keys,
    unwrap_custom_tool_arguments,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
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
_ITEMS: Final = TypeAdapter(list[object])
_UNENCODED_HEADERS: Final = frozenset({"content-encoding", "content-length"})


def _responses_url(api_base: str) -> str:
    base_url: Final = httpx.URL(api_base)
    responses_path: Final = "/" + "/".join((*foundry_project_root(base_url), *_RESPONSES_PATH))
    return str(base_url.copy_with(path=responses_path, params=without_api_version(base_url)))


def _router_function_tool(tool: object) -> object:
    if not isinstance(tool, Mapping):
        return tool
    fields: Final = _RESPONSE_FIELDS.validate_python(tool)
    function_tool: Final = convert_custom_tool_to_function_tool(fields)
    if function_tool is None:
        return fields
    return {**{key: value for key, value in function_tool.items() if key != "function"}, **function_tool["function"]}


def _router_tool(tool: object) -> object:
    match tool:
        case {"type": "namespace", "tools": list()}:
            fields: Final = _RESPONSE_FIELDS.validate_python(tool)
            members: Final = _REQUEST_TOOLS.validate_python(fields["tools"])
            return {**fields, "tools": [_router_function_tool(member) for member in members]}
        case _:
            return _router_function_tool(tool)


def _router_choice_member(member: object) -> object:
    match member:
        case {"type": "custom"}:
            return {**_RESPONSE_FIELDS.validate_python(member), "type": "function"}
        case _:
            return member


def _router_tool_choice(tool_choice: object) -> object:
    match tool_choice:
        case {"type": "allowed_tools", "tools": list()}:
            fields: Final = _RESPONSE_FIELDS.validate_python(tool_choice)
            members: Final = _REQUEST_TOOLS.validate_python(fields["tools"])
            return {**fields, "tools": [_router_choice_member(member) for member in members]}
        case _:
            return _router_choice_member(tool_choice)


def _function_call_item(item: Mapping[str, object]) -> Mapping[str, object]:
    namespace: Final = item.get("namespace")
    return {
        "type": "function_call",
        "call_id": item.get("call_id"),
        "name": item.get("name"),
        **({"namespace": namespace} if isinstance(namespace, str) else {}),
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


def _router_request(model: str, request: Mapping[str, object]) -> Mapping[str, object]:
    if AzureFoundryModelInfo.get_azure_ai_route(model) != "model_router":
        return request
    request_input: Final = request.get("input")
    tools: Final = request.get("tools")
    return {
        **request,
        **(
            {"input": [_router_input_item(item) for item in _ITEMS.validate_python(request_input)]}
            if isinstance(request_input, list)
            else {}
        ),
        **(
            {"tools": [_router_tool(tool) for tool in _REQUEST_TOOLS.validate_python(tools)]}
            if isinstance(tools, list)
            else {}
        ),
        **({"tool_choice": _router_tool_choice(request["tool_choice"])} if "tool_choice" in request else {}),
    }


class _RequestedTools(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    tools: tuple[object, ...] | None = None


def _custom_tool_keys(model: str, logging_obj: "LiteLLMLoggingObj") -> frozenset[CustomToolKey]:
    if AzureFoundryModelInfo.get_azure_ai_route(model) != "model_router":
        return frozenset()
    requested: Final = _RequestedTools.model_validate(logging_obj.model_call_details.get("optional_params") or {})
    return extract_custom_tool_keys(requested.tools)


def _call_key(fields: Mapping[str, object], name: str) -> CustomToolKey:
    namespace: Final = fields.get("namespace")
    return (namespace if isinstance(namespace, str) else None, name)


def _caller_output_item(item: object, custom_tool_keys: frozenset[CustomToolKey]) -> object:
    match item:
        case {"type": "function_call", "name": str() as name}:
            fields: Final = _RESPONSE_FIELDS.validate_python(item)
            if _call_key(fields, name) not in custom_tool_keys:
                return item
            arguments: Final = fields.get("arguments")
            return {
                **{key: value for key, value in fields.items() if key != "arguments"},
                "type": "custom_tool_call",
                "input": unwrap_custom_tool_arguments(arguments if isinstance(arguments, str) else ""),
            }
        case _:
            return item


def _caller_response(
    response: Mapping[str, object], custom_tool_keys: frozenset[CustomToolKey]
) -> Mapping[str, object]:
    output: Final = response.get("output")
    if not custom_tool_keys or not isinstance(output, list):
        return response
    return {
        **response,
        "output": [_caller_output_item(item, custom_tool_keys) for item in _ITEMS.validate_python(output)],
    }


def _with_custom_tool_calls(raw_response: httpx.Response, custom_tool_keys: frozenset[CustomToolKey]) -> httpx.Response:
    if not custom_tool_keys:
        return raw_response
    try:
        body: Final = _RESPONSE_FIELDS.validate_json(raw_response.content)
    except ValidationError:
        return raw_response
    return httpx.Response(
        status_code=raw_response.status_code,
        headers={key: value for key, value in raw_response.headers.items() if key.lower() not in _UNENCODED_HEADERS},
        json=_caller_response(body, custom_tool_keys),
        request=raw_response.request,
    )


def _caller_event(model: str, event: Mapping[str, object], logging_obj: "LiteLLMLoggingObj") -> Mapping[str, object]:
    match event:
        case {"type": "response.output_item.done", "item": item}:
            return {**event, "item": _caller_output_item(item, _custom_tool_keys(model, logging_obj))}
        case {"response": Mapping()}:
            fields: Final = _RESPONSE_FIELDS.validate_python(event["response"])
            return {**event, "response": _caller_response(fields, _custom_tool_keys(model, logging_obj))}
        case _:
            return event


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
        api_base: Final = AzureFoundryModelInfo.get_api_base(params.api_base)
        on_project: Final = api_base is not None and bool(foundry_project_root(httpx.URL(api_base)))
        auth_params: Final = with_foundry_project_scope(params, api_base)
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
        return url, dict(_router_request(model, body))

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
        return dict(_router_request(model, request))

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
        response: Final = super().transform_response_api_response(
            model, _with_custom_tool_calls(raw_response, _custom_tool_keys(model, logging_obj)), logging_obj
        )
        _stamp_routed_model(model, response)
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: "LiteLLMLoggingObj",
    ) -> ResponsesAPIStreamingResponse:
        event: Final = super().transform_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # parent types parsed_chunk as a bare dict
            model=model, parsed_chunk=dict(_caller_event(model, parsed_chunk, logging_obj)), logging_obj=logging_obj
        )
        response: Final = getattr(event, "response", None)
        if isinstance(response, ResponsesAPIResponse):
            _stamp_routed_model(model, response)
        return event

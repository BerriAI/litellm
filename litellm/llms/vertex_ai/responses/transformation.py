import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

import httpx
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.openai.chat.gpt_reasoning_transformation import unsupported_reasoning_effort_message
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig, drops_params
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase, graft_default_vertex_path
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse, ResponsesAPIStreamingResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_FIELDS: Final = TypeAdapter(dict[str, object])
_PARAM_NAMES: Final = TypeAdapter(tuple[str, ...])
_OBJECTS: Final = TypeAdapter(tuple[object, ...])
_CREDENTIALS: Final[TypeAdapter[str | Mapping[str, JsonValue] | None]] = TypeAdapter(
    str | Mapping[str, JsonValue] | None
)
_AUTH: Final = VertexBase()
_MUSE_PREFIX: Final = "meta/muse-"
_STORED_RESPONSE_PARAMS: Final = frozenset({"background", "conversation", "previous_response_id"})
_MUSE_TOOL_TYPES: Final = frozenset({"function", "namespace"})
_NO_NAMESPACES: Final[Mapping[str, frozenset[str]]] = MappingProxyType({})
_MUSE_PUBLISHER_PATH: Final = "/publishers/meta/v1"
_MUSE_RESPONSES_PATH: Final = f"{_MUSE_PUBLISHER_PATH}/responses"
MUSE_CHAT_PARAMS: Final = (
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "user",
    "store",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "extra_headers",
)


class _SentMember(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str | None = None


class _SentTool(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str | None = None
    name: str | None = None
    tools: tuple[_SentMember, ...] = ()


class _SentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    tools: tuple[_SentTool, ...] = ()


class _SentArgs(BaseModel):
    model_config = ConfigDict(frozen=True)
    complete_input_dict: _SentRequest = _SentRequest()


class _LoggedCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    additional_args: _SentArgs = _SentArgs()


def is_vertex_muse_model(model: str | None) -> bool:
    return (model or "").removeprefix("vertex_ai/").startswith(_MUSE_PREFIX)


def _reasoning_fields(reasoning: object) -> Mapping[str, object]:
    return _FIELDS.validate_python(reasoning) if isinstance(reasoning, Mapping) else {}


def _keeps_muse_tool(tool: object) -> bool:
    return not isinstance(tool, Mapping) or _FIELDS.validate_python(tool).get("type") in _MUSE_TOOL_TYPES


def _muse_tools(tools: object) -> tuple[object, ...] | None:
    if not isinstance(tools, list | tuple):
        return None
    return tuple(tool for tool in _OBJECTS.validate_python(tools) if _keeps_muse_tool(tool))


def _refused_reasoning_effort(model: str, reasoning: Mapping[str, object]) -> str | None:
    from litellm.utils import is_explicitly_disabled_factory

    effort: Final = reasoning.get("effort")
    if not isinstance(effort, str):
        return None
    disabled: Final = is_explicitly_disabled_factory(
        model=f"vertex_ai/{model.removeprefix('vertex_ai/')}",
        custom_llm_provider=None,
        key=f"supports_{effort}_reasoning_effort",
    )
    return effort if disabled else None


def _sent_namespaces(logging_obj: "Logging") -> Mapping[str, frozenset[str]]:
    try:
        tools: Final = _LoggedCall.model_validate(
            logging_obj.model_call_details
        ).additional_args.complete_input_dict.tools
    except ValidationError:
        return _NO_NAMESPACES
    return {
        tool.name: frozenset(member.name for member in tool.tools if member.name)
        for tool in tools
        if tool.type == "namespace" and tool.name
    }


def _split_member_name(name: str, namespaces: Mapping[str, frozenset[str]]) -> tuple[str, str] | None:
    return next(
        (
            (namespace, name.removeprefix(f"{namespace}."))
            for namespace, members in namespaces.items()
            if name.startswith(f"{namespace}.") and name.removeprefix(f"{namespace}.") in members
        ),
        None,
    )


def _namespaced_call(item: object, namespaces: Mapping[str, frozenset[str]]) -> object:
    if isinstance(item, ResponseFunctionToolCall):
        typed_split: Final = None if item.namespace else _split_member_name(item.name, namespaces)
        return (
            item
            if typed_split is None
            else item.model_copy(update={"namespace": typed_split[0], "name": typed_split[1]})
        )
    if not isinstance(item, Mapping):
        return item
    fields: Final = _FIELDS.validate_python(item)
    name: Final = fields.get("name")
    split: Final = (
        _split_member_name(name, namespaces)
        if fields.get("type") == "function_call" and not fields.get("namespace") and isinstance(name, str)
        else None
    )
    return fields if split is None else {**fields, "namespace": split[0], "name": split[1]}


def _namespaced_output(response: object, namespaces: Mapping[str, frozenset[str]]) -> object:
    if not isinstance(response, Mapping):
        return response
    fields: Final = _FIELDS.validate_python(response)
    output: Final = fields.get("output")
    if not isinstance(output, list | tuple):
        return fields
    return {**fields, "output": tuple(_namespaced_call(item, namespaces) for item in _OBJECTS.validate_python(output))}


def _namespaced_event(chunk: Mapping[str, object], namespaces: Mapping[str, frozenset[str]]) -> Mapping[str, object]:
    return {
        **chunk,
        **({"item": _namespaced_call(chunk["item"], namespaces)} if "item" in chunk else {}),
        **({"response": _namespaced_output(chunk["response"], namespaces)} if "response" in chunk else {}),
    }


class VertexAIMuseResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self, vertex_auth: VertexBase | None = None) -> None:
        super().__init__()
        self._vertex_auth = vertex_auth or _AUTH

    @classmethod
    def for_model(cls, model: str | None) -> "VertexAIMuseResponsesAPIConfig | None":
        return cls() if is_vertex_muse_model(model) else None

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base API requires a list return
        names: Final = _PARAM_NAMES.validate_python(
            super().get_supported_openai_params(  # pyright: ignore[reportUnknownMemberType]  # legacy list is untyped
                model
            )
        )
        return [name for name in names if name not in _STORED_RESPONSE_PARAMS]

    def _access_token(self, params: Mapping[str, object]) -> tuple[str, str]:
        fields: Final = dict(params)
        credentials: Final = _CREDENTIALS.validate_python(
            VertexBase.safe_get_vertex_ai_credentials(  # pyright: ignore[reportUnknownMemberType]  # untyped bag
                fields
            )
        )
        return self._vertex_auth.ensure_access_token(
            credentials=json.dumps(dict(credentials)) if isinstance(credentials, Mapping) else credentials,
            project_id=VertexBase.safe_get_vertex_ai_project(  # pyright: ignore[reportUnknownMemberType]  # untyped bag
                fields
            ),
            custom_llm_provider="vertex_ai",
        )

    def validate_environment(
        self, headers: Mapping[str, str], model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, str]:  # mutable-ok: Responses handler mutates the returned headers
        params: Final = litellm_params or GenericLiteLLMParams()
        token, _ = self._access_token(_FIELDS.validate_python(params.model_dump()))
        return {**headers, "Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        explicit_project: Final = VertexBase.safe_get_vertex_ai_project(  # pyright: ignore[reportUnknownMemberType]  # untyped bag
            dict(litellm_params)
        )
        project: Final = explicit_project or self._access_token(litellm_params)[1]
        default_url: Final = (
            f"https://aiplatform.googleapis.com/v1/projects/{encode_url_path_segment(project)}"
            "/locations/global/publishers/meta/v1/responses"
        )
        if not api_base:
            return default_url
        parsed_api_base: Final = urlparse(api_base)
        api_base_path: Final = parsed_api_base.path.rstrip("/")
        if api_base_path.endswith(_MUSE_RESPONSES_PATH):
            return api_base
        if api_base_path.endswith(_MUSE_PUBLISHER_PATH):
            return parsed_api_base._replace(path=api_base_path + "/responses").geturl()
        if api_base_path.endswith(("/v1", "/v1beta1")):
            return graft_default_vertex_path(api_base, default_url)
        return parsed_api_base._replace(path=api_base_path + urlparse(default_url).path).geturl()

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        merged: Final = super().transform_extra_body(extra_body, request, model, litellm_params)
        return {key: value for key, value in merged.items() if key != "model"}

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: Responses handler mutates the returned request
        drops_unsupported: Final = drops_params(litellm_params)
        if response_api_optional_request_params.get("store") is True and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse supports stateless Responses requests only. Set store=False or drop_params=True.",
            )
        location: Final = VertexBase.explicit_vertex_ai_location(_FIELDS.validate_python(litellm_params.model_dump()))
        if location not in (None, "global") and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse is served from the global location only. Set vertex_location to global "
                "or drop_params=True.",
            )
        truncates: Final = response_api_optional_request_params.get("truncation") not in (None, "disabled")
        if truncates and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse supports only truncation='disabled'. Set drop_params=True to omit it.",
            )
        tool_choice: Final = response_api_optional_request_params.get("tool_choice")
        if tool_choice is not None and tool_choice != "auto" and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message="Vertex Muse supports only tool_choice='auto'. Set drop_params=True to normalize unsupported choices.",
            )
        reasoning: Final = _reasoning_fields(response_api_optional_request_params.get("reasoning"))
        refused_effort: Final = _refused_reasoning_effort(model, reasoning)
        if refused_effort is not None and not drops_unsupported:
            raise UnsupportedParamsError(
                status_code=400,
                message=unsupported_reasoning_effort_message("reasoning.effort", refused_effort, model),
            )
        kept_reasoning: Final = {key: value for key, value in reasoning.items() if key != "effort"}
        kept_tools: Final = (
            _muse_tools(response_api_optional_request_params.get("tools")) if drops_unsupported else None
        )
        dropped_params: Final = (
            (_STORED_RESPONSE_PARAMS if drops_unsupported else frozenset[str]())
            | ({"tools"} if tool_choice == "none" or kept_tools == () else frozenset[str]())
            | ({"truncation"} if truncates else frozenset[str]())
            | ({"reasoning"} if refused_effort is not None else frozenset[str]())
        )
        tool_params: Final = {
            key: value for key, value in response_api_optional_request_params.items() if key not in dropped_params
        }
        normalized_params: Final = {
            **tool_params,
            **({"tools": kept_tools} if kept_tools and "tools" in tool_params else {}),
            **({"tool_choice": "auto"} if tool_choice is not None else {}),
            **({"reasoning": kept_reasoning} if refused_effort is not None and kept_reasoning else {}),
        }
        request: Final = _FIELDS.validate_python(
            super().transform_responses_api_request(  # pyright: ignore[reportUnknownMemberType]  # untyped bags
                model, input, normalized_params, litellm_params, dict(headers)
            )
        )
        return {
            **request,
            "model": model.removeprefix("vertex_ai/").removeprefix("meta/"),
            "store": False,
        }

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> ResponsesAPIResponse:
        namespaces: Final = _sent_namespaces(logging_obj)
        response: Final = super().transform_response_api_response(model, raw_response, logging_obj)
        if not namespaces:
            return response
        return response.model_copy(
            update={
                "output": [_namespaced_call(item, namespaces) for item in _OBJECTS.validate_python(response.output)]
            }
        )

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: "Logging",
    ) -> ResponsesAPIStreamingResponse:
        carries_items: Final = "item" in parsed_chunk or "response" in parsed_chunk
        namespaces: Final = _sent_namespaces(logging_obj) if carries_items else _NO_NAMESPACES
        return super().transform_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # base takes an untyped dict
            model, dict(_namespaced_event(parsed_chunk, namespaces) if namespaces else parsed_chunk), logging_obj
        )

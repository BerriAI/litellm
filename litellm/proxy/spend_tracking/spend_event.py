"""Compact, typed success event handed from an inference worker to the spend sidecar.

``build_spend_event`` runs on the inference worker right after ``Logging.async_success_handler``
has built the ``standard_logging_object`` (so the cost is already known). It validates the success
callback's ``kwargs`` into the projection ``_PROXY_track_cost_callback`` and
``DBSpendUpdateWriter.update_database`` actually read: identities and metadata, timings, usage, the
standard logging payload without its prompt/response bodies, and the tool names. The request
messages, the raw ``proxy_server_request`` body and the full response travel only when spend logs
are configured to store prompts and responses. The cache key is the preset key the caching layer
already computed, never a fresh hash over the request body.

``spend_event_callback_args`` rebuilds the ``(kwargs, response_obj, start_time, end_time)`` tuple
the existing cost pipeline consumes, so the sidecar runs the unchanged pipeline against the event.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.spend_log_tool_index import response_tool_call_names
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.utils import LiteLLMBatch, Usage

SPEND_EVENT_VERSION: Final = 1
CACHE_OFF_KEY: Final = "Cache OFF"

ObjectMapping: TypeAlias = Mapping[str, object]

_UNSERIALIZABLE_METADATA_KEYS: Final = frozenset({"user_api_key_auth", "litellm_parent_otel_span"})
_STANDARD_LOGGING_BODY_KEYS: Final = frozenset({"messages", "response"})
_STANDARD_LOGGING_DROPPED_KEYS: Final = frozenset({"model_parameters"})
_NOT_OFFLOADED_RESPONSE_TYPES: Final = (LiteLLMBatch, InteractionsAPIResponse)


class _LitellmParams(TypedDict, total=False):
    api_base: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    litellm_call_id: ReadOnly[str | None]
    user_api_key_end_user_id: ReadOnly[str | None]
    metadata: ReadOnly[ObjectMapping | None]
    litellm_metadata: ReadOnly[ObjectMapping | None]
    proxy_server_request: ReadOnly[ObjectMapping | None]
    preset_cache_key: ReadOnly[str | None]


class _DynamicParams(TypedDict, total=False):
    turn_off_message_logging: ReadOnly[bool | None]


class _RequestBody(TypedDict, total=False):
    tools: ReadOnly[Sequence[ObjectMapping] | None]


class _PassthroughPayload(TypedDict, total=False):
    request_body: ReadOnly[_RequestBody | None]


class _ToolCallFunction(TypedDict):
    name: ReadOnly[str]
    arguments: ReadOnly[str]


class _ToolCall(TypedDict):
    id: ReadOnly[str | None]
    type: ReadOnly[Literal["function"]]
    function: ReadOnly[_ToolCallFunction]


class _ToolCallMessage(TypedDict):
    role: ReadOnly[Literal["assistant"]]
    content: ReadOnly[None]
    tool_calls: ReadOnly[Sequence[_ToolCall]]


class _ToolCallChoice(TypedDict):
    index: ReadOnly[int]
    finish_reason: ReadOnly[Literal["tool_calls"]]
    message: ReadOnly[_ToolCallMessage]


class CompactResponse(TypedDict, total=False):
    """What the spend pipeline reads off a response: its id, usage and which tools it called."""

    id: ReadOnly[object]
    model: ReadOnly[object]
    usage: ReadOnly[object]
    usage_info: ReadOnly[object]
    status: ReadOnly[object]
    background: ReadOnly[object]
    choices: ReadOnly[Sequence[_ToolCallChoice]]


class _SuccessKwargs(TypedDict, total=False):
    """The success callback's ``kwargs`` (``Logging.model_call_details``), validated and projected."""

    litellm_call_id: ReadOnly[str | None]
    call_type: ReadOnly[str | None]
    model: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    stream: ReadOnly[bool | None]
    complete_streaming_response: ReadOnly[object]
    cache_hit: ReadOnly[bool | None]
    response_cost: ReadOnly[float | None]
    completion_start_time: ReadOnly[datetime | None]
    agent_id: ReadOnly[str | None]
    litellm_trace_id: ReadOnly[str | None]
    litellm_params: ReadOnly[_LitellmParams]
    standard_logging_object: ReadOnly[ObjectMapping | None]
    standard_callback_dynamic_params: ReadOnly[_DynamicParams | None]
    combined_usage_object: ReadOnly[Usage | None]
    realtime_tools: ReadOnly[Sequence[object] | None]
    realtime_tool_calls: ReadOnly[Sequence[object] | None]
    tools: ReadOnly[Sequence[ObjectMapping] | None]
    passthrough_logging_payload: ReadOnly[_PassthroughPayload | None]


class _FunctionToolFunction(TypedDict):
    name: ReadOnly[str]


class _FunctionTool(TypedDict):
    type: ReadOnly[Literal["function"]]
    function: ReadOnly[_FunctionToolFunction]


class SpendCallbackKwargs(TypedDict):
    """The ``kwargs`` handed to ``_PROXY_track_cost_callback`` on the sidecar."""

    litellm_call_id: ReadOnly[str | None]
    call_type: ReadOnly[str | None]
    model: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    stream: ReadOnly[bool | None]
    cache_hit: ReadOnly[bool | None]
    response_cost: ReadOnly[float | None]
    completion_start_time: ReadOnly[datetime | None]
    agent_id: ReadOnly[str | None]
    litellm_trace_id: ReadOnly[str | None]
    litellm_params: ReadOnly[_LitellmParams]
    standard_logging_object: ReadOnly[ObjectMapping | None]
    standard_callback_dynamic_params: ReadOnly[_DynamicParams | None]
    combined_usage_object: ReadOnly[Usage | None]
    realtime_tools: ReadOnly[Sequence[object] | None]
    realtime_tool_calls: ReadOnly[Sequence[object] | None]
    tools: ReadOnly[Sequence[_FunctionTool] | None]
    complete_streaming_response: NotRequired[ReadOnly[CompactResponse | None]]


_NO_LITELLM_PARAMS: Final[_LitellmParams] = {}
_SUCCESS_KWARGS: Final = TypeAdapter(_SuccessKwargs)
_OBJECT_MAPPING: Final = TypeAdapter(ObjectMapping)
_COMPACT_RESPONSE: Final = TypeAdapter(CompactResponse)


class SpendEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1]
    litellm_call_id: str | None
    call_type: str | None
    model: str | None
    custom_llm_provider: str | None
    stream: bool | None
    complete_streaming_response: bool
    cache_hit: bool | None
    response_cost: float | None
    start_time: datetime
    end_time: datetime
    completion_start_time: datetime | None
    agent_id: str | None
    litellm_trace_id: str | None
    litellm_params: _LitellmParams
    standard_logging_object: ObjectMapping | None
    standard_callback_dynamic_params: _DynamicParams | None
    response: CompactResponse | None
    combined_usage: ObjectMapping | None
    realtime_tools: Sequence[object] | None
    realtime_tool_calls: Sequence[object] | None
    request_tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpendEventCallbackArgs:
    kwargs: SpendCallbackKwargs
    response_obj: CompactResponse | None
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True, slots=True)
class SpendEventBuildError:
    reason: str


@dataclass(frozen=True, slots=True)
class SpendEventDecodeError:
    reason: str


def is_offloadable_success(response_obj: object) -> bool:
    """Batch retrieves and interaction polls branch on the concrete response class, so they stay in-process."""
    return not isinstance(response_obj, _NOT_OFFLOADED_RESPONSE_TYPES)


def _json_fallback(value: object) -> str:
    return str(value)


def _mapping_or_none(value: object) -> ObjectMapping | None:
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return None


def _drop_keys(mapping: ObjectMapping, keys: frozenset[str]) -> ObjectMapping:
    return MappingProxyType({key: value for key, value in mapping.items() if key not in keys})


def _budget_reservation(metadata: ObjectMapping) -> ObjectMapping | None:
    """The admission-time reservation, wherever the request setup left it, so the sidecar can reconcile it."""
    direct: Final = _mapping_or_none(metadata.get("user_api_key_budget_reservation"))
    if direct is not None:
        return direct
    auth: Final = metadata.get("user_api_key_auth")
    if isinstance(auth, UserAPIKeyAuth):
        return auth.budget_reservation
    auth_mapping: Final = _mapping_or_none(auth)
    return _mapping_or_none(auth_mapping.get("budget_reservation")) if auth_mapping is not None else None


def _metadata_for_event(
    metadata: ObjectMapping | None, budget_reservation: ObjectMapping | None
) -> ObjectMapping | None:
    if metadata is None:
        return None
    kept: Final = _drop_keys(metadata, _UNSERIALIZABLE_METADATA_KEYS)
    if budget_reservation is None:
        return kept
    return MappingProxyType({**kept, "user_api_key_budget_reservation": budget_reservation})


def _litellm_params_for_event(
    litellm_params: _LitellmParams, cache_key: str | None, store_bodies: bool
) -> _LitellmParams:
    metadata: Final = litellm_params.get("metadata")
    litellm_metadata: Final = litellm_params.get("litellm_metadata")
    budget_reservation: Final = next(
        (
            reservation
            for source in (litellm_metadata, metadata)
            if source is not None and (reservation := _budget_reservation(source)) is not None
        ),
        None,
    )
    projected: Final[_LitellmParams] = {
        "api_base": litellm_params.get("api_base"),
        "custom_llm_provider": litellm_params.get("custom_llm_provider"),
        "litellm_call_id": litellm_params.get("litellm_call_id"),
        "user_api_key_end_user_id": litellm_params.get("user_api_key_end_user_id"),
        "metadata": _metadata_for_event(metadata, budget_reservation),
        "litellm_metadata": _metadata_for_event(litellm_metadata, budget_reservation),
        "proxy_server_request": litellm_params.get("proxy_server_request") if store_bodies else None,
        "preset_cache_key": cache_key,
    }
    return projected


def _standard_logging_for_event(sl_object: ObjectMapping | None, store_bodies: bool) -> ObjectMapping | None:
    if sl_object is None:
        return None
    dropped: Final = (
        _STANDARD_LOGGING_DROPPED_KEYS if store_bodies else _STANDARD_LOGGING_DROPPED_KEYS | _STANDARD_LOGGING_BODY_KEYS
    )
    return _drop_keys(sl_object, dropped)


def _tool_call(name: str) -> _ToolCall:
    tool_call: Final[_ToolCall] = {"id": None, "type": "function", "function": {"name": name, "arguments": "{}"}}
    return tool_call


def _tool_call_choice(names: Sequence[str]) -> _ToolCallChoice:
    choice: Final[_ToolCallChoice] = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": None, "tool_calls": tuple(_tool_call(name) for name in names)},
    }
    return choice


def _compact_response(response_obj: object) -> CompactResponse | None:
    """Usage, identity and tool calls of the response, in chat-completions shape, without the content."""
    dumped: Final = response_obj.model_dump() if isinstance(response_obj, BaseModel) else _mapping_or_none(response_obj)
    if dumped is None:
        return None
    scalars: Final = _COMPACT_RESPONSE.validate_python(_drop_keys(dumped, frozenset({"choices"})))
    tool_call_names: Final = response_tool_call_names(response_obj)
    if not tool_call_names:
        return scalars
    with_tool_calls: Final[CompactResponse] = {**scalars, "choices": (_tool_call_choice(tool_call_names),)}
    return with_tool_calls


def _tool_name(tool: ObjectMapping) -> str | None:
    """Chat tools nest the name under ``function``; Anthropic and Responses API tools keep it at the top."""
    function: Final = _mapping_or_none(tool.get("function"))
    name: Final = function.get("name") if function is not None else tool.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _request_tool_names(kwargs: _SuccessKwargs) -> tuple[str, ...]:
    passthrough: Final = kwargs.get("passthrough_logging_payload")
    request_body: Final = passthrough.get("request_body") if passthrough is not None else None
    passthrough_tools: Final = request_body.get("tools") if request_body is not None else None
    return tuple(
        name
        for source in (kwargs.get("tools"), passthrough_tools)
        if source is not None
        for tool in source
        if (name := _tool_name(tool)) is not None
    )


def preset_spend_log_cache_key(litellm_params: _LitellmParams) -> str | None:
    """The key the caching layer already stored in ``litellm_params``, or ``Cache OFF``; never hashes the body."""
    if litellm.cache is None:
        return CACHE_OFF_KEY
    return litellm_params.get("preset_cache_key")


def _function_tool(name: str) -> _FunctionTool:
    tool: Final[_FunctionTool] = {"type": "function", "function": {"name": name}}
    return tool


def build_spend_event(
    raw_kwargs: ObjectMapping, response_obj: object, start_time: datetime, end_time: datetime, store_bodies: bool
) -> bytes | SpendEventBuildError:
    """Validate the success callback's kwargs and serialize the event once, as a single JSON line."""
    try:
        kwargs: Final = _SUCCESS_KWARGS.validate_python(raw_kwargs)
    except ValidationError as error:
        return SpendEventBuildError(reason=str(error))
    litellm_params: Final = kwargs.get("litellm_params", _NO_LITELLM_PARAMS)
    sl_object: Final = kwargs.get("standard_logging_object")
    cache_key: Final = preset_spend_log_cache_key(litellm_params)
    response_cost: Final = sl_object.get("response_cost") if sl_object is not None else kwargs.get("response_cost")
    combined_usage: Final = kwargs.get("combined_usage_object")
    event: Final = SpendEvent(
        version=SPEND_EVENT_VERSION,
        litellm_call_id=kwargs.get("litellm_call_id"),
        call_type=kwargs.get("call_type"),
        model=kwargs.get("model"),
        custom_llm_provider=kwargs.get("custom_llm_provider"),
        stream=kwargs.get("stream"),
        complete_streaming_response="complete_streaming_response" in kwargs,
        cache_hit=kwargs.get("cache_hit"),
        response_cost=response_cost if isinstance(response_cost, (int, float)) else None,
        start_time=start_time,
        end_time=end_time,
        completion_start_time=kwargs.get("completion_start_time"),
        agent_id=kwargs.get("agent_id"),
        litellm_trace_id=kwargs.get("litellm_trace_id"),
        litellm_params=_litellm_params_for_event(litellm_params, cache_key, store_bodies),
        standard_logging_object=_standard_logging_for_event(sl_object, store_bodies),
        standard_callback_dynamic_params=kwargs.get("standard_callback_dynamic_params"),
        response=_compact_response(response_obj),
        combined_usage=combined_usage.model_dump() if combined_usage is not None else None,
        realtime_tools=kwargs.get("realtime_tools"),
        realtime_tool_calls=kwargs.get("realtime_tool_calls"),
        request_tool_names=_request_tool_names(kwargs),
    )
    return event.model_dump_json(fallback=_json_fallback).encode() + b"\n"


def decode_spend_event(line: bytes) -> SpendEvent | SpendEventDecodeError:
    try:
        return SpendEvent.model_validate_json(line)
    except ValidationError as error:
        return SpendEventDecodeError(reason=str(error))


def spend_event_callback_args(event: SpendEvent) -> SpendEventCallbackArgs:
    """The ``(kwargs, response_obj, start_time, end_time)`` the in-process cost callback receives."""
    tools: Final = tuple(_function_tool(name) for name in event.request_tool_names)
    kwargs: Final[SpendCallbackKwargs] = {
        "litellm_call_id": event.litellm_call_id,
        "call_type": event.call_type,
        "model": event.model,
        "custom_llm_provider": event.custom_llm_provider,
        "stream": event.stream,
        "cache_hit": event.cache_hit,
        "response_cost": event.response_cost,
        "completion_start_time": event.completion_start_time,
        "agent_id": event.agent_id,
        "litellm_trace_id": event.litellm_trace_id,
        "litellm_params": event.litellm_params,
        "standard_logging_object": event.standard_logging_object,
        "standard_callback_dynamic_params": event.standard_callback_dynamic_params,
        "combined_usage_object": Usage.model_validate(event.combined_usage)
        if event.combined_usage is not None
        else None,
        "realtime_tools": event.realtime_tools,
        "realtime_tool_calls": event.realtime_tool_calls,
        "tools": tools or None,
    }
    if not event.complete_streaming_response:
        return SpendEventCallbackArgs(kwargs, event.response, event.start_time, event.end_time)
    streaming_kwargs: Final[SpendCallbackKwargs] = {**kwargs, "complete_streaming_response": event.response}
    return SpendEventCallbackArgs(streaming_kwargs, event.response, event.start_time, event.end_time)

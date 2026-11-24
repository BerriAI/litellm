import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, get_args

from openai.types.responses import ResponseStreamEvent
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ResponsesAPIStreamEvents

if TYPE_CHECKING:
    from prometheus_client import Counter


_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_ARRAY: Final = TypeAdapter(list[object])


def _object(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _array(value: object) -> Sequence[object] | None:
    if value is None:
        return None
    try:
        return _JSON_ARRAY.validate_python(value)
    except ValidationError:
        return None


def _event_type_of(model: type[BaseModel]) -> str:
    values: Final[tuple[object, ...]] = get_args(model.model_fields["type"].annotation)
    return str(values[0])


def _sdk_event_models() -> Mapping[str, type[BaseModel]]:
    union: Final[tuple[object, ...]] = get_args(ResponseStreamEvent)
    members: Final[tuple[object, ...]] = get_args(union[0])
    models: Final = tuple(m for m in members if isinstance(m, type) and issubclass(m, BaseModel))
    return MappingProxyType({_event_type_of(m): m for m in models})


SDK_EVENT_MODELS: Final = _sdk_event_models()
_LITELLM_ONLY_EVENT_TYPES: Final = frozenset(e.value for e in ResponsesAPIStreamEvents) - SDK_EVENT_MODELS.keys()

_SPEC_EVENT_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "response.mcp_call.arguments.delta": ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DELTA.value,
        "response.mcp_call.arguments.done": ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DONE.value,
    }
)
_STRING_TOOL_INPUT_FIELD_BY_EVENT: Final[Mapping[str, str]] = MappingProxyType(
    {
        ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA.value: "delta",
        ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE.value: "arguments",
        ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DELTA.value: "delta",
        ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DONE.value: "arguments",
        ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DELTA.value: "delta",
        ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DONE.value: "input",
    }
)
_MISSING_FIELD_DEFAULTS: Final[Mapping[str, Callable[[], object]]] = MappingProxyType(
    {"output_index": int, "content_index": int, "logprobs": list}
)
_ERROR_FIELDS_SHARED_WITH_TOP_LEVEL: Final = ("message", "code", "param")
_TOOL_SECRET_FIELDS: Final = frozenset({"headers", "authorization"})


@dataclass(frozen=True, slots=True)
class StreamContext:
    previous_sequence_number: int | None
    stream_started_at: int
    request: Mapping[str, object]
    function_names: Mapping[str, str]


_Repair = Callable[[Mapping[str, object], StreamContext], Mapping[str, object]]


def _event_type(chunk: Mapping[str, object]) -> str:
    return str(chunk.get("type"))


def _absent(values: Mapping[str, object], defaults: Mapping[str, object]) -> Mapping[str, object]:
    return {name: default for name, default in defaults.items() if values.get(name) is None}


def _spec_event_name(chunk: Mapping[str, object], _: StreamContext) -> Mapping[str, object]:
    spec_name: Final = _SPEC_EVENT_NAMES.get(_event_type(chunk))
    return chunk if spec_name is None else {**chunk, "type": spec_name}


def _string_tool_input(chunk: Mapping[str, object], _: StreamContext) -> Mapping[str, object]:
    field_name: Final = _STRING_TOOL_INPUT_FIELD_BY_EVENT.get(_event_type(chunk))
    if field_name is None:
        return chunk
    value: Final = chunk.get(field_name)
    if isinstance(value, str):
        return chunk
    return {**chunk, field_name: "" if value is None else json.dumps(value, ensure_ascii=False)}


def _requires(model: type[BaseModel], field_name: str) -> bool:
    field: Final = model.model_fields.get(field_name)
    return field is not None and field.is_required()


def _missing_required_fields(chunk: Mapping[str, object], _: StreamContext) -> Mapping[str, object]:
    model: Final = SDK_EVENT_MODELS.get(_event_type(chunk))
    if model is None:
        return chunk
    filled: Final = {
        name: default()
        for name, default in _MISSING_FIELD_DEFAULTS.items()
        if chunk.get(name) is None and _requires(model, name)
    }
    return {**chunk, **filled} if filled else chunk


def _sequence_number(chunk: Mapping[str, object], context: StreamContext) -> Mapping[str, object]:
    if chunk.get("sequence_number") is not None or _event_type(chunk) not in SDK_EVENT_MODELS:
        return chunk
    previous: Final = context.previous_sequence_number
    return {**chunk, "sequence_number": 0 if previous is None else previous + 1}


def _function_name(chunk: Mapping[str, object], context: StreamContext) -> Mapping[str, object]:
    if _event_type(chunk) != ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE.value or "name" in chunk:
        return chunk
    name: Final = context.function_names.get(str(chunk.get("item_id")))
    return chunk if name is None else {**chunk, "name": name}


def function_call_name(chunk: Mapping[str, object]) -> tuple[str, str] | None:
    """The (item id, function name) a function_call output item announces, for later argument events to carry."""
    item: Final = _object(chunk.get("item"))
    if item is None or item.get("type") != "function_call":
        return None
    item_id: Final = item.get("id")
    name: Final = item.get("name")
    return (item_id, name) if isinstance(item_id, str) and isinstance(name, str) else None


def _details(details: object, defaults: Mapping[str, object]) -> Mapping[str, object]:
    present: Final = _object(details) or {}
    return {**present, **_absent(present, defaults)}


def _usage(usage: Mapping[str, object]) -> Mapping[str, object]:
    input_tokens: Final = usage.get("input_tokens")
    output_tokens: Final = usage.get("output_tokens")
    total: Final[Mapping[str, object]] = (
        {"total_tokens": input_tokens + output_tokens}
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else {}
    )
    return {
        **usage,
        **_absent(usage, total),
        "input_tokens_details": _details(usage.get("input_tokens_details"), {"cached_tokens": 0}),
        "output_tokens_details": _details(usage.get("output_tokens_details"), {"reasoning_tokens": 0}),
    }


def _echoed_tools(
    requested: object,
) -> list[object]:  # mutable-ok: provider readers validate chunks as JSON, which has no tuples
    return [_without_credentials(tool) for tool in _array(requested) or ()]


def _without_credentials(tool: object) -> object:
    fields: Final = _object(tool)
    return tool if fields is None else {key: value for key, value in fields.items() if key not in _TOOL_SECRET_FIELDS}


def _response_object(chunk: Mapping[str, object], context: StreamContext) -> Mapping[str, object]:
    response: Final = _object(chunk.get("response"))
    if response is None:
        return chunk
    usage: Final = _object(response.get("usage"))
    request: Final = context.request
    defaults: Final[Mapping[str, object]] = {
        "object": "response",
        "created_at": context.stream_started_at,
        "output": list[object](),
        "tools": _echoed_tools(request.get("tools")),
        "tool_choice": request.get("tool_choice") or "auto",
        "parallel_tool_calls": request.get("parallel_tool_calls") is not False,
    }
    return {
        **chunk,
        "response": {
            **response,
            **_absent(response, defaults),
            **({"truncation": "disabled"} if response.get("truncation") == "" else {}),
            **({} if usage is None else {"usage": _usage(usage)}),
        },
    }


def _output_item(chunk: Mapping[str, object], _: StreamContext) -> Mapping[str, object]:
    item: Final = _object(chunk.get("item"))
    if item is None:
        return chunk
    status: Final = (
        "completed" if _event_type(chunk) == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value else "in_progress"
    )
    match item.get("type"):
        case "message":
            return {**chunk, "item": {**item, **_absent(item, {"content": list[object](), "status": status})}}
        case "reasoning":
            return {**chunk, "item": {**item, **_absent(item, {"summary": list[object]()})}}
        case _:
            return chunk


def _content_part(chunk: Mapping[str, object], _: StreamContext) -> Mapping[str, object]:
    part: Final = _object(chunk.get("part"))
    if part is None or part.get("type") != "output_text":
        return chunk
    return {**chunk, "part": {**part, **_absent(part, {"annotations": list[object]()})}}


def _error_fields(chunk: Mapping[str, object], _: StreamContext) -> Mapping[str, object]:
    error: Final = _object(chunk.get("error"))
    if _event_type(chunk) != ResponsesAPIStreamEvents.ERROR.value or error is None:
        return chunk
    nested: Final[Mapping[str, object]] = {**error, **_absent(error, {"code": "unknown_error"})}
    top_level: Final = {
        name: nested[name] for name in _ERROR_FIELDS_SHARED_WITH_TOP_LEVEL if name in nested and chunk.get(name) is None
    }
    return {**chunk, **top_level, "error": nested}


_REPAIRS: Final[tuple[_Repair, ...]] = (
    _spec_event_name,
    _string_tool_input,
    _error_fields,
    _missing_required_fields,
    _sequence_number,
    _function_name,
    _response_object,
    _output_item,
    _content_part,
)


def repair_stream_chunk(chunk: Mapping[str, object], context: StreamContext) -> Mapping[str, object]:
    """Rewrite known provider deviations from the OpenAI Responses stream spec into spec shape."""
    return reduce(lambda repaired, repair: repair(repaired, context), _REPAIRS, chunk)


@dataclass(frozen=True, slots=True)
class ContractViolation:
    event_type: str
    reason: Literal["invalid", "unknown_type"]
    detail: str


def _validation_summary(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
        for detail in error.errors(include_input=False, include_url=False)[:5]
    )


def contract_violation(payload: Mapping[str, object]) -> ContractViolation | None:
    event_type: Final = _event_type(payload)
    if event_type in _LITELLM_ONLY_EVENT_TYPES:
        return None
    model: Final = SDK_EVENT_MODELS.get(event_type)
    if model is None:
        return ContractViolation(event_type, "unknown_type", "not an OpenAI Responses stream event type")
    try:
        _ = model.model_validate(payload)
    except ValidationError as error:
        return ContractViolation(event_type, "invalid", _validation_summary(error))
    return None


def _violation_counter() -> "Counter | None":
    try:
        # prometheus-client ships only in the proxy-runtime extra
        from prometheus_client import Counter
    except ImportError:
        return None
    return Counter(
        "litellm_responses_stream_contract_violations",
        "Streamed /v1/responses events that do not match the OpenAI Responses stream event types",
        ("provider", "event_type", "reason"),
    )


_VIOLATIONS: Final = _violation_counter()
_METRIC_LABEL_VALUE: Final = re.compile(r"[a-z0-9_.]{1,80}")


def _label(value: str) -> str:
    return value if _METRIC_LABEL_VALUE.fullmatch(value) else "other"


def report_contract_violation(provider: str, violation: ContractViolation, *, log: bool) -> None:
    if _VIOLATIONS is not None:
        _VIOLATIONS.labels(
            provider=_label(provider), event_type=_label(violation.event_type), reason=violation.reason
        ).inc()
    if log:
        verbose_logger.warning(
            "Provider %s streamed a /v1/responses event that does not match the OpenAI spec (%s %s): %s. "
            "It was passed to the client as received.",
            provider,
            violation.reason,
            violation.event_type,
            violation.detail,
        )

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Final, Literal, Protocol

from openai._streaming import ServerSentEvent
from openai.types.responses import ResponseToolSearchCall
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    CachedResponsesAPIStreamingIterator,
    MockResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.responses.tool_search.lowering import TOOL_SEARCH_FUNCTION_NAME
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)

_TOOL_SEARCH_CALL_ID_PREFIX: Final = "tsc_"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_ITEM_EVENTS: Final = frozenset({ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE})
_ARGUMENT_EVENTS: Final = frozenset(
    {ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA, ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE}
)
_TERMINAL_EVENTS: Final = frozenset(
    {ResponsesAPIStreamEvents.RESPONSE_COMPLETED, ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE}
)


class _FunctionCall(BaseModel):
    type: Literal["function_call"]
    name: str
    namespace: str | None = None
    id: str | None = None
    call_id: str
    arguments: str = ""


class _Event(BaseModel):
    type: str = ""
    item_id: str | None = None
    item: object = None
    response: dict[str, object] | None = None


class _Response(BaseModel):
    output: tuple[object, ...] = ()


class _HasOutput(Protocol):
    @property
    def output(self) -> Sequence[object]: ...


def _tool_search_function_call(item: object) -> _FunctionCall | None:
    plain: Final = item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item
    try:
        call: Final = _FunctionCall.model_validate(plain)
    except ValidationError:
        return None
    return call if call.name == TOOL_SEARCH_FUNCTION_NAME and not call.namespace else None


def _search_arguments(arguments: str) -> dict[str, object]:
    try:
        return _JSON_OBJECT.validate_json(arguments)
    except ValidationError:
        verbose_logger.warning(
            "tool_search was called with arguments that aren't a JSON object, using them as the query"
        )
        return {"query": arguments}


def _tool_search_call_item_id(call: _FunctionCall) -> str:
    source: Final = call.id or call.call_id
    return f"{_TOOL_SEARCH_CALL_ID_PREFIX}{source.partition('_')[2] or source}"


def _tool_search_call(call: _FunctionCall, status: Literal["in_progress", "completed"]) -> dict[str, object]:
    return {
        "type": "tool_search_call",
        "id": _tool_search_call_item_id(call),
        "call_id": call.call_id,
        "execution": "client",
        "status": status,
        "arguments": {} if status == "in_progress" else _search_arguments(call.arguments),
    }


def _lifted_output_item(item: object) -> object:
    call: Final = _tool_search_function_call(item)
    if call is None:
        return item
    return ResponseToolSearchCall.model_validate(_tool_search_call(call, "completed"))


def _output_items(response: _HasOutput) -> tuple[object, ...]:
    return tuple(response.output)


def lift_tool_search_calls(response: ResponsesAPIResponse) -> ResponsesAPIResponse:
    output: Final = _output_items(response)
    lifted: Final = [_lifted_output_item(item) for item in output]
    if all(new is old for new, old in zip(lifted, output, strict=True)):
        return response
    return response.model_copy(update={"output": lifted})


def _lifted_output_dict(item: object) -> object:
    call: Final = _tool_search_function_call(item)
    return item if call is None else _tool_search_call(call, "completed")


def _lifted_response_dict(response: dict[str, object]) -> dict[str, object]:
    output: Final = _Response.model_validate(response).output
    return {**response, "output": [_lifted_output_dict(item) for item in output]}


class ToolSearchEventLifter:
    def __init__(self) -> None:
        self._tool_search_item_ids: frozenset[str] = frozenset()

    def lift(self, data: str) -> str | None:
        try:
            raw: Final = _JSON_OBJECT.validate_json(data)
        except ValidationError:
            return data
        event: Final = _Event.model_validate(raw)
        if event.type in _ARGUMENT_EVENTS and event.item_id in self._tool_search_item_ids:
            return None
        if event.type in _TERMINAL_EVENTS and event.response is not None:
            return json.dumps({**raw, "response": _lifted_response_dict(event.response)})
        call: Final = _tool_search_function_call(event.item) if event.type in _ITEM_EVENTS else None
        if call is None:
            return data
        if call.id is not None:
            self._tool_search_item_ids = self._tool_search_item_ids | {call.id}
        status: Final = "in_progress" if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED else "completed"
        return json.dumps({**raw, "item": _tool_search_call(call, status)})


def _lifted_event(event: ServerSentEvent, lifter: ToolSearchEventLifter) -> ServerSentEvent | None:
    data: Final = lifter.lift(event.data)
    if data is None:
        return None
    return ServerSentEvent(event=event.event, data=data, id=event.id, retry=event.retry)


async def _alifted_events(
    events: AsyncIterator[ServerSentEvent], lifter: ToolSearchEventLifter
) -> AsyncIterator[ServerSentEvent]:
    async for event in events:
        if (lifted := _lifted_event(event, lifter)) is not None:
            yield lifted


def _lifted_events(events: Iterator[ServerSentEvent], lifter: ToolSearchEventLifter) -> Iterator[ServerSentEvent]:
    lifted_events: Final = (_lifted_event(event, lifter) for event in events)
    return (lifted for lifted in lifted_events if lifted is not None)


def _lift_synthetic_events(stream: MockResponsesAPIStreamingIterator | CachedResponsesAPIStreamingIterator) -> None:
    terminal_event: Final = stream.completed_response
    if not isinstance(terminal_event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
        return
    stream._set_events_from_response(  # pyright: ignore[reportPrivateUsage]  # the only way to rebuild a replayed stream
        transformed=lift_tool_search_calls(terminal_event.response), logging_obj=stream.logging_obj
    )


def lift_tool_search_stream(stream: BaseResponsesAPIStreamingIterator) -> BaseResponsesAPIStreamingIterator:
    match stream:
        case ResponsesAPIStreamingIterator():
            stream.stream_iterator = _alifted_events(  # rebind-ok: lift events before the stream logs its own output
                stream.stream_iterator, ToolSearchEventLifter()
            )
        case SyncResponsesAPIStreamingIterator():
            stream.stream_iterator = _lifted_events(  # rebind-ok: lift events before the stream logs its own output
                stream.stream_iterator, ToolSearchEventLifter()
            )
        case MockResponsesAPIStreamingIterator() | CachedResponsesAPIStreamingIterator():
            _lift_synthetic_events(stream)
        case _:
            verbose_logger.debug("tool_search lifting skipped for %s", type(stream).__name__)
    return stream

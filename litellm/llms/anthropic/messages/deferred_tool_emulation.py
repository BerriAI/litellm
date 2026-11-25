import inspect
import itertools
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Container,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Final, TypeVar, cast  # noqa: TID251  # see cast-ok markers

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping, loads_json_object
from litellm.llms.anthropic.messages.deferred_tools import (
    DeferredToolsMode,
    client_tool_search_name,
    deferred_tools_mode,
    is_deferred,
)
from litellm.llms.anthropic.messages.native.streaming_iterator import sse_frame_event, sse_frames
from litellm.llms.anthropic.messages.utils import anthropic_sse_frame
from litellm.llms.custom_httpx.stream_close import aclose_if_supported, close_if_supported
from litellm.responses.streaming_iterator import partial_responses_incomplete_event_on_disconnect
from litellm.types.llms.openai import ResponseIncompleteEvent

DISPATCHER_NAME: Final = "CallLoadedTool"
_Result = TypeVar("_Result")
_Item = TypeVar("_Item", bound=Mapping[str, object])
_DISPATCHER_DESCRIPTION: Final = (
    "Call a deferred tool that ToolSearch loaded earlier in this conversation. Pass the loaded tool's exact name "
    "and an input object matching the input schema ToolSearch returned for it."
)


def _unused_dispatcher_name(tool_names: Container[str]) -> str:
    candidates: Final = (DISPATCHER_NAME if n == 1 else f"{DISPATCHER_NAME}{n}" for n in itertools.count(1))
    return next(name for name in candidates if name not in tool_names)


def _dispatcher_tool(dispatcher: str, forced: str | None) -> Mapping[str, object]:
    name_schema: Final = {"type": "string"} if forced is None else {"type": "string", "enum": [forced]}
    return {
        "name": dispatcher,
        "description": _DISPATCHER_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {"name": name_schema, "input": {"type": "object"}},
            "required": ["name", "input"],
        },
    }


@dataclass(frozen=True, slots=True)
class DispatchedCall:
    name: str
    input: Mapping[str, object]


def dispatched_call(dispatcher_input: object, catalog: Container[str]) -> DispatchedCall | None:
    arguments: Final = as_str_mapping(dispatcher_input) or {}
    name: Final = arguments.get("name")
    raw_input: Final = arguments.get("input", {})
    tool_input: Final = loads_json_object(raw_input) if isinstance(raw_input, str) else as_str_mapping(raw_input)
    if not isinstance(name, str) or name not in catalog or tool_input is None:
        return None
    return DispatchedCall(name=name, input=tool_input)


@dataclass(frozen=True, slots=True)
class Dispatcher:
    name: str
    catalog: Container[str]


def _restored_block(block: object, dispatcher: Dispatcher) -> object:
    tool_use: Final = as_str_mapping(block)
    if tool_use is None or tool_use.get("type") != "tool_use" or tool_use.get("name") != dispatcher.name:
        return block
    call: Final = dispatched_call(tool_use.get("input"), dispatcher.catalog)
    return block if call is None else {**tool_use, "name": call.name, "input": call.input}


def restored_response(response: Mapping[str, object], dispatcher: Dispatcher) -> Mapping[str, object]:
    content: Final = as_object_list(response.get("content"))
    if content is None:
        return response
    return {**response, "content": [_restored_block(block, dispatcher) for block in content]}


def _partial_json(event: Mapping[str, object]) -> str:
    partial: Final = (as_str_mapping(event.get("delta")) or {}).get("partial_json")
    return partial if isinstance(partial, str) else ""


def _opens_dispatcher_block(event: Mapping[str, object], dispatcher: str) -> bool:
    block: Final = as_str_mapping(event.get("content_block")) or {}
    return (
        event.get("type") == "content_block_start"
        and block.get("type") == "tool_use"
        and block.get("name") == dispatcher
    )


class DispatcherBlockRestorer:
    def __init__(self, dispatcher: Dispatcher) -> None:
        self._dispatcher: Final = dispatcher
        self._start: Mapping[str, object] | None = None
        self._held: tuple[bytes, ...] = ()
        self._partial_json: str = ""

    def frames(self, frame: bytes) -> tuple[bytes, ...]:
        event: Final = sse_frame_event(frame)
        if self._start is None:
            if event is None or not _opens_dispatcher_block(event, self._dispatcher.name):
                return (frame,)
            self._start = event
            self._held = (frame,)
            self._partial_json = ""
            return ()
        if event is None or event.get("index") != self._start.get("index"):
            return (frame,)
        self._held = (*self._held, frame)
        if event.get("type") == "content_block_delta":
            self._partial_json += _partial_json(event)
            return ()
        if event.get("type") != "content_block_stop":
            return ()
        restored: Final = self._restored(self._start, frame)
        self._start = None
        self._held = ()
        return restored

    def rest(self) -> tuple[bytes, ...]:
        return self._held

    def _restored(self, start: Mapping[str, object], stop_frame: bytes) -> tuple[bytes, ...]:
        block: Final = as_str_mapping(start.get("content_block")) or {}
        call: Final = dispatched_call(
            loads_json_object(self._partial_json) if self._partial_json else block.get("input"),
            self._dispatcher.catalog,
        )
        if call is None:
            return self._held
        return (
            anthropic_sse_frame({**start, "content_block": {**block, "name": call.name, "input": {}}}),
            anthropic_sse_frame(
                {
                    "type": "content_block_delta",
                    "index": start.get("index"),
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(call.input)},
                }
            ),
            stop_frame,
        )


async def _restored_async_frames(stream: AsyncIterator[bytes], dispatcher: Dispatcher) -> AsyncGenerator[bytes, None]:
    restorer: Final = DispatcherBlockRestorer(dispatcher)
    async for frame in sse_frames(stream):
        if not isinstance(frame, (bytes, bytearray)):  # pyright: ignore[reportUnnecessaryIsInstance]  # bridge-path chunks are not always bytes
            yield frame
            continue
        for restored in restorer.frames(bytes(frame)):
            yield restored
    for held in restorer.rest():
        yield held


def _restored_sync_frames(stream: Iterator[bytes], dispatcher: Dispatcher) -> Generator[bytes, None, None]:
    restorer: Final = DispatcherBlockRestorer(dispatcher)
    for frame in stream:
        yield from restorer.frames(frame)
    yield from restorer.rest()


class DispatcherRestoringSyncStream:
    def __init__(self, stream: Iterator[bytes], dispatcher: Dispatcher) -> None:
        self._stream: Final = stream
        self._frames: Final = _restored_sync_frames(stream, dispatcher)

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)  # pyright: ignore[reportAny]  # forwards whatever the wrapped stream exposes

    def __iter__(self) -> "DispatcherRestoringSyncStream":
        return self

    def __next__(self) -> bytes:
        return next(self._frames)

    def close(self) -> None:
        self._frames.close()
        close_if_supported(self._stream)


class DispatcherRestoringStream:
    def __init__(self, stream: AsyncIterator[bytes], dispatcher: Dispatcher) -> None:
        self._stream: Final = stream
        self._frames: Final = _restored_async_frames(stream, dispatcher)

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)  # pyright: ignore[reportAny]  # forwards whatever the wrapped stream exposes

    def __aiter__(self) -> "DispatcherRestoringStream":
        return self

    async def __anext__(self) -> bytes:
        return await self._frames.__anext__()

    async def aclose(self) -> None:
        await self._frames.aclose()
        await aclose_if_supported(self._stream)

    def partial_response_on_disconnect(self) -> ResponseIncompleteEvent | None:
        return partial_responses_incomplete_event_on_disconnect(self._stream)


def _loaded_tool_text(name: str, tool: Mapping[str, object] | None, dispatcher: str) -> str:
    if tool is None:
        return f"Loaded tool {name}."
    return (
        f"Loaded tool {name}: {tool.get('description', '')} Input schema: {json.dumps(tool.get('input_schema', {}))}. "
        f"Call it with {dispatcher}, name {name!r}."
    )


def _forced_tool_name(tool_choice: Mapping[str, object] | None) -> object:
    return None if tool_choice is None or tool_choice.get("type") != "tool" else tool_choice.get("name")


def _reference_name(part: object) -> str | None:
    reference: Final = as_str_mapping(part) or {}
    name: Final = reference.get("tool_name")
    return name if reference.get("type") == "tool_reference" and isinstance(name, str) else None


def _block_reference_names(block: object) -> Iterator[str | None]:
    tool_result: Final = as_str_mapping(block) or {}
    if tool_result.get("type") != "tool_result":
        return iter(())
    return map(_reference_name, as_object_list(tool_result.get("content")) or ())


def _message_reference_names(message: Mapping[str, object]) -> Iterator[str | None]:
    return itertools.chain.from_iterable(map(_block_reference_names, as_object_list(message.get("content")) or ()))


def _loaded_tool_names(messages: Iterable[Mapping[str, object]]) -> frozenset[str]:
    names: Final = itertools.chain.from_iterable(map(_message_reference_names, messages))
    return frozenset(name for name in names if name is not None)


@dataclass(frozen=True, slots=True)
class DeferredToolEmulation:
    deferred: Mapping[str, Mapping[str, object]]
    dispatcher: str = DISPATCHER_NAME
    forced: str | None = None
    loaded: frozenset[str] = frozenset()

    def tools(self, tools: Iterable[_Item]) -> tuple[_Item, ...]:
        dispatcher: Final = cast(  # cast-ok: the dispatcher is a tool like the others
            _Item, _dispatcher_tool(self.dispatcher, self.forced)
        )
        return (*(tool for tool in tools if not is_deferred(tool)), dispatcher)

    def tool_choice(self, tool_choice: _Item | None) -> _Item | None:
        if tool_choice is None or self.forced is None or _forced_tool_name(tool_choice) != self.forced:
            return tool_choice
        return cast(_Item, {**tool_choice, "name": self.dispatcher})  # cast-ok: only the forced tool's name changed

    def messages(self, messages: Iterable[_Item]) -> tuple[_Item, ...]:
        return tuple(self._message(message) for message in messages)

    def _message(self, message: _Item) -> _Item:
        content: Final = as_object_list(message.get("content"))
        if content is None:
            return message
        rewritten: Final = {**message, "content": [self._block(block) for block in content]}
        return cast(_Item, rewritten)  # cast-ok: only the content blocks changed

    def _block(self, block: object) -> object:
        mapping: Final = as_str_mapping(block)
        if mapping is None:
            return block
        name: Final = mapping.get("name")
        if mapping.get("type") == "tool_use" and isinstance(name, str) and name in self.deferred:
            return {**mapping, "name": self.dispatcher, "input": {"name": name, "input": mapping.get("input", {})}}
        content: Final = as_object_list(mapping.get("content"))
        if mapping.get("type") != "tool_result" or content is None:
            return block
        return {**mapping, "content": [self._reference_as_text(part) for part in content]}

    def _reference_as_text(self, part: object) -> object:
        name: Final = _reference_name(part)
        if name is None:
            return part
        return {"type": "text", "text": _loaded_tool_text(name, self.deferred.get(name), self.dispatcher)}

    def strict_input_refusal(self) -> str | None:
        strict: Final = tuple(name for name, tool in self.deferred.items() if tool.get("strict") is True)
        if not strict:
            return None
        return (
            f"model_info.deferred_tools emulate sends deferred tools through one {self.dispatcher} tool, which cannot "
            f"enforce strict input for {', '.join(strict)}. Remove strict from those tools, pick another mode for "
            "this deployment, or set drop_params to send them without strict"
        )

    def restored(self, result: _Result) -> _Result:
        return cast(_Result, self._restored(result))  # cast-ok: each branch hands back the shape it was given

    def _call_restoration(self) -> Dispatcher:
        return Dispatcher(name=self.dispatcher, catalog=self.loaded if self.forced is None else (self.forced,))

    def _restored(self, result: object) -> object:
        if inspect.isawaitable(result):
            return self._restored_later(result)
        if isinstance(result, Mapping):
            response: Final = cast(Mapping[str, object], result)  # cast-ok: a bridged Messages response
            return restored_response(response, self._call_restoration())
        if isinstance(result, Iterator):
            sync_stream: Final = cast(Iterator[bytes], result)  # cast-ok: a bridged SSE stream
            return DispatcherRestoringSyncStream(sync_stream, self._call_restoration())
        if isinstance(result, AsyncIterator):
            stream: Final = cast(AsyncIterator[bytes], result)  # cast-ok: a bridged SSE stream
            return DispatcherRestoringStream(stream, self._call_restoration())
        return result

    async def _restored_later(self, pending: Awaitable[object]) -> object:
        return self.restored(await pending)


def deferred_tools_messages(
    messages: Sequence[_Item], emulation: DeferredToolEmulation | None, deferred_tools: DeferredToolsMode
) -> Sequence[_Item]:
    if emulation is not None:
        return emulation.messages(messages)
    if deferred_tools == "passthrough":
        return DeferredToolEmulation(deferred={}).messages(messages)
    return messages


def deferred_tool_emulation(
    model_info: object,
    *,
    tools: Sequence[object] | None,
    tool_choice: Mapping[str, object] | None,
    messages: Iterable[Mapping[str, object]],
) -> DeferredToolEmulation | None:
    if deferred_tools_mode(model_info) != "emulate" or client_tool_search_name(tools) is None:
        return None
    tool_mappings: Final = tuple(as_str_mapping(tool) or {} for tool in tools or ())
    deferred: Final = {str(tool.get("name")): tool for tool in tool_mappings if is_deferred(tool)}
    tool_names: Final = frozenset(
        name for name in (tool.get("name") for tool in tool_mappings) if isinstance(name, str)
    )
    forced: Final = _forced_tool_name(tool_choice)
    return DeferredToolEmulation(
        deferred=deferred,
        dispatcher=_unused_dispatcher_name(tool_names),
        forced=forced if isinstance(forced, str) and forced in deferred else None,
        loaded=frozenset(name for name in _loaded_tool_names(messages) if name in deferred),
    )

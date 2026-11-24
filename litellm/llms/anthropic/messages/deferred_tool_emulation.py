import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Container, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeVar, cast  # noqa: TID251  # see cast-ok markers

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping, loads_json_object
from litellm.llms.anthropic.messages.deferred_tools import client_tool_search_name, deferred_tools_mode, is_deferred
from litellm.llms.anthropic.messages.native.streaming_iterator import sse_frame_event, sse_frames
from litellm.llms.anthropic.messages.utils import anthropic_sse_frame
from litellm.llms.custom_httpx.stream_close import aclose_if_supported

DISPATCHER_NAME: Final = "CallLoadedTool"
_Result = TypeVar("_Result")
_Item = TypeVar("_Item", bound=Mapping[str, object])
_DISPATCHER_DESCRIPTION: Final = (
    "Call a deferred tool that ToolSearch loaded earlier in this conversation. Pass the loaded tool's exact name "
    "and an input object matching the input schema ToolSearch returned for it."
)


def _dispatcher_tool() -> Mapping[str, object]:
    return {
        "name": DISPATCHER_NAME,
        "description": _DISPATCHER_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
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


def _restored_block(block: object, catalog: Container[str]) -> object:
    tool_use: Final = as_str_mapping(block)
    if tool_use is None or tool_use.get("type") != "tool_use" or tool_use.get("name") != DISPATCHER_NAME:
        return block
    call: Final = dispatched_call(tool_use.get("input"), catalog)
    return block if call is None else {**tool_use, "name": call.name, "input": call.input}


def restored_response(response: Mapping[str, object], catalog: Container[str]) -> Mapping[str, object]:
    content: Final = as_object_list(response.get("content"))
    if content is None:
        return response
    return {**response, "content": [_restored_block(block, catalog) for block in content]}


def _partial_json(event: Mapping[str, object]) -> str:
    partial: Final = (as_str_mapping(event.get("delta")) or {}).get("partial_json")
    return partial if isinstance(partial, str) else ""


def _opens_dispatcher_block(event: Mapping[str, object]) -> bool:
    block: Final = as_str_mapping(event.get("content_block")) or {}
    return (
        event.get("type") == "content_block_start"
        and block.get("type") == "tool_use"
        and block.get("name") == DISPATCHER_NAME
    )


class DispatcherBlockRestorer:
    def __init__(self, catalog: Container[str]) -> None:
        self._catalog: Final = catalog
        self._start: Mapping[str, object] | None = None
        self._held: tuple[bytes, ...] = ()
        self._partial_json: str = ""

    def frames(self, frame: bytes) -> tuple[bytes, ...]:
        event: Final = sse_frame_event(frame)
        if self._start is None:
            if event is None or not _opens_dispatcher_block(event):
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
            loads_json_object(self._partial_json) if self._partial_json else block.get("input"), self._catalog
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


async def _restored_async_frames(stream: AsyncIterator[bytes], catalog: Container[str]) -> AsyncGenerator[bytes, None]:
    restorer: Final = DispatcherBlockRestorer(catalog)
    async for frame in sse_frames(stream):
        if not isinstance(frame, (bytes, bytearray)):  # pyright: ignore[reportUnnecessaryIsInstance]  # bridge-path chunks are not always bytes
            yield frame
            continue
        for restored in restorer.frames(bytes(frame)):
            yield restored
    for held in restorer.rest():
        yield held


def _restored_sync_frames(stream: Iterator[bytes], catalog: Container[str]) -> Iterator[bytes]:
    restorer: Final = DispatcherBlockRestorer(catalog)
    for frame in stream:
        yield from restorer.frames(frame)
    yield from restorer.rest()


class DispatcherRestoringStream:
    def __init__(self, stream: AsyncIterator[bytes], catalog: Container[str]) -> None:
        self._stream: Final = stream
        self._frames: Final = _restored_async_frames(stream, catalog)

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)  # pyright: ignore[reportAny]  # forwards whatever the wrapped stream exposes

    def __aiter__(self) -> "DispatcherRestoringStream":
        return self

    async def __anext__(self) -> bytes:
        return await self._frames.__anext__()

    async def aclose(self) -> None:
        await self._frames.aclose()
        await aclose_if_supported(self._stream)


def _loaded_tool_text(name: str, tool: Mapping[str, object] | None) -> str:
    if tool is None:
        return f"Loaded tool {name}."
    return (
        f"Loaded tool {name}: {tool.get('description', '')} Input schema: {json.dumps(tool.get('input_schema', {}))}. "
        f"Call it with {DISPATCHER_NAME}, name {name!r}."
    )


@dataclass(frozen=True, slots=True)
class DeferredToolEmulation:
    deferred: Mapping[str, Mapping[str, object]]

    def tools(self, tools: Iterable[_Item]) -> tuple[_Item, ...]:
        dispatcher: Final = cast(_Item, _dispatcher_tool())  # cast-ok: the dispatcher is a tool like the others
        return (*(tool for tool in tools if not is_deferred(tool)), dispatcher)

    def tool_choice(self, tool_choice: _Item | None) -> _Item | None:
        if tool_choice is None or tool_choice.get("name") not in self.deferred:
            return tool_choice
        return cast(_Item, {**tool_choice, "name": DISPATCHER_NAME})  # cast-ok: only the forced tool's name changed

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
            return {**mapping, "name": DISPATCHER_NAME, "input": {"name": name, "input": mapping.get("input", {})}}
        content: Final = as_object_list(mapping.get("content"))
        if mapping.get("type") != "tool_result" or content is None:
            return block
        return {**mapping, "content": [self._reference_as_text(part) for part in content]}

    def _reference_as_text(self, part: object) -> object:
        reference: Final = as_str_mapping(part) or {}
        name: Final = reference.get("tool_name")
        if reference.get("type") != "tool_reference" or not isinstance(name, str):
            return part
        return {"type": "text", "text": _loaded_tool_text(name, self.deferred.get(name))}

    def restored(self, result: _Result) -> _Result:
        return cast(_Result, self._restored(result))  # cast-ok: each branch hands back the shape it was given

    def _restored(self, result: object) -> object:
        if inspect.isawaitable(result):
            return self._restored_later(result)
        if isinstance(result, Mapping):
            response: Final = cast(Mapping[str, object], result)  # cast-ok: a bridged Messages response
            return restored_response(response, self.deferred)
        if isinstance(result, Iterator):
            return _restored_sync_frames(cast(Iterator[bytes], result), self.deferred)  # cast-ok: a bridged SSE stream
        if isinstance(result, AsyncIterator):
            stream: Final = cast(AsyncIterator[bytes], result)  # cast-ok: a bridged SSE stream
            return DispatcherRestoringStream(stream, self.deferred)
        return result

    async def _restored_later(self, pending: Awaitable[object]) -> object:
        return self.restored(await pending)


def deferred_tool_emulation(model_info: object, tools: Sequence[object] | None) -> DeferredToolEmulation | None:
    if deferred_tools_mode(model_info) != "emulate" or client_tool_search_name(tools) is None:
        return None
    tool_mappings: Final = (as_str_mapping(tool) or {} for tool in tools or ())
    deferred: Final = {str(tool.get("name")): tool for tool in tool_mappings if is_deferred(tool)}
    return DeferredToolEmulation(deferred=deferred)

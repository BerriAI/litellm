from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from .engine import PrivaiteEngine, RestoreBuffer, RestoreMapping
from .payload import as_array, as_object, index_of, read_field, read_list, write_field

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream

TEXT_FIELDS: Final = ("reasoning_content", "reasoning", "refusal")
"""Text fields restored besides `content`. A refusal can quote the request, so it carries placeholders too."""

DELTA_CHANNELS: Final = ("content", *TEXT_FIELDS, "audio", "tool_calls", "function_call")

JSON_CHANNELS: Final = frozenset({"tool", "function_call"})
"""Channels that stream JSON source text, where a restored original must arrive JSON-escaped."""


def json_escape(value: str) -> str:
    """The JSON string-literal encoding of value without its quotes, as privaite.streaming.buffer does."""
    return json.dumps(value, ensure_ascii=False)[1:-1]


async def _restore_json_tree(engine: PrivaiteEngine, value: object, mapping: RestoreMapping) -> object:
    """Restores every string leaf of a parsed JSON document. Keys are left as they are, as in the core engine."""
    if isinstance(value, str):
        return await engine.process_response(value, mapping)
    fields: Final = as_object(value)
    if fields is not None:
        return {key: await _restore_json_tree(engine, item, mapping) for key, item in fields.items()}
    items: Final = as_array(value)
    if items is not None:
        return [await _restore_json_tree(engine, item, mapping) for item in items]
    return value


async def restore_arguments(engine: PrivaiteEngine, arguments: str, mapping: RestoreMapping) -> str:
    """Restores a tool call's `arguments`, a JSON document inside a string.

    Plain substitution would splice a raw quote, backslash or newline from the original
    into a JSON string literal, and the client's parse of the arguments would fail. So
    the restore runs on the parsed tree and re-encodes it, as PIIEngine does. When nothing
    changes, or the string is not JSON, the plain restore is returned as is.
    """
    plain: Final = await engine.process_response(arguments, mapping)
    if plain == arguments:
        return arguments
    try:
        parsed: Final[object] = json.loads(arguments)  # pyright: ignore[reportAny]  # json.loads is untyped
    except ValueError:
        return plain
    return json.dumps(await _restore_json_tree(engine, parsed, mapping), ensure_ascii=False)


async def _restore_text(holder: object, name: str, engine: PrivaiteEngine, mapping: RestoreMapping) -> None:
    value: Final = read_field(holder, name)
    if isinstance(value, str) and value:
        write_field(holder, name, await engine.process_response(value, mapping))


async def _restore_call_arguments(holder: object, engine: PrivaiteEngine, mapping: RestoreMapping) -> None:
    arguments: Final = read_field(holder, "arguments")
    if isinstance(arguments, str) and arguments:
        write_field(holder, "arguments", await restore_arguments(engine, arguments, mapping))


async def restore_message(message: object, engine: PrivaiteEngine, mapping: RestoreMapping) -> None:
    """Restores one response message: content, the text fields, the audio transcript, and tool-call arguments."""
    for name in ("content", *TEXT_FIELDS):
        await _restore_text(message, name, engine, mapping)
    await _restore_text(read_field(message, "audio"), "transcript", engine, mapping)
    for tool_call in read_list(message, "tool_calls"):
        await _restore_call_arguments(read_field(tool_call, "function"), engine, mapping)
    await _restore_call_arguments(read_field(message, "function_call"), engine, mapping)


async def restore_responses_output(response: object, engine: PrivaiteEngine, mapping: RestoreMapping) -> None:
    """Restores a Responses API result: output_text blocks and function_call arguments."""
    for item in read_list(response, "output"):
        for block in read_list(item, "content"):
            await _restore_text(block, "text", engine, mapping)
        await _restore_call_arguments(item, engine, mapping)


@dataclass(frozen=True, slots=True)
class Channel:
    """One streamed segment of one choice. Each keeps its own restore buffer."""

    kind: str
    choice: int
    tool_call: int = 0


Restore: TypeAlias = Callable[[Channel, str, bool], str]


def _append_text(holder: object, name: str, tail: str) -> None:
    existing: Final = read_field(holder, name)
    write_field(holder, name, (existing if isinstance(existing, str) else "") + tail)


def _append_nested(delta: object, carrier: str, name: str, tail: str) -> None:
    """Appends to delta.<carrier>.<name>, creating the carrier when a bare finish delta has none."""
    holder: Final = read_field(delta, carrier)
    if holder is None:
        write_field(delta, carrier, {name: tail})
        return
    _append_text(holder, name, tail)


def _append_tool_tail(delta: object, tool_call: int, tail: str) -> None:
    """Appends an argument tail under its tool_call index, since the client reassembles arguments per index."""
    calls: Final = as_array(read_field(delta, "tool_calls"))
    added: Final = {"index": tool_call, "function": {"arguments": tail}}
    if calls is None:
        write_field(delta, "tool_calls", [added])
        return
    existing: Final = next((call for call in calls if index_of(call) == tool_call), None)
    if existing is None:
        calls.append(added)
        return
    _append_nested(existing, "function", "arguments", tail)


def append_tail(delta: object, channel: Channel, tail: str) -> None:
    """Appends a restore buffer's held tail onto a streamed delta."""
    match channel.kind:
        case "tool":
            _append_tool_tail(delta, channel.tool_call, tail)
        case "function_call":
            _append_nested(delta, "function_call", "arguments", tail)
        case "audio":
            _append_nested(delta, "audio", "transcript", tail)
        case _:
            _append_text(delta, channel.kind, tail)


def _restore_delta_audio(delta: object, choice: int, finished: bool, restore: Restore) -> None:
    """Restores a streamed audio transcript, creating the carrier on a finish delta so a held tail is kept."""
    audio: Final = read_field(delta, "audio")
    transcript: Final = read_field(audio, "transcript")
    fragment: Final = transcript if isinstance(transcript, str) else ""
    if not fragment and not finished:
        return
    restored: Final = restore(Channel("audio", choice), fragment, finished)
    if audio is not None:
        write_field(audio, "transcript", restored)
    elif restored:
        write_field(delta, "audio", {"transcript": restored})


def restore_delta(delta: object, choice: int, finished: bool, restore: Restore) -> None:
    """Restores one streamed delta: content, the text fields, the audio transcript and argument fragments."""
    content: Final = read_field(delta, "content")
    text: Final = content if isinstance(content, str) else ""
    restored: Final = restore(Channel("content", choice), text, finished)
    # A fragment held back whole must not stay visible raw, but a terminal None with
    # nothing held back keeps its None rather than becoming "".
    if text or restored:
        write_field(delta, "content", restored)
    for name in TEXT_FIELDS:
        value = read_field(delta, name)
        if isinstance(value, str) and (value or finished):
            write_field(delta, name, restore(Channel(name, choice), value, finished))
    _restore_delta_audio(delta, choice, finished, restore)
    for tool_call in read_list(delta, "tool_calls"):
        function = read_field(tool_call, "function")
        arguments = read_field(function, "arguments")
        if isinstance(arguments, str) and arguments:
            channel = Channel("tool", choice, index_of(tool_call))
            write_field(function, "arguments", restore(channel, arguments, finished))
    function_call: Final = read_field(delta, "function_call")
    call_arguments: Final = read_field(function_call, "arguments")
    if isinstance(call_arguments, str) and call_arguments:
        write_field(function_call, "arguments", restore(Channel("function_call", choice), call_arguments, finished))


def drain_chunk(template: ModelResponseStream, channel: Channel, tail: str) -> ModelResponseStream:
    """A trailing chunk carrying one held tail, for a stream the provider closed without a finish_reason.

    It is a copy of the last chunk of that choice, so id, model and object match the rest
    of the stream, reduced to that one choice with its delta cleared and no usage.
    """
    chunk: Final = copy.deepcopy(template)
    choice: Final = next(item for item in read_list(chunk, "choices") if index_of(item) == channel.choice)
    write_field(chunk, "choices", [choice])
    if read_field(chunk, "usage") is not None:
        write_field(chunk, "usage", None)
    write_field(choice, "finish_reason", None)
    delta: Final = read_field(choice, "delta")
    for name in DELTA_CHANNELS:
        if read_field(delta, name) is not None:
            write_field(delta, name, None)
    append_tail(delta, channel, tail)
    return chunk


class StreamRestorer:
    """Restore state for one streamed response.

    With n>1 the provider interleaves chunks of different choices, and tool-call arguments
    stream as fragments per tool_call index, so every channel keeps its own buffer and a
    placeholder split across chunks reassembles without mixing channels. A held tail is
    flushed onto the chunk carrying finish_reason, or, when the provider closes the stream
    without one, drained as trailing chunks. It is never dropped.
    """

    def __init__(
        self,
        mapping: RestoreMapping,
        json_mapping: RestoreMapping,
        make_buffer: Callable[[RestoreMapping], RestoreBuffer],
    ) -> None:
        self._mapping: Final = mapping
        self._json_mapping: Final = json_mapping
        self._make_buffer: Final = make_buffer
        self._buffers: Final[dict[Channel, RestoreBuffer]] = {}  # mutable-ok: per-stream restore state
        self._last_chunk: Final[dict[int, ModelResponseStream]] = {}  # mutable-ok: per-stream restore state

    def restore(self, channel: Channel, text: str, finished: bool) -> str:
        if channel not in self._buffers:
            mapping: Final = self._json_mapping if channel.kind in JSON_CHANNELS else self._mapping
            self._buffers[channel] = self._make_buffer(mapping)
        restored: Final = self._buffers[channel].feed(text) if text else ""
        if not finished:
            return restored
        return restored + self._buffers.pop(channel).flush()

    def _close(self, choice: int | None) -> tuple[tuple[Channel, str], ...]:
        """Flushes and drops every buffer still open for a choice, or for all of them; returns the non-empty tails."""
        closing: Final = tuple(channel for channel in self._buffers if choice is None or channel.choice == choice)
        flushed: Final = tuple((channel, self._buffers.pop(channel).flush()) for channel in closing)
        return tuple((channel, tail) for channel, tail in flushed if tail)

    def restore_chunk(self, chunk: ModelResponseStream) -> None:
        """Restores every choice of a chunk, and appends the tails of the channels a finish delta does not carry."""
        for choice in read_list(chunk, "choices"):
            delta = read_field(choice, "delta")
            if delta is None:
                continue
            index = index_of(choice)
            self._last_chunk[index] = chunk
            finished = read_field(choice, "finish_reason") is not None
            restore_delta(delta, index, finished, self.restore)
            if finished:
                for channel, tail in self._close(index):
                    append_tail(delta, channel, tail)

    def drain(self) -> Iterator[ModelResponseStream]:
        for channel, tail in self._close(None):
            template = self._last_chunk.get(channel.choice)
            if template is not None:
                yield drain_chunk(template, channel, tail)

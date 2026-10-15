import copy
import functools
import itertools
import json
import re
from enum import Enum
from types import MappingProxyType
from typing import (
    Final,
    TypeAlias,
)

from .payload import (
    JsonBody,
    MutableRequest,
    Rehydrate,
    SlotSink,
    StreamStep,
    as_object,
    collect_response_item,
    read_field,
    read_list,
    rehydrate_slots,
    write_field,
)

ANTHROPIC_DELTA_FIELDS: Final = MappingProxyType({"text_delta": "text", "input_json_delta": "partial_json"})

SSE_EVENT_BOUNDARY: Final = re.compile(rb"(\r?\n\r?\n)")

SSE_OPENINGS: Final = (b"event:", b"data:", b"id:", b"retry:", b":")

RESPONSES_BINARY_DELTAS: Final = frozenset(("response.audio.delta",))

RESPONSES_STRUCTURAL_FIELDS: Final = frozenset(
    ("type", "id", "item_id", "call_id", "name", "server_label", "status", "obfuscation")
)

RESPONSES_TERMINAL_EVENTS: Final = frozenset(("response.completed", "response.incomplete"))

CarryKey: TypeAlias = tuple[int, int | None]
CarryWindows: TypeAlias = dict[CarryKey, str]

ResponsesStreamKey: TypeAlias = tuple[str, object, object, object]


def carry_sort_key(key: CarryKey) -> tuple[int, int]:
    """Orders streaming windows without ever comparing None to an int.

    `sorted()` over the raw keys raises as soon as one choice holds both a content window
    and a tool-call window, because `None < 0` is not orderable. Content sorts first, then
    tool calls by their index.
    """
    choice_index, tool_index = key
    return (choice_index, -1 if tool_index is None else tool_index)


def continuation_delta(tool_index: int, text: str) -> list[dict[str, object]]:
    """A `tool_calls` delta carrying `text` as an index-only continuation.

    Clients concatenate tool-call fragments by index, so no id or name is needed.
    """
    return [{"index": tool_index, "function": {"arguments": text}}]


def opens_like_sse(head: bytes) -> bool | None:
    """Whether a raw stream is SSE, judged by its opening bytes; None while undecidable.

    An SSE stream opens with a field name or a `:` comment. Anything else -- a JSON array
    streamed in pieces, say -- has no event boundaries to wait for. A chunk that ends
    partway through a field name decides nothing yet, so that case waits for more.
    """
    opening: Final = head.lstrip()
    if not opening:
        return None
    if opening.startswith(SSE_OPENINGS):
        return True
    if any(field.startswith(opening) for field in SSE_OPENINGS):
        return None
    return False


def responses_event_type(chunk: object) -> str | None:
    """The event type of a Responses API stream event, or None for any other chunk.

    The type arrives as a plain string on dicts and as a str-valued Enum on LiteLLM's
    event models. The Enum is unwrapped because it does not hash like its value, so it
    would miss every lookup in the event tables above.
    """
    if isinstance(chunk, (bytes, str)):
        return None
    kind: Final = read_field(chunk, "type")
    value: Final = kind.value if isinstance(kind, Enum) else kind
    return value if isinstance(value, str) and value.startswith("response.") else None


class AnthropicSSERestorer:
    """Restores an Anthropic `/v1/messages` stream, which reaches the hook as raw SSE.

    Each content block is its own token stream with its own window, keyed by the block's
    `index`: `text_delta` carries prose and `input_json_delta` a tool call's arguments.
    When a block stops, whatever its window still holds is emitted as one more delta for
    that block, just ahead of the `content_block_stop` frame, so the client has the whole
    block before it is told the block is complete.

    Frames are processed whole. A network chunk can end in the middle of an event, so the
    unfinished tail is kept until the rest arrives; that delays one partial event, never
    a completed one. A frame that is not an Anthropic event -- another endpoint's SSE, or
    anything that fails to parse -- is passed through byte for byte, and a raw stream that
    does not open like SSE at all is passed through chunk by chunk, never buffered.
    """

    def __init__(self, step: StreamStep) -> None:
        self._step: Final = step
        self._carries: Final[dict[int, str]] = {}  # mutable-ok: per-block windows advanced in place.
        self._delta_types: Final[dict[int, str]] = {}  # mutable-ok: each block's delta type, for its flush.
        self._pending = b""
        self._as_text = False
        self._is_sse: bool | None = None

    async def feed(self, chunk: bytes | str) -> tuple[bytes | str, ...]:
        """Restores every event this chunk completes; holds back an unfinished tail."""
        if isinstance(chunk, str):
            self._as_text = True
        if self._is_sse is False:
            return (chunk,)
        raw: Final = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        buffered: Final = self._pending + raw
        if self._is_sse is None:
            self._is_sse = opens_like_sse(buffered)
            if self._is_sse is None:
                self._pending = buffered
                return ()
            if not self._is_sse:
                self._pending = b""
                return self._emit(buffered)
        boundaries: Final = tuple(SSE_EVENT_BOUNDARY.finditer(buffered))
        if not boundaries:
            self._pending = buffered
            return ()
        cut: Final = boundaries[-1].end()
        self._pending = buffered[cut:]
        parts: Final = SSE_EVENT_BOUNDARY.split(buffered[:cut])
        restored: Final = tuple(
            [await self._restore_event(parts[index]) + parts[index + 1] for index in range(0, len(parts) - 1, 2)]
        )
        return self._emit(b"".join(restored))

    async def finish(self) -> tuple[bytes | str, ...]:
        """Emits an unterminated final event and any window a block never closed."""
        held: Final = self._pending
        self._pending = b""
        if not self._is_sse:
            return self._emit(held)
        tail: Final = await self._restore_event(held) if held.strip() else held
        flushed: Final = await self._flush_all()
        separator: Final = b"\n\n" if tail.strip() and flushed else b""
        return self._emit(tail + separator + flushed)

    def _emit(self, frames: bytes) -> tuple[bytes | str, ...]:
        if not frames:
            return ()
        return (frames.decode("utf-8") if self._as_text else frames,)

    async def _restore_event(self, block: bytes) -> bytes:
        """Rewrites one SSE event, or returns it untouched if it carries nothing to restore."""
        try:
            lines: Final = block.decode("utf-8").split("\n")
        except UnicodeDecodeError:
            return block
        data_lines: Final = tuple(index for index, line in enumerate(lines) if line.startswith("data:"))
        if len(data_lines) != 1:
            return block
        line: Final = lines[data_lines[0]]
        try:
            parsed: Final[object] = json.loads(line[len("data:") :])
        except ValueError:
            return block
        event: Final = as_object(parsed)
        if event is None:
            return block
        kind: Final = event.get("type")
        index: Final = event.get("index")
        if kind == "content_block_stop" and isinstance(index, int):
            return await self._flush(index) + block
        if kind == "message_stop":
            return await self._flush_all() + block
        if kind != "content_block_delta" or not await self._restore_delta(event):
            return block
        ending: Final = "\r" if line.endswith("\r") else ""
        rewritten: Final = (
            *lines[: data_lines[0]],
            f"data: {json.dumps(event, ensure_ascii=False)}{ending}",
            *lines[data_lines[0] + 1 :],
        )
        return "\n".join(rewritten).encode("utf-8")

    async def _restore_delta(self, event: MutableRequest) -> bool:
        """Advances one block's window through this delta. False if it holds no text."""
        index: Final = event.get("index")
        delta: Final = event.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            return False
        delta_type: Final = delta.get("type")
        if not isinstance(delta_type, str):
            return False
        field: Final = ANTHROPIC_DELTA_FIELDS.get(delta_type)
        text: Final = delta.get(field) if field is not None else None
        if field is None or not isinstance(text, str) or not text:
            return False
        emitted, remaining = await self._step(text, self._carries.get(index, ""), False)
        self._carries[index] = remaining
        self._delta_types[index] = delta_type
        delta[field] = emitted
        return True

    async def _flush(self, index: int) -> bytes:
        """One synthetic delta frame carrying whatever `index`'s window still holds."""
        carry: Final = self._carries.pop(index, "")
        delta_type: Final = self._delta_types.pop(index, None)
        field: Final = ANTHROPIC_DELTA_FIELDS.get(delta_type) if isinstance(delta_type, str) else None
        if not carry or field is None:
            return b""
        text, _ = await self._step("", carry, True)
        if not text:
            return b""
        event: Final[JsonBody] = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": delta_type, field: text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

    async def _flush_all(self) -> bytes:
        flushed: Final = tuple([await self._flush(index) for index in tuple(self._carries)])
        return b"".join(flushed)


class ResponsesStreamRestorer:
    """Restores a Responses API event stream.

    The event families are matched by shape rather than listed, so a text stream the
    API adds later is restored by default instead of leaking a placeholder:

    - Any `*.delta` event whose `delta` is a string is a token stream (output_text,
      refusal, function-call and MCP arguments, reasoning summaries, ...). Each gets its
      own window, keyed by the family, the item id and the part index.
    - Any `*.done` event closes the stream of the same family. Whatever its window still
      holds goes out first, as a copy of that stream's last delta event -- so it carries
      the stream's own ids, and repeats that event's `sequence_number`. Then every text
      field on the done event is restored in full: its string fields other than
      identifiers, plus any `part` or `item` it repeats.
    - `response.completed` / `response.incomplete` repeat the whole reply, and are
      restored the same way the non-streaming reply is.
    """

    def __init__(self, step: StreamStep, rehydrate: Rehydrate) -> None:
        self._step: Final = step
        self._rehydrate: Final = rehydrate
        self._carries: Final[dict[ResponsesStreamKey, str]] = {}  # mutable-ok: per-stream windows advanced in place.
        self._last_deltas: Final[dict[ResponsesStreamKey, object]] = {}  # mutable-ok: newest delta per stream.

    async def restore(self, event: object) -> tuple[object, ...]:
        """The events to emit in place of `event`: any flush, then the event itself."""
        kind: Final = responses_event_type(event)
        if kind is None:
            return (event,)
        if kind.endswith(".delta") and kind not in RESPONSES_BINARY_DELTAS:
            await self._restore_delta(event, kind)
            return (event,)
        slots: Final[SlotSink] = []
        flushed: Final = await self._flush(responses_stream_key(event, kind)) if kind.endswith(".done") else ()
        if kind.endswith(".done"):
            collect_event_text(event, slots)
            part: Final = read_field(event, "part")
            if part is not None:
                collect_response_item({"content": [part]}, slots)
            collect_response_item(read_field(event, "item"), slots)
        elif kind in RESPONSES_TERMINAL_EVENTS:
            for item in read_list(read_field(event, "response"), "output"):
                collect_response_item(item, slots)
        await rehydrate_slots(slots, self._rehydrate)
        return (*flushed, event)

    async def finish(self) -> tuple[object, ...]:
        """Flushes every stream the provider never closed, e.g. a truncated reply."""
        flushed: Final = tuple([await self._flush(key) for key in tuple(self._carries)])
        return tuple(itertools.chain.from_iterable(flushed))

    async def _restore_delta(self, event: object, kind: str) -> None:
        text: Final = read_field(event, "delta")
        if not isinstance(text, str) or not text:
            return
        key: Final = responses_stream_key(event, kind)
        emitted, remaining = await self._step(text, self._carries.get(key, ""), False)
        self._carries[key] = remaining
        self._last_deltas[key] = event
        write_field(event, "delta", emitted)

    async def _flush(self, key: ResponsesStreamKey) -> tuple[object, ...]:
        carry: Final = self._carries.pop(key, "")
        template: Final = self._last_deltas.pop(key, None)
        if not carry or template is None:
            return ()
        text, _ = await self._step("", carry, True)
        if not text:
            return ()
        flush: Final = copy.deepcopy(template)
        write_field(flush, "delta", text)
        return (flush,)


def responses_stream_key(event: object, kind: str) -> ResponsesStreamKey:
    """Identifies the delta stream an event belongs to, the same for its delta and done.

    The family is the event type without its `.delta` / `.done` suffix, so an output_text
    stream and a refusal stream on the same part never share a window.
    """
    family: Final = kind.rsplit(".", 1)[0]
    part_index: Final = read_field(event, "content_index")
    summary_index: Final = read_field(event, "summary_index")
    return (
        family,
        read_field(event, "item_id"),
        read_field(event, "output_index"),
        part_index if part_index is not None else summary_index,
    )


def collect_event_text(event: object, slots: SlotSink) -> None:
    """Collects every top-level text field of a Responses API event, dict or model.

    Scan by default, with identifiers excluded, rather than a list of known fields: the
    `.done` event of each stream family names its text differently (`text`, `refusal`,
    `arguments`, ...), and a family added upstream would otherwise leak a placeholder.
    """
    attributes: Final[object] = getattr(event, "__dict__", None)
    fields: Final = as_object(event) or as_object(attributes)
    if fields is None:
        return
    for name, value in tuple(fields.items()):
        if name in RESPONSES_STRUCTURAL_FIELDS or name.endswith("_id"):
            continue
        if isinstance(value, str) and value:
            slots.append((value, functools.partial(write_field, event, name)))

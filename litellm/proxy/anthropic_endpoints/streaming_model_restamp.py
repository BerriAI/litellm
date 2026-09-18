"""
Restamp the public ``model`` on the Anthropic Messages ``message_start`` event, the only
stream event carrying a model, so streamed responses report the requested model like
non-streaming ones do.

Chunks reach the serializer either as already-encoded SSE frames (``bytes``/``str``, the
provider passthrough path) or as event dicts (fake-stream and agentic paths).
"""

import json
import re
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

_MESSAGE_START_EVENT: Final = "message_start"
_MESSAGE_START_MARKER: Final = b"message_start"
_SSE_DATA_FIELD: Final = "data:"
_SSE_FRAME_END_PATTERN: Final = re.compile(rb"\r\n\r\n|\r\r|\n\n")
_MAX_HELD_BYTES: Final = 65536
_PING_MARKERS: Final = (b"event: ping", b'"type": "ping"', b'"type":"ping"')

_EVENT_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _restamped_event(event: Mapping[str, object], requested_model: str) -> Mapping[str, object] | None:
    message: Final = event.get("message")
    if event.get("type") != _MESSAGE_START_EVENT or not isinstance(message, dict):
        return None
    if message.get("model") == requested_model:
        return None
    return {**event, "message": {**message, "model": requested_model}}  # mutable-ok: SSE payload, re-serialized as is


def _restamped_data_line(line: str, requested_model: str) -> str | None:
    stripped: Final = line.strip()
    if not stripped.startswith(_SSE_DATA_FIELD):
        return None
    payload: Final = stripped[len(_SSE_DATA_FIELD) :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        event: Final = _EVENT_ADAPTER.validate_json(payload)
    except ValidationError:
        return None
    restamped: Final = _restamped_event(event, requested_model)
    if restamped is None:
        return None
    terminator: Final = line[len(line.rstrip("\r\n")) :]
    return f"data: {json.dumps(restamped, separators=(',', ':'))}{terminator}"


def _restamped_frame(frame: str, requested_model: str) -> str | None:
    lines: Final = frame.splitlines(keepends=True)
    restamped: Final = tuple(_restamped_data_line(line, requested_model) for line in lines)
    if all(line is None for line in restamped):
        return None
    return "".join(new if new is not None else old for new, old in zip(restamped, lines))


def restamp_anthropic_stream_chunk_model(chunk: object, requested_model: str) -> object:
    """
    Return ``chunk`` with the ``message_start`` model replaced by ``requested_model``.

    Chunks that carry no model are returned unchanged.
    """
    if isinstance(chunk, dict):
        try:
            event: Final = _EVENT_ADAPTER.validate_python(chunk)
        except ValidationError:
            return chunk
        return _restamped_event(event, requested_model) or chunk

    if isinstance(chunk, (bytes, bytearray)):
        if _MESSAGE_START_EVENT.encode() not in chunk:
            return chunk
        restamped_bytes: Final = _restamped_frame(chunk.decode("utf-8", errors="ignore"), requested_model)
        return chunk if restamped_bytes is None else restamped_bytes.encode("utf-8")

    if isinstance(chunk, str):
        if _MESSAGE_START_EVENT not in chunk:
            return chunk
        restamped_text: Final = _restamped_frame(chunk, requested_model)
        return chunk if restamped_text is None else restamped_text

    return chunk


def _is_ping_frame(frame: bytes) -> bool:
    return any(marker in frame for marker in _PING_MARKERS)


class AnthropicStreamModelRestamper:
    """
    Per-stream restamper for the encoded passthrough path, where chunks are raw
    transport reads: the ``message_start`` SSE frame can arrive split across
    chunks or coalesced with later frames. Complete frames (``\\n\\n``,
    ``\\r\\n\\r\\n``, or ``\\r\\r`` terminated) are emitted as their terminator
    closes them and an incomplete tail is held until it completes, so the
    restamp never misses a torn frame; ``flush`` returns whatever is still held
    when the stream ends so no bytes are swallowed. Once ``message_start`` has
    been handled, or the first real event proves the stream carries none, every
    later chunk passes through untouched.
    """

    def __init__(self, requested_model: str) -> None:
        self._requested_model: Final = requested_model
        self._held = b""
        self._armed = True

    def process(self, chunk: object) -> object:
        if not self._armed:
            return chunk
        if isinstance(chunk, (bytes, bytearray)):
            return self._process_encoded(bytes(chunk))
        if isinstance(chunk, str):
            return self._process_encoded(chunk.encode("utf-8"))
        restamped: Final = restamp_anthropic_stream_chunk_model(chunk, self._requested_model)
        if isinstance(chunk, dict) and chunk.get("type") not in (None, "ping"):
            self._armed = False
        return restamped

    def flush(self) -> bytes:
        held: Final = self._held
        self._held = b""
        self._armed = False
        if not held:
            return b""
        restamped: Final = restamp_anthropic_stream_chunk_model(held, self._requested_model)
        return restamped if isinstance(restamped, bytes) else held

    def _process_encoded(self, data: bytes) -> bytes:
        combined: Final = self._held + data
        boundaries: Final = tuple(match.end() for match in _SSE_FRAME_END_PATTERN.finditer(combined))
        if not boundaries:
            if len(combined) > _MAX_HELD_BYTES:
                self._held = b""
                self._armed = False
                return combined
            self._held = combined
            return b""
        emitted: Final = self._restamped_closed_block(combined[: boundaries[-1]])
        tail: Final = combined[boundaries[-1] :]
        if not self._armed:
            self._held = b""
            return emitted + tail
        self._held = tail
        return emitted

    def _restamped_closed_block(self, closed: bytes) -> bytes:
        boundaries: Final = tuple(match.end() for match in _SSE_FRAME_END_PATTERN.finditer(closed))
        frames: Final = tuple(closed[start:end] for start, end in zip((0, *boundaries[:-1]), boundaries))
        decider: Final = next(
            (
                index
                for index, frame in enumerate(frames)
                if _MESSAGE_START_MARKER in frame or (b"data:" in frame and not _is_ping_frame(frame))
            ),
            None,
        )
        if decider is None:
            return closed
        self._armed = False
        if _MESSAGE_START_MARKER not in frames[decider]:
            return closed
        restamped_text: Final = _restamped_frame(
            frames[decider].decode("utf-8", errors="ignore"), self._requested_model
        )
        if restamped_text is None:
            return closed
        return b"".join(
            restamped_text.encode("utf-8") if index == decider else frame for index, frame in enumerate(frames)
        )

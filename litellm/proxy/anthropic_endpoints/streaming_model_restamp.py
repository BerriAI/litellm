"""
Restamp the public ``model`` on the Anthropic Messages ``message_start`` event, the only
stream event carrying a model, so streamed responses report the requested model like
non-streaming ones do.

Chunks reach the serializer either as already-encoded SSE frames (``bytes``/``str``, the
provider passthrough path) or as event dicts (fake-stream and agentic paths).
"""

import json
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

_MESSAGE_START_EVENT: Final = "message_start"
_SSE_DATA_FIELD: Final = "data:"

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
    return f"data: {json.dumps(restamped, separators=(',', ':'))}"


def _restamped_frame(frame: str, requested_model: str) -> str | None:
    lines: Final = frame.split("\n")
    restamped: Final = tuple(_restamped_data_line(line, requested_model) for line in lines)
    if all(line is None for line in restamped):
        return None
    return "\n".join(new if new is not None else old for new, old in zip(restamped, lines))


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

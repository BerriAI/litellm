import base64
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final

from botocore.eventstream import EventStreamBuffer, EventStreamMessage
from pydantic import BaseModel, Field, JsonValue, TypeAdapter


class _EventHeaders(BaseModel):
    message_type: str = Field(alias=":message-type")
    event_type: str = Field(default="", alias=":event-type")
    exception_type: str = Field(default="", alias=":exception-type")


class _InvokeChunk(BaseModel):
    bytes: str


class _PreludeLength(BaseModel):
    total_length: int


@dataclass(frozen=True, slots=True)
class BedrockEvent:
    payload: str
    error: str | None = None
    error_code: str | None = None


_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _decode_event(event: EventStreamMessage) -> BedrockEvent:
    headers: Final = _EventHeaders.model_validate(event.headers)
    if headers.message_type != "event":
        return BedrockEvent(
            payload=event.payload.decode(),
            error=f"{headers.exception_type or headers.message_type}: {event.payload.decode()}",
            error_code=headers.exception_type or headers.message_type,
        )
    if headers.event_type == "chunk":
        encoded: Final = _InvokeChunk.model_validate_json(event.payload)
        return BedrockEvent(payload=base64.b64decode(encoded.bytes, validate=True).decode())
    return BedrockEvent(payload=_JSON.dump_json({headers.event_type: _JSON.validate_json(event.payload)}).decode())


class _CompleteEventStream:
    def __init__(self) -> None:
        self._buffer: Final = EventStreamBuffer()
        self._pending_bytes: int = 0

    def feed(self, chunk: bytes) -> Iterator[BedrockEvent]:
        self._pending_bytes += len(chunk)
        self._buffer.add_data(chunk)
        for event in self._buffer:
            self._pending_bytes -= _PreludeLength.model_validate(event.prelude, from_attributes=True).total_length
            yield _decode_event(event)

    def finish(self) -> None:
        assert self._pending_bytes == 0, f"incomplete Bedrock frame: {self._pending_bytes} trailing bytes"


def decode_bedrock_stream(chunks: Iterable[bytes]) -> Iterator[BedrockEvent]:
    stream: Final = _CompleteEventStream()
    for chunk in chunks:
        yield from stream.feed(chunk)
    stream.finish()

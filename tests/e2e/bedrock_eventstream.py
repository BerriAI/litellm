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


@dataclass(frozen=True, slots=True)
class BedrockEvent:
    payload: str
    error: str | None = None


_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _decode_event(event: EventStreamMessage) -> BedrockEvent:
    headers: Final = _EventHeaders.model_validate(event.headers)
    if headers.message_type != "event":
        return BedrockEvent(
            payload=event.payload.decode(),
            error=f"{headers.exception_type or headers.message_type}: {event.payload.decode()}",
        )
    if headers.event_type == "chunk":
        encoded: Final = _InvokeChunk.model_validate_json(event.payload)
        return BedrockEvent(payload=base64.b64decode(encoded.bytes, validate=True).decode())
    return BedrockEvent(payload=_JSON.dump_json({headers.event_type: _JSON.validate_json(event.payload)}).decode())


def decode_bedrock_stream(chunks: Iterable[bytes]) -> Iterator[BedrockEvent]:
    buffer: Final = EventStreamBuffer()
    for chunk in chunks:
        buffer.add_data(chunk)
        for event in buffer:
            yield _decode_event(event)

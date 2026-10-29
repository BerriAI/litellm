from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.llm_response_utils.get_headers import provider_request_id_headers

if TYPE_CHECKING:
    from botocore.eventstream import EventStreamMessage

STREAM_HEAD_BYTES: Final = 200
_NO_EVENT_HEADERS: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class UndecodedEventStream:
    status_code: int
    message: str
    headers: httpx.Headers | None


def _string_header(headers: Mapping[str, object] | None, name: str) -> str | None:
    value: Final = None if headers is None else headers.get(name)
    return value if isinstance(value, str) else None


class EventStreamTally:
    def __init__(self, provider: str, event_statuses: Mapping[str, int | None] | None = None) -> None:
        self.provider = provider
        self.event_statuses = event_statuses
        self.bytes_received = 0
        self.bytes_decoded = 0
        self.events = 0
        self.recognized_events = 0
        self.unrecognized_event_types: frozenset[str] = frozenset()
        self.unrecognized_head = b""
        self.head = b""

    def add_chunk(self, chunk: bytes) -> None:
        self.bytes_received += len(chunk)
        if len(self.head) < STREAM_HEAD_BYTES:
            self.head = (self.head + chunk)[:STREAM_HEAD_BYTES]

    def add_event(self, event: "EventStreamMessage", headers: Mapping[str, object] = _NO_EVENT_HEADERS) -> None:
        self.events += 1
        self.bytes_decoded += event.prelude.total_length
        event_type: Final = _string_header(headers, ":event-type")
        if (
            self.event_statuses is None
            or _string_header(headers, ":message-type") != "event"
            or (event_type is not None and event_type in self.event_statuses)
        ):
            self.recognized_events += 1
            return
        self.unrecognized_event_types = self.unrecognized_event_types | {event_type or "<missing>"}
        if not self.unrecognized_head:
            self.unrecognized_head = event.payload[:STREAM_HEAD_BYTES]

    def undecoded_stream(self, response_headers: Mapping[str, str] | None) -> UndecodedEventStream | None:
        undecoded: Final = self.bytes_received - self.bytes_decoded
        if self.recognized_events and not undecoded:
            return None
        detail: Final = (
            f"content-type={_string_header(response_headers, 'content-type')!r}, "
            f"x-amzn-requestid={_string_header(response_headers, 'x-amzn-requestid')!r}, "
            f"{self.bytes_received} bytes received"
        )
        request_id_headers: Final = provider_request_id_headers(response_headers)
        if not self.events:
            return UndecodedEventStream(
                status_code=502,
                message=(
                    f"{self.provider} answered the stream with HTTP 200 but its body decoded to no events "
                    f"({detail}, first bytes={self.head!r})"
                ),
                headers=request_id_headers,
            )
        if not self.recognized_events:
            return UndecodedEventStream(
                status_code=502,
                message=(
                    f"{self.provider} answered the stream with HTTP 200 but none of its {self.events} events carried "
                    f"a known event type (event types={sorted(self.unrecognized_event_types)}, {detail}, "
                    f"first payload={self.unrecognized_head!r})"
                ),
                headers=request_id_headers,
            )
        return UndecodedEventStream(
            status_code=502,
            message=(
                f"{self.provider} stream ended with {undecoded} undecoded bytes after {self.events} events ({detail})"
            ),
            headers=request_id_headers,
        )

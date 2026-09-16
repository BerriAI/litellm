from __future__ import annotations

import base64
import hashlib
import hmac
import io
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit

from e2e_http import (
    NetworkError,
    StreamChunk,
    StreamHead,
    StreamStep,
    StreamTruncation,
    forward_prepared_stream,
    forward_stream,
    prepare_forward,
    primed_steps,
)
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

LIFETIME_SECONDS: Final = 86_400
MAX_REQUEST_BYTES: Final = 256 * 1024
MAX_RESPONSE_BYTES: Final = 8 * 1024 * 1024
UNRECORDED_RESPONSE_HEADERS: Final = frozenset({"set-cookie"})
JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class CacheHit:
    payload: bytes
    valid_until: float


@dataclass(frozen=True, slots=True)
class CaptureLease:
    token: str
    captured_at_ms: int
    expires_at_ms: int


@dataclass(frozen=True, slots=True)
class CacheBusy:
    pass


@dataclass(frozen=True, slots=True)
class CacheUnavailable:
    pass


type CacheLookup = CacheHit | CaptureLease | CacheBusy | CacheUnavailable


class ResponseStore(Protocol):
    def lookup(self, key: str) -> CacheLookup: ...

    def publish(self, key: str, lease: CaptureLease, payload: bytes) -> bool: ...

    def release(self, key: str, lease: CaptureLease) -> bool: ...

    def discard(self, key: str, payload: bytes) -> bool: ...


class CachedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    format_version: Literal[1] = 1
    request_key: str
    status_code: int
    headers: dict[str, str]
    chunks: tuple[str, ...]


class SignedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    response: str
    signature: str


def exact_key(secret: bytes, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> str:
    fields: Final = (
        b"provider-cache-exact-v1", method.encode(), url.encode(),
        *(part.encode() for pair in sorted(headers.items()) for part in pair),
        b"no-body" if body is None else b"body", b"" if body is None else body,
    )
    encoded: Final = b"".join(len(part).to_bytes(8, "big") + part for part in fields)
    return hmac.new(secret, encoded, hashlib.sha256).hexdigest()


def cacheable_endpoint(method: str, url: str, body: bytes | None) -> bool:
    return (
        method == "POST"
        and urlsplit(url).path in {"/v1/chat/completions", "/v1/messages"}
        and body is not None
        and len(body) <= MAX_REQUEST_BYTES
    )


def successful_response(url: str, status: int, headers: Mapping[str, str], body: bytes) -> bool:
    if not 200 <= status < 300 or len(body) > MAX_RESPONSE_BYTES:
        return False
    streaming: Final = "text/event-stream" in headers.get("content-type", "").lower()
    if streaming:
        try:
            text: Final = body.decode("utf-8").replace("\r\n", "\n")
            if not text.endswith("\n\n"):
                return False
            events: Final = tuple(
                "\n".join(line[5:].removeprefix(" ") for line in event.split("\n") if line.startswith("data:"))
                for event in text.split("\n\n") if any(line.startswith("data:") for line in event.split("\n"))
            )
            values: Final = tuple(JSON_VALUE.validate_json(event) for event in events if event != "[DONE]")
        except (UnicodeDecodeError, ValidationError):
            return False
        if not values or any(not isinstance(value, dict) or "error" in value or value.get("type") == "error" for value in values):
            return False
        if urlsplit(url).path == "/v1/chat/completions":
            return events[-1] == "[DONE]" and "[DONE]" not in events[:-1] and complete_chat_stream(values)
        return (
            "[DONE]" not in events
            and isinstance(values[0], dict) and values[0].get("type") == "message_start"
            and isinstance(values[-1], dict) and values[-1].get("type") == "message_stop"
            and any(
                isinstance(value, dict) and value.get("type") == "message_delta"
                and isinstance(delta := value.get("delta"), dict) and isinstance(delta.get("stop_reason"), str)
                for value in values
            )
        )
    try:
        value: Final = JSON_VALUE.validate_json(body)
    except ValidationError:
        return False
    if not isinstance(value, dict) or "error" in value:
        return False
    if urlsplit(url).path == "/v1/messages":
        return value.get("type") == "message" and isinstance(value.get("content"), list) and isinstance(value.get("stop_reason"), str)
    choices: Final = value.get("choices")
    return isinstance(choices, list) and bool(choices) and all(
        isinstance(choice, dict) and isinstance(choice.get("message"), dict) and isinstance(choice.get("finish_reason"), str)
        for choice in choices
    )


def complete_chat_stream(values: tuple[JsonValue, ...]) -> bool:
    if any(not isinstance(value, dict) or not isinstance(value.get("choices"), list) for value in values):
        return False
    choices: Final = tuple(
        choice for value in values if isinstance(value, dict)
        if isinstance(items := value.get("choices"), list) for choice in items
    )
    if not choices or any(
        not isinstance(choice, dict) or type(choice.get("index")) is not int
        or not isinstance(choice.get("delta"), dict)
        for choice in choices
    ):
        return False
    indices: Final = frozenset(choice["index"] for choice in choices if isinstance(choice, dict))
    return all(
        isinstance(tuple(choice for choice in choices if isinstance(choice, dict) and choice["index"] == index)[-1].get("finish_reason"), str)
        for index in indices
    )


def encode_response(secret: bytes, response: CachedResponse) -> bytes:
    raw: Final = response.model_dump_json()
    return SignedResponse(response=raw, signature=hmac.new(secret, raw.encode(), hashlib.sha256).hexdigest()).model_dump_json().encode()


def decode_response(secret: bytes, key: str, payload: bytes, url: str) -> CachedResponse | None:
    if len(payload) > 2 * MAX_RESPONSE_BYTES:
        return None
    try:
        signed: Final = SignedResponse.model_validate_json(payload)
        if not hmac.compare_digest(signed.signature.encode(), hmac.new(secret, signed.response.encode(), hashlib.sha256).hexdigest().encode()):
            return None
        response: Final = CachedResponse.model_validate_json(signed.response)
        chunks: Final = tuple(base64.b64decode(chunk, validate=True) for chunk in response.chunks)
    except (ValidationError, ValueError):
        return None
    if response.request_key != key or not successful_response(url, response.status_code, response.headers, b"".join(chunks)):
        return None
    return response


@dataclass(slots=True)
class CacheCounters:
    counts: tuple[tuple[str, int], ...] = ()
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, name: str) -> None:
        with self.lock:
            current: Final = dict(self.counts)
            self.counts = tuple((current | {name: current.get(name, 0) + 1}).items())


@dataclass(slots=True)
class ResponseCapture:
    buffer: io.BytesIO = field(default_factory=io.BytesIO)
    size: int = 0
    eligible: bool = True

    def observe(self, step: StreamStep) -> None:
        if not self.eligible:
            return
        if isinstance(step, StreamTruncation) or self.size + len(step.data) + 8 > MAX_RESPONSE_BYTES:
            self.eligible = False
            self.buffer.close()
            return
        self.buffer.write(len(step.data).to_bytes(8, "big"))
        self.buffer.write(step.data)
        self.size += len(step.data) + 8

    def chunks(self) -> tuple[bytes, ...]:
        self.buffer.seek(0)
        return tuple(self.buffer.read(int.from_bytes(size, "big")) for size in iter(lambda: self.buffer.read(8), b""))


def response_steps(response: CachedResponse) -> Generator[StreamStep, None, None]:
    for chunk in response.chunks:
        yield StreamChunk(base64.b64decode(chunk, validate=True))


@dataclass(frozen=True, slots=True)
class CacheEdge:
    store: ResponseStore
    secret: bytes = field(repr=False)
    counters: CacheCounters = field(default_factory=CacheCounters)
    wait_seconds: float = 2.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def lookup(self, key: str) -> CacheLookup:
        deadline: Final = self.clock() + self.wait_seconds
        while isinstance(result := self.store.lookup(key), CacheBusy) and self.clock() < deadline:
            self.sleep(min(0.05, max(0, deadline - self.clock())))
        return result

    def forward(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> StreamHead | NetworkError:
        if not cacheable_endpoint(method, url, body):
            self.counters.increment("bypass")
            self.counters.increment("upstream_attempts")
            return forward_stream(method, url, headers=headers, body=body, timeout=timeout)
        prepared: Final = prepare_forward(method, url, headers, body)
        if isinstance(prepared, NetworkError):
            self.counters.increment("rejected")
            return prepared
        key: Final = exact_key(self.secret, method, url, prepared.headers, body)
        found: Final = self.lookup(key)
        if isinstance(found, CacheHit):
            response: Final = decode_response(self.secret, key, found.payload, url)
            if response is not None and self.clock() < found.valid_until:
                self.counters.increment("hits")
                return StreamHead(response.status_code, response.headers, response_steps(response))
            self.counters.increment("corrupt" if response is None else "expired")
            self.store.discard(key, found.payload)
        capture_slot: Final = self.lookup(key) if isinstance(found, CacheHit) else found
        self.counters.increment("misses")
        if isinstance(capture_slot, CacheUnavailable):
            self.counters.increment("cache_errors")
        self.counters.increment("upstream_attempts")
        head: Final = forward_prepared_stream(prepared, timeout)
        if not isinstance(capture_slot, CaptureLease):
            return head
        if isinstance(head, NetworkError):
            self.store.release(key, capture_slot)
            self.counters.increment("rejected")
            return head
        return StreamHead(head.status_code, head.headers, primed_steps(self.capture(key, capture_slot, url, head)))

    def capture(self, key: str, lease: CaptureLease, url: str, head: StreamHead) -> Generator[StreamStep, None, None]:
        capture: Final = ResponseCapture()
        try:
            with closing(head.steps):
                yield StreamChunk(b"")
                for step in head.steps:
                    yield step
                    capture.observe(step)
            chunks: Final = capture.chunks() if capture.eligible else ()
            headers: Final = {
                name: value for name, value in head.headers.items() if name.lower() not in UNRECORDED_RESPONSE_HEADERS
            }
            if not capture.eligible or not successful_response(url, head.status_code, headers, b"".join(chunks)):
                self.counters.increment("rejected")
                return
            response: Final = CachedResponse(
                request_key=key, status_code=head.status_code, headers=headers,
                chunks=tuple(base64.b64encode(chunk).decode("ascii") for chunk in chunks),
            )
            published: Final = self.store.publish(key, lease, encode_response(self.secret, response))
            self.counters.increment("writes" if published else "write_failures")
        finally:
            self.store.release(key, lease)
            capture.buffer.close()

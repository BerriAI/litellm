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
from types import MappingProxyType
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
from fixture_canonical import MARKER_PATTERN, MARKER_PLACEHOLDER
from fixture_mode import SESSION_TEST_KEY, current_test_key
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

LIFETIME_SECONDS: Final = 86_400
MAX_REQUEST_BYTES: Final = 256 * 1024
MAX_RESPONSE_BYTES: Final = 8 * 1024 * 1024
UNRECORDED_RESPONSE_HEADERS: Final = frozenset({"set-cookie"})
SIGNATURE_HEADERS: Final = frozenset(
    {"authorization", "x-amz-date", "x-amz-security-token", "x-amz-content-sha256"}
)
BEDROCK_MOUNT_PREFIX: Final = "bedrock"
OPENAI_JSON_PATHS: Final = frozenset({"/v1/chat/completions", "/v1/messages", "/v1/embeddings", "/v1/responses"})
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
type RequestSigner = Callable[[str, str, Mapping[str, str], bytes | None], dict[str, str]]


@dataclass(frozen=True, slots=True)
class MountPolicy:
    """What a mount needs beyond plain forwarding.

    ``sign`` mints a fresh credential over the upstream URL, for providers whose
    auth covers the Host the edge rewrote. ``unkeyed_headers`` names headers that
    must stay out of the cache key because they change on every call and would
    otherwise make the mount a permanent miss: a minted signature, or an OAuth
    token the provider rotates. Naming one costs the guarantee that a recording
    can never cross credentials, so a mount with a rotating token relies on the
    environment holding one identity for that provider. Mounts with a static API
    key name nothing here and keep the guarantee whole."""

    sign: RequestSigner | None = None
    unkeyed_headers: frozenset[str] = frozenset()


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


def canonical_text(value: str) -> str:
    return MARKER_PATTERN.sub(MARKER_PLACEHOLDER, value)


def canonical_body(body: bytes) -> bytes:
    try:
        return canonical_text(body.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:
        return body


def request_identity(
    secret: bytes, test_key: str, method: str, url: str, headers: Mapping[str, str], body: bytes | None,
) -> str:
    fields: Final = (
        b"provider-cache-canonical-v2", test_key.encode(), method.encode(), canonical_text(url).encode(),
        *(part.encode() for pair in sorted(headers.items()) for part in pair),
        b"no-body" if body is None else b"body", b"" if body is None else canonical_body(body),
    )
    encoded: Final = b"".join(len(part).to_bytes(8, "big") + part for part in fields)
    return hmac.new(secret, encoded, hashlib.sha256).hexdigest()


def slotted_key(secret: bytes, identity: str, slot: int) -> str:
    return hmac.new(secret, f"{identity}:{slot}".encode(), hashlib.sha256).hexdigest()


def is_bedrock(mount: str) -> bool:
    return mount.partition("/")[0] == BEDROCK_MOUNT_PREFIX


def cacheable_endpoint(mount: str, method: str, url: str, body: bytes | None) -> bool:
    if method != "POST" or body is None or len(body) > MAX_REQUEST_BYTES:
        return False
    path: Final = urlsplit(url).path
    if is_bedrock(mount):
        return path.startswith("/model/") and path.endswith(("/converse", "/invoke"))
    return path in OPENAI_JSON_PATHS


def successful_response(mount: str, url: str, status: int, headers: Mapping[str, str], body: bytes) -> bool:
    if not 200 <= status < 300 or len(body) > MAX_RESPONSE_BYTES:
        return False
    if is_bedrock(mount):
        return complete_bedrock_response(url, body)
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
        if urlsplit(url).path == "/v1/responses":
            return complete_responses_stream(values)
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
    path: Final = urlsplit(url).path
    if path == "/v1/messages":
        return value.get("type") == "message" and isinstance(value.get("content"), list) and isinstance(value.get("stop_reason"), str)
    if path == "/v1/embeddings":
        data: Final = value.get("data")
        return isinstance(data, list) and bool(data) and isinstance(value.get("usage"), dict) and all(
            isinstance(item, dict) and isinstance(item.get("embedding"), list) and bool(item["embedding"])
            for item in data
        )
    if path == "/v1/responses":
        return value.get("object") == "response" and value.get("status") == "completed"
    choices: Final = value.get("choices")
    return isinstance(choices, list) and bool(choices) and all(
        isinstance(choice, dict) and isinstance(choice.get("message"), dict) and isinstance(choice.get("finish_reason"), str)
        for choice in choices
    )


def complete_bedrock_response(url: str, body: bytes) -> bool:
    """Converse answers with ``output`` plus a ``stopReason``; InvokeModel on an
    Anthropic model answers the Anthropic message shape. Either way a truncated
    or error body is missing the terminator field, which is what makes it safe to
    record. The streaming variants never reach here: they are not cacheable."""
    try:
        value: Final = JSON_VALUE.validate_json(body)
    except ValidationError:
        return False
    if not isinstance(value, dict) or "message" in value:
        return False
    if urlsplit(url).path.endswith("/converse"):
        return isinstance(value.get("output"), dict) and isinstance(value.get("stopReason"), str)
    return (
        value.get("type") == "message"
        and isinstance(value.get("content"), list)
        and isinstance(value.get("stop_reason"), str)
    )


def complete_responses_stream(values: tuple[JsonValue, ...]) -> bool:
    """The Responses API streams typed events and ends with ``response.completed``.
    A run that failed, was cancelled, or ran out of tokens ends with a different
    terminal event, so requiring that one keeps a half-finished response out."""
    last: Final = values[-1]
    return isinstance(last, dict) and last.get("type") == "response.completed"


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


def decode_response(secret: bytes, key: str, payload: bytes, mount: str, url: str) -> CachedResponse | None:
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
    if response.request_key != key or not successful_response(
        mount, url, response.status_code, response.headers, b"".join(chunks)
    ):
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
class SlotCounter:
    """FIFO position of a request among the canonically identical ones its test
    has already sent. Two calls in one test that differ only by ``unique_marker``
    canonicalize the same, so without this they would share one recording and the
    second would replay the first's provider response id."""

    counts: tuple[tuple[str, int], ...] = ()
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, identity: str) -> int:
        with self.lock:
            current: Final = dict(self.counts)
            taken: Final = current.get(identity, 0)
            self.counts = tuple((current | {identity: taken + 1}).items())
            return taken


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


NO_POLICIES: Final[Mapping[str, MountPolicy]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class CacheEdge:
    store: ResponseStore
    secret: bytes = field(repr=False)
    counters: CacheCounters = field(default_factory=CacheCounters)
    slots: SlotCounter = field(default_factory=SlotCounter)
    policies: Mapping[str, MountPolicy] = NO_POLICIES
    wait_seconds: float = 2.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    test_key: Callable[[], str] = current_test_key

    def lookup(self, key: str) -> CacheLookup:
        deadline: Final = self.clock() + self.wait_seconds
        while isinstance(result := self.store.lookup(key), CacheBusy) and self.clock() < deadline:
            self.sleep(min(0.05, max(0, deadline - self.clock())))
        return result

    def count(self, mount: str, name: str) -> None:
        self.counters.increment(name)
        self.counters.increment(f"mount:{mount}:{name}")

    def outbound(self, mount: str, method: str, url: str, headers: dict[str, str], body: bytes | None) -> dict[str, str]:
        """The headers actually sent upstream. A signing mount gets a signature
        minted over the upstream URL, because the edge rewrote the Host the proxy
        signed and the provider verifies it."""
        signer: Final = self.policies.get(mount, MountPolicy()).sign
        return headers if signer is None else signer(method, url, headers, body)

    def keyed(self, mount: str, headers: Mapping[str, str]) -> Mapping[str, str]:
        """Headers the cache key is built from. A mount keeps its credentials in
        the key unless its policy names them unkeyed, so by default one account
        can never read another's recording."""
        unkeyed: Final = self.policies.get(mount, MountPolicy()).unkeyed_headers
        if not unkeyed:
            return headers
        return {name: value for name, value in headers.items() if name.lower() not in unkeyed}

    def forward(
        self, mount: str, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float,
    ) -> StreamHead | NetworkError:
        test_key: Final = self.test_key()
        if test_key == SESSION_TEST_KEY or not cacheable_endpoint(mount, method, url, body):
            self.count(mount, "bypass")
            self.count(mount, "upstream_attempts")
            return forward_stream(
                method, url, headers=self.outbound(mount, method, url, headers, body), body=body, timeout=timeout,
            )
        prepared: Final = prepare_forward(method, url, self.outbound(mount, method, url, headers, body), body)
        if isinstance(prepared, NetworkError):
            self.count(mount, "rejected")
            return prepared
        identity: Final = request_identity(
            self.secret, test_key, method, url, self.keyed(mount, prepared.headers), body,
        )
        key: Final = slotted_key(self.secret, identity, self.slots.take(identity))
        found: Final = self.lookup(key)
        if isinstance(found, CacheHit):
            response: Final = decode_response(self.secret, key, found.payload, mount, url)
            if response is not None and self.clock() < found.valid_until:
                self.count(mount, "hits")
                return StreamHead(response.status_code, response.headers, response_steps(response))
            self.count(mount, "corrupt" if response is None else "expired")
            self.store.discard(key, found.payload)
        capture_slot: Final = self.lookup(key) if isinstance(found, CacheHit) else found
        self.count(mount, "misses")
        if isinstance(capture_slot, CacheUnavailable):
            self.count(mount, "cache_errors")
        self.count(mount, "upstream_attempts")
        head: Final = forward_prepared_stream(prepared, timeout)
        if not isinstance(capture_slot, CaptureLease):
            return head
        if isinstance(head, NetworkError):
            self.store.release(key, capture_slot)
            self.count(mount, "rejected")
            return head
        return StreamHead(
            head.status_code, head.headers, primed_steps(self.capture(mount, key, capture_slot, url, head)),
        )

    def capture(
        self, mount: str, key: str, lease: CaptureLease, url: str, head: StreamHead,
    ) -> Generator[StreamStep, None, None]:
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
            if not capture.eligible or not successful_response(mount, url, head.status_code, headers, b"".join(chunks)):
                self.count(mount, "rejected")
                return
            response: Final = CachedResponse(
                request_key=key, status_code=head.status_code, headers=headers,
                chunks=tuple(base64.b64encode(chunk).decode("ascii") for chunk in chunks),
            )
            published: Final = self.store.publish(key, lease, encode_response(self.secret, response))
            self.count(mount, "writes" if published else "write_failures")
        finally:
            self.store.release(key, lease)
            capture.buffer.close()

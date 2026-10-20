import io
import sys
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

import anyio.to_thread
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.constants import (
    MAX_DECOMPRESSED_REQUEST_BODIES_IN_FLIGHT_MB,
    MAX_DECOMPRESSED_REQUEST_BODY_SIZE_MB,
    MAX_REQUEST_BODY_DECOMPRESSION_RATIO,
)
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.middleware.request_size_limit_middleware import MaxRequestSizeGetter, send_request_too_large

RFC_9659_ZSTD_WINDOW_LOG_MAX: Final = 23
GZIP_MAGIC: Final = b"\x1f\x8b"
ZSTD_FRAME_MAGIC: Final = b"\x28\xb5\x2f\xfd"
ZSTD_SKIPPABLE_FRAME_MAGIC_HIGH_BYTES: Final = b"\x2a\x4d\x18"
SIGNATURE_LENGTH: Final = len(ZSTD_FRAME_MAGIC)
GZIP_HEADER_AND_TRAILER_LENGTH: Final = 18
RAW_BODY_ROUTE_PREFIXES: Final = (
    "/plugin-proxy/",
    "/claude_code_gateway/v1/metrics",
    "/claude_code_gateway/v1/logs",
    "/claude_code_gateway/v1/traces",
)
CODING_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"x-gzip": "gzip"})
_REPLACED_REQUEST_HEADERS: Final = frozenset({b"content-encoding", b"content-length", b"transfer-encoding"})
_CONTENT_ENCODING_HEADER: Final = frozenset({b"content-encoding"})


class _Decompressor(Protocol):
    @property
    def eof(self) -> bool: ...

    @property
    def unused_data(self) -> bytes: ...

    def decompress(self, data: bytes, /, max_length: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class Codec:
    name: str
    new_decompressor: Callable[[], _Decompressor]
    error: type[Exception]
    declared_size: Callable[[bytes], int | None]


def _new_gzip_decompressor() -> _Decompressor:
    return zlib.decompressobj(wbits=31)


def _gzip_declared_size(data: bytes) -> int | None:
    if len(data) < GZIP_HEADER_AND_TRAILER_LENGTH:
        return None
    return int.from_bytes(data[-4:], "little")


GZIP: Final = Codec(
    name="gzip",
    new_decompressor=_new_gzip_decompressor,
    error=zlib.error,
    declared_size=_gzip_declared_size,
)


def _starts_a_gzip_member(data: bytes) -> bool:
    return data.startswith(GZIP_MAGIC)


def _starts_a_zstd_frame(data: bytes) -> bool:
    if data.startswith(ZSTD_FRAME_MAGIC):
        return True
    return len(data) >= 4 and data[0] & 0xF0 == 0x50 and data[1:4] == ZSTD_SKIPPABLE_FRAME_MAGIC_HIGH_BYTES


def load_zstd_codec() -> Codec | None:
    try:
        if sys.version_info >= (3, 14):
            from compression.zstd import DecompressionParameter, ZstdDecompressor, ZstdError, get_frame_info
        else:
            from backports.zstd import DecompressionParameter, ZstdDecompressor, ZstdError, get_frame_info
    except ImportError:
        return None

    def new_zstd_decompressor() -> _Decompressor:
        return ZstdDecompressor(options={DecompressionParameter.window_log_max: RFC_9659_ZSTD_WINDOW_LOG_MAX})

    def zstd_declared_size(data: bytes) -> int | None:
        try:
            return get_frame_info(data).decompressed_size
        except ZstdError:
            return None

    return Codec(
        name="zstd",
        new_decompressor=new_zstd_decompressor,
        error=ZstdError,
        declared_size=zstd_declared_size,
    )


_ZSTD: Final = load_zstd_codec()
CODING_SIGNATURES: Final[Mapping[str, Callable[[bytes], bool]]] = MappingProxyType(
    {"gzip": _starts_a_gzip_member, "zstd": _starts_a_zstd_frame}
)
AVAILABLE_CODECS: Final[Mapping[str, Codec]] = MappingProxyType(
    {codec.name: codec for codec in (GZIP, _ZSTD) if codec is not None}
)


@dataclass(frozen=True, slots=True)
class _Prefix:
    data: bytes
    more_body: bool


@dataclass(frozen=True, slots=True)
class _Body:
    data: bytes


@dataclass(frozen=True, slots=True)
class _TooLarge:
    pass


@dataclass(frozen=True, slots=True)
class _DecodedTooLarge:
    pass


@dataclass(frozen=True, slots=True)
class _OverRatio:
    pass


@dataclass(frozen=True, slots=True)
class _ClientDisconnected:
    pass


@dataclass(frozen=True, slots=True)
class _Undecodable:
    message: str


@dataclass(frozen=True, slots=True)
class _Limits:
    max_mb: float
    max_ratio: int

    @property
    def max_bytes(self) -> int:
        return int(self.max_mb * 1024 * 1024)


_ReceivedBody: TypeAlias = _Body | _TooLarge | _ClientDisconnected
_DecodedBody: TypeAlias = _Body | _DecodedTooLarge | _OverRatio | _Undecodable


class _DecodedBodyBudget:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes: Final = max_bytes
        self.reserved_bytes = 0

    def try_reserve(self, size: int) -> bool:
        if self.reserved_bytes > 0 and self.reserved_bytes + size > self.max_bytes:
            return False
        self.hold(size)
        return True

    def hold(self, size: int) -> None:
        self.reserved_bytes += size

    def release(self, size: int) -> None:
        self.reserved_bytes -= size


def _coding_for(scope: Scope) -> str | None:
    if scope["type"] != "http":
        return None
    content_encodings: Final = Headers(scope=scope).getlist("content-encoding")
    if len(content_encodings) != 1:
        return None
    named_coding: Final = content_encodings[0].strip().lower()
    coding: Final = CODING_ALIASES.get(named_coding, named_coding)
    if coding not in CODING_SIGNATURES:
        return None
    if get_request_route(Request(scope)).startswith(RAW_BODY_ROUTE_PREFIXES):
        return None
    return coding


def _request_chunk(message: Message) -> _Prefix | _ClientDisconnected:
    match message:
        case {"type": "http.disconnect"}:
            return _ClientDisconnected()
        case {"body": bytes() as body, "more_body": True}:
            return _Prefix(body, more_body=True)
        case {"body": bytes() as body}:
            return _Prefix(body, more_body=False)
        case {"more_body": True}:
            return _Prefix(b"", more_body=True)
        case _:
            return _Prefix(b"", more_body=False)


async def _receive_prefix(receive: Receive, received: bytes = b"") -> _Prefix | _ClientDisconnected:
    match _request_chunk(await receive()):
        case _ClientDisconnected() as disconnected:
            return disconnected
        case _Prefix(data=data, more_body=more_body):
            prefix: Final = _Prefix(received + data, more_body)
            if len(prefix.data) >= SIGNATURE_LENGTH or not more_body:
                return prefix
            return await _receive_prefix(receive, prefix.data)


async def _receive_body(scope: Scope, receive: Receive, prefix: _Prefix, max_bytes: int) -> _ReceivedBody:
    if len(prefix.data) > max_bytes:
        return _TooLarge()
    if not prefix.more_body:
        return _Body(prefix.data)
    buffer: Final = io.BytesIO()
    buffer.write(prefix.data)
    try:
        async for chunk in Request(scope, receive).stream():
            buffer.write(chunk)
            if buffer.tell() > max_bytes:
                return _TooLarge()
    except ClientDisconnect:
        return _ClientDisconnected()
    return _Body(buffer.getvalue())


def _max_decoded_length(codec: Codec, data: bytes, limits: _Limits) -> int:
    max_by_limits: Final = min(limits.max_bytes, len(data) * limits.max_ratio)
    declared: Final = codec.declared_size(data)
    return max_by_limits if declared is None else min(max_by_limits, declared)


def _decode(codec: Codec, data: bytes, limits: _Limits) -> _DecodedBody:
    max_by_ratio: Final = len(data) * limits.max_ratio
    declared: Final = codec.declared_size(data)
    max_length: Final = _max_decoded_length(codec, data, limits)
    decompressor: Final = codec.new_decompressor()
    try:
        decoded: Final = decompressor.decompress(data, max_length=max_length + 1)
    except codec.error as error:
        return _Undecodable(f"Request body is not valid {codec.name} data: {error}")
    if len(decoded) > limits.max_bytes:
        return _DecodedTooLarge()
    if len(decoded) > max_by_ratio:
        return _OverRatio()
    if declared is not None and len(decoded) > declared:
        return _Undecodable(f"Request body decodes past the size its {codec.name} data declares")
    if not decompressor.eof:
        return _Undecodable(f"Request body ends before the end of its {codec.name} stream")
    if decompressor.unused_data:
        return _Undecodable(f"Request body has data after the end of its {codec.name} stream")
    return _Body(decoded)


def _headers_without(scope: Scope, dropped: frozenset[bytes]) -> tuple[tuple[bytes, bytes], ...]:
    return tuple((name, value) for name, value in Headers(scope=scope).raw if name.lower() not in dropped)


def _identity_scope(scope: Scope, body_length: int) -> Scope:
    kept_headers: Final = _headers_without(scope, _REPLACED_REQUEST_HEADERS)
    return {**scope, "headers": [*kept_headers, (b"content-length", str(body_length).encode("latin-1"))]}


def _replay_body(body: bytes, receive: Receive, more_body: bool = False) -> Receive:
    body_message: Final[Message] = {"type": "http.request", "body": body, "more_body": more_body}
    pending: Final = iter((body_message,))

    async def replay() -> Message:
        return next(pending, None) or await receive()

    return replay


def _error_response(
    message: str,
    status_code: int,
    error_type: str = "invalid_request_error",
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "param": None, "code": str(status_code)}},
        status_code=status_code,
        headers=headers,
    )


def _held_length(decoded: _DecodedBody) -> int:
    return len(decoded.data) if isinstance(decoded, _Body) else 0


def _effective_max_mb(max_decompressed_body_size_mb: float, operator_max_mb: float | None) -> float:
    if operator_max_mb is None or operator_max_mb <= 0:
        return max_decompressed_body_size_mb
    return min(max_decompressed_body_size_mb, operator_max_mb)


def _no_operator_limit() -> None:
    return None


class RequestBodyDecompressionMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        get_max_request_size_mb: MaxRequestSizeGetter = _no_operator_limit,
        max_decompressed_body_size_mb: float = MAX_DECOMPRESSED_REQUEST_BODY_SIZE_MB,
        max_decompression_ratio: int = MAX_REQUEST_BODY_DECOMPRESSION_RATIO,
        max_decoded_bodies_in_flight_mb: float = MAX_DECOMPRESSED_REQUEST_BODIES_IN_FLIGHT_MB,
        codecs: Mapping[str, Codec] = AVAILABLE_CODECS,
    ) -> None:
        self.app = app
        self.get_max_request_size_mb: Final = get_max_request_size_mb
        self.max_decompressed_body_size_mb: Final = max_decompressed_body_size_mb
        self.max_decompression_ratio: Final = max_decompression_ratio
        self.budget: Final = _DecodedBodyBudget(int(max_decoded_bodies_in_flight_mb * 1024 * 1024))
        self.codecs: Final = codecs

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        coding: Final = _coding_for(scope)
        if coding is None:
            await self.app(scope, receive, send)
            return
        match await _receive_prefix(receive):
            case _ClientDisconnected():
                return
            case _Prefix() as prefix:
                await self._decode_if_encoded(scope, receive, send, coding, prefix)

    async def _decode_if_encoded(
        self, scope: Scope, receive: Receive, send: Send, coding: str, prefix: _Prefix
    ) -> None:
        if not CODING_SIGNATURES[coding](prefix.data):
            sent_as_is: Final = {**scope, "headers": [*_headers_without(scope, _CONTENT_ENCODING_HEADER)]}
            await self.app(sent_as_is, _replay_body(prefix.data, receive, more_body=prefix.more_body), send)
            return
        codec: Final = self.codecs.get(coding)
        if codec is None:
            supported: Final = ", ".join(sorted(self.codecs))
            await _error_response(
                f"This proxy cannot decode {coding} request bodies. Supported encodings: {supported}",
                415,
                headers={"accept-encoding": supported},
            )(scope, receive, send)
            return
        await self._decode_then_call(scope, receive, send, codec, prefix)

    async def _decode_then_call(
        self, scope: Scope, receive: Receive, send: Send, codec: Codec, prefix: _Prefix
    ) -> None:
        max_mb: Final = _effective_max_mb(self.max_decompressed_body_size_mb, self.get_max_request_size_mb())
        limits: Final = _Limits(max_mb=max_mb, max_ratio=self.max_decompression_ratio)
        match await _receive_body(scope, receive, prefix, limits.max_bytes):
            case _Body(data=data):
                await self._decode_within_budget(scope, receive, send, codec, data, limits)
            case _TooLarge():
                await send_request_too_large(send=send, max_request_size_mb=max_mb)
            case _ClientDisconnected():
                return

    async def _decode_within_budget(
        self, scope: Scope, receive: Receive, send: Send, codec: Codec, data: bytes, limits: _Limits
    ) -> None:
        reservation: Final = _max_decoded_length(codec, data, limits)
        if not self.budget.try_reserve(reservation):
            await _error_response(
                "The proxy is decoding too many compressed request bodies right now. "
                "Retry later, or send this body uncompressed",
                503,
                error_type="overloaded_error",
                headers={"retry-after": "1"},
            )(scope, receive, send)
            return
        try:
            decoded: Final = await anyio.to_thread.run_sync(_decode, codec, data, limits)
        finally:
            self.budget.release(reservation)
        held: Final = _held_length(decoded)
        self.budget.hold(held)
        try:
            await self._respond(scope, receive, send, decoded, limits)
        finally:
            self.budget.release(held)

    async def _respond(
        self, scope: Scope, receive: Receive, send: Send, decoded: _DecodedBody, limits: _Limits
    ) -> None:
        match decoded:
            case _Body(data=body):
                await self.app(_identity_scope(scope, len(body)), _replay_body(body, receive), send)
            case _DecodedTooLarge():
                await _error_response(f"Request body is larger than {limits.max_mb:g} MB when decoded", 413)(
                    scope, receive, send
                )
            case _OverRatio():
                await _error_response(f"Request body expands more than {limits.max_ratio} times when decoded", 413)(
                    scope, receive, send
                )
            case _Undecodable(message=message):
                await _error_response(message, 400)(scope, receive, send)

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
RAW_BODY_ROUTE_PREFIXES: Final = ("/plugin-proxy/",)
_REPLACED_REQUEST_HEADERS: Final = frozenset({b"content-encoding", b"content-length", b"transfer-encoding"})


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
    is_encoded: Callable[[bytes], bool]


def _new_gzip_decompressor() -> _Decompressor:
    return zlib.decompressobj(wbits=31)


GZIP: Final = Codec(
    name="gzip",
    new_decompressor=_new_gzip_decompressor,
    error=zlib.error,
    is_encoded=lambda data: data.startswith(GZIP_MAGIC),
)


def _starts_a_zstd_frame(data: bytes) -> bool:
    if data.startswith(ZSTD_FRAME_MAGIC):
        return True
    return len(data) >= 4 and data[0] & 0xF0 == 0x50 and data[1:4] == ZSTD_SKIPPABLE_FRAME_MAGIC_HIGH_BYTES


def load_zstd_codec() -> Codec | None:
    try:
        if sys.version_info >= (3, 14):
            from compression.zstd import DecompressionParameter, ZstdDecompressor, ZstdError
        else:
            from backports.zstd import DecompressionParameter, ZstdDecompressor, ZstdError
    except ImportError:
        return None

    def new_zstd_decompressor() -> _Decompressor:
        return ZstdDecompressor(options={DecompressionParameter.window_log_max: RFC_9659_ZSTD_WINDOW_LOG_MAX})

    return Codec(
        name="zstd",
        new_decompressor=new_zstd_decompressor,
        error=ZstdError,
        is_encoded=_starts_a_zstd_frame,
    )


_ZSTD: Final = load_zstd_codec()
DECODABLE_CODINGS: Final = frozenset({"gzip", "zstd"})
AVAILABLE_CODECS: Final[Mapping[str, Codec]] = MappingProxyType(
    {codec.name: codec for codec in (GZIP, _ZSTD) if codec is not None}
)


@dataclass(frozen=True, slots=True)
class _Unsupported:
    coding: str


@dataclass(frozen=True, slots=True)
class _Body:
    data: bytes


@dataclass(frozen=True, slots=True)
class _NotEncoded:
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


_Coding: TypeAlias = Codec | _Unsupported | None
_ReceivedBody: TypeAlias = _Body | _TooLarge | _ClientDisconnected
_DecodedBody: TypeAlias = _Body | _NotEncoded | _DecodedTooLarge | _OverRatio | _Undecodable


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


def _coding_for(scope: Scope, codecs: Mapping[str, Codec]) -> _Coding:
    if scope["type"] != "http":
        return None
    content_encodings: Final = Headers(scope=scope).getlist("content-encoding")
    if len(content_encodings) != 1:
        return None
    coding: Final = content_encodings[0].strip().lower()
    if coding not in DECODABLE_CODINGS:
        return None
    if get_request_route(Request(scope)).startswith(RAW_BODY_ROUTE_PREFIXES):
        return None
    return codecs.get(coding) or _Unsupported(coding)


async def _receive_body(scope: Scope, receive: Receive, max_bytes: int) -> _ReceivedBody:
    buffer: Final = io.BytesIO()
    try:
        async for chunk in Request(scope, receive).stream():
            buffer.write(chunk)
            if buffer.tell() > max_bytes:
                return _TooLarge()
    except ClientDisconnect:
        return _ClientDisconnected()
    return _Body(buffer.getvalue())


def _max_decoded_length(data: bytes, limits: _Limits) -> int:
    return min(limits.max_bytes, len(data) * limits.max_ratio)


def _decode(codec: Codec, data: bytes, limits: _Limits) -> _DecodedBody:
    if not data:
        return _Body(b"")
    if not codec.is_encoded(data):
        return _NotEncoded(data)
    max_by_ratio: Final = len(data) * limits.max_ratio
    max_length: Final = _max_decoded_length(data, limits)
    decompressor: Final = codec.new_decompressor()
    try:
        decoded: Final = decompressor.decompress(data, max_length=max_length + 1)
    except codec.error as error:
        return _Undecodable(f"Request body is not valid {codec.name} data: {error}")
    if len(decoded) > limits.max_bytes:
        return _DecodedTooLarge()
    if len(decoded) > max_by_ratio:
        return _OverRatio()
    if not decompressor.eof:
        return _Undecodable(f"Request body ends before the end of its {codec.name} stream")
    if decompressor.unused_data:
        return _Undecodable(f"Request body has data after the end of its {codec.name} stream")
    return _Body(decoded)


def _identity_scope(scope: Scope, body_length: int) -> Scope:
    kept_headers: Final = [
        (name, value) for name, value in Headers(scope=scope).raw if name.lower() not in _REPLACED_REQUEST_HEADERS
    ]
    return {**scope, "headers": [*kept_headers, (b"content-length", str(body_length).encode("latin-1"))]}


def _replay_body(body: bytes, receive: Receive) -> Receive:
    body_message: Final[Message] = {"type": "http.request", "body": body, "more_body": False}
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
        match _coding_for(scope, self.codecs):
            case None:
                await self.app(scope, receive, send)
            case _Unsupported(coding=coding):
                supported: Final = ", ".join(sorted(self.codecs))
                response: Final = _error_response(
                    f"This proxy cannot decode {coding} request bodies. Supported encodings: {supported}",
                    415,
                    headers={"accept-encoding": supported},
                )
                await response(scope, receive, send)
            case Codec() as codec:
                await self._decode_then_call(scope, receive, send, codec)

    async def _decode_then_call(self, scope: Scope, receive: Receive, send: Send, codec: Codec) -> None:
        max_mb: Final = _effective_max_mb(self.max_decompressed_body_size_mb, self.get_max_request_size_mb())
        limits: Final = _Limits(max_mb=max_mb, max_ratio=self.max_decompression_ratio)
        match await _receive_body(scope, receive, limits.max_bytes):
            case _Body(data=data):
                await self._decode_within_budget(scope, receive, send, codec, data, limits)
            case _TooLarge():
                await send_request_too_large(send=send, max_request_size_mb=max_mb)
            case _ClientDisconnected():
                return

    async def _decode_within_budget(
        self, scope: Scope, receive: Receive, send: Send, codec: Codec, data: bytes, limits: _Limits
    ) -> None:
        reservation: Final = _max_decoded_length(data, limits)
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
            case _Body(data=body) | _NotEncoded(data=body):
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

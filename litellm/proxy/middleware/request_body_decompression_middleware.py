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

from litellm.constants import MAX_DECOMPRESSED_REQUEST_BODY_SIZE_MB
from litellm.proxy.middleware.request_size_limit_middleware import send_request_too_large

if sys.version_info >= (3, 14):
    from compression.zstd import DecompressionParameter, ZstdDecompressor, ZstdError
else:
    from backports.zstd import DecompressionParameter, ZstdDecompressor, ZstdError

RFC_9659_ZSTD_WINDOW_LOG_MAX: Final = 23
_REPLACED_REQUEST_HEADERS: Final = frozenset({b"content-encoding", b"content-length", b"transfer-encoding"})


class _Decompressor(Protocol):
    @property
    def eof(self) -> bool: ...

    @property
    def unused_data(self) -> bytes: ...

    def decompress(self, data: bytes, /, max_length: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class _Codec:
    name: str
    new_decompressor: Callable[[], _Decompressor]
    error: type[Exception]


def _new_gzip_decompressor() -> _Decompressor:
    return zlib.decompressobj(wbits=31)


def _new_zstd_decompressor() -> _Decompressor:
    return ZstdDecompressor(options={DecompressionParameter.window_log_max: RFC_9659_ZSTD_WINDOW_LOG_MAX})


CODECS: Final[Mapping[str, _Codec]] = MappingProxyType(
    {
        "gzip": _Codec(name="gzip", new_decompressor=_new_gzip_decompressor, error=zlib.error),
        "zstd": _Codec(name="zstd", new_decompressor=_new_zstd_decompressor, error=ZstdError),
    }
)


@dataclass(frozen=True, slots=True)
class _Body:
    data: bytes


@dataclass(frozen=True, slots=True)
class _TooLarge:
    pass


@dataclass(frozen=True, slots=True)
class _ClientDisconnected:
    pass


@dataclass(frozen=True, slots=True)
class _Undecodable:
    message: str


_ReceivedBody: TypeAlias = _Body | _TooLarge | _ClientDisconnected
_DecodedBody: TypeAlias = _Body | _TooLarge | _Undecodable


def _codec_for(scope: Scope) -> _Codec | None:
    if scope["type"] != "http":
        return None
    content_encodings: Final = Headers(scope=scope).getlist("content-encoding")
    if len(content_encodings) != 1:
        return None
    return CODECS.get(content_encodings[0].strip().lower())


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


def _decode(codec: _Codec, data: bytes, max_bytes: int) -> _DecodedBody:
    if not data:
        return _Body(b"")
    decompressor: Final = codec.new_decompressor()
    try:
        decoded: Final = decompressor.decompress(data, max_length=max_bytes + 1)
    except codec.error as error:
        return _Undecodable(f"Request body is not valid {codec.name} data: {error}")
    if len(decoded) > max_bytes:
        return _TooLarge()
    if not decompressor.eof:
        return _Undecodable(f"Request body ends before the end of its {codec.name} stream")
    if decompressor.unused_data:
        return _Undecodable(f"Request body has data after the end of its {codec.name} stream")
    return _Body(decoded)


async def _receive_decoded_body(
    scope: Scope, receive: Receive, codec: _Codec, max_bytes: int
) -> _DecodedBody | _ClientDisconnected:
    received: Final = await _receive_body(scope, receive, max_bytes)
    if not isinstance(received, _Body):
        return received
    return await anyio.to_thread.run_sync(_decode, codec, received.data, max_bytes)


def _decoded_scope(scope: Scope, body_length: int) -> Scope:
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


def _invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": "400"}},
        status_code=400,
    )


class RequestBodyDecompressionMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        max_decompressed_body_size_mb: float = MAX_DECOMPRESSED_REQUEST_BODY_SIZE_MB,
    ) -> None:
        self.app = app
        self.max_decompressed_body_size_mb: Final = max_decompressed_body_size_mb
        self.max_decompressed_body_bytes: Final = int(max_decompressed_body_size_mb * 1024 * 1024)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        codec: Final = _codec_for(scope)
        if codec is None:
            await self.app(scope, receive, send)
            return
        match await _receive_decoded_body(scope, receive, codec, self.max_decompressed_body_bytes):
            case _Body(data=body):
                await self.app(_decoded_scope(scope, len(body)), _replay_body(body, receive), send)
            case _TooLarge():
                await send_request_too_large(send=send, max_request_size_mb=self.max_decompressed_body_size_mb)
            case _Undecodable(message=message):
                await _invalid_request(message)(scope, receive, send)
            case _ClientDisconnected():
                return

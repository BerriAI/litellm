"""
Optional Rust-backed realtime WebSocket upstream.

Enable with ``litellm.use_litellm_rust(realtime=...)`` or
``litellm.use_litellm_rust()``. The proxy realtime route keeps URL construction,
OpenAI-Beta forwarding, pre-call logging, guardrails, and spend logging in
Python, while the upstream WebSocket transport is supplied by the compiled
``litellm_python_bridge`` extension.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from ssl import SSLContext
from types import TracebackType
from typing import (
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    Final,
    Mapping,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)


@runtime_checkable
class RustRealtimeUpstream(Protocol):
    def send(self, message: str | bytes) -> Awaitable[None]: ...
    def recv(self) -> Awaitable[str | bytes]: ...
    def close(self) -> Awaitable[None]: ...


class RustRealtimeConnect(Protocol):
    def __call__(
        self, url: str, headers: dict[str, str], max_size: Optional[int] = None
    ) -> Awaitable[RustRealtimeUpstream]: ...


class _Unset:
    """Sentinel so ``realtime=None`` clears while omission preserves."""


_UNSET: Final[_Unset] = _Unset()

_rust_realtime_enabled = False
_rust_realtime_impl: RustRealtimeConnect | None = None


def set_rust_realtime(
    enabled: bool, *, connect: RustRealtimeConnect | None | _Unset = _UNSET
) -> None:
    global _rust_realtime_enabled, _rust_realtime_impl
    _rust_realtime_enabled = enabled
    if not isinstance(connect, _Unset):
        _rust_realtime_impl = connect


def rust_realtime_enabled() -> bool:
    return _rust_realtime_enabled


def load_rust_realtime() -> RustRealtimeConnect | None:
    if _rust_realtime_impl is not None:
        return _rust_realtime_impl
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    return cast(
        RustRealtimeConnect, getattr(litellm_python_bridge, "realtime_connect", None)
    )


class RustBackendWebsocket:
    """Subset of ``websockets.ClientConnection`` used by ``RealTimeStreaming``."""

    def __init__(self, upstream: RustRealtimeUpstream) -> None:
        self._upstream = upstream
        self._closed = False

    async def send(self, message: str | bytes) -> None:
        await self._upstream.send(message)

    async def recv(self, *_args: object, **_kwargs: object) -> str | bytes:
        try:
            return await self._upstream.recv()
        except StopAsyncIteration:
            import websockets

            raise websockets.exceptions.ConnectionClosedOK(None, None) from None

    def __aiter__(self) -> RustBackendWebsocket:
        return self

    async def __anext__(self) -> str | bytes:
        return await self._upstream.recv()

    async def close(self, *_args: object, **_kwargs: object) -> None:
        if self._closed:
            return
        self._closed = True
        await self._upstream.close()

    async def __aenter__(self) -> RustBackendWebsocket:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exc_type, exc, traceback)
        await self.close()


RustBackendConnectFactory = Callable[..., AsyncContextManager[RustBackendWebsocket]]


def rust_backend_connect_factory(
    connect: RustRealtimeConnect,
) -> RustBackendConnectFactory:
    @asynccontextmanager
    async def _connect(
        url: str,
        *,
        additional_headers: Mapping[str, str],
        max_size: Optional[int] = None,
        ssl: bool | SSLContext | str | None = None,
    ) -> AsyncIterator[RustBackendWebsocket]:
        if not rust_supports_ssl_config(ssl):
            raise NotImplementedError(
                "Rust realtime path does not support custom TLS configuration"
            )
        try:
            upstream = await connect(url, dict(additional_headers), max_size)
        except RuntimeError as exc:
            status_code = _rust_realtime_http_status_code(str(exc))
            if status_code is not None:
                import websockets.exceptions as websocket_exceptions
                from websockets.datastructures import Headers

                raise websocket_exceptions.InvalidStatusCode(
                    status_code, Headers()
                ) from exc
            raise
        backend = RustBackendWebsocket(upstream)
        try:
            yield backend
        finally:
            await backend.close()

    return _connect


def rust_supports_ssl_config(ssl_config: bool | SSLContext | str | None) -> bool:
    return ssl_config is None or ssl_config is True


def _rust_realtime_http_status_code(message: str) -> int | None:
    prefix = "upstream websocket HTTP status "
    start = message.find(prefix)
    if start == -1:
        return None
    start += len(prefix)
    end = start
    while end < len(message) and message[end].isdigit():
        end += 1
    if end == start:
        return None
    return int(message[start:end])

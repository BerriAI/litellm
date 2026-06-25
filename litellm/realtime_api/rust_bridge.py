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
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Final,
    Mapping,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)


@runtime_checkable
class RustRealtimeUpstream(Protocol):
    def send(self, text: str) -> Awaitable[None]: ...
    def recv(self) -> Awaitable[str]: ...
    def close(self) -> Awaitable[None]: ...


class RustRealtimeConnect(Protocol):
    def __call__(
        self, url: str, headers: dict[str, str]
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

    async def send(self, message: str) -> None:
        await self._upstream.send(message)

    async def recv(self, *_args: Any, **_kwargs: Any) -> str:
        try:
            return await self._upstream.recv()
        except StopAsyncIteration:
            import websockets

            raise websockets.exceptions.ConnectionClosedOK(None, None) from None

    def __aiter__(self) -> "RustBackendWebsocket":
        return self

    async def __anext__(self) -> str:
        return await self._upstream.recv()

    async def close(self, *_args: Any, **_kwargs: Any) -> None:
        if self._closed:
            return
        self._closed = True
        await self._upstream.close()

    async def __aenter__(self) -> "RustBackendWebsocket":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


def rust_backend_connect_factory(connect: RustRealtimeConnect):
    @asynccontextmanager
    async def _connect(
        url: str,
        *,
        additional_headers: Mapping[str, str],
        max_size: Optional[int] = None,
        ssl: Any = None,
    ) -> AsyncIterator[RustBackendWebsocket]:
        _ = max_size
        if ssl is False:
            raise NotImplementedError(
                "Rust realtime path does not support disabling TLS verification"
            )
        upstream = await connect(url, dict(additional_headers))
        backend = RustBackendWebsocket(upstream)
        try:
            yield backend
        finally:
            await backend.close()

    return _connect


def rust_supports_ssl_config(ssl_config: Any) -> bool:
    return ssl_config is not False

"""
Optional Rust-backed realtime WebSocket upstream.

Enable with ``litellm.use_litellm_rust(realtime=True)``; the proxy's async
realtime entrypoint then dials the upstream provider through the compiled
``litellm_python_bridge`` extension instead of the ``websockets`` library, while
Python keeps owning URL construction, ``OpenAI-Beta`` passthrough, ``pre_call``
logging, guardrails, and the ``RealTimeStreaming`` accumulator.

No module-level ``litellm`` imports so this stays a leaf module
(``litellm/realtime_api/main.py`` imports it statically without forming a
cycle).
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
    """Subset of the compiled ``litellm_python_bridge.RustRealtimeUpstream``
    class this adapter depends on.

    Each method returns a Python awaitable backed by tokio on the Rust side.
    ``recv`` raises ``StopAsyncIteration`` on a clean close so it composes with
    ``async for``.
    """

    def send(self, text: str) -> Awaitable[None]: ...
    def recv(self) -> Awaitable[str]: ...
    def close(self) -> Awaitable[None]: ...


class RustRealtimeConnect(Protocol):
    """Signature of the compiled ``litellm_python_bridge.realtime_connect``
    entrypoint: ``connect(url, headers) -> RustRealtimeUpstream``."""

    def __call__(
        self, url: str, headers: dict[str, str]
    ) -> Awaitable[RustRealtimeUpstream]: ...


class _Unset:
    """Sentinel so ``realtime=None`` can clear a prior injection while omission
    preserves it (same convention as ``ocr/rust_bridge.py``)."""


_UNSET: Final[_Unset] = _Unset()

_rust_realtime_enabled = False
_rust_realtime_impl: Optional[RustRealtimeConnect] = None


def set_rust_realtime(
    enabled: bool, *, connect: RustRealtimeConnect | None | _Unset = _UNSET
) -> None:
    """Toggle the Rust realtime path; optionally inject a bridge callable.

    Called from :func:`litellm.use_litellm_rust` so users have one knob. Passing
    ``connect=None`` explicitly clears a prior injection; omitting it preserves
    it.
    """
    global _rust_realtime_enabled, _rust_realtime_impl
    _rust_realtime_enabled = enabled
    if not isinstance(connect, _Unset):
        _rust_realtime_impl = connect


def rust_realtime_enabled() -> bool:
    return _rust_realtime_enabled


def load_rust_realtime() -> RustRealtimeConnect | None:
    """Return the Rust ``realtime_connect`` callable, or ``None``.

    Prefers an injected implementation, otherwise loads the compiled
    ``litellm_python_bridge`` extension; a missing extension yields ``None`` so
    callers can fall back to the Python path instead of hard-failing.
    """
    if _rust_realtime_impl is not None:
        return _rust_realtime_impl
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    return cast(RustRealtimeConnect, litellm_python_bridge.realtime_connect)


class RustBackendWebsocket:
    """Quacks like ``websockets.asyncio.client.ClientConnection`` for the
    subset of methods ``RealTimeStreaming`` uses on the backend leg.

    The Rust upstream raises ``StopAsyncIteration`` on a clean close;
    ``recv`` translates that into ``websockets.exceptions.ConnectionClosedOK``
    so the existing ``except websockets.exceptions.ConnectionClosed`` block in
    ``RealTimeStreaming.backend_to_client_send_messages`` keeps firing
    unchanged.
    """

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
    """Adapt the Rust connector to ``websockets.connect``'s call shape.

    The returned async context manager has the same ``(url, *, additional_headers,
    max_size, ssl)`` keyword signature so ``OpenAIRealtime.async_realtime`` can
    drop it in without branching its dial site. ``max_size`` is honored by the
    Rust transport's own frame limits; ``ssl`` must be the default (``None`` or
    ``True``) because the Rust path uses rustls native roots and does not
    accept a custom ``ssl.SSLContext`` yet.
    """

    @asynccontextmanager
    async def _connect(
        url: str,
        *,
        additional_headers: Mapping[str, str],
        max_size: Optional[int] = None,
        ssl: Any = None,
    ) -> AsyncIterator[RustBackendWebsocket]:
        if ssl is False:
            raise NotImplementedError(
                "Rust realtime path does not support disabling TLS "
                "verification; fall back to the Python path"
            )
        upstream = await connect(url, dict(additional_headers))
        backend = RustBackendWebsocket(upstream)
        try:
            yield backend
        finally:
            await backend.close()

    return _connect


def rust_supports_ssl_config(ssl_config: Any) -> bool:
    """True when ``ssl_config`` is one the Rust connector can replicate today.

    The Python path consults ``get_shared_realtime_ssl_context`` which returns
    ``None``/``True``/an ``ssl.SSLContext`` (default verification path, the
    common case) or ``False`` when the user has explicitly disabled
    verification. The Rust transport uses rustls native roots and verifies
    every connection, so verification-off configurations must keep using
    ``websockets`` to honor the operator's intent (a Python user opting out
    of TLS verification expects exactly that, not silent re-verification).
    """
    return ssl_config is not False

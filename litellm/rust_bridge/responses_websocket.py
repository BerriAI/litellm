"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Final, Protocol

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.callbacks import SessionCallbackHandle
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import (
    RustResponsesWebSocket,
    RustResponsesWebSocketConnection,
    RustRouteDecline,
)
from litellm.rust_bridge.request import (
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    NativeResponsesWebSocketRequest,
    PreparedNativeCall,
    call_native,
    with_capabilities,
)
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointBinding,
    assess_route,
    async_none,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds

_RESPONSES_WEBSOCKET: Final[EndpointBinding[RustResponsesWebSocketConnection]] = EndpointBinding.native(
    route="responses_websocket",
    select=lambda native: native.ResponsesWebSocketConnection,
    enabled=rust_enabled,
)


_PREFLIGHT: Final[EndpointBinding[RustRouteDecline]] = EndpointBinding.native(
    route="responses_websocket",
    select=lambda native: native.responses_websocket_decline,
    enabled=rust_enabled,
)


def set_rust_responses_websocket(
    *,
    connection: RustResponsesWebSocketConnection | None | Unchanged = UNCHANGED,
    decline: RustRouteDecline | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(decline, Unchanged):
        if decline is None:
            _PREFLIGHT.reset()
        else:
            _PREFLIGHT.override(decline)
    if not isinstance(connection, Unchanged):
        if connection is None:
            _RESPONSES_WEBSOCKET.reset()
        else:
            _RESPONSES_WEBSOCKET.override(connection)


class Connection(Protocol):
    async def send(self, text: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


class _ConnectionAdapter:
    def __init__(self, connection: RustResponsesWebSocket):
        self._connection: Final[RustResponsesWebSocket] = connection

    async def send(self, text: str) -> None:
        await self._connection.send_text(text)

    async def recv(self) -> str:
        message: Final = await self._connection.recv_text()
        if message is None:
            raise ConnectionClosedOK(None, None)
        return message

    async def close(self) -> None:
        await self._connection.close()


async def connect(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
    model: str = "responses websocket",
    provider: str = "openai",
    callback_adapter: SessionCallbackHandle | None = None,
    fallback: Callable[[], Awaitable[Connection | None]] = async_none,
    context: NativeRequestContext | None = None,
) -> Connection | None:
    return await _RESPONSES_WEBSOCKET.ainvoke(
        prepare=lambda: PreparedNativeCall[NativeResponsesWebSocketRequest, SessionCallbackHandle](
            NativeResponsesWebSocketRequest(
                url=url,
            ),
            options=NativeRequestOptions(
                extra_headers=headers,
                timeout_seconds=timeout_to_seconds(timeout),
                custom_llm_provider=provider,
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="async",
                    websocket_mode="native",
                    requires_connection=True,
                ),
            ),
            callback_adapter=callback_adapter,
        ),
        call=lambda connection_type, request: call_native(connection_type.connect, request),
        preflight=lambda: assess_route(_PREFLIGHT, model, provider),
        fallback=fallback,
        adapt=_ConnectionAdapter,
        error_context=BridgeErrorContext(provider=provider, model=model),
    )


@asynccontextmanager
async def open_connection(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
    model: str,
    provider: str,
    callback_adapter: SessionCallbackHandle | None = None,
    fallback: Callable[[], AbstractAsyncContextManager[Connection]],
    context: NativeRequestContext | None = None,
) -> AsyncGenerator[Connection]:
    async with AsyncExitStack() as stack:

        async def python_connection() -> Connection:
            return await stack.enter_async_context(fallback())

        backend: Final = await connect(
            url=url,
            headers=headers,
            timeout=timeout,
            model=model,
            provider=provider,
            callback_adapter=callback_adapter,
            fallback=python_connection,
            context=context,
        )
        if backend is None:
            raise RuntimeError("WebSocket connection returned no connection")
        if isinstance(backend, _ConnectionAdapter):
            stack.push_async_callback(backend.close)
        yield backend

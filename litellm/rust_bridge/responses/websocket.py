"""Native Responses WebSocket bindings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Final, Protocol, TypeVar, cast  # noqa: TID251  # native class is validated at load time

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.configuration import CapabilityContext, DeliveryMode
from litellm.rust_bridge.responses.definition import COMPONENT
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustResponsesWebSocket(Protocol):
    def send_text(self, text: str) -> Awaitable[None]: ...

    def recv_text(self) -> Awaitable[str | None]: ...

    def close(self) -> Awaitable[None]: ...


class ResponsesWebSocket(Protocol):
    def send(self, text: str) -> Awaitable[None]: ...

    def recv(self) -> Awaitable[str | bytes]: ...

    def close(self) -> Awaitable[None]: ...


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    def connect(
        cls,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
        custom_llm_provider: str | None,
    ) -> Awaitable[RustResponsesWebSocket]: ...


def _as_connection(value: object) -> RustResponsesWebSocketConnection | None:
    if not callable(getattr(value, "connect", None)):
        return None
    return cast(RustResponsesWebSocketConnection, value)  # cast-ok: native connection factory validated above


_CONNECTION: Final = COMPONENT.bind("ResponsesWebSocketConnection", validate=_as_connection)


def set_rust_responses_websocket(
    *,
    connection: RustResponsesWebSocketConnection | None | BindingUnset = BINDING_UNSET,
) -> None:
    _CONNECTION.configure(connection)


def load_rust_responses_websocket() -> RustResponsesWebSocketConnection | None:
    return COMPONENT.resolve(CapabilityContext(delivery=DeliveryMode.WEBSOCKET)).select(_CONNECTION)


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


ConnectionT = TypeVar("ConnectionT", bound=ResponsesWebSocket)


async def connect(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    model: str,
    python_fallback: Callable[[], Awaitable[ConnectionT]],
) -> ResponsesWebSocket:
    execution: Final = COMPONENT.resolve(
        CapabilityContext(
            provider=custom_llm_provider or "",
            model=model,
            delivery=DeliveryMode.WEBSOCKET,
        )
    )
    connection_type: Final = execution.select(_CONNECTION)

    native_call: Final[Callable[[], Awaitable[RustResponsesWebSocket]] | None] = (
        (
            lambda: connection_type.connect(
                url=url,
                custom_llm_provider=custom_llm_provider,
                headers=headers,
                timeout_seconds=timeout_to_seconds(timeout),
            )
        )
        if connection_type is not None
        else None
    )

    async def adapt(connection: RustResponsesWebSocket) -> ResponsesWebSocket:
        return _ConnectionAdapter(connection)

    return await ainvoke(
        execution=execution,
        native_call=native_call,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
    )

from __future__ import annotations

import os
from importlib import import_module
from typing import Final, Protocol, cast

import pytest


class StreamRoute(Protocol):
    def __call__(self, *, request: object, provider: str) -> object: ...


class ResponsesWebSocketSession(Protocol):
    @classmethod
    def connect(cls, *, provider: str) -> object: ...


class NativeBridge(Protocol):
    RustBridgeDeclined: type[Exception]
    ResponsesWebSocketSession: type[ResponsesWebSocketSession]
    chat_completions_stream: StreamRoute
    achat_completions_stream: StreamRoute
    messages_stream: StreamRoute
    amessages_stream: StreamRoute
    responses_stream: StreamRoute
    aresponses_stream: StreamRoute


try:
    native: Final = cast(NativeBridge, import_module("litellm.rust_bridge._native"))
except ImportError:
    if os.getenv("LITELLM_REQUIRE_NATIVE_BRIDGE") == "1":
        raise
    pytest.skip("the native bridge has not been built", allow_module_level=True)

STREAM_ROUTES: Final[tuple[tuple[StreamRoute, str], ...]] = (
    (native.chat_completions_stream, "anthropic"),
    (native.achat_completions_stream, "anthropic"),
    (native.messages_stream, "anthropic"),
    (native.amessages_stream, "anthropic"),
    (native.responses_stream, "openai"),
    (native.aresponses_stream, "openai"),
)


@pytest.mark.parametrize(("function", "provider"), STREAM_ROUTES)
def test_stream_routes_preserve_validation_errors(
    function: StreamRoute,
    provider: str,
) -> None:
    with pytest.raises(ValueError):
        function(request=object(), provider=provider)


def test_native_websocket_connect_preserves_decline() -> None:
    with pytest.raises(native.RustBridgeDeclined):
        native.ResponsesWebSocketSession.connect(provider="unsupported")

import json
from typing import cast
from unittest.mock import MagicMock, patch

import pytest


class _RecordingClientWebSocket:
    scope: dict[str, list[tuple[bytes, bytes]]] = {"headers": []}

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: list[tuple[int, str | None]] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))


@pytest.mark.asyncio
async def test_async_realtime_upstream_handshake_refusal_sends_error_event_then_policy_close():
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime
    from litellm.types.realtime import RealtimeErrorEvent

    handler = AzureOpenAIRealtime()
    model = "gpt-realtime"

    dummy_websocket = _RecordingClientWebSocket()
    dummy_logging_obj = MagicMock()

    refused = InvalidStatus(Response(401, "Unauthorized", Headers()))

    with patch("websockets.connect", side_effect=refused):
        await handler.async_realtime(  # pyright: ignore[reportUnknownMemberType]  # handler's websocket param is a Protocol here but the mock connect type is incomplete
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base="https://example.openai.azure.com",
            api_key="bad-key",
            api_version="2025-08-28",
            query_params={"model": model},
        )

    assert len(dummy_websocket.sent) == 1
    event = cast(RealtimeErrorEvent, json.loads(dummy_websocket.sent[0]))
    assert event["type"] == "error"
    assert event["error"]["type"] == "server_error"
    assert "401" in event["error"]["message"]
    assert dummy_websocket.closed and dummy_websocket.closed[0][0] == 1008


@pytest.mark.asyncio
async def test_async_realtime_unexpected_error_sends_error_event_then_internal_close():
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime
    from litellm.types.realtime import RealtimeErrorEvent

    handler = AzureOpenAIRealtime()
    model = "gpt-realtime"

    dummy_websocket = _RecordingClientWebSocket()
    dummy_logging_obj = MagicMock()

    with patch("websockets.connect", side_effect=OSError("connection reset")):
        await handler.async_realtime(  # pyright: ignore[reportUnknownMemberType]  # same as above
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base="https://example.openai.azure.com",
            api_key="bad-key",
            api_version="2025-08-28",
            query_params={"model": model},
        )

    assert len(dummy_websocket.sent) == 1
    event = cast(RealtimeErrorEvent, json.loads(dummy_websocket.sent[0]))
    assert event["type"] == "error"
    assert event["error"]["type"] == "server_error"
    assert event["error"]["message"] == "Internal server error"
    assert "connection reset" not in dummy_websocket.sent[0]
    assert dummy_websocket.closed and dummy_websocket.closed[0] == (1011, "Internal server error")

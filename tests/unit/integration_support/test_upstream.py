from collections import deque
from pathlib import Path
from typing import Final

import pytest
from starlette.types import Message, Scope
from starlette.websockets import WebSocket


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (OSError("client disconnected"), RuntimeError("unexpected upstream failure")))
async def test_realtime_disconnect_after_session_creation_keeps_unexpected_errors_visible(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.upstream import Provider
    from integration.cost_calculation.cost_tracking_case import RealtimeResponse

    provider: Final = Provider()
    response: Final = RealtimeResponse(content_type="application/x-realtime", events=({"type": "response.done"},))
    provider.scenario_store.put("owned-scenario", response)
    incoming: Final[deque[Message]] = deque(
        ({"type": "websocket.connect"}, {"type": "websocket.receive", "text": '{"type":"response.create"}'})
    )
    outgoing: Final[deque[Message]] = deque()

    async def receive() -> Message:
        return incoming.popleft()

    async def send(message: Message) -> None:
        if message["type"] == "websocket.send" and any(item["type"] == "websocket.send" for item in outgoing):
            raise failure
        outgoing.append(message)

    scope: Final[Scope] = {
        "type": "websocket",
        "path": "/v1/realtime",
        "headers": [(b"authorization", b"Bearer owned-scenario")],
        "query_string": b"model=synthetic",
    }
    socket: Final = WebSocket(scope, receive, send)
    if isinstance(failure, RuntimeError):
        with pytest.raises(RuntimeError, match="unexpected upstream failure"):
            await provider.realtime(socket)
    else:
        await provider.realtime(socket)
    assert [item["type"] for item in outgoing] == ["websocket.accept", "websocket.send"]
    observed: Final = tuple(provider.observations.get_nowait() for _ in range(provider.observations.qsize()))
    assert [(item.method, item.body) for item in observed] == [
        ("WEBSOCKET", {"query": [["model", "synthetic"]]}),
        ("WEBSOCKET_FRAME", {"type": "response.create"}),
    ]

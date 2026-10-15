import asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method,status", (("POST", 200), ("POST", 400), ("DELETE", 200)))
async def test_mcp_notification_ordering_preserves_other_responses(
    method: str, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import _acknowledge_notifications_after_dispatch
    from starlette.types import Receive, Send

    outgoing: Final[list[Message]] = []
    messages: Final[tuple[Message, ...]] = (
        {"type": "http.response.start", "status": status, "headers": []},
        {"type": "http.response.body", "body": b"result", "more_body": True},
        {"type": "http.response.body", "body": b"", "more_body": False},
    )

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        outgoing.append(message)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        for index, message in enumerate(messages):
            await send(message)
            assert outgoing == list(messages[: index + 1]), "Ordinary response streaming was buffered"

    await _acknowledge_notifications_after_dispatch(app)({"type": "http", "method": method}, receive, send)
    assert outgoing == list(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (RuntimeError("dispatch failed"), asyncio.CancelledError()))
async def test_mcp_notification_dispatch_failure_is_not_acknowledged(
    failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import _acknowledge_notifications_after_dispatch
    from starlette.types import Receive, Send

    outgoing: Final[list[Message]] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        outgoing.append(message)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"] == "DELETE":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b""})
        raise failure

    ordered: Final = _acknowledge_notifications_after_dispatch(app)
    scope: Final[Scope] = {"type": "http", "headers": [(b"mcp-session-id", b"failed-session")]}
    with pytest.raises(type(failure)) as caught:
        await ordered({**scope, "method": "POST"}, receive, send)
    assert caught.value is failure
    assert outgoing == [], "Failed dispatch must not send an acknowledgement"
    await asyncio.wait_for(ordered({**scope, "method": "DELETE"}, receive, send), 2)
    assert outgoing[0]["status"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses",
    (
        (),
        ({"type": "http.response.body", "body": b"unexpected"},),
        ({"type": "http.response.body", "body": b"", "more_body": True},),
        ({"type": "http.response.start", "status": 500, "headers": []},),
        ({"type": "http.response.body", "body": b""}, {"type": "http.response.body", "body": b""}),
    ),
)
async def test_mcp_notification_ordering_rejects_invalid_acknowledgements(
    responses: tuple[Message, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import _acknowledge_notifications_after_dispatch
    from starlette.types import Receive, Send

    outgoing: Final[list[Message]] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        outgoing.append(message)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 202, "headers": []})
        for message in responses:
            await send(message)

    with pytest.raises(RuntimeError, match="MCP notification acknowledgement"):
        await _acknowledge_notifications_after_dispatch(app)({"type": "http", "method": "POST"}, receive, send)
    assert outgoing == [], "Invalid acknowledgement must not be reported as accepted"


@pytest.mark.asyncio
async def test_mcp_session_deletion_waits_for_concurrent_notification_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import _acknowledge_notifications_after_dispatch
    from starlette.types import Receive, Send

    started: Final = [asyncio.Event(), asyncio.Event()]
    release: Final = asyncio.Event()
    deletion_started: Final = asyncio.Event()
    deleted: Final = asyncio.Event()
    other_deleted: Final = asyncio.Event()
    completed: Final[list[str]] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        return None

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"] == "POST":
            started[int(scope["path"])].set()
            await release.wait()
            completed.append(scope["path"])
            await send({"type": "http.response.start", "status": 202, "headers": []})
            await send({"type": "http.response.body", "body": b""})
        else:
            if scope["headers"][0][1] == b"other-session":
                other_deleted.set()
            else:
                deleted.set()
                completed.append("DELETE")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    ordered: Final = _acknowledge_notifications_after_dispatch(app)
    scope: Final[Scope] = {"type": "http", "headers": [(b"mcp-session-id", b"owned-session")]}
    posts: Final = [
        asyncio.create_task(ordered({**scope, "method": "POST", "path": str(index)}, receive, send))
        for index in range(2)
    ]

    async def delete() -> None:
        deletion_started.set()
        await ordered({**scope, "method": "DELETE"}, receive, send)

    deletion = None
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
        deletion = asyncio.create_task(delete())
        await deletion_started.wait()
        await asyncio.wait_for(
            ordered({**scope, "method": "DELETE", "headers": [(b"mcp-session-id", b"other-session")]}, receive, send), 2
        )
        assert other_deleted.is_set(), "Unrelated session deletion was blocked"
        assert not deleted.is_set(), "Session deletion overtook in-flight notification dispatch"
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*posts, *([deletion] if deletion is not None else [])), 2)
    assert set(completed[:2]) == {"0", "1"}
    assert completed[-1] == "DELETE"


@pytest.mark.asyncio
async def test_mcp_session_deletion_can_terminate_streaming_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import _acknowledge_notifications_after_dispatch
    from starlette.types import Receive, Send

    streaming: Final = asyncio.Event()
    deleted: Final = asyncio.Event()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        return None

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        if scope["method"] == "POST":
            streaming.set()
            await deleted.wait()
        else:
            deleted.set()
        await send({"type": "http.response.body", "body": b""})

    ordered: Final = _acknowledge_notifications_after_dispatch(app)
    scope: Final[Scope] = {"type": "http", "headers": [(b"mcp-session-id", b"streaming-session")]}
    post: Final = asyncio.create_task(ordered({**scope, "method": "POST"}, receive, send))
    try:
        await asyncio.wait_for(streaming.wait(), 2)
        await asyncio.wait_for(ordered({**scope, "method": "DELETE"}, receive, send), 2)
        assert deleted.is_set(), "Session deletion must terminate an ordinary streaming request"
    finally:
        deleted.set()
        await asyncio.wait_for(post, 2)

import asyncio
import json
from typing import Final

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Message

from litellm.proxy.common_utils.active_request import active_request, active_request_disconnected
from litellm.proxy.middleware.active_request_middleware import ActiveRequestMiddleware

BODY: Final = json.dumps({"prompt": "draw a cat"}).encode()


class DisconnectProbe:
    """A non-chat route that reads its body, then asks the fairness helper whether the client is still there."""

    def __init__(self) -> None:
        self.observed: list[bool] = []

    async def route(self, request: Request) -> Response:
        await request.json()
        self.observed.append(await active_request_disconnected())
        return Response(status_code=204)

    def app(self) -> ActiveRequestMiddleware:
        return ActiveRequestMiddleware(
            Starlette(routes=[Route("/v1/images/generations", self.route, methods=["POST"])])
        )


def _scope() -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/images/generations",
        "raw_path": b"/v1/images/generations",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(BODY)).encode())],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }


async def _run(app: ActiveRequestMiddleware, messages_after_body: tuple[Message, ...]) -> None:
    pending: Final = [{"type": "http.request", "body": BODY, "more_body": False}, *messages_after_body]
    forever: Final = asyncio.Event()

    async def receive() -> Message:
        if pending:
            return pending.pop(0)
        await forever.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        del message

    await app(_scope(), receive, send)


@pytest.mark.asyncio
async def test_non_chat_route_sees_client_disconnect_through_active_request():
    probe: Final = DisconnectProbe()
    await _run(probe.app(), ({"type": "http.disconnect"},))
    assert probe.observed == [True]


@pytest.mark.asyncio
async def test_connected_client_is_not_reported_as_disconnected():
    probe: Final = DisconnectProbe()
    await _run(probe.app(), ())
    assert probe.observed == [False]


@pytest.mark.asyncio
async def test_active_request_is_cleared_once_the_response_is_sent():
    await _run(DisconnectProbe().app(), ())
    assert active_request.get() is None

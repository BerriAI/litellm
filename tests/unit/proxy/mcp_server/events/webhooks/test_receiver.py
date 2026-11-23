import base64
import json
import time
from typing import Final

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.webhooks.receiver import MAX_BODY_BYTES, gateway_delivery_id, receive_webhook
from litellm.proxy.mcp_server.events.webhooks.signing import signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow, SubscriptionStatus

from ..fixtures import SETTINGS
from .memory_store import MemoryStore

KEY: Final = bytes(range(32))
SECRET: Final = "whsec_" + base64.b64encode(KEY).decode()


def _row(status: SubscriptionStatus = "active") -> SubscriptionRow:
    return SubscriptionRow(
        id="sub_1",
        principal="key:1",
        server_id="srv-ci",
        event_name="build.failed",
        client_url="https://client.example/hook",
        client_secret=SECRET,
        upstream_subscription_id="up_1",
        upstream_secret=SECRET,
        status=status,
    )


def _app(store: MemoryStore, settings: EventsSettings = SETTINGS) -> Starlette:
    async def endpoint(request: Request) -> Response:
        return await receive_webhook(
            request.path_params["sid"], request, settings=lambda: settings, store=lambda: store
        )

    return Starlette(routes=[Route("/hooks/{sid}", endpoint, methods=["POST"])])


async def _post(
    app: Starlette,
    body: bytes,
    *,
    key: bytes = KEY,
    sid: str = "sub_1",
    upstream_id: str = "up_1",
    webhook_id: str = "msg_a",
) -> httpx.Response:
    headers: Final = dict(signed_headers(key, webhook_id, int(time.time()), body, upstream_id))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        return await client.post(f"/hooks/{sid}", content=body, headers=headers)


EVENT: Final = json.dumps({"eventId": "e1", "name": "build.failed", "timestamp": "t", "data": {}}).encode()


@pytest.mark.asyncio
async def test_an_event_is_queued_durably_before_204() -> None:
    store: Final = MemoryStore(rows={"sub_1": _row()})
    response: Final = await _post(_app(store), EVENT)
    assert response.status_code == 204
    queued: Final = store.queue[gateway_delivery_id("sub_1", "msg_a")]
    assert (queued.subscription_id, queued.body) == ("sub_1", EVENT.decode())


@pytest.mark.asyncio
async def test_an_upstream_retry_with_the_same_webhook_id_is_queued_once() -> None:
    store: Final = MemoryStore(rows={"sub_1": _row()})
    first: Final = await _post(_app(store), EVENT)
    second: Final = await _post(_app(store), EVENT)
    assert (first.status_code, second.status_code, len(store.queue)) == (204, 204, 1)


@pytest.mark.parametrize("status", ("pending", "active"))
@pytest.mark.asyncio
async def test_a_signed_verification_challenge_is_echoed(status: SubscriptionStatus) -> None:
    body: Final = json.dumps({"type": "verification", "challenge": "nonce-1"}).encode()
    response: Final = await _post(_app(MemoryStore(rows={"sub_1": _row(status)})), body, upstream_id="not-yet-known")
    assert (response.status_code, response.json()) == (200, {"challenge": "nonce-1"})


@pytest.mark.parametrize(
    ("rows", "body", "kwargs", "status"),
    (
        ({}, EVENT, {}, 410),
        ({"sub_1": _row("terminated")}, EVENT, {}, 410),
        ({"sub_1": _row()}, EVENT, {"key": bytes(32)}, 401),
        ({"sub_1": _row()}, EVENT, {"upstream_id": "someone-else"}, 401),
        ({"sub_1": _row("pending")}, EVENT, {}, 425),
        ({"sub_1": _row()}, b"not json", {}, 400),
        pytest.param({"sub_1": _row()}, b"x" * (MAX_BODY_BYTES + 1), {}, 413, id="oversized-body"),
    ),
)
@pytest.mark.asyncio
async def test_rejections(
    rows: dict[str, SubscriptionRow], body: bytes, kwargs: dict[str, object], status: int
) -> None:
    store: Final = MemoryStore(rows=dict(rows))
    response: Final = await _post(_app(store), body, **kwargs)  # pyright: ignore[reportArgumentType]  # per-case overrides
    assert response.status_code == status
    assert store.queue == {}


@pytest.mark.asyncio
async def test_events_for_a_suspended_subscription_are_dropped_until_reactivation() -> None:
    store: Final = MemoryStore(rows={"sub_1": _row("suspended")})
    gap: Final = json.dumps({"type": "gap", "cursor": "c"}).encode()
    assert (await _post(_app(store), EVENT)).status_code == 204
    assert store.queue == {}
    assert (await _post(_app(store), gap, webhook_id="msg_gap")).status_code == 204
    assert len(store.queue) == 1


@pytest.mark.asyncio
async def test_a_failed_durable_write_asks_the_upstream_to_retry() -> None:
    store: Final = MemoryStore(rows={"sub_1": _row()}, fail_enqueue=True)
    assert (await _post(_app(store), EVENT)).status_code == 503


@pytest.mark.asyncio
async def test_the_route_is_not_found_while_events_are_off() -> None:
    store: Final = MemoryStore(rows={"sub_1": _row()})
    assert (await _post(_app(store, EventsSettings()), EVENT)).status_code == 404

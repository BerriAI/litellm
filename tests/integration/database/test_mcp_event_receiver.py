import base64
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
import time_machine
from pydantic import JsonValue
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.webhooks.receiver import MAX_BODY_BYTES, gateway_delivery_id, receive_webhook
from litellm.proxy.mcp_server.events.webhooks.signing import signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import (
    Activation,
    PrismaEventStore,
    SubscriptionRow,
    SubscriptionStatus,
)
from tests.integration._support.database import read_rows
from tests.integration._support.prisma_clients import connected_proxy_prisma

pytestmark = pytest.mark.asyncio
RECEIVED_AT: Final = datetime(2026, 11, 20, tzinfo=timezone.utc)
TIMESTAMP: Final = int(RECEIVED_AT.timestamp())
KEY: Final = bytes(range(32))
SECRET: Final = "whsec_" + base64.b64encode(KEY).decode()
EVENT: Final = b'{"eventId":"e1","name":"build.failed","timestamp":"2026-11-20T00:00:00Z","data":{}}'


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-receiver-test-key")


@asynccontextmanager
async def _subscription(status: SubscriptionStatus | None) -> AsyncIterator[tuple[PrismaEventStore, str]]:
    sid: Final = f"sub_{uuid.uuid4().hex}"
    async with connected_proxy_prisma() as client:
        store: Final = PrismaEventStore(client)
        row: Final = SubscriptionRow(
            id=sid,
            principal="key:1",
            server_id="srv-ci",
            event_name="build.failed",
            client_url="https://client.example/hook",
            client_secret=SECRET,
            upstream_secret=SECRET,
        )
        try:
            if status is not None:
                assert await store.reserve(row)
            if status in ("active", "suspended"):
                assert await store.activate(
                    sid,
                    Activation(
                        client_secret=SECRET,
                        upstream_subscription_id="up_1",
                        upstream_refresh_before=None,
                        refresh_before=None,
                        cursor=None,
                    ),
                )
            if status == "suspended":
                await store.mark_failed(sid, "timeout", RECEIVED_AT, suspend_after=1)
            if status == "terminated":
                await store.terminate(sid)
            yield store, sid
        finally:
            await store.delete(sid)


async def _post(
    store: PrismaEventStore,
    sid: str,
    body: bytes = EVENT,
    *,
    key: bytes = KEY,
    upstream_id: str = "up_1",
    timestamp: int = TIMESTAMP,
    enabled: bool = True,
) -> httpx.Response:
    async def endpoint(request: Request) -> Response:
        return await receive_webhook(
            request.path_params["sid"], request, settings=lambda: EventsSettings(enabled=enabled), store=lambda: store
        )

    app: Final = Starlette(routes=[Route("/hooks/{sid}", endpoint, methods=["POST"])])
    headers: Final = signed_headers(key, "msg_a", timestamp, body, upstream_id)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        with time_machine.travel(RECEIVED_AT, tick=False):
            return await client.post(f"/hooks/{sid}", content=body, headers=headers)


def _outbox(sid: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT delivery_id, webhook_id, body FROM "LiteLLM_MCPEventDelivery" WHERE subscription_id = %s', (sid,)
    )


async def test_callback_is_committed_before_ack_and_retry_keeps_the_same_row() -> None:
    async with _subscription("active") as (store, sid):
        delivery_id: Final = gateway_delivery_id(sid, "msg_a")
        expected: Final = [{"delivery_id": delivery_id, "webhook_id": delivery_id, "body": EVENT.decode()}]
        assert (await _post(store, sid)).status_code == 204
        assert _outbox(sid) == expected
        assert (await _post(store, sid)).status_code == 204
        assert _outbox(sid) == expected


@pytest.mark.parametrize("status", ("pending", "active"))
async def test_a_signed_verification_challenge_is_echoed(status: SubscriptionStatus) -> None:
    async with _subscription(status) as (store, sid):
        response: Final = await _post(
            store, sid, b'{"type":"verification","challenge":"nonce-1"}', upstream_id="unknown"
        )
        assert (response.status_code, response.json(), _outbox(sid)) == (200, {"challenge": "nonce-1"}, [])


@pytest.mark.parametrize(
    ("state", "body", "key", "upstream_id", "timestamp", "enabled", "code"),
    (
        pytest.param(None, EVENT, KEY, "up_1", TIMESTAMP, True, 410, id="missing"),
        pytest.param("terminated", EVENT, KEY, "up_1", TIMESTAMP, True, 410, id="terminated"),
        pytest.param("active", EVENT, bytes(32), "up_1", TIMESTAMP, True, 401, id="wrong-key"),
        pytest.param("active", EVENT, KEY, "another", TIMESTAMP, True, 401, id="wrong-upstream"),
        pytest.param("active", EVENT, KEY, "up_1", TIMESTAMP - 301, True, 401, id="expired-signature"),
        pytest.param("pending", EVENT, KEY, "up_1", TIMESTAMP, True, 425, id="pending"),
        pytest.param("active", b"not json", KEY, "up_1", TIMESTAMP, True, 400, id="malformed"),
        pytest.param("active", b"x" * (MAX_BODY_BYTES + 1), KEY, "up_1", TIMESTAMP, True, 413, id="oversized-body"),
        pytest.param("active", EVENT, KEY, "up_1", TIMESTAMP, False, 404, id="disabled"),
    ),
)
async def test_rejection_leaves_no_delivery(
    state: SubscriptionStatus | None,
    body: bytes,
    key: bytes,
    upstream_id: str,
    timestamp: int,
    enabled: bool,
    code: int,
) -> None:
    async with _subscription(state) as (store, sid):
        response: Final = await _post(
            store, sid, body, key=key, upstream_id=upstream_id, timestamp=timestamp, enabled=enabled
        )
        assert (response.status_code, _outbox(sid)) == (code, [])


async def test_a_suspended_subscription_only_queues_the_gap() -> None:
    async with _subscription("suspended") as (store, sid):
        assert (await _post(store, sid)).status_code == 204
        assert _outbox(sid) == []
        gap: Final = b'{"type":"gap","cursor":"c"}'
        assert (await _post(store, sid, gap)).status_code == 204
        assert [r["body"] for r in _outbox(sid)] == [gap.decode()]


async def test_a_failed_database_write_asks_the_upstream_to_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _subscription("active") as (_, sid):
        database_url: Final = httpx.URL(os.environ["DATABASE_URL"]).copy_merge_params(
            {"options": "-c default_transaction_read_only=on"}
        )
        with monkeypatch.context() as env:
            env.setenv("DATABASE_URL", str(database_url))
            async with connected_proxy_prisma() as reader:
                response: Final = await _post(PrismaEventStore(reader), sid)
        assert (response.status_code, _outbox(sid)) == (503, [])

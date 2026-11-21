import base64
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.webhooks.store import (
    Activation,
    PrismaEventStore,
    QueuedDelivery,
    SubscriptionRow,
)
from tests.integration._support.prisma_clients import connected_proxy_prisma

SECRET: Final = "whsec_" + base64.b64encode(bytes(range(32))).decode()
ROTATED: Final = "whsec_" + base64.b64encode(bytes(range(1, 33))).decode()
NOW: Final = datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-store-key")


@asynccontextmanager
async def _connected() -> AsyncIterator[PrismaEventStore]:
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as client:
        yield PrismaEventStore(client)


def _row(status: str = "active") -> SubscriptionRow:
    return SubscriptionRow(
        id=f"sub_{uuid.uuid4().hex}",
        principal=f"key:{uuid.uuid4().hex}",
        key_hash="abc",
        server_id="srv-ci",
        event_name="build.failed",
        arguments={"repo": "acme/web"},
        client_url="https://client.example/hook",
        client_secret=SECRET,
        upstream_secret=SECRET,
        status=status,  # pyright: ignore[reportArgumentType]  # the table's status values
    )


def _delivery(subscription_id: str, delivery_id: str | None = None) -> QueuedDelivery:
    return QueuedDelivery(
        delivery_id=delivery_id or f"dlv_{uuid.uuid4().hex}",
        subscription_id=subscription_id,
        body='{"eventId":"evt_1"}',
        webhook_id="msg_gw",
        next_attempt_at=NOW,
    )


async def _active(store: PrismaEventStore) -> SubscriptionRow:
    row: Final = _row()
    assert await store.reserve(row)
    assert await store.activate(
        row.id,
        Activation(
            client_secret=SECRET,
            upstream_subscription_id="up_1",
            upstream_refresh_before=None,
            refresh_before=NOW + timedelta(hours=1),
            cursor="c1",
        ),
    )
    return row


@pytest.mark.asyncio
async def test_a_reserved_id_refuses_a_second_reservation_until_it_is_terminated() -> None:
    async with _connected() as store:
        row: Final = _row()
        assert await store.reserve(row) is True
        assert await store.reserve(row) is False
        await store.terminate(row.id)
        assert await store.reserve(row) is True
        reserved: Final = await store.get(row.id)
        assert reserved is not None and reserved.status == "pending" and reserved.refresh_before is not None
        await store.delete(row.id)


@pytest.mark.asyncio
async def test_discarding_only_removes_a_row_that_is_still_pending() -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        await store.discard_pending(row.id)
        assert await store.get(row.id) is not None
        pending: Final = _row()
        assert await store.reserve(pending)
        await store.discard_pending(pending.id)
        assert await store.get(pending.id) is None
        await store.delete(row.id)


@pytest.mark.asyncio
async def test_no_writer_revives_a_terminated_subscription() -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        await store.terminate(row.id)
        refreshed: Final = Activation(
            client_secret=ROTATED,
            upstream_subscription_id="up_2",
            upstream_refresh_before=None,
            refresh_before=None,
            cursor="c2",
        )
        assert await store.activate(row.id, refreshed) is False
        await store.mark_delivered(row.id, NOW)
        await store.mark_failed(row.id, "timeout", NOW, 1)
        after: Final = await store.get(row.id)
        assert after is not None
        assert (after.status, after.client_secret, after.consecutive_failures, after.last_delivery_at) == (
            "terminated",
            SECRET,
            0,
            None,
        )
        await store.delete(row.id)


@pytest.mark.asyncio
async def test_a_refresh_rotates_the_secret_encrypted_at_rest() -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        rotated: Final = Activation(
            client_secret=ROTATED,
            upstream_subscription_id="up_1",
            upstream_refresh_before=None,
            refresh_before=None,
            cursor="c2",
        )
        assert await store.activate(row.id, rotated)
        after: Final = await store.get(row.id)
        assert after is not None and (after.client_secret, after.cursor) == (ROTATED, "c2")
        raw: Final = await store._subscriptions.find_unique(where={"id": row.id})  # pyright: ignore[reportPrivateUsage]  # the stored ciphertext is the unit under test
        assert raw is not None and raw.client_secret != ROTATED
        await store.delete(row.id)


@pytest.mark.asyncio
async def test_failures_count_up_set_failed_since_once_and_suspend_at_the_threshold() -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        await store.mark_failed(row.id, "timeout", NOW, 2)
        await store.mark_failed(row.id, "http_5xx", NOW + timedelta(minutes=1), 2)
        after: Final = await store.get(row.id)
        assert after is not None
        assert (after.consecutive_failures, after.last_error, after.status) == (2, "http_5xx", "suspended")
        assert after.failed_since is not None and after.failed_since.replace(tzinfo=timezone.utc) == NOW
        await store.mark_delivered(row.id, NOW)
        cleared: Final = await store.get(row.id)
        assert cleared is not None and (cleared.consecutive_failures, cleared.failed_since, cleared.last_error) == (
            0,
            None,
            None,
        )
        await store.delete(row.id)


@pytest.mark.asyncio
async def test_a_delivered_message_is_never_claimed_again_and_its_replay_stays_a_duplicate() -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        delivery: Final = _delivery(row.id)
        assert await store.enqueue(delivery)
        claimed: Final = await store.claim(NOW, NOW + timedelta(seconds=60), 100)
        assert delivery.delivery_id in {item.delivery_id for item in claimed}
        await store.prepare(delivery.delivery_id, '{"eventId":"evt_1","name":"ci-build.failed"}', "msg_built")
        await store.finish(delivery.delivery_id, NOW)
        later: Final = await store.claim(NOW + timedelta(minutes=5), NOW + timedelta(minutes=6), 100)
        assert delivery.delivery_id not in {item.delivery_id for item in later}
        assert await store.enqueue(delivery) is False
        await store.purge_delivered(NOW + timedelta(minutes=11))
        assert await store.enqueue(delivery) is True
        await store.delete(row.id)


@pytest.mark.asyncio
async def test_a_prepared_delivery_is_claimed_with_its_built_body() -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        delivery: Final = _delivery(row.id)
        assert await store.enqueue(delivery)
        await store.prepare(delivery.delivery_id, '{"eventId":"evt_1","name":"ci-build.failed"}', "msg_built")
        await store.reschedule(delivery.delivery_id, 1, NOW)
        (claimed,) = [
            item
            for item in await store.claim(NOW, NOW + timedelta(seconds=60), 100)
            if item.delivery_id == delivery.delivery_id
        ]
        assert (claimed.prepared, claimed.webhook_id, claimed.attempt) == (True, "msg_built", 1)
        await store.delete(row.id)

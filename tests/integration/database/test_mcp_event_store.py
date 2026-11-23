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
async def test_an_unreadable_signing_key_never_becomes_a_plaintext_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _connected() as store:
        row: Final = await _active(store)
        try:
            monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-different-store-key")
            loaded: Final = await store.get(row.id)
            assert loaded is not None and (loaded.client_secret, loaded.upstream_secret) == ("", "")
        finally:
            await store.delete(row.id)

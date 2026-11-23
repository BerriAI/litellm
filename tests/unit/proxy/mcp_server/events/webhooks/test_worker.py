import base64
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import seal_cursor
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.outgoing import binding_of
from litellm.proxy.mcp_server.events.webhooks.reauth import Authorized, Reauthorization
from litellm.proxy.mcp_server.events.webhooks.store import QueuedDelivery, SubscriptionRow
from litellm.proxy.mcp_server.events.webhooks.worker import RETRY_DELAYS_SECONDS, DeliveryWorker
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import CallbackFailure
from litellm.types.mcp_server.mcp_server_manager import MCPServer

from ..fakes import SETTINGS, occurrence, runtime, server
from .memory_store import MemoryStore

KEY: Final = bytes(range(32))
SECRET: Final = "whsec_" + base64.b64encode(KEY).decode()
NOW: Final = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
ROW: Final = SubscriptionRow(
    id="sub_1",
    principal="key:1",
    key_hash="abc",
    server_id="srv-ci",
    event_name="build.failed",
    arguments={"repo": "acme/web"},
    client_url="https://client.example/hook",
    client_secret=SECRET,
    upstream_subscription_id="up_1",
    upstream_secret=SECRET,
    status="active",
    cursor="downstream-watermark",
)


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-worker-key")


class _Client:
    def __init__(self, *statuses: int | CallbackFailure) -> None:
        self.statuses = list(statuses) or [200]
        self.posts: list[tuple[bytes, Mapping[str, str]]] = []  # mutable-ok: records deliveries

    async def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> Result[CallbackResponse, CallbackFailure]:
        self.posts.append((body, headers))
        status: Final = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return Ok(CallbackResponse(status=status, body=b"")) if isinstance(status, int) else Error(status)

    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(body) for body, _ in self.posts]


def _caller(outcome: Reauthorization):  # noqa: ANN202  # test helper
    async def load(_row: SubscriptionRow) -> Reauthorization:
        return outcome

    return load


def _delivery(body: object, attempt: int = 0) -> QueuedDelivery:
    return QueuedDelivery(
        delivery_id="d1",
        subscription_id="sub_1",
        body=json.dumps(body),
        webhook_id="msg_gw",
        attempt=attempt,
        next_attempt_at=NOW,
    )


def _worker(
    store: MemoryStore,
    client: _Client,
    outcome: Reauthorization = Authorized(UserAPIKeyAuth(api_key="abc")),
    target: MCPServer | None = None,
    screen=None,  # noqa: ANN001  # test double
    settings: EventsSettings = SETTINGS,
) -> DeliveryWorker:
    upstream_server: Final = target or server()
    return DeliveryWorker(
        store,
        settings=lambda: settings,
        runtime=runtime((upstream_server,), screen=screen),
        load_caller=_caller(outcome),
        post=client,
        clock=lambda: NOW,
        lookup=lambda _: upstream_server,
    )


def _store(row: SubscriptionRow = ROW, body: object = None, attempt: int = 0) -> MemoryStore:
    event: Final = body if body is not None else occurrence("evt_1", "41").model_dump(by_alias=True, mode="json")
    return MemoryStore(rows={row.id: row}, queue={"d1": _delivery(event, attempt)})


def test_a_sealed_gap_cursor_falls_back_to_the_last_known_watermark() -> None:
    from litellm.proxy.mcp_server.events.webhooks.outgoing import gap

    stored: Final = seal_cursor(binding_of(ROW), "12", int(NOW.timestamp()))
    assert isinstance(stored, Ok)
    outgoing: Final = gap(ROW.model_copy(update={"cursor": stored.ok}), None, int(NOW.timestamp()))
    assert json.loads(outgoing.body) == {"type": "gap", "cursor": stored.ok}


@pytest.mark.asyncio
async def test_the_last_scheduled_retry_is_delayed_eight_minutes() -> None:
    store: Final = _store(attempt=2)
    (delivery,) = await store.claim(NOW, NOW + timedelta(minutes=5), 1)
    await _worker(store, _Client(503)).process(delivery)
    queued: Final = store.queue["d1"]
    assert (queued.attempt, queued.next_attempt_at) == (3, NOW + timedelta(minutes=8))


@pytest.mark.asyncio
async def test_a_temporarily_unreadable_signing_secret_keeps_the_event_for_retry() -> None:
    store: Final = _store(ROW.model_copy(update={"client_secret": ""}))
    client: Final = _Client()
    (delivery,) = await store.claim(NOW, NOW + timedelta(minutes=5), 1)
    await _worker(store, client).process(delivery)
    assert not client.posts and not store.delivered
    retry: Final = store.queue["d1"]
    assert retry.claim_id is None and retry.attempt == 0 and retry.next_attempt_at > NOW


class _SlowClient(_Client):
    """A callback that lets something else change the subscription while the POST is in flight."""

    def __init__(self, during: Callable[[], Awaitable[None]], *statuses: int | CallbackFailure) -> None:
        super().__init__(*statuses)
        self.during = during

    async def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> Result[CallbackResponse, CallbackFailure]:
        await self.during()
        return await super().__call__(url, body, headers)


ROTATED: Final = "whsec_" + base64.b64encode(bytes(range(1, 33))).decode()


@pytest.mark.asyncio
async def test_an_abandoned_event_stays_queued_when_its_gap_cannot_be_written() -> None:
    store: Final = _store(attempt=len(RETRY_DELAYS_SECONDS))
    store.fail_enqueue = True
    (claimed,) = await store.claim(NOW, NOW + timedelta(minutes=5), 1)
    with pytest.raises(ConnectionError):
        await _worker(store, _Client("timeout")).process(claimed)
    assert "d1" in store.queue


@pytest.mark.parametrize("status,attempt", [(200, 0), (500, len(RETRY_DELAYS_SECONDS))])
@pytest.mark.asyncio
async def test_worker_losing_a_claim_in_flight_cannot_settle(status: int, attempt: int) -> None:
    store: Final = _store(attempt=attempt)
    (stale,) = await store.claim(NOW, NOW + timedelta(seconds=1), 1)

    async def take_over() -> None:
        await store.claim(NOW + timedelta(seconds=2), NOW + timedelta(minutes=5), 1)

    await _worker(store, _SlowClient(take_over, status)).process(stale)
    assert (store.rows["sub_1"], list(store.queue), store.delivered) == (ROW, ["d1"], {})
    assert store.queue["d1"].claim_id != stale.claim_id


@pytest.mark.asyncio
async def test_a_worker_reading_an_expired_deadline_preserves_a_concurrently_renewed_delivery() -> None:
    class RenewOnRead(MemoryStore):
        async def get(self, subscription_id: str) -> SubscriptionRow | None:
            observed: Final = await super().get(subscription_id)
            assert observed is not None
            self.rows[subscription_id] = observed.model_copy(
                update={"generation": "renewed", "refresh_before": NOW + timedelta(hours=1)}
            )
            return observed

    before: Final = _store(ROW.model_copy(update={"refresh_before": NOW - timedelta(seconds=1)}))
    store: Final = RenewOnRead(rows=before.rows, queue=before.queue)
    (delivery,) = await store.claim(NOW, NOW + timedelta(minutes=5), 1)
    client: Final = _Client()
    await _worker(store, client).process(delivery)
    assert not client.posts and store.queue["d1"].body == delivery.body and not store.delivered

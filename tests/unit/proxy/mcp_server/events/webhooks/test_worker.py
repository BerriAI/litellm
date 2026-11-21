import base64
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import open_cursor, seal_cursor
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.outgoing import binding_of
from litellm.proxy.mcp_server.events.webhooks.reauth import Authorized, Reauthorization, Revoked, Unavailable
from litellm.proxy.mcp_server.events.webhooks.signing import verify
from litellm.proxy.mcp_server.events.webhooks.store import Activation, QueuedDelivery, SubscriptionRow
from litellm.proxy.mcp_server.events.webhooks.worker import RETRY_DELAYS_SECONDS, SUSPEND_AFTER_FAILURES, DeliveryWorker
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import CallbackFailure
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

from ..fakes import SETTINGS, occurrence, runtime, screen_blocking, server
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


@pytest.mark.asyncio
async def test_an_event_is_signed_with_the_client_secret_renamed_and_resealed() -> None:
    store: Final = _store()
    client: Final = _Client(200)
    assert await _worker(store, client).run_once() == 1
    body, headers = client.posts[0]
    assert verify((KEY,), headers, body, int(time.time()))
    assert headers["x-mcp-subscription-id"] == "sub_1" and headers["webhook-id"] == "msg_gw"
    delivered: Final = json.loads(body)
    assert (delivered["name"], delivered["eventId"]) == ("ci-build.failed", "evt_1")
    opened: Final = open_cursor(delivered["cursor"], binding_of(ROW), int(time.time()))
    assert isinstance(opened, Ok) and opened.ok.upstream == "41"
    assert store.queue == {} and store.rows["sub_1"].last_delivery_at == NOW


@pytest.mark.asyncio
async def test_the_legacy_flag_reuses_the_event_id_as_the_webhook_id() -> None:
    client: Final = _Client(200)
    legacy: Final = server(events=MCPServerEventsConfig(enabled=True, legacy_webhook_id_equals_event_id=True))
    await _worker(_store(), client, target=legacy).run_once()
    assert client.posts[0][1]["webhook-id"] == "evt_1"


@pytest.mark.parametrize(("attempt", "delay"), tuple(enumerate(RETRY_DELAYS_SECONDS)))
@pytest.mark.asyncio
async def test_failures_retry_on_the_backoff_ladder(attempt: int, delay: int) -> None:
    store: Final = _store(attempt=attempt)
    await _worker(store, _Client(503)).run_once()
    assert (store.queue["d1"].attempt, store.queue["d1"].next_attempt_at) == (
        attempt + 1,
        NOW + timedelta(seconds=delay),
    )


@pytest.mark.asyncio
async def test_an_event_that_exhausts_its_retries_is_counted_and_reported_as_a_gap() -> None:
    store: Final = _store(attempt=len(RETRY_DELAYS_SECONDS))
    client: Final = _Client("timeout")
    await _worker(store, client).run_once()
    row: Final = store.rows["sub_1"]
    assert (row.consecutive_failures, row.last_error, row.failed_since, row.status) == (1, "timeout", NOW, "active")
    (queued,) = store.queue.values()
    sent_cursor: Final = client.bodies()[0]["cursor"]
    assert (queued.prepared, json.loads(queued.body)) == (True, {"type": "gap", "cursor": sent_cursor})


@pytest.mark.asyncio
async def test_sustained_failure_suspends_the_subscription() -> None:
    store: Final = _store(ROW.model_copy(update={"consecutive_failures": SUSPEND_AFTER_FAILURES - 1}), attempt=3)
    await _worker(store, _Client(500)).run_once()
    assert store.rows["sub_1"].status == "suspended"


@pytest.mark.parametrize("status", (410, 413))
@pytest.mark.asyncio
async def test_gone_and_too_large_are_never_retried(status: int) -> None:
    store: Final = _store()
    await _worker(store, _Client(status)).run_once()
    assert store.queue == {} and store.rows["sub_1"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_a_revoked_caller_ends_the_subscription_with_a_signed_terminated_envelope() -> None:
    store: Final = _store()
    client: Final = _Client(200)
    await _worker(store, client, outcome=Revoked("key")).run_once()
    assert client.bodies() == [
        {"type": "terminated", "error": {"code": -32024, "message": "Forbidden", "data": {"reason": "Access revoked"}}}
    ]
    assert store.rows["sub_1"].status == "terminated" and store.queue == {}


@pytest.mark.asyncio
async def test_a_transient_reauthorization_failure_delays_without_spending_an_attempt() -> None:
    store: Final = _store()
    client: Final = _Client(200)
    await _worker(store, client, outcome=Unavailable("database down")).run_once()
    assert client.posts == [] and store.rows["sub_1"].status == "active"
    assert (store.queue["d1"].attempt, store.queue["d1"].next_attempt_at) == (0, NOW + timedelta(seconds=30))


@pytest.mark.asyncio
async def test_a_blocked_event_becomes_a_gap_with_the_skipped_position() -> None:
    client: Final = _Client(200)
    await _worker(_store(), client, screen=screen_blocking()).run_once()
    delivered: Final = client.bodies()[0]
    assert delivered["type"] == "gap"
    opened: Final = open_cursor(str(delivered["cursor"]), binding_of(ROW), int(time.time()))
    assert isinstance(opened, Ok) and opened.ok.upstream == "41"


@pytest.mark.asyncio
async def test_events_over_the_caller_budget_become_a_gap() -> None:
    client: Final = _Client(200)
    store: Final = _store()
    second: Final = _delivery(occurrence("evt_2", "42").model_dump(by_alias=True, mode="json")).model_copy(
        update={"delivery_id": "d2", "webhook_id": "msg_gw2"}
    )
    store.queue["d2"] = second
    await _worker(store, client, settings=EventsSettings(enabled=True, max_events_per_minute=1)).run_once()
    assert sorted(str(body.get("type", body.get("eventId"))) for body in client.bodies()) == ["evt_1", "gap"]


@pytest.mark.asyncio
async def test_an_upstream_terminated_is_forwarded_and_ends_the_subscription() -> None:
    client: Final = _Client(200)
    store: Final = _store(
        body={"type": "terminated", "error": {"code": -32023, "message": "NotFound", "data": {"kind": "event"}}}
    )
    await _worker(store, client).run_once()
    assert client.bodies()[0]["type"] == "terminated" and store.rows["sub_1"].status == "terminated"
    assert client.posts[0][1]["webhook-id"].startswith("msg_terminated_")


@pytest.mark.asyncio
async def test_an_upstream_gap_is_forwarded_with_a_resealed_cursor() -> None:
    client: Final = _Client(200)
    await _worker(_store(body={"type": "gap", "cursor": "77"}), client).run_once()
    opened: Final = open_cursor(str(client.bodies()[0]["cursor"]), binding_of(ROW), int(time.time()))
    assert isinstance(opened, Ok) and opened.ok.upstream == "77"


@pytest.mark.parametrize("status", ("terminated", "pending"))
@pytest.mark.asyncio
async def test_deliveries_for_inactive_rows_are_dropped(status: str) -> None:
    store: Final = _store(ROW.model_copy(update={"status": status}))
    client: Final = _Client(200)
    await _worker(store, client).run_once()
    assert client.posts == [] and store.queue == {}


@pytest.mark.asyncio
async def test_expired_subscriptions_are_terminated_quietly() -> None:
    store: Final = _store(ROW.model_copy(update={"refresh_before": NOW - timedelta(seconds=1)}))
    client: Final = _Client(200)
    await _worker(store, client).expire()
    assert store.rows["sub_1"].status == "terminated" and client.posts == [] and store.queue == {}


def test_a_sealed_gap_cursor_falls_back_to_the_last_known_watermark() -> None:
    from litellm.proxy.mcp_server.events.webhooks.outgoing import gap

    stored: Final = seal_cursor(binding_of(ROW), "12", int(time.time()))
    assert isinstance(stored, Ok)
    outgoing: Final = gap(ROW.model_copy(update={"cursor": stored.ok}), None, int(time.time()))
    assert json.loads(outgoing.body) == {"type": "gap", "cursor": stored.ok}


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


@pytest.mark.parametrize("status", (200, 503))
@pytest.mark.asyncio
async def test_a_termination_during_an_inflight_delivery_stays_terminated(status: int) -> None:
    store: Final = _store()

    async def admin_terminates() -> None:
        await store.terminate("sub_1")

    await _worker(store, _SlowClient(admin_terminates, status)).run_once()
    assert store.rows["sub_1"].status == "terminated"


@pytest.mark.asyncio
async def test_a_refresh_during_an_inflight_delivery_keeps_the_new_secret_and_cursor() -> None:
    store: Final = _store()
    activation: Final = Activation(
        client_secret=ROTATED,
        upstream_subscription_id="up_1",
        upstream_refresh_before=None,
        refresh_before=None,
        cursor="mcp_state_v1.fresh",
    )

    async def client_refreshes() -> None:
        assert await store.activate("sub_1", activation)

    await _worker(store, _SlowClient(client_refreshes)).run_once()
    assert (store.rows["sub_1"].client_secret, store.rows["sub_1"].cursor) == (ROTATED, "mcp_state_v1.fresh")


@pytest.mark.asyncio
async def test_a_retry_resends_the_first_build_without_spending_the_event_budget_again() -> None:
    store: Final = _store()
    client: Final = _Client(503, 200)
    tight: Final = EventsSettings(enabled=True, public_base_url="https://gw.example", max_events_per_minute=1)
    worker: Final = _worker(store, client, settings=tight)
    await worker.run_once()
    store.queue["d1"] = store.queue["d1"].model_copy(update={"next_attempt_at": NOW})
    await worker.run_once()
    (first_body, first_headers), (second_body, second_headers) = client.posts
    assert (second_body, second_headers["webhook-id"]) == (first_body, first_headers["webhook-id"])
    assert json.loads(second_body)["eventId"] == "evt_1"


@pytest.mark.asyncio
async def test_an_upstream_gap_keeps_its_webhook_id_across_retries() -> None:
    store: Final = _store(body={"type": "gap", "cursor": "41"})
    client: Final = _Client(503, 200)
    worker: Final = _worker(store, client)
    await worker.run_once()
    store.queue["d1"] = store.queue["d1"].model_copy(update={"next_attempt_at": NOW})
    await worker.run_once()
    assert client.posts[0][1]["webhook-id"] == client.posts[1][1]["webhook-id"]
    assert client.posts[0][1]["webhook-id"].startswith("msg_gap_")


@pytest.mark.asyncio
async def test_a_subscription_past_its_deadline_receives_nothing() -> None:
    store: Final = _store(row=ROW.model_copy(update={"refresh_before": NOW - timedelta(seconds=1)}))
    client: Final = _Client(200)
    await _worker(store, client).run_once()
    assert client.posts == [] and store.queue == {}


@pytest.mark.asyncio
async def test_a_delivered_message_stays_a_duplicate_until_its_tombstone_is_purged() -> None:
    store: Final = _store()
    await _worker(store, _Client(200)).run_once()
    replay: Final = _delivery(occurrence().model_dump(by_alias=True, mode="json"))
    assert await store.enqueue(replay) is False
    await store.purge_delivered(NOW + timedelta(minutes=11))
    assert await store.enqueue(replay) is True

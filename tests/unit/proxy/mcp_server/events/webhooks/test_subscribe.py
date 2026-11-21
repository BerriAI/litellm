import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from mcp.shared.exceptions import MCPError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.access import principal_of
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.policy import subscription_id
from litellm.proxy.mcp_server.events.webhooks.signing import parse_secret
from litellm.proxy.mcp_server.events.webhooks.store import Activation
from litellm.proxy.mcp_server.events.webhooks.subscribe import (
    _prepare,  # pyright: ignore[reportPrivateUsage]  # the reservation step is the unit under test
    _Request,  # pyright: ignore[reportPrivateUsage]  # see above
    _reserve,  # pyright: ignore[reportPrivateUsage]  # see above
    subscribe,
    unsubscribe,
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import (
    CallbackFailure,
    EventsErrorCode,
    SubscribeEventsParams,
    UnsubscribeDelivery,
    UnsubscribeEventsParams,
    WebhookDelivery,
)

from ..fakes import CALLER, JWT, SETTINGS, FakeUpstream, context, runtime, server
from .memory_store import MemoryStore

SECRET: Final = "whsec_" + base64.b64encode(bytes(range(32))).decode()
HOOK: Final = "https://client.example/hook"
ROTATED: Final = "whsec_" + base64.b64encode(bytes(range(1, 33))).decode()
ARGS: Final = {"repo": "acme/web"}
SID: Final = subscription_id(principal_of(CALLER) or "", HOOK, "ci-build.failed", ARGS)
UPSTREAM_OK: Final = {"id": "up_1", "refreshBefore": "2099-01-01T00:00:00Z", "cursor": "c0", "truncated": False}


def _request(store: MemoryStore) -> _Request:
    secret: Final = parse_secret(SECRET)
    assert secret is not None
    return _Request(principal=principal_of(CALLER) or "", secret=secret, store=store)


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-subscribe-key")


class _Client:
    def __init__(self, echo: bool = True, failure: CallbackFailure | None = None) -> None:
        self.echo = echo
        self.failure = failure
        self.posts: list[bytes] = []  # mutable-ok: records verification posts

    async def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> Result[CallbackResponse, CallbackFailure]:
        self.posts.append(body)
        if self.failure is not None:
            return Error(self.failure)
        challenge: Final = json.loads(body)["challenge"] if self.echo else "wrong"
        return Ok(CallbackResponse(status=200, body=json.dumps({"challenge": challenge}).encode()))


def _params(ttl_ms: int | None = 600_000, url: str = HOOK, secret: str = SECRET) -> SubscribeEventsParams:
    return SubscribeEventsParams(
        name="ci-build.failed", arguments=ARGS, delivery=WebhookDelivery(url=url, secret=secret), ttl_ms=ttl_ms
    )


async def _subscribe(store: MemoryStore, upstream: FakeUpstream, client: _Client, **overrides: object):  # noqa: ANN202  # test helper
    return await subscribe(
        overrides.get("context", context()),  # pyright: ignore[reportArgumentType]  # test override
        overrides.get("params", _params()),  # pyright: ignore[reportArgumentType]  # test override
        None,
        overrides.get("settings", SETTINGS),  # pyright: ignore[reportArgumentType]  # test override
        runtime((server(),), upstream=upstream),
        store,
        client,
    )


@pytest.mark.asyncio
async def test_a_new_subscription_verifies_the_client_then_subscribes_upstream_with_the_gateway_callback() -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    client: Final = _Client()
    result: Final = await _subscribe(store, upstream, client)
    assert isinstance(result, Ok)
    assert result.ok.id == SID and result.ok.refresh_before is not None
    assert len(client.posts) == 1 and json.loads(client.posts[0])["type"] == "verification"
    method, params, _headers = upstream.calls[0]
    delivery: Final = params["delivery"]
    assert method == "events/subscribe" and isinstance(delivery, Mapping)
    assert delivery["url"] == f"https://gateway.example/mcp/events/webhooks/{SID}"
    assert delivery["secret"] != SECRET
    row: Final = store.rows[SID]
    assert (row.status, row.upstream_subscription_id, row.client_secret) == ("active", "up_1", SECRET)


@pytest.mark.asyncio
async def test_a_refresh_keeps_the_id_skips_verification_and_reports_delivery_status() -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    client: Final = _Client()
    first: Final = await _subscribe(store, upstream, client)
    store.rows[SID] = store.rows[SID].model_copy(
        update={"last_delivery_at": datetime(2026, 11, 20, tzinfo=timezone.utc)}
    )
    second: Final = await _subscribe(store, upstream, client)
    assert isinstance(first, Ok) and isinstance(second, Ok) and first.ok.id == second.ok.id
    assert len(client.posts) == 1
    assert second.ok.delivery_status is not None and second.ok.delivery_status.active is True
    assert second.ok.delivery_status.last_delivery_at == "2026-11-20T00:00:00Z"


@pytest.mark.asyncio
async def test_refreshing_a_suspended_subscription_reactivates_it_and_queues_a_gap() -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    await _subscribe(store, upstream, _Client())
    store.rows[SID] = store.rows[SID].model_copy(update={"status": "suspended", "consecutive_failures": 20})
    refreshed: Final = await _subscribe(store, upstream, _Client())
    assert isinstance(refreshed, Ok) and refreshed.ok.delivery_status is not None
    assert refreshed.ok.delivery_status.active is False
    assert store.rows[SID].status == "active"
    (gap,) = store.queue.values()
    assert (gap.prepared, json.loads(gap.body)) == (True, {"type": "gap", "cursor": refreshed.ok.cursor})


def test_no_expiry_is_written_as_an_explicit_null() -> None:
    from litellm.types.mcp_server.events import SubscribeEventsResult

    assert SubscribeEventsResult(id="s", refresh_before=None).model_dump(by_alias=True, exclude_none=True) == {
        "id": "s",
        "truncated": False,
        "refreshBefore": None,
    }


@pytest.mark.parametrize(
    ("overrides", "code"),
    (
        ({"settings": EventsSettings(enabled=True)}, EventsErrorCode.UNSUPPORTED),
        ({"context": context(UserAPIKeyAuth())}, EventsErrorCode.FORBIDDEN),
        ({"context": context(UserAPIKeyAuth(api_key=JWT, user_id="u"))}, EventsErrorCode.FORBIDDEN),
        ({"params": _params(url="http://client.example/hook")}, EventsErrorCode.INVALID_PARAMS),
        ({"params": _params(secret="whsec_short")}, EventsErrorCode.INVALID_PARAMS),
        ({"settings": EventsSettings(enabled=True, public_base_url="https://gw", max_subscriptions_per_key=1)}, None),
    ),
)
@pytest.mark.asyncio
async def test_invalid_subscriptions_are_rejected_before_any_upstream_work(
    overrides: dict[str, object], code: EventsErrorCode | None
) -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    result: Final = await _subscribe(store, upstream, _Client(), **overrides)
    if code is None:
        assert isinstance(result, Ok)
        other: Final = await _subscribe(
            store, upstream, _Client(), params=_params(url="https://other.example/h"), **overrides
        )
        assert isinstance(other, Error) and other.error.code is EventsErrorCode.RESOURCE_EXHAUSTED
        return
    assert isinstance(result, Error) and result.error.code is code
    assert upstream.calls == [] and store.rows == {}


@pytest.mark.parametrize(
    ("client", "reason"), ((_Client(echo=False), "challenge_failed"), (_Client(failure="timeout"), "timeout"))
)
@pytest.mark.asyncio
async def test_a_client_that_fails_verification_gets_callback_endpoint_error(client: _Client, reason: str) -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    result: Final = await _subscribe(store, upstream, client)
    assert isinstance(result, Error)
    assert (result.error.code, result.error.data) == (EventsErrorCode.CALLBACK_ENDPOINT_ERROR, {"reason": reason})
    assert upstream.calls == [] and store.rows == {}


@pytest.mark.parametrize(
    "failure",
    (MCPError(code=-32024, message="Forbidden"), TimeoutError("upstream hung"), MCPError(code=-32027, message="x")),
)
@pytest.mark.asyncio
async def test_an_upstream_failure_leaves_no_pending_row(failure: Exception) -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": failure})
    if isinstance(failure, MCPError):
        result = await _subscribe(store, upstream, _Client())
        assert isinstance(result, Error)
        assert result.error.code is (
            EventsErrorCode.FORBIDDEN if failure.error.code == -32024 else EventsErrorCode.INTERNAL_ERROR
        )
    else:
        with pytest.raises(TimeoutError):
            await _subscribe(store, upstream, _Client())
    assert store.rows == {}


@pytest.mark.asyncio
async def test_unsubscribe_deletes_the_row_and_ends_the_upstream_subscription() -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK, "events/unsubscribe": {}})
    await _subscribe(store, upstream, _Client())
    params: Final = UnsubscribeEventsParams(
        name="ci-build.failed", arguments=ARGS, delivery=UnsubscribeDelivery(url=HOOK)
    )
    result: Final = await unsubscribe(context(), params, None, SETTINGS, runtime((server(),), upstream=upstream), store)
    assert result == Ok(None) and store.rows == {}
    method, sent, _ = upstream.calls[-1]
    assert method == "events/unsubscribe"
    assert sent["delivery"] == {"url": f"https://gateway.example/mcp/events/webhooks/{SID}"}
    again: Final = await unsubscribe(context(), params, None, SETTINGS, runtime((server(),), upstream=upstream), store)
    assert isinstance(again, Error) and (again.error.code, again.error.data) == (
        EventsErrorCode.NOT_FOUND,
        {"kind": "subscription"},
    )


@pytest.mark.asyncio
async def test_a_subscribe_racing_one_still_being_created_is_told_to_retry_and_touches_nothing() -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    await _subscribe(store, upstream, _Client())
    store.rows[SID] = store.rows[SID].model_copy(update={"status": "pending"})
    calls_before: Final = len(upstream.calls)
    second: Final = await _subscribe(store, upstream, _Client(), params=_params(secret=ROTATED))
    assert isinstance(second, Error) and second.error.code is EventsErrorCode.INTERNAL_ERROR
    assert (len(upstream.calls), store.rows[SID].status, store.rows[SID].client_secret) == (
        calls_before,
        "pending",
        SECRET,
    )


@pytest.mark.asyncio
async def test_a_reservation_past_the_cap_backs_itself_out() -> None:
    capped: Final = EventsSettings(enabled=True, public_base_url="https://gw", max_subscriptions_per_key=1)
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    prepared: Final = await _prepare(
        context(), _params(), None, capped, runtime((server(),), upstream=upstream), _request(store), _Client()
    )
    assert isinstance(prepared, Ok)
    await store.reserve(prepared.ok.base.model_copy(update={"id": "sub_other"}))
    reserved: Final = await _reserve(prepared.ok, capped)
    assert isinstance(reserved, Error) and reserved.error.code is EventsErrorCode.RESOURCE_EXHAUSTED
    assert set(store.rows) == {"sub_other"}


@pytest.mark.asyncio
async def test_a_failed_subscribe_never_deletes_a_row_another_request_activated() -> None:
    store: Final = MemoryStore()
    ok_upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    await _subscribe(store, ok_upstream, _Client())
    await store.discard_pending(SID)
    assert store.rows[SID].status == "active"


@pytest.mark.asyncio
async def test_a_subscription_terminated_while_upstream_answers_is_ended_upstream_too() -> None:
    class _TerminatingStore(MemoryStore):
        async def activate(self, subscription_id: str, activation: Activation) -> bool:
            await self.terminate(subscription_id)
            return await super().activate(subscription_id, activation)

    store: Final = _TerminatingStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK, "events/unsubscribe": {}})
    result: Final = await _subscribe(store, upstream, _Client())
    assert isinstance(result, Error) and result.error.code is EventsErrorCode.NOT_FOUND
    assert [method for method, *_ in upstream.calls] == ["events/subscribe", "events/unsubscribe"]


@pytest.mark.asyncio
async def test_a_cursor_that_cannot_be_sealed_is_refused_before_any_upstream_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    result: Final = await _subscribe(store, upstream, _Client())
    assert isinstance(result, Error)
    assert (upstream.calls, store.rows) == ([], {})


@pytest.mark.asyncio
async def test_a_name_now_served_by_another_server_ends_the_old_subscription_and_starts_fresh() -> None:
    store: Final = MemoryStore()
    upstream: Final = FakeUpstream(responses={"events/subscribe": UPSTREAM_OK})
    await _subscribe(store, upstream, _Client())
    store.rows[SID] = store.rows[SID].model_copy(update={"server_id": "srv-old", "consecutive_failures": 3})
    result: Final = await _subscribe(store, upstream, _Client())
    assert isinstance(result, Ok)
    assert (store.rows[SID].server_id, store.rows[SID].status, store.rows[SID].consecutive_failures) == (
        "srv-ci",
        "active",
        0,
    )

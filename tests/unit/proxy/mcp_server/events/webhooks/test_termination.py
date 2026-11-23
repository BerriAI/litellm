import asyncio
import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from mcp.shared.exceptions import MCPError

from litellm.proxy.mcp_server.events.errors import forbidden
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.policy import callback_url
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow
from litellm.proxy.mcp_server.events.webhooks.termination import terminate
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.types.mcp_server.events import CallbackFailure, EventsErrorCode, JsonObject

from ..fakes import SETTINGS, FakeUpstream, runtime, server
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


@pytest.mark.parametrize("notify", (True, False))
@pytest.mark.asyncio
async def test_terminating_twice_notifies_once_and_always_ends_the_upstream_subscription(notify: bool) -> None:
    store: Final = MemoryStore(rows={"sub_1": ROW})
    upstream: Final = FakeUpstream(responses={"events/unsubscribe": {}})
    client: Final = _Client()
    events_runtime: Final = runtime((server(),), upstream=upstream)
    for _ in range(2):
        await terminate(
            ROW, forbidden("Key revoked"), store, SETTINGS, events_runtime, client, lambda _id: server(), notify=notify
        )
    assert (len(client.posts), [method for method, *_ in upstream.calls], store.rows["sub_1"].status) == (
        1 if notify else 0,
        ["events/unsubscribe"],
        "terminated",
    )


@pytest.mark.parametrize("interrupted_at", ("local_commit", "client", "upstream"))
@pytest.mark.asyncio
async def test_interrupted_termination_can_finish_upstream_without_repeating_the_notification(
    interrupted_at: str,
) -> None:
    entered: Final = asyncio.Event()

    async def pause(stage: str) -> None:
        if stage == interrupted_at and not entered.is_set():
            entered.set()
            await asyncio.Future()

    class InterruptedStore(MemoryStore):
        async def end_if_current(
            self, subscription_id: str, upstream_subscription_id: str | None, generation: str | None = None
        ) -> bool:
            ended: Final = await super().end_if_current(subscription_id, upstream_subscription_id, generation)
            await pause("local_commit")
            return ended

    class InterruptedUpstream(FakeUpstream):
        async def call(
            self, target: SubscriptionTarget, method: str, params: JsonObject, headers: Mapping[str, str]
        ) -> Mapping[str, object]:
            await pause("upstream")
            return await super().call(target, method, params, headers)

    class InterruptedClient(_Client):
        async def __call__(
            self, url: str, body: bytes, headers: Mapping[str, str]
        ) -> Result[CallbackResponse, CallbackFailure]:
            result: Final = await super().__call__(url, body, headers)
            await pause("client")
            return result

    store: Final = InterruptedStore(rows={ROW.id: ROW})
    client: Final = InterruptedClient()
    upstream: Final = InterruptedUpstream(responses={"events/unsubscribe": {}})
    target: Final = runtime((server(),), upstream=upstream)

    async def end() -> bool:
        current: Final = await store.get(ROW.id)
        assert current is not None
        return await terminate(current, forbidden("Key revoked"), store, SETTINGS, target, client, lambda _: server())

    task: Final = asyncio.create_task(end())
    await entered.wait()
    assert store.rows[ROW.id].status == "terminated"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await end() and await end()
    assert (len(client.posts), [m for m, *_ in upstream.calls], store.rows[ROW.id].upstream_subscription_id) == (
        0 if interrupted_at == "local_commit" else 1,
        ["events/unsubscribe"],
        None,
    )


@pytest.mark.parametrize("status", ("active", "terminated"))
@pytest.mark.asyncio
async def test_delayed_teardown_acknowledgement_preserves_a_reused_upstream_id(status: str) -> None:
    store: Final = MemoryStore(rows={ROW.id: ROW})
    replacement: Final = ROW.model_copy(
        update={
            "upstream_secret": "whsec_" + base64.b64encode(KEY[::-1]).decode(),
            "generation": "replacement",
            "status": status,
        }
    )

    class ReplacingUpstream(FakeUpstream):
        async def call(
            self, target: SubscriptionTarget, method: str, params: JsonObject, headers: Mapping[str, str]
        ) -> Mapping[str, object]:
            assert store.rows[ROW.id].status == "terminated"
            store.rows[ROW.id] = replacement
            return await super().call(target, method, params, headers)

    upstream: Final = ReplacingUpstream(responses={"events/unsubscribe": {}})
    target: Final = runtime((server(),), upstream=upstream)
    assert await terminate(ROW, forbidden("Key revoked"), store, SETTINGS, target, _Client(), lambda _: server())
    stale: Final = ROW.model_copy(update={"status": "terminated"})
    assert not await terminate(stale, forbidden("Key revoked"), store, SETTINGS, target, _Client(), lambda _: server())
    assert [(m, p["delivery"]) for m, p, _ in upstream.calls] == [
        ("events/unsubscribe", {"url": callback_url(SETTINGS, ROW.id, ROW.upstream_secret)})
    ]
    assert store.rows[ROW.id] == replacement


@pytest.mark.asyncio
async def test_ambiguous_unsubscribe_is_retried_until_the_peer_confirms_it_is_gone() -> None:
    store: Final = MemoryStore(rows={ROW.id: ROW})
    client: Final = _Client()
    retry: Final = MCPError(code=EventsErrorCode.NOT_FOUND, message="NotFound", data={"kind": "subscription"})
    outcomes: Final = (TimeoutError("reply lost"), retry, AssertionError("already acknowledged"))
    upstream: Final = FakeUpstream()
    for outcome in outcomes:
        upstream.responses = {"events/unsubscribe": outcome}
        assert (current := await store.get(ROW.id)) is not None
        assert await terminate(
            current,
            forbidden("Key revoked"),
            store,
            SETTINGS,
            runtime((server(),), upstream=upstream),
            client,
            lambda _: server(),
        )
    assert len(client.posts) == 1
    assert [params["delivery"] for _, params, _ in upstream.calls] == [
        {"url": callback_url(SETTINGS, ROW.id, ROW.upstream_secret)}
    ] * 2
    assert store.rows[ROW.id].upstream_subscription_id is None

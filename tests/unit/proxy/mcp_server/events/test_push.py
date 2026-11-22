from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Final

import anyio
import pytest
from mcp.shared.exceptions import MCPError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant, principal_of
from litellm.proxy.mcp_server.events.catalog import prepare_event
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest, open_cursor
from litellm.proxy.mcp_server.events.frames import Overflow, StreamFrame
from litellm.proxy.mcp_server.events.guardrails import ScreenOutcome
from litellm.proxy.mcp_server.events.observability import EventActivity
from litellm.proxy.mcp_server.events.push import PushRelay, granted_heartbeat, push
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.outbound_credentials.result import Ok
from litellm.types.mcp_server.events import JsonObject, StreamEventsParams

from .fakes import CALLER, SETTINGS, FakeUpstream, context, occurrence, runtime, screen_blocking, server

ARGS: Final[JsonObject] = {"repo": "acme/web"}
BINDING: Final = CursorBinding("srv-ci", "build.failed", arguments_digest(ARGS), principal_of(CALLER) or "")
SUBSCRIPTION: Final = "stream-7"


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-push-key")


def _active(cursor: str = "1", truncated: bool = False) -> StreamFrame:
    return StreamFrame("notifications/events/active", {"cursor": cursor, "truncated": truncated, "heartbeatMs": 30000})


def _event(event_id: str = "evt_1", cursor: str = "2") -> StreamFrame:
    return StreamFrame(
        "notifications/events/event", occurrence(event_id, cursor).model_dump(by_alias=True, mode="json")
    )


async def _authorized() -> OperationContext:
    return context()


async def _revoked() -> OperationContext:
    raise PermissionError("key deleted")


async def _run(
    frames: tuple[object, ...],
    authorize: Callable[[], Awaitable[OperationContext]] = _authorized,
    screen: Callable[..., Awaitable[ScreenOutcome]] | None = None,
    settings: EventsSettings = SETTINGS,
    records: list[EventActivity] | None = None,
) -> tuple[list[tuple[str, JsonObject]], FakeUpstream]:
    sent: Final[list[tuple[str, JsonObject]]] = []  # mutable-ok: collects downstream notifications

    async def send(method: str, params: JsonObject) -> None:
        sent.append((method, params))

    upstream: Final = FakeUpstream(frames=frames)
    await push(
        send,
        SUBSCRIPTION,
        context(),
        StreamEventsParams(name="ci-build.failed", arguments=ARGS, heartbeat_ms=1),
        None,
        settings,
        authorize,
        runtime((server(),), upstream=upstream, screen=screen, records=records),
    )
    return sent, upstream


def _opened(cursor: object) -> str | None:
    opened: Final = open_cursor(cursor if isinstance(cursor, str) else None, BINDING, 1_800_000_000)
    return opened.ok.upstream if isinstance(opened, Ok) else None


@pytest.mark.asyncio
async def test_the_stream_forwards_active_and_events_with_sealed_cursors_and_the_subscription_id() -> None:
    sent, upstream = await _run((_active("1"), _event("evt_1", "2")))
    assert [method for method, _ in sent] == ["notifications/events/active", "notifications/events/event"]
    assert all(params["_meta"] == {"io.modelcontextprotocol/subscriptionId": SUBSCRIPTION} for _, params in sent)
    active, event = sent[0][1], sent[1][1]
    assert (_opened(active["cursor"]), active["heartbeatMs"], active["truncated"]) == ("1", 5000, False)
    assert (event["name"], event["eventId"], _opened(event["cursor"])) == ("ci-build.failed", "evt_1", "2")
    assert (upstream.stream_calls[0]["name"], upstream.stream_calls[0]["heartbeatMs"]) == ("build.failed", 5000)


@pytest.mark.asyncio
async def test_revocation_before_an_event_terminates_with_forbidden_and_stops() -> None:
    sent, _ = await _run((_active(), _event("evt_1"), _event("evt_2")), authorize=_revoked)
    assert [method for method, _ in sent] == ["notifications/events/active", "notifications/events/terminated"]
    assert sent[1][1]["error"] == {"code": -32024, "message": "Forbidden", "data": {"reason": "Access revoked"}}


@pytest.mark.asyncio
async def test_an_overflow_becomes_a_truncated_active_with_the_latest_skipped_cursor() -> None:
    sent, _ = await _run((_active("1"), Overflow(cursor="40")))
    assert [(method, params["truncated"], _opened(params["cursor"])) for method, params in sent] == [
        ("notifications/events/active", False, "1"),
        ("notifications/events/active", True, "40"),
    ]


@pytest.mark.asyncio
async def test_an_upstream_terminated_is_forwarded_and_ends_the_stream() -> None:
    terminated: Final = StreamFrame(
        "notifications/events/terminated", {"error": {"code": -32023, "message": "NotFound"}}
    )
    sent, _ = await _run((_active(), terminated, _event("evt_late")))
    assert [method for method, _ in sent] == ["notifications/events/active", "notifications/events/terminated"]


@pytest.mark.asyncio
async def test_a_blocked_event_becomes_a_truncated_active() -> None:
    records: Final[list[EventActivity]] = []  # mutable-ok: collects recorded activity
    sent, _ = await _run((_active("1"), _event("evt_1", "2")), screen=screen_blocking(), records=records)
    assert [(method, params.get("truncated"), _opened(params.get("cursor"))) for method, params in sent] == [
        ("notifications/events/active", False, "1"),
        ("notifications/events/active", True, "2"),
    ]
    assert "blocked" in [item.outcome for item in records]
    assert "throttled" not in [item.outcome for item in records]


@pytest.mark.parametrize(("requested", "granted"), ((None, 30000), (1, 5000), (15000, 15000), (600000, 60000)))
def test_the_heartbeat_grant_is_clamped(requested: int | None, granted: int) -> None:
    assert granted_heartbeat(requested) == granted


async def _relay(
    heartbeat_ms: int, clock: Callable[[], float] = lambda: 1_800_000_000.0
) -> tuple[PushRelay, list[tuple[str, JsonObject]]]:
    sent: Final[list[tuple[str, JsonObject]]] = []  # mutable-ok: collects downstream notifications

    async def send(method: str, params: JsonObject) -> None:
        sent.append((method, params))

    prepared: Final = await prepare_event(context(), "ci-build.failed", runtime((server(),)))
    assert isinstance(prepared, Ok)
    relay: Final = PushRelay(
        send,
        SUBSCRIPTION,
        prepared.ok,
        BINDING,
        SETTINGS,
        _authorized,
        heartbeat_ms,
        False,
        runtime((server(),), clock=clock),
        CALLER,
    )
    return relay, sent


@pytest.mark.asyncio
async def test_the_gateway_sends_heartbeats_only_after_the_upstream_confirms() -> None:
    started: Final = anyio.current_time()
    relay, sent = await _relay(heartbeat_ms=1, clock=lambda: 1_800_000_000.0 + anyio.current_time() - started)
    send_stream, receive_stream = anyio.create_memory_object_stream[StreamFrame | Overflow](4)
    with anyio.move_on_after(0.02):  # idle ticks pass; with no upstream active none of them may heartbeat
        await relay.consume(receive_stream)
    assert sent == []
    send_stream.send_nowait(_active("3"))
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(relay.consume, receive_stream)
            while not any(method == "notifications/events/heartbeat" for method, _ in sent):
                await anyio.sleep(0)
            group.cancel_scope.cancel()
    methods: Final = [method for method, _ in sent]
    assert methods[0] == "notifications/events/active"
    assert "notifications/events/heartbeat" in methods[1:]
    heartbeat: Final = next(params for method, params in sent if method == "notifications/events/heartbeat")
    assert _opened(heartbeat["cursor"]) == "3"


@pytest.mark.asyncio
async def test_an_unreadable_upstream_event_becomes_a_truncated_active() -> None:
    unreadable: Final = StreamFrame("notifications/events/event", {"cursor": "3", "not": "an occurrence"})
    sent, _ = await _run((_active("1"), unreadable))
    assert [(method, params.get("truncated"), _opened(params.get("cursor"))) for method, params in sent] == [
        ("notifications/events/active", False, "1"),
        ("notifications/events/active", True, "3"),
    ]


async def _push_with(
    events_runtime: EventsRuntime, frames: tuple[object, ...] | None = None
) -> list[tuple[str, JsonObject]]:
    sent: Final[list[tuple[str, JsonObject]]] = []  # mutable-ok: collects downstream notifications

    async def send(method: str, params: JsonObject) -> None:
        sent.append((method, params))

    await push(
        send,
        SUBSCRIPTION,
        context(),
        StreamEventsParams(name="ci-build.failed", arguments=ARGS, heartbeat_ms=1),
        None,
        SETTINGS,
        _authorized,
        events_runtime if frames is None else replace(events_runtime, stream=FakeUpstream(frames=frames).stream),
    )
    return sent


@pytest.mark.asyncio
async def test_a_permission_lookup_that_fails_mid_stream_terminates_the_stream() -> None:
    calls: Final[list[str]] = []  # mutable-ok: counts grant lookups

    async def flaky_grant(_server_id: str, _caller: UserAPIKeyAuth | None) -> EventGrant:
        calls.append("grant")
        if len(calls) > 1:
            raise ConnectionError("permission store unreachable")
        return None

    sent: Final = await _push_with(
        replace(runtime((server(),)), grant=flaky_grant), (_active("1"), _event("evt_1", "2"))
    )
    assert [method for method, _ in sent][-1] == "notifications/events/terminated"


@pytest.mark.asyncio
async def test_a_stream_is_refused_before_any_upstream_work_when_cursors_cannot_be_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    upstream: Final = FakeUpstream(frames=(_active("1"),))
    with pytest.raises(MCPError):
        await _push_with(runtime((server(),), upstream=upstream))
    assert upstream.stream_calls == []


@pytest.mark.asyncio
async def test_an_open_stream_ends_when_an_operator_turns_events_off() -> None:
    sent: Final[list[tuple[str, JsonObject]]] = []  # mutable-ok: collects downstream notifications
    ticks: Final[list[float]] = [1_800_000_000.0]  # mutable-ok: a clock the test advances

    async def send(method: str, params: JsonObject) -> None:
        sent.append((method, params))

    prepared: Final = await prepare_event(context(), "ci-build.failed", runtime((server(),)))
    assert isinstance(prepared, Ok)
    relay: Final = PushRelay(
        send,
        SUBSCRIPTION,
        prepared.ok,
        BINDING,
        SETTINGS,
        _authorized,
        30_000,
        False,
        runtime((server(),), clock=lambda: ticks[0]),
        CALLER,
        enabled=lambda: False,
    )
    send_stream, receive_stream = anyio.create_memory_object_stream[StreamFrame | Overflow](4)
    send_stream.send_nowait(_active("1"))
    ticks[0] += 3600
    send_stream.send_nowait(_event("evt_1", "2"))
    send_stream.close()
    await relay.consume(receive_stream)
    methods: Final = [method for method, _ in sent]
    assert methods == ["notifications/events/terminated"]
    assert sent[-1][1]["error"]["data"] == {"feature": "push"}


@pytest.mark.asyncio
async def test_the_relay_seals_progress_and_identifies_the_client_subscription() -> None:
    relay, sent = await _relay(heartbeat_ms=5000)
    await relay.relay(_active("a"))
    await relay.relay(_event("evt_1", "b"))
    assert [(method, _opened(params["cursor"]), params["_meta"]) for method, params in sent] == [
        ("notifications/events/active", "a", {"io.modelcontextprotocol/subscriptionId": SUBSCRIPTION}),
        ("notifications/events/event", "b", {"io.modelcontextprotocol/subscriptionId": SUBSCRIPTION}),
    ]
    assert sent[1][1]["name"] == "ci-build.failed"

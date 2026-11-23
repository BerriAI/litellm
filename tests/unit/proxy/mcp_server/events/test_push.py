from typing import Final

import pytest

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import principal_of
from litellm.proxy.mcp_server.events.catalog import prepare_event
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest, open_cursor
from litellm.proxy.mcp_server.events.frames import StreamFrame
from litellm.proxy.mcp_server.events.push import PushRelay, granted_heartbeat
from litellm.proxy.mcp_server.outbound_credentials.result import Ok
from litellm.types.mcp_server.events import JsonObject

from .fakes import CALLER, SETTINGS, context, occurrence, runtime, server

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


def _opened(cursor: object) -> str | None:
    opened: Final = open_cursor(cursor if isinstance(cursor, str) else None, BINDING, 1_800_000_000)
    return opened.ok.upstream if isinstance(opened, Ok) else None


@pytest.mark.parametrize(("requested", "granted"), ((None, 30000), (1, 5000), (15000, 15000), (600000, 60000)))
def test_the_heartbeat_grant_is_clamped(requested: int | None, granted: int) -> None:
    assert granted_heartbeat(requested) == granted


async def _relay(heartbeat_ms: int) -> tuple[PushRelay, list[tuple[str, JsonObject]]]:
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
        runtime((server(),)),
        CALLER,
    )
    return relay, sent


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

import json
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.cursors import open_cursor
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.outgoing import DeliveryContext, Outgoing, binding_of, build_outgoing
from litellm.proxy.mcp_server.events.webhooks.store import QueuedDelivery, SubscriptionRow
from litellm.proxy.mcp_server.outbound_credentials.result import Ok
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

from ..fakes import occurrence, runtime, screen_blocking, server
from .test_worker import NOW, ROW

CALLER: Final = UserAPIKeyAuth(api_key="sk-caller", user_id="user-1")
NOW_SECONDS: Final = 1_800_000_000


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-outgoing-key")


def _context(body: str, target: MCPServer | None = None, per_minute: int = 100) -> DeliveryContext:
    delivery: Final = QueuedDelivery(
        delivery_id="d1", subscription_id="sub_1", body=body, webhook_id="msg_gw", next_attempt_at=NOW
    )
    return DeliveryContext(ROW, delivery, CALLER, target or server(), "ci-build.failed", per_minute)


def _cursor(outgoing: Outgoing | None, row: SubscriptionRow = ROW) -> str | None:
    assert outgoing is not None
    sealed: Final = json.loads(outgoing.body).get("cursor")
    opened: Final = open_cursor(sealed, binding_of(row), NOW_SECONDS)
    return opened.ok.upstream if isinstance(opened, Ok) else None


def _event_body(cursor: str = "7") -> str:
    upstream_meta: Final = {"io.example/trace": "upstream-only"}
    return occurrence("evt_1", cursor).model_copy(update={"meta": upstream_meta}).model_dump_json(by_alias=True)


@pytest.mark.parametrize(
    "body",
    ("not json", json.dumps({"type": "terminated"}), json.dumps({"eventId": "evt_1"})),
)
@pytest.mark.asyncio
async def test_an_unreadable_queued_body_builds_nothing(body: str) -> None:
    assert await build_outgoing(_context(body), runtime((server(),))) is None


@pytest.mark.asyncio
async def test_a_terminated_notice_is_terminal_with_a_fresh_webhook_id() -> None:
    body: Final = json.dumps({"type": "terminated", "error": {"code": -32024, "message": "Forbidden"}})
    built: Final = await build_outgoing(_context(body), runtime((server(),)))
    assert built is not None
    assert (built.terminal, built.webhook_id.startswith("msg_"), built.webhook_id != "msg_gw") == (True, True, True)
    assert json.loads(built.body)["error"]["message"] == "Forbidden"


@pytest.mark.asyncio
async def test_a_gap_carries_its_upstream_position_sealed_for_this_subscription() -> None:
    built: Final = await build_outgoing(_context(json.dumps({"type": "gap", "cursor": "42"})), runtime((server(),)))
    assert (json.loads(built.body)["type"] if built else None, _cursor(built)) == ("gap", "42")


@pytest.mark.asyncio
async def test_an_event_is_renamed_sealed_and_stripped_of_upstream_meta() -> None:
    built: Final = await build_outgoing(_context(_event_body("7")), runtime((server(),)))
    assert built is not None
    body: Final = json.loads(built.body)
    assert (body["name"], "_meta" in body, _cursor(built), built.webhook_id, built.event_id) == (
        "ci-build.failed",
        False,
        "7",
        "msg_gw",
        "evt_1",
    )


@pytest.mark.asyncio
async def test_a_legacy_server_reuses_the_event_id_as_the_webhook_id() -> None:
    legacy: Final = server(events=MCPServerEventsConfig(enabled=True, legacy_webhook_id_equals_event_id=True))
    built: Final = await build_outgoing(_context(_event_body(), target=legacy), runtime((legacy,)))
    assert built is not None and built.webhook_id == "evt_1"


@pytest.mark.asyncio
async def test_a_blocked_event_becomes_a_gap_at_its_position() -> None:
    built: Final = await build_outgoing(_context(_event_body("9")), runtime((server(),), screen=screen_blocking()))
    assert (json.loads(built.body)["type"] if built else None, _cursor(built)) == ("gap", "9")


@pytest.mark.asyncio
async def test_an_event_over_the_callers_budget_becomes_a_gap() -> None:
    events_runtime: Final[EventsRuntime] = runtime((server(),))
    first: Final = await build_outgoing(_context(_event_body("1"), per_minute=1), events_runtime)
    second: Final = await build_outgoing(_context(_event_body("2"), per_minute=1), events_runtime)
    assert (first.event_id if first else None, json.loads(second.body)["type"] if second else None) == ("evt_1", "gap")

from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest
from litellm.proxy.mcp_server.events.guardrails import Blocked
from litellm.proxy.mcp_server.events.observability import EventMetrics
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.signing import new_webhook_id
from litellm.proxy.mcp_server.events.webhooks.store import QueuedDelivery, SubscriptionRow
from litellm.types.mcp_server.events import EventOccurrence, GapEnvelope, TerminatedEnvelope
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class Outgoing:
    body: bytes
    webhook_id: str
    terminal: bool
    event_id: str | None
    cursor: str | None = None


def envelope_type(body: str) -> str | None:
    """The control envelope's ``type``; None for an event occurrence or an unreadable body."""
    try:
        kind: Final = _JSON_OBJECT.validate_json(body).get("type")
    except ValidationError:
        return None
    return kind if isinstance(kind, str) else None


def prepared_outgoing(delivery: QueuedDelivery) -> Outgoing | None:
    """The delivery as it was first built, so a retry resends the same bytes and webhook-id."""
    try:
        envelope: Final = _JSON_OBJECT.validate_json(delivery.body)
    except ValidationError:
        return None
    event_id: Final = envelope.get("eventId")
    cursor: Final = envelope.get("cursor")
    return Outgoing(
        delivery.body.encode(),
        delivery.webhook_id,
        envelope.get("type") == "terminated",
        event_id if isinstance(event_id, str) else None,
        cursor if isinstance(cursor, str) else None,
    )


@dataclass(frozen=True, slots=True)
class DeliveryContext:
    row: SubscriptionRow
    delivery: QueuedDelivery
    caller: UserAPIKeyAuth
    server: MCPServer | None
    gateway_name: str
    max_events_per_minute: int


def binding_of(row: SubscriptionRow) -> CursorBinding:
    return CursorBinding(row.server_id, row.event_name, arguments_digest(row.arguments), row.principal)


def gap(row: SubscriptionRow, upstream_cursor: JsonValue | None, now: int) -> Outgoing:
    return Outgoing(GapEnvelope(cursor=row.cursor).model_dump_json().encode(), new_webhook_id("gap"), False, None)


def _legacy_webhook_id(server: MCPServer | None) -> bool:
    return server is not None and server.events is not None and server.events.legacy_webhook_id_equals_event_id


async def _event(context: DeliveryContext, occurrence: EventOccurrence, runtime: EventsRuntime) -> Outgoing:
    row: Final = context.row
    if not runtime.budget.take(row.principal, context.max_events_per_minute):
        EventMetrics.count(row.server_id, context.gateway_name, "throttled")
        return gap(row, occurrence.cursor, runtime.now())
    server_name: Final = context.server.name if context.server is not None else row.server_id
    screened: Final = await runtime.screen(occurrence, server_name, context.gateway_name, row.id, context.caller)
    if isinstance(screened, Blocked):
        EventMetrics.count(row.server_id, context.gateway_name, "blocked")
        return gap(row, occurrence.cursor, runtime.now())
    body: Final = occurrence.model_copy(
        update={
            "name": context.gateway_name,
            "data": screened.data,
            "cursor": row.cursor,
            "meta": None,
        }
    ).model_dump_json(by_alias=True, exclude_none=True)
    webhook_id: Final = occurrence.event_id if _legacy_webhook_id(context.server) else context.delivery.webhook_id
    return Outgoing(body.encode(), webhook_id, False, occurrence.event_id, row.cursor)


async def build_outgoing(context: DeliveryContext, runtime: EventsRuntime) -> Outgoing | None:
    try:
        envelope: Final = _JSON_OBJECT.validate_json(context.delivery.body)
    except ValidationError:
        return None
    kind: Final = envelope.get("type")
    if kind == "terminated":
        try:
            terminated: Final = TerminatedEnvelope.model_validate(envelope)
        except ValidationError:
            return None
        return Outgoing(
            terminated.model_dump_json(exclude_none=True).encode(), new_webhook_id("terminated"), True, None
        )
    if kind == "gap":
        return gap(context.row, envelope.get("cursor"), runtime.now())
    try:
        occurrence: Final = EventOccurrence.model_validate(envelope)
    except ValidationError:
        return None
    return await _event(context, occurrence, runtime)

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from mcp.shared.inbound import MCP_NAME_HEADER, decode_header_value, validate_mcp_param_headers

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import grant_allows, principal_of
from litellm.proxy.mcp_server.events.catalog import PreparedEvent, event_servers, offered_modes, prepare_event
from litellm.proxy.mcp_server.events.config import EventsSettings, server_allows_event
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest
from litellm.proxy.mcp_server.events.errors import EventsFailure, exhausted, header_mismatch, unsupported
from litellm.proxy.mcp_server.events.guardrails import Blocked
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Result
from litellm.types.mcp_server.events import DeliveryMode, EventOccurrence, JsonObject


@dataclass(frozen=True, slots=True)
class Admitted:
    events: tuple[EventOccurrence, ...]
    throttled: int
    blocked: int

    @property
    def dropped(self) -> int:
        return self.throttled + self.blocked


def caller_key(context: OperationContext) -> str:
    return principal_of(context.user_api_key_auth) or "anonymous"


def check_name_header(headers: Mapping[str, str] | None, name: str) -> EventsFailure | None:
    if headers is None or decode_header_value(headers.get(MCP_NAME_HEADER)) == name:
        return None
    return header_mismatch("Mcp-Name")


def check_routing_headers(
    headers: Mapping[str, str] | None, name: str, arguments: JsonObject, prepared: PreparedEvent
) -> EventsFailure | None:
    if headers is None:
        return None
    named: Final = check_name_header(headers, name)
    if named is not None:
        return named
    rejection: Final = validate_mcp_param_headers(dict(prepared.descriptor.input_schema), arguments, headers)
    return header_mismatch(_header_named_in(rejection.message)) if rejection is not None else None


def _header_named_in(message: str) -> str:
    """The SDK names the offending header first, e.g. ``Mcp-Param-Repo header is missing ...``."""
    return message.split(" header", 1)[0]


async def admit(
    context: OperationContext,
    name: str,
    arguments: JsonObject,
    mode: DeliveryMode,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    runtime: EventsRuntime,
) -> Result[PreparedEvent, EventsFailure]:
    misnamed: Final = check_name_header(headers, name)
    if misnamed is not None:
        return Error(misnamed)
    prepared: Final = await prepare_event(context, name, runtime)
    if isinstance(prepared, Error):
        return prepared
    mismatch: Final = check_routing_headers(headers, name, arguments, prepared.ok)
    if mismatch is not None:
        return Error(mismatch)
    server: Final = prepared.ok.resolved.target.server
    if mode not in offered_modes(prepared.ok.descriptor, server, settings, context):
        return Error(unsupported("deliveryMode", mode))
    if await runtime.rate_limited(server, context.user_api_key_auth):
        return Error(exhausted("rate", server.rpm or 0))
    return prepared


def binding_for(context: OperationContext, prepared: PreparedEvent, arguments: JsonObject) -> CursorBinding:
    return CursorBinding(
        server_id=prepared.resolved.target.server.server_id,
        event=prepared.resolved.upstream_name,
        args_hash=arguments_digest(arguments),
        caller=caller_key(context),
    )


async def still_authorized(context: OperationContext, prepared: PreparedEvent, runtime: EventsRuntime) -> bool:
    server_id: Final = prepared.resolved.target.server.server_id
    current: Final = next(
        (item for item in event_servers(await runtime.allowed_servers(context)) if item.server.server_id == server_id),
        None,
    )
    upstream_name: Final = prepared.resolved.upstream_name
    if current is None or not server_allows_event(current.config, upstream_name):
        return False
    return grant_allows(await runtime.grant(server_id, context.user_api_key_auth), upstream_name)


async def screen_all(
    occurrences: tuple[EventOccurrence, ...],
    context: OperationContext,
    prepared: PreparedEvent,
    settings: EventsSettings,
    subscription_id: str | None,
    runtime: EventsRuntime,
) -> Admitted:
    async def one(occurrence: EventOccurrence) -> EventOccurrence | Literal["throttled", "blocked"]:
        if not runtime.budget.take(caller_key(context), settings.max_events_per_minute):
            return "throttled"
        outcome: Final = await runtime.screen(
            occurrence,
            prepared.resolved.target.server.name,
            prepared.resolved.gateway_name,
            subscription_id,
            context.user_api_key_auth,
        )
        if isinstance(outcome, Blocked):
            return "blocked"
        return occurrence.model_copy(
            update={"name": prepared.resolved.gateway_name, "data": outcome.data, "cursor": None, "meta": None}
        )

    screened: Final = tuple([await one(occurrence) for occurrence in occurrences])
    return Admitted(
        events=tuple(item for item in screened if isinstance(item, EventOccurrence)),
        throttled=screened.count("throttled"),
        blocked=screened.count("blocked"),
    )

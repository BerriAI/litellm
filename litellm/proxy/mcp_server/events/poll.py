from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.admission import Admitted, admit, binding_for, screen_all
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import ensure_sealable, open_cursor, seal_cursor
from litellm.proxy.mcp_server.events.errors import EventsFailure, internal_error
from litellm.proxy.mcp_server.events.observability import EventActivity, EventOutcome
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.upstream import routing_headers
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import PollEventsParams, PollEventsResult

_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def _outcome(admitted: Admitted) -> EventOutcome:
    if admitted.blocked:
        return "blocked"
    return "throttled" if admitted.throttled else "delivered"


async def poll(
    context: OperationContext,
    params: PollEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    runtime: EventsRuntime,
) -> Result[PollEventsResult, EventsFailure]:
    prepared: Final = await admit(context, params.name, params.arguments, "poll", headers, settings, runtime)
    if isinstance(prepared, Error):
        return Error(prepared.error)
    resolved: Final = prepared.ok.resolved
    binding: Final = binding_for(context, prepared.ok, params.arguments)
    unsealable: Final = ensure_sealable(binding, runtime.now())
    if unsealable is not None:
        return Error(unsealable)
    opened: Final = open_cursor(params.cursor, binding, runtime.now())
    if isinstance(opened, Error):
        return Error(opened.error)
    upstream_params: Final = PollEventsParams(
        name=resolved.upstream_name,
        arguments=params.arguments,
        cursor=opened.ok.upstream,
        max_age_ms=params.max_age_ms,
        max_events=params.max_events,
    )
    raw: Final = await runtime.call(
        prepared.ok.upstream,
        "events/poll",
        _JSON_OBJECT.validate_python(upstream_params.model_dump(by_alias=True, mode="json", exclude_none=True)),
        routing_headers(resolved.upstream_name, params.arguments, prepared.ok.descriptor),
    )
    try:
        upstream: Final = PollEventsResult.model_validate(raw)
    except ValidationError:
        return Error(internal_error("Upstream returned an invalid events/poll result"))
    admitted: Final = await screen_all(upstream.events, context, prepared.ok, settings, None, runtime)
    sealed: Final = seal_cursor(binding, upstream.cursor, runtime.now())
    if isinstance(sealed, Error):
        return Error(sealed.error)
    if admitted.events or admitted.dropped:
        await runtime.record(
            EventActivity(
                call_type="mcp_event_poll",
                mode="poll",
                server_id=resolved.target.server.server_id,
                event_name=resolved.gateway_name,
                outcome=_outcome(admitted),
                caller=context.user_api_key_auth,
                event_ids=tuple(item.event_id for item in admitted.events),
            )
        )
    return Ok(
        PollEventsResult(
            events=admitted.events,
            cursor=sealed.ok,
            truncated=upstream.truncated or opened.ok.expired or admitted.dropped > 0,
            has_more=upstream.has_more,
            next_poll_ms=max(upstream.next_poll_ms or 0, resolved.target.config.min_poll_interval_ms),
        )
    )

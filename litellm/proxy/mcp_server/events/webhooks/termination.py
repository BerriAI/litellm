import time
from collections.abc import Callable
from typing import Final

from mcp.shared.exceptions import MCPError

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.errors import EventsFailure
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.upstream import routing_headers
from litellm.proxy.mcp_server.events.webhooks.delivery import post_callback
from litellm.proxy.mcp_server.events.webhooks.policy import callback_url, credential_outlives_request
from litellm.proxy.mcp_server.events.webhooks.signing import new_webhook_id, parse_secret, signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import EventStore, SubscriptionRow
from litellm.proxy.mcp_server.events.webhooks.verification import Poster
from litellm.types.mcp_server.events import EventsError, TerminatedEnvelope
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def registered_server(server_id: str) -> MCPServer | None:
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    return global_mcp_server_manager.get_mcp_server_by_id(server_id)


async def end_upstream(
    row: SubscriptionRow,
    settings: EventsSettings,
    runtime: EventsRuntime,
    lookup: Callable[[str], MCPServer | None] = registered_server,
) -> None:
    server: Final = lookup(row.server_id)
    if server is None or row.upstream_subscription_id is None or not credential_outlives_request(server):
        return
    try:
        target: Final = await runtime.prepare_target(OperationContext(_caller=None), server, (server,))
        descriptor: Final = next(
            (item for item in await runtime.descriptors(target) if item.name == row.event_name), None
        )
        if descriptor is None:
            return
        await runtime.call(
            target,
            "events/unsubscribe",
            {
                "name": row.event_name,
                "arguments": dict(row.arguments),
                "delivery": {"url": callback_url(settings, row.id, row.upstream_secret)},
            },
            routing_headers(row.event_name, row.arguments, descriptor),
        )
    except (MCPError, OSError, TimeoutError):
        return


def terminated_body(failure: EventsFailure) -> bytes:
    error: Final = EventsError(code=int(failure.code), message=failure.message, data=failure.data)
    return TerminatedEnvelope(error=error).model_dump_json(exclude_none=True).encode()


async def terminate(
    row: SubscriptionRow,
    failure: EventsFailure,
    store: EventStore,
    settings: EventsSettings,
    runtime: EventsRuntime,
    post: Poster = post_callback,
    lookup: Callable[[str], MCPServer | None] = registered_server,
    *,
    notify: bool = True,
) -> None:
    if row.status == "terminated" or not await store.end_if_current(
        row.id, row.upstream_subscription_id, row.generation
    ):
        return
    secret: Final = parse_secret(row.client_secret)
    if notify and secret is not None and row.status != "pending":
        body: Final = terminated_body(failure)
        await post(
            row.client_url, body, signed_headers(secret, new_webhook_id("terminated"), int(time.time()), body, row.id)
        )
    await end_upstream(row, settings, runtime, lookup)

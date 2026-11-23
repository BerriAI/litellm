from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import principal_of
from litellm.proxy.mcp_server.events.catalog import event_servers, offered_modes, publishable
from litellm.proxy.mcp_server.events.config import events_settings
from litellm.proxy.mcp_server.events.runtime import DEFAULT_RUNTIME, EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.delivery import post_callback
from litellm.proxy.mcp_server.events.webhooks.store import (
    EventStore,
    SubscriptionQuery,
    SubscriptionRow,
    SubscriptionStatus,
    default_store,
)
from litellm.proxy.mcp_server.events.webhooks.termination import registered_server
from litellm.proxy.mcp_server.events.webhooks.verification import Poster
from litellm.types.mcp_server.events import CallbackFailure, EventType, JsonObject
from litellm.types.mcp_server.mcp_server_manager import MCPServer

router: Final = APIRouter(prefix="/events", tags=["mcp"])
_ADMIN_ROLES: Final = frozenset({LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY})


class SubscriptionView(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    server_id: str
    event_name: str
    arguments: JsonObject
    callback_host: str
    status: SubscriptionStatus
    user_id: str | None
    team_id: str | None
    refresh_before: datetime | None
    last_delivery_at: datetime | None
    last_error: CallbackFailure | None
    failed_since: datetime | None
    created_at: datetime | None


class SubscriptionList(BaseModel):
    model_config = ConfigDict(frozen=True)

    subscriptions: tuple[SubscriptionView, ...]
    next_offset: int | None = None


def _view(row: SubscriptionRow) -> SubscriptionView:
    from urllib.parse import urlparse

    return SubscriptionView(
        id=row.id,
        server_id=row.server_id,
        event_name=row.event_name,
        arguments=row.arguments,
        callback_host=urlparse(row.client_url).hostname or "",
        status=row.status,
        user_id=row.user_id,
        team_id=row.team_id,
        refresh_before=row.refresh_before,
        last_delivery_at=row.last_delivery_at,
        last_error=row.last_error,
        failed_since=row.failed_since,
        created_at=row.created_at,
    )


@dataclass(frozen=True, slots=True)
class AdminServices:
    store: EventStore | None
    runtime: EventsRuntime
    post: Poster
    lookup: Callable[[str], MCPServer | None]
    key_for_alias: Callable[[str], Awaitable[str | None]]

    def require_store(self) -> EventStore:
        if self.store is None:
            raise HTTPException(status_code=503, detail="MCP event subscriptions need a connected database")
        return self.store


def admin_services() -> AdminServices:
    return AdminServices(
        store=default_store(),
        runtime=DEFAULT_RUNTIME,
        post=post_callback,
        lookup=registered_server,
        key_for_alias=key_for_alias,
    )


Services: TypeAlias = Annotated[AdminServices, Depends(admin_services)]


def _is_admin(caller: UserAPIKeyAuth) -> bool:
    return caller.user_role in _ADMIN_ROLES


@router.get("/types", response_model=tuple[EventType, ...])
async def list_event_types(
    request: Request,
    server_id: Annotated[str, Query()],
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    services: Services,
) -> tuple[EventType, ...]:
    """List the event types one events-enabled MCP server publishes to this caller, under their upstream names."""
    from litellm.proxy.auth.ip_address_utils import IPAddressUtils

    runtime: Final = services.runtime
    context: Final = OperationContext(
        _caller=caller, client_ip=IPAddressUtils.get_mcp_client_ip(request), raw_headers=dict(request.headers)
    )
    allowed: Final = await runtime.allowed_servers(context)
    server: Final = next((item for item in event_servers(allowed) if item.server.server_id == server_id), None)
    if server is None:
        raise HTTPException(status_code=404, detail="No events-enabled MCP server with this id")
    descriptors: Final = await runtime.descriptors(await runtime.prepare_target(context, server.server, allowed))
    grant: Final = await runtime.grant(server.server.server_id, caller)
    webhooks_enabled: Final = services.store is not None and events_settings().public_base_url is not None
    return tuple(
        item.model_copy(update={"delivery": modes})
        for item in descriptors
        if publishable(item, server.config, grant)
        and (modes := offered_modes(item, server.server, webhooks_enabled, context))
    )


@router.get("/subscriptions", response_model=SubscriptionList)
async def list_subscriptions(
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    services: Services,
    server_id: str | None = None,
    status: SubscriptionStatus | None = None,
    key_alias: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SubscriptionList:
    store: Final = services.require_store()
    by_alias: Final = key_alias is not None and _is_admin(caller)
    key_hash: Final = await services.key_for_alias(key_alias) if by_alias and key_alias is not None else None
    if by_alias and key_hash is None:
        return SubscriptionList(subscriptions=())
    query: Final = SubscriptionQuery(
        principal=None if _is_admin(caller) else principal_of(caller) or "-",
        key_hash=key_hash,
        server_id=server_id,
        status=status,
        limit=limit + 1,
        offset=offset,
    )
    rows: Final = await store.find(query)
    return SubscriptionList(
        subscriptions=tuple(_view(row) for row in rows[:limit]),
        next_offset=offset + limit if len(rows) > limit else None,
    )


async def key_for_alias(alias: str) -> str | None:
    """The stored key hash for an alias; 400 when the alias names more than one key, as the key lookup does."""
    from litellm.proxy.proxy_server import prisma_client
    from litellm.repositories.verification_token_repository import VerificationTokenRepository

    if prisma_client is None:
        return None
    records: Final = await VerificationTokenRepository(prisma_client).table.find_many(
        where={"key_alias": alias}, take=2
    )
    if len(records) > 1:
        raise HTTPException(status_code=400, detail="More than one key has this alias; filter by key instead")
    return records[0].token if records else None

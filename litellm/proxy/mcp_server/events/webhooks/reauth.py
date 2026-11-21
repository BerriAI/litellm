from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, TypeAlias

from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import grant_allows
from litellm.proxy.mcp_server.events.catalog import event_servers
from litellm.proxy.mcp_server.events.config import server_allows_event
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow


@dataclass(frozen=True, slots=True)
class Authorized:
    caller: UserAPIKeyAuth


@dataclass(frozen=True, slots=True)
class Revoked:
    reason: str


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: str


Reauthorization: TypeAlias = Authorized | Revoked | Unavailable
CallerLoader: TypeAlias = Callable[[SubscriptionRow], Awaitable[Reauthorization]]
UserReloader: TypeAlias = Callable[[str], Awaitable[UserAPIKeyAuth]]


def _key_revoked(key: UserAPIKeyAuth) -> bool:
    expires: Final = key.expires
    expired: Final = isinstance(expires, datetime) and expires.astimezone(timezone.utc) <= datetime.now(timezone.utc)
    return bool(key.blocked) or expired


async def _reload_live_user(user_id: str) -> UserAPIKeyAuth:
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

    return await MCPRequestHandler.reload_admitted_user(user_id, requires_fresh_policy=True)


async def admitted_user(user_id: str, reload: UserReloader = _reload_live_user) -> Reauthorization:
    try:
        return Authorized(await reload(user_id))
    except HTTPException as error:
        return Revoked("user") if error.status_code == 401 else Unavailable(str(error.detail))
    except Exception as error:  # noqa: BLE001  # a transient lookup failure delays delivery instead of ending the subscription
        return Unavailable(str(error))


async def stored_caller(row: SubscriptionRow) -> Reauthorization:
    from litellm.proxy.auth.auth_checks import get_key_object
    from litellm.proxy.proxy_server import master_key, prisma_client, proxy_logging_obj, user_api_key_cache
    from litellm.proxy.utils import hash_token

    if row.key_hash is None:
        return await admitted_user(row.user_id) if row.user_id else Revoked("caller")
    if isinstance(master_key, str) and row.key_hash == hash_token(master_key):
        return Authorized(UserAPIKeyAuth(api_key=row.key_hash, user_role=LitellmUserRoles.PROXY_ADMIN))
    try:
        key: Final = await get_key_object(
            hashed_token=row.key_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=True,
        )
    except ProxyException as error:
        if error.type == ProxyErrorTypes.token_not_found_in_db:
            return Revoked("key")
        return Unavailable(error.message)
    except Exception as error:  # noqa: BLE001  # a transient lookup failure delays delivery instead of ending the subscription
        return Unavailable(str(error))
    return Revoked("key") if _key_revoked(key) else Authorized(key)


async def authorize_row(
    row: SubscriptionRow, runtime: EventsRuntime, load: CallerLoader = stored_caller
) -> Reauthorization:
    loaded: Final = await load(row)
    if not isinstance(loaded, Authorized):
        return loaded
    try:
        allowed: Final = await runtime.allowed_servers(OperationContext(_caller=loaded.caller))
        grant: Final = await runtime.grant(row.server_id, loaded.caller)
    except Exception as error:  # noqa: BLE001  # a transient lookup failure delays delivery instead of ending the subscription
        return Unavailable(str(error))
    server: Final = next((item for item in event_servers(allowed) if item.server.server_id == row.server_id), None)
    if server is None or not server_allows_event(server.config, row.event_name):
        return Revoked("server")
    return loaded if grant_allows(grant, row.event_name) else Revoked("event")

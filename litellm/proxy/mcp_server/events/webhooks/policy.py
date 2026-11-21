import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Final

from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.events import JsonObject, SubscribeEventsParams
from litellm.types.mcp_server.mcp_server_manager import MCPServer

DEFAULT_TTL_MS: Final = 60 * 60 * 1000
MIN_TTL_MS: Final = 60 * 1000
MAX_TTL_MS: Final = 24 * 60 * 60 * 1000
REQUEST_SCOPED_TTL_MS: Final = 60 * 60 * 1000
_STORED_CREDENTIALS: Final = frozenset(
    {
        MCPAuth.none,
        MCPAuth.api_key,
        MCPAuth.bearer_token,
        MCPAuth.basic,
        MCPAuth.authorization,
        MCPAuth.token,
        MCPAuth.aws_sigv4,
    }
)


def subscription_id(principal: str, url: str, name: str, arguments: JsonObject) -> str:
    material: Final = json.dumps([principal, url, name, arguments], sort_keys=True, separators=(",", ":"))
    return "sub_" + hashlib.sha256(material.encode()).hexdigest()[:32]


def callback_url(settings: EventsSettings, subscription: str) -> str:
    return f"{(settings.public_base_url or '').rstrip('/')}/mcp/events/webhooks/{subscription}"


def _forwards_client_credentials(server: MCPServer) -> bool:
    return bool(server.extra_headers) or server.oauth_passthrough or server.delegate_auth_to_upstream or server.is_byok


def credential_outlives_request(server: MCPServer) -> bool:
    if _forwards_client_credentials(server):
        return False
    if server.auth_type is None or server.auth_type in _STORED_CREDENTIALS:
        return True
    return server.auth_type == MCPAuth.oauth2 and server.has_client_credentials


def granted_ttl_ms(params: SubscribeEventsParams, server: MCPServer) -> int | None:
    if not credential_outlives_request(server):
        return min(params.ttl_ms or REQUEST_SCOPED_TTL_MS, REQUEST_SCOPED_TTL_MS)
    if "ttl_ms" in params.model_fields_set and params.ttl_ms is None:
        return None
    return min(max(params.ttl_ms or DEFAULT_TTL_MS, MIN_TTL_MS), MAX_TTL_MS)


def client_grant(now: datetime, ttl_ms: int | None, upstream: datetime | None) -> datetime | None:
    local: Final = now + timedelta(milliseconds=ttl_ms) if ttl_ms is not None else None
    candidates: Final = tuple(item for item in (local, upstream) if item is not None)
    return min(candidates) if candidates else None


def iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value is not None else None


def parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed: Final = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

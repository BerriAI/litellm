from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Final

from litellm.experimental_mcp_client.client import MCPClient, strip_auth_scheme
from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import raise_public
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import CredError
from litellm.proxy._experimental.mcp_server.utils import merge_openapi_headers
from litellm.types.mcp import MCPAuth, MCPAuthType, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_STATIC_MODES: Final = frozenset(
    (MCPAuth.api_key, MCPAuth.bearer_token, MCPAuth.basic, MCPAuth.token, MCPAuth.authorization)
)


def _usable_credential_value(auth_type: MCPAuthType, name: str, value: str) -> bool:
    if not value:
        return False
    if auth_type == MCPAuth.authorization or (auth_type == MCPAuth.api_key and name != "authorization"):
        return True
    if value.lower() in ("bearer", "basic", "token", "apikey"):
        return False
    if auth_type in (MCPAuth.bearer_token, MCPAuth.token):
        scheme: Final = "Bearer" if auth_type == MCPAuth.bearer_token else "token"
        credential: Final = strip_auth_scheme(value, scheme).strip()
        return bool(credential) and credential.lower() != scheme.lower()
    if auth_type == MCPAuth.basic:
        parts: Final = value.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "basic":
            return False
        try:
            decoded: Final = base64.b64decode(parts[1], validate=True).strip()
            return b":" in decoded
        except ValueError:
            return False
    return True


def validate_static_credential(server: MCPServer, headers: Mapping[str, str]) -> Result[None, CredError]:
    if server.auth_type not in _STATIC_MODES or server.transport == MCPTransport.stdio:
        return Ok(None)
    default_slot: Final = "X-API-Key" if server.auth_type == MCPAuth.api_key else "Authorization"
    slots: Final = frozenset(
        name.lower()
        for name in (
            server.upstream_token_header or default_slot,
            default_slot,
            "Authorization",
        )
    )
    values: Final = tuple((name.lower(), value.strip()) for name, value in headers.items() if name.lower() in slots)
    if any(_usable_credential_value(server.auth_type, name, value) for name, value in values):
        return Ok(None)
    return Error(CredError.of_misconfigured(f"{server.auth_type} requires a usable upstream credential"))


async def prepare_mcp_client(server: MCPServer, client: MCPClient) -> MCPClient:
    if server.auth_type not in _STATIC_MODES or client.transport_type == MCPTransport.stdio:
        return client
    request: Final = await client.prepare_request_auth()
    match validate_static_credential(server, request.headers):
        case Error(error):
            raise_public(error)
        case Ok():
            return client


def validate_openapi_credentials(
    server: MCPServer,
    resolved_headers: Mapping[str, str] | None,
    forwarded_headers: Mapping[str, str] | None,
    caller_authorization: str | None,
) -> None:
    headers: Final = merge_openapi_headers(
        server.static_headers or {}, forwarded_headers, caller_authorization, resolved_headers
    )
    match validate_static_credential(server, headers):
        case Error(error):
            raise_public(error)
        case Ok():
            return

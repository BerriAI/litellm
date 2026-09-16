"""Gateway-side sign-in for MCP connects.

A guardrail that evaluates tool calls in the caller's own identity (an On-Behalf-Of exchange of the caller's
bearer) needs the caller signed in with its identity provider before the first tool call, and a tool call's
JSON-RPC error cannot carry ``WWW-Authenticate``. Such guardrails implement :class:`GatewaySignInProvider`;
the MCP transport asks the registered providers whether a connect must answer with the RFC 9728 challenge and
which issuers and scopes the protected-resource metadata advertises. The MCP package never imports a concrete
provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


@runtime_checkable
class GatewaySignInProvider(Protocol):
    def gateway_authorization_servers(
        self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None
    ) -> tuple[str, ...]:
        """Issuers the caller signs in with before calling ``server``; empty when this provider does not gate
        ``server`` for the caller (``None`` is the anonymous metadata fetch that follows a challenge)."""
        ...

    def gateway_scopes_supported(self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> tuple[str, ...]:
        """Scopes the caller requests from those issuers when the admin set none on ``server``."""
        ...

    async def gateway_sign_in_required(
        self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None, oauth2_headers: Mapping[str, str] | None
    ) -> bool:
        """Whether the connect carries no assertion this provider can use for ``server``, or one its issuer
        refuses, so the caller must sign in (again)."""
        ...


def _providers() -> tuple[GatewaySignInProvider, ...]:
    return tuple(
        callback
        for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(CustomGuardrail)
        if isinstance(callback, GatewaySignInProvider)
    )


def gateway_authorization_servers(server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            issuer
            for provider in _providers()
            for issuer in provider.gateway_authorization_servers(server, user_api_key_auth)
        )
    )


def gateway_scopes_supported(server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> tuple[str, ...]:
    if server.scopes:
        return tuple(server.scopes)
    return tuple(
        dict.fromkeys(
            scope for provider in _providers() for scope in provider.gateway_scopes_supported(server, user_api_key_auth)
        )
    )


async def gateway_sign_in_required(
    server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None, oauth2_headers: Mapping[str, str] | None
) -> bool:
    for provider in _providers():
        if await provider.gateway_sign_in_required(server, user_api_key_auth, oauth2_headers):
            return True
    return False

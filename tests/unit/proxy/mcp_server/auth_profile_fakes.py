from typing import Final

import httpx2
from mcp.types import JSONRPCMessage, JSONRPCRequest
from pydantic import TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.auth_profile_selection import AuthProfileGrantSource, AuthProfileOwnerSource
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.mcp_server.outbound_credentials.client_credentials import (
    ClientCredentialsTokenSource,
    TokenEndpointOutcome,
    TokenEndpointSuccess,
)
from litellm.proxy.mcp_server.outbound_credentials.resolver import UpstreamCredentialProvider
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, MCPServer

_JSONRPC: Final = TypeAdapter(JSONRPCMessage)

BOT_PROFILE: Final = MCPAuthProfile(
    id="ana-bots",
    auth_type=MCPAuth.oauth2,
    oauth2_flow="client_credentials",
    client_id="bot-app",
    client_secret="bot-secret",
    token_url="https://idp.example/bot-token",
)


def profiled_analytics_server(**overrides: object) -> MCPServer:
    return MCPServer.model_validate(
        {
            "server_id": "analytics",
            "name": "analytics",
            "url": "https://mcp.analytics.example/v1/mcp",
            "transport": MCPTransport.http,
            "auth_type": MCPAuth.oauth2,
            "oauth2_flow": "authorization_code",
            "client_id": "person-app",
            "authorization_url": "https://idp.example/authorize",
            "token_url": "https://idp.example/person-token",
            "additional_auth": (BOT_PROFILE,),
            **overrides,
        }
    )


class AnalyticsUpstream:
    def __init__(self, rejected_tokens: frozenset[str] = frozenset()) -> None:
        self.rejected_tokens = rejected_tokens
        self.requests: tuple[tuple[str, str], ...] = ()

    async def respond(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        authorization: Final = request.headers.get("authorization", "")
        if authorization in self.rejected_tokens:
            self.requests = (*self.requests, ("rejected", authorization))
            return httpx2.Response(401, headers={"WWW-Authenticate": "Bearer"})
        payload: Final = _JSONRPC.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        self.requests = (*self.requests, (payload.method, authorization))
        result: Final[dict[str, object]] = {
            "initialize": {
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "analytics", "version": "1"},
                "capabilities": {"tools": {}, "resources": {}},
            },
            "tools/list": {"tools": [{"name": "get_issue", "inputSchema": {"type": "object"}}]},
            "tools/call": {"content": [{"type": "text", "text": f"issue read with {authorization}"}], "isError": False},
            "resources/list": {"resources": []},
        }[payload.method]
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload.id, "result": result})

    def calls(self, method: str) -> tuple[str, ...]:
        return tuple(authorization for seen, authorization in self.requests if seen == method)


class BotTokenEndpoint:
    def __init__(self, outcome: TokenEndpointOutcome | None = None) -> None:
        self.outcome = outcome
        self.client_ids: tuple[str, ...] = ()

    async def __call__(self, url: str, form: dict[str, str], headers: dict[str, str]) -> TokenEndpointOutcome:
        self.client_ids = (*self.client_ids, form["client_id"])
        minted: Final = f"{form['client_id']}-token-{len(self.client_ids)}"
        return self.outcome or TokenEndpointSuccess(body={"access_token": minted, "expires_in": 300})


def granting(*grants: str) -> AuthProfileGrantSource:
    async def caller_grants(user_api_key_auth: UserAPIKeyAuth | None) -> tuple[tuple[str, ...], ...]:
        return (grants,) if grants else ()

    return caller_grants


def profile_aware_manager(
    server: MCPServer,
    upstream: AnalyticsUpstream,
    token_endpoint: BotTokenEndpoint,
    *grants: str,
    owners: AuthProfileOwnerSource | None = None,
) -> MCPServerManager:
    manager: Final = MCPServerManager(
        cred_provider=UpstreamCredentialProvider(
            client_credentials_source=ClientCredentialsTokenSource(token_endpoint)
        ),
        upstream_transport=httpx2.MockTransport(upstream.respond),
        auth_profile_grants=granting(*grants),
        auth_profile_owners=owners,
    )
    manager.registry[server.server_id] = server
    manager.tool_name_to_mcp_server_name_mapping[f"{server.server_id}-get_issue"] = server.server_id
    return manager

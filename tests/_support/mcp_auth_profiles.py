import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final

import httpx2
import pytest
from mcp.types import CallToolResult, InputRequiredResult

from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server import mcp_server_manager as mcp_server_manager_module
from litellm.proxy.mcp_server import operations
from litellm.proxy.mcp_server.auth_profiles import parse_auth_profiles
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.mcp_server.outbound_credentials.client_credentials import (
    ClientCredentialsTokenSource,
    TokenEndpointOutcome,
    TokenEndpointSuccess,
)
from litellm.proxy.mcp_server.outbound_credentials.resolver import UpstreamCredentialProvider
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from tests.unit.proxy.rust_auth_stores import RustAuthStores, RustStoreRows, install_rust_auth_stores

DEFAULT_CREDENTIAL: Final = "default-credential"
MINTED_PROFILE_TOKEN: Final = "ops-minted-token"
TICKETS_PROFILES: Final = (
    {"id": "bots", "auth_type": "bearer_token", "auth_value": "bots-token"},
    {
        "id": "ops",
        "auth_type": "oauth2",
        "oauth2_flow": "client_credentials",
        "client_id": "ops-client",
        "client_secret": "ops-secret",
        "token_url": "https://issuer.example/token",
    },
    {"id": "empty", "auth_type": "bearer_token"},
)


def credential_label(authorization: str | None) -> str:
    return "anonymous" if authorization is None else authorization.removeprefix("Bearer ").replace("-", "_")


class ProfileUpstream:
    def __init__(self, unauthorized_calls: int = 0) -> None:
        self.requests: Final[list[tuple[str, str | None]]] = []  # mutable-ok: the fake records what it was sent
        self._unauthorized_calls = unauthorized_calls

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(200)
        payload: Final = json.loads(request.content)
        authorization: Final = request.headers.get("authorization")
        self.requests.append((payload["method"], authorization))
        if "id" not in payload:
            return httpx2.Response(202)
        if payload["method"] == "tools/call" and self._unauthorized_calls > 0:
            self._unauthorized_calls -= 1
            return httpx2.Response(401)
        label: Final = credential_label(authorization)
        results: Final[Mapping[str, Mapping[str, object]]] = {
            "initialize": {
                "protocolVersion": payload.get("params", {}).get("protocolVersion"),
                "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
                "serverInfo": {"name": "tickets", "version": "1"},
                "instructions": f"instructions for {label}",
            },
            "tools/list": {"tools": [{"name": f"tool_{label}", "inputSchema": {"type": "object"}}], "ttlMs": 60000},
            "tools/call": {"content": [{"type": "text", "text": "added"}]},
            "prompts/list": {"prompts": [{"name": f"prompt_{label}"}], "ttlMs": 60000},
            "prompts/get": {"messages": [{"role": "user", "content": {"type": "text", "text": label}}]},
            "resources/list": {"resources": [{"name": f"resource_{label}", "uri": f"file:///{label}"}]},
            "resources/templates/list": {
                "resourceTemplates": [{"name": f"template_{label}", "uriTemplate": f"file:///{label}/{{id}}"}]
            },
            "resources/read": {"contents": [{"uri": "file:///thing", "text": label}]},
        }
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": payload["id"], "result": results.get(payload["method"], {})}
        )

    def authorizations(self, method: str) -> tuple[str | None, ...]:
        return tuple(authorization for seen_method, authorization in self.requests if seen_method == method)

    def credentials_seen(self) -> frozenset[str | None]:
        return frozenset(authorization for _, authorization in self.requests)

    def request_count(self) -> int:
        return len(self.requests)


@dataclass(frozen=True, slots=True)
class TokenRequest:
    authorization: str | None
    form: Mapping[str, str]


def minted_token(mint: int) -> str:
    return MINTED_PROFILE_TOKEN if mint == 1 else f"{MINTED_PROFILE_TOKEN}-{mint}"


class RecordingTokenEndpoint:
    def __init__(self) -> None:
        self.requests: Final[list[TokenRequest]] = []  # mutable-ok: the fake records what it was sent

    async def __call__(self, url: str, form: Mapping[str, str], headers: Mapping[str, str]) -> TokenEndpointOutcome:
        self.requests.append(TokenRequest(headers.get("Authorization"), dict(form)))
        return TokenEndpointSuccess(body={"access_token": minted_token(len(self.requests)), "expires_in": 3600})


def tickets_server(
    profiles: Sequence[Mapping[str, object]] = TICKETS_PROFILES,
    overrides: Mapping[str, object] = MappingProxyType({}),
) -> MCPServer:
    parsed: Final = parse_auth_profiles("tickets", tuple(profiles))
    assert isinstance(parsed, tuple)
    return MCPServer.model_validate(
        {
            "server_id": "tickets",
            "name": "tickets",
            "alias": "tickets",
            "transport": MCPTransport.http,
            "url": "https://tickets.example.com/mcp",
            "auth_type": MCPAuth.bearer_token,
            "authentication_token": DEFAULT_CREDENTIAL,
            "additional_auth": parsed or None,
            "allow_all_keys": True,
            "mcp_info": {"tools_list_cache_ttl_seconds": 60},
            **overrides,
        }
    )


def caller_granted(granted: str | None) -> UserAPIKeyAuth:
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-1", mcp_auth_profiles=None if granted is None else {"tickets": granted}
    )
    return UserAPIKeyAuth(api_key="hashed-key", object_permission=permission, object_permission_id="perm-1")


def stored_caller_granted(granted: str | None) -> UserAPIKeyAuth:
    """A key whose permission row `perm-<granted>` is read from the store, as `grants_database` provides it."""
    return UserAPIKeyAuth(api_key="hashed-key", object_permission_id=f"perm-{granted}")


def install_grants_database(monkeypatch: pytest.MonkeyPatch) -> dict[str, LiteLLM_ObjectPermissionTable]:
    """Stored key permissions `perm-<profile>` granting that profile of tickets; `perm-None` grants none."""
    permissions: Final[dict[str, LiteLLM_ObjectPermissionTable]] = {  # mutable-ok: tests edit a stored grant
        f"perm-{granted}": LiteLLM_ObjectPermissionTable(
            object_permission_id=f"perm-{granted}",
            mcp_auth_profiles=None if granted is None else {"tickets": granted},
        )
        for granted in (None, "default", "bots", "ops", "ghost", "empty", "a", "b", "shared")
    }
    install_rust_auth_stores(
        monkeypatch, RustAuthStores(object_permission=RustStoreRows(reader=permissions, writer=permissions))
    )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(proxy_server, "should_load_db_object", lambda _: False)
    return permissions


def plain_server() -> MCPServer:
    return MCPServer(
        server_id="plain",
        name="plain",
        alias="plain",
        transport=MCPTransport.http,
        url="https://plain.example.com/mcp",
        allow_all_keys=True,
    )


def profile_manager(
    server: MCPServer,
    upstream: ProfileUpstream,
    token_endpoint: RecordingTokenEndpoint | None = None,
    *,
    warm: bool = True,
) -> MCPServerManager:
    provider: Final = UpstreamCredentialProvider(
        client_credentials_source=ClientCredentialsTokenSource(token_endpoint or RecordingTokenEndpoint())
    )
    manager: Final = MCPServerManager(cred_provider=provider, upstream_transport=httpx2.MockTransport(upstream))
    manager.config_mcp_servers = {server.server_id: server}
    manager.tool_name_to_mcp_server_name_mapping = {"add": "tickets", "tickets-add": "tickets"} if warm else {}
    return manager


def install_profile_manager(monkeypatch: pytest.MonkeyPatch, manager: MCPServerManager) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)
    monkeypatch.setattr(mcp_server_manager_module, "global_mcp_server_manager", manager)


async def call_tickets_tool(
    name: str,
    server: MCPServer,
    caller: UserAPIKeyAuth | None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
    allowed_mcp_servers: Sequence[MCPServer] | None = None,
) -> CallToolResult | InputRequiredResult:
    return await operations.execute_mcp_tool(
        name=name,
        arguments={},
        allowed_mcp_servers=[server] if allowed_mcp_servers is None else list(allowed_mcp_servers),
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        user_api_key_auth=caller,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
    )


async def call_tickets_add(
    server: MCPServer,
    caller: UserAPIKeyAuth | None,
    mcp_auth_header: str | None = None,
    mcp_server_auth_headers: dict[str, dict[str, str]] | None = None,
) -> CallToolResult | InputRequiredResult:
    return await call_tickets_tool("tickets-add", server, caller, mcp_auth_header, mcp_server_auth_headers)

"""Bridge: route v1 MCP auth resolution through the v2 UpstreamCredentialProvider.

Strangler-fig graft. When ``LITELLM_USE_V2_MCP_RESOLVER`` is enabled (set by the
``--use_v2_migration_resolver`` CLI flag), ``resolve_mcp_auth()`` routes the ``none`` and
``api_key`` modes through the clean-room v2 resolver instead of v1's logic, returning the
resolved credential as a header dict (which ``MCPClient`` merges verbatim). Every other mode,
and any v2 error, falls back to v1. The v2 output is header-for-header identical to v1 for these
two modes; parity is the integration method, so the graft is observable but behavior-preserving.

This lives on the v1 side so the v2 core keeps its no-v1-imports invariant.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Dict, Optional

import httpx
from pydantic import SecretStr

from litellm._logging import verbose_logger
from litellm.proxy.gateway.mcp.outbound_credentials.clock import SystemClock
from litellm.proxy.gateway.mcp.outbound_credentials.credential_store import (
    InMemoryCredentialStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.resolver import (
    UpstreamCredentialProvider,
)
from litellm.proxy.gateway.mcp.outbound_credentials.service_token_store import (
    InMemoryServiceTokenStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_store import (
    InMemoryTokenStore,
    StoredToken,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AwsSigV4Config,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)
from litellm.proxy.gateway.mcp.result import Error, Result
from litellm.types.mcp import MCPAuth

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

_V2_ENV_FLAG = "LITELLM_USE_V2_MCP_RESOLVER"


def v2_resolver_enabled() -> bool:
    return os.getenv(_V2_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


class _Unwired:
    """Fail-closed stand-ins for the ports the grafted modes (none, api_key) never touch."""

    async def refresh(
        self, config: AuthorizationCodeConfig, refresh_token: SecretStr
    ) -> Result[StoredToken, CredError]:
        return Error(CredError.of_not_implemented("token_refresher not wired"))

    async def fetch(
        self, config: ClientCredentialsConfig
    ) -> Result[StoredToken, CredError]:
        return Error(
            CredError.of_not_implemented("client_credentials_fetcher not wired")
        )

    async def exchange(
        self, config: TokenExchangeConfig, subject_token: SecretStr, resource: str
    ) -> Result[StoredToken, CredError]:
        return Error(CredError.of_not_implemented("token_exchanger not wired"))

    async def build(self, config: AwsSigV4Config) -> Result[httpx.Auth, CredError]:
        return Error(CredError.of_not_implemented("signer_factory not wired"))


@functools.lru_cache(maxsize=1)
def _provider() -> UpstreamCredentialProvider:
    # none + api_key resolve from config alone, so the stores/ports below are inert placeholders;
    # real bodies get wired in as their modes are grafted.
    unwired = _Unwired()
    return UpstreamCredentialProvider(
        credential_store=InMemoryCredentialStore(),
        token_store=InMemoryTokenStore(),
        token_refresher=unwired,
        clock=SystemClock(),
        service_token_store=InMemoryServiceTokenStore(),
        client_credentials_fetcher=unwired,
        token_exchanger=unwired,
        signer_factory=unwired,
    )


def _to_server_spec(server: MCPServer) -> Optional[ServerSpec]:
    resource = server.url or server.server_id
    if server.auth_type in (None, MCPAuth.none):
        return ServerSpec(
            server_id=server.server_id, resource=resource, config=NoneConfig()
        )
    if server.auth_type == MCPAuth.api_key:
        token = server.authentication_token
        if not token:
            return None  # api_key with no key: let v1 handle it (parity-safe)
        return ServerSpec(
            server_id=server.server_id,
            resource=resource,
            config=ApiKeyConfig(
                header_name="X-API-Key",
                value_prefix="",
                key_source=SharedKey(value=SecretStr(token)),
            ),
        )
    return None  # other modes are not grafted yet


def _added_headers(auth: httpx.Auth) -> Dict[str, str]:
    # Use .raw (not .items()) so the auth's original header casing survives, e.g. X-API-Key
    # rather than httpx's lowercased x-api-key, keeping byte-parity with v1.
    request = httpx.Request("GET", "https://placeholder.invalid")
    base = {(name.lower(), value) for name, value in request.headers.raw}
    signed = next(auth.auth_flow(request))
    return {
        name.decode(): value.decode()
        for name, value in signed.headers.raw
        if (name.lower(), value) not in base
    }


async def resolve_v2_auth_value(server: MCPServer) -> Optional[Dict[str, str]]:
    """Resolve `none`/`api_key` via the v2 resolver, or return None to defer to v1."""
    if not v2_resolver_enabled():
        return None
    spec = _to_server_spec(server)
    if spec is None:
        return None
    result = await _provider().resolve(
        Subject(tenant_id="", subject_id="", inbound_token=None), spec
    )
    if isinstance(result, Error):
        verbose_logger.warning(
            "v2 MCP resolver failed for server %s: %s; falling back to v1",
            server.server_id,
            result.error.summary,
        )
        return None
    return _added_headers(result.ok)

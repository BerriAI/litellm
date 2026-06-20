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
from litellm.proxy._experimental.mcp_server.v2_port_bodies import (
    HttpxClientCredentialsFetcher,
    HttpxSigV4Signer,
    HttpxTokenExchanger,
    V1ByokCredentialStore,
    V1OAuthTokenStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.clock import SystemClock
from litellm.proxy.gateway.mcp.outbound_credentials.resolver import (
    UpstreamCredentialProvider,
)
from litellm.proxy.gateway.mcp.outbound_credentials.service_token_store import (
    InMemoryServiceTokenStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_store import StoredToken
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    Ambient,
    ApiKeyConfig,
    AssumeRole,
    AuthorizationCodeConfig,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    ServerSpec,
    SharedKey,
    StaticKeys,
    Subject,
    TokenExchangeConfig,
)
from litellm.proxy.gateway.mcp.result import Error, Result
from litellm.types.mcp import MCPAuth

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
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
def provider() -> UpstreamCredentialProvider:
    # Real bodies are wired as their modes are grafted. token_refresher stays unwired: the bridge's
    # V1OAuthTokenStore returns currently-valid tokens (v1 refreshes on read), so the resolver's
    # proactive-refresh path is inert until v1 retires.
    unwired = _Unwired()
    return UpstreamCredentialProvider(
        credential_store=V1ByokCredentialStore(),
        token_store=V1OAuthTokenStore(),
        token_refresher=unwired,
        clock=SystemClock(),
        service_token_store=InMemoryServiceTokenStore(),
        client_credentials_fetcher=HttpxClientCredentialsFetcher(),
        token_exchanger=HttpxTokenExchanger(),
        signer_factory=HttpxSigV4Signer(),
    )


def to_server_spec(server: MCPServer) -> Optional[ServerSpec]:
    resource = server.url or server.server_id
    if server.auth_type in (None, MCPAuth.none):
        # A none server opted into upstream OAuth passthrough forwards the caller's bearer; the
        # passthrough arm sends the inbound token. Otherwise no upstream credential.
        if server.is_oauth_passthrough:
            return ServerSpec(
                server_id=server.server_id,
                resource=resource,
                config=PassthroughConfig(),
            )
        return ServerSpec(
            server_id=server.server_id, resource=resource, config=NoneConfig()
        )
    if server.auth_type == MCPAuth.api_key:
        if server.is_byok:
            # Per-user key; the arm pulls it from the CredentialStore keyed by the Subject.
            return ServerSpec(
                server_id=server.server_id,
                resource=resource,
                config=ApiKeyConfig(
                    header_name="X-API-Key",
                    value_prefix="",
                    key_source=Byok(),
                ),
            )
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
    if server.auth_type == MCPAuth.aws_sigv4:
        # Reuse the SigV4 config builder; the resolver's aws_sigv4 arm turns it into the botocore
        # signer (an httpx.Auth) that signs each upstream request.
        aws_config = _to_aws_sigv4_config(server)
        if aws_config is None:
            return None  # AssumeRole with explicit base keys: not representable yet, defer to v1
        return ServerSpec(
            server_id=server.server_id, resource=resource, config=aws_config
        )
    if server.has_token_exchange_config:
        # token_exchange takes precedence over client_credentials (matches v1's cascade); the arm
        # binds the exchanged token to this resource (audience, RFC 8707).
        return ServerSpec(
            server_id=server.server_id,
            resource=server.audience or resource,
            config=TokenExchangeConfig(
                subject_token_type=server.subject_token_type,
                token_exchange_endpoint=server.token_exchange_endpoint
                or server.token_url,
                client_id=server.client_id,
                client_secret=(
                    SecretStr(server.client_secret) if server.client_secret else None
                ),
                scopes=tuple(server.scopes or ()),
            ),
        )
    if (
        server.has_client_credentials
        and server.client_id
        and server.client_secret
        and server.token_url
    ):
        return ServerSpec(
            server_id=server.server_id,
            resource=resource,
            config=ClientCredentialsConfig(
                client_id=server.client_id,
                client_secret=SecretStr(server.client_secret),
                token_url=server.token_url,
                scopes=tuple(server.scopes or ()),
            ),
        )
    if server.needs_user_oauth_token:
        # Interactive oauth2 (3LO), not M2M/exchange (handled above). delegate_auth_to_upstream
        # forwards the caller token (passthrough); otherwise the gateway holds a per-user token
        # (authorization_code). This split is the LIT-3795 fix: a non-delegated interactive oauth2
        # server must not forward the caller JWT upstream.
        if server.delegate_auth_to_upstream:
            return ServerSpec(
                server_id=server.server_id,
                resource=resource,
                config=PassthroughConfig(),
            )
        return ServerSpec(
            server_id=server.server_id,
            resource=resource,
            config=AuthorizationCodeConfig(
                client_id=server.client_id,
                client_secret=(
                    SecretStr(server.client_secret) if server.client_secret else None
                ),
                authorization_url=server.authorization_url,
                token_url=server.token_url,
                scopes=tuple(server.scopes or ()),
            ),
        )
    return None  # aws_sigv4 uses the aws_auth seam; misconfigured modes defer to v1


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


def to_subject(
    user_api_key_auth: Optional[UserAPIKeyAuth], subject_token: Optional[str]
) -> Subject:
    """Map v1's authenticated principal onto the v2 Subject.

    Isolated so it can later swap to auth_v2's Principal. tenant_id/subject_id are empty for an
    unauthenticated caller; the per-user arms (BYOK api_key, token_exchange, authorization_code)
    must reject an empty subject_id rather than share one credential slot across callers.
    """
    inbound = SecretStr(subject_token) if subject_token else None
    if user_api_key_auth is None:
        return Subject(tenant_id="", subject_id="", inbound_token=inbound)
    return Subject(
        tenant_id=user_api_key_auth.org_id or user_api_key_auth.team_id or "",
        subject_id=user_api_key_auth.user_id or "",
        inbound_token=inbound,
    )


async def resolve_v2_auth_value(
    server: MCPServer,
    user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    subject_token: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Resolve `none`/`api_key` via the v2 resolver, or return None to defer to v1."""
    if not v2_resolver_enabled():
        return None
    spec = to_server_spec(server)
    if spec is None:
        return None
    result = await provider().resolve(
        to_subject(user_api_key_auth, subject_token), spec
    )
    if isinstance(result, Error):
        verbose_logger.warning(
            "v2 MCP resolver failed for server %s: %s; falling back to v1",
            server.server_id,
            result.error.summary,
        )
        return None
    headers = _added_headers(result.ok)
    verbose_logger.info(
        "v2 MCP resolver handled server %s (auth_type=%s): %s",
        server.server_id,
        server.auth_type,
        f"attached {sorted(headers)}" if headers else "no auth header",
    )
    return headers


def _to_aws_sigv4_config(server: MCPServer) -> Optional[AwsSigV4Config]:
    region = server.aws_region_name or "us-east-1"
    service = server.aws_service_name or "bedrock-agentcore"
    if server.aws_role_name:
        # v2's AssumeRole assumes via the ambient/default chain. v1 also supports assuming with
        # explicit base keys, which v2 can't represent yet, so defer that case to v1.
        if server.aws_access_key_id or server.aws_secret_access_key:
            return None
        return AwsSigV4Config(
            region=region,
            service=service,
            credentials=AssumeRole(
                role_arn=server.aws_role_name,
                session_name=server.aws_session_name,
            ),
        )
    if server.aws_access_key_id and server.aws_secret_access_key:
        return AwsSigV4Config(
            region=region,
            service=service,
            credentials=StaticKeys(
                access_key_id=server.aws_access_key_id,
                secret_access_key=SecretStr(server.aws_secret_access_key),
                session_token=(
                    SecretStr(server.aws_session_token)
                    if server.aws_session_token
                    else None
                ),
            ),
        )
    return AwsSigV4Config(region=region, service=service, credentials=Ambient())


async def resolve_v2_aws_auth(server: MCPServer) -> Optional[httpx.Auth]:
    """Resolve `aws_sigv4` via the v2 resolver into the SigV4 signer, or None to defer to v1.

    SigV4 signs every request, so (unlike the header modes) the result is attached to the
    connection as `MCPClient.aws_auth` rather than extracted into a header.
    """
    if not v2_resolver_enabled():
        return None
    if server.auth_type != MCPAuth.aws_sigv4:
        return None
    config = _to_aws_sigv4_config(server)
    if config is None:
        return None
    result = await provider().resolve(
        Subject(tenant_id="", subject_id="", inbound_token=None),
        ServerSpec(
            server_id=server.server_id,
            resource=server.url or server.server_id,
            config=config,
        ),
    )
    if isinstance(result, Error):
        verbose_logger.warning(
            "v2 MCP resolver failed to build aws_sigv4 signer for server %s: %s; "
            "falling back to v1",
            server.server_id,
            result.error.summary,
        )
        return None
    verbose_logger.info(
        "v2 MCP resolver handled server %s (auth_type=aws_sigv4): attached SigV4 signer",
        server.server_id,
    )
    return result.ok

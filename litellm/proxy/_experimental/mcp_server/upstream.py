from __future__ import annotations

import os
from typing import Final

import httpx2
from fastapi import HTTPException

from litellm.constants import MCP_CLIENT_TIMEOUT, MCP_NPM_CACHE_DIR, MCP_STDIO_ALLOWED_COMMANDS
from litellm.experimental_mcp_client.client import MCPClient, MCPSigV4Auth
from litellm.proxy._experimental.mcp_server.legacy_callbacks import ElicitationCallback, SamplingCallback
from litellm.proxy._experimental.mcp_server.mcp_debug import record_auth_resolution
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import resolve_mcp_auth, resolved_token_header
from litellm.proxy._experimental.mcp_server.outbound_credentials import Error, Ok, UpstreamCredentialProvider
from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (
    prepare_mcp_client,
    raise_public,
    raise_token_exchange_challenge,
    raise_user_oauth_challenge,
    to_server_spec,
    to_subject,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.resolver import resolve_credentials_with_source
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    DEFAULT_CREDENTIAL_HEADER,
    AuthorizationCodeConfig,
    AuthResolution,
    ClientCredentialsConfig,
    CredError,
    IdJagConfig,
    PassthroughConfig,
    ServerSpec,
    TokenExchangeConfig,
)
from litellm.proxy._experimental.mcp_server.stdio_gate import MCP_STDIO_DISABLED_MESSAGE, is_mcp_stdio_enabled
from litellm.proxy._types import MCPTransport, UserAPIKeyAuth
from litellm.types.mcp import (
    MCPAuth,
    MCPStdioConfig,
    MCPTransportType,
    MCPUpstreamProtocol,
    has_header,
    without_header,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def take_forwarded_authorization(
    headers: dict[str, str] | None,
) -> tuple[str | None, dict[str, str] | None]:
    """Pop the ``Authorization`` value out of ``headers`` (case-insensitive), returning it with the
    remaining headers, so the passthrough resolver arm is the single Authorization source rather than
    the header also riding in ``extra_headers`` (which the resolved auth would then defer to)."""
    if not headers:
        return None, headers
    value: Final = next((v for k, v in headers.items() if k.lower() == "authorization"), None)
    return value, without_header(headers, DEFAULT_CREDENTIAL_HEADER)


def passthrough_token_from_mcp_auth_header(
    mcp_auth_header: str | dict[str, str] | None,
) -> str | None:
    """The caller's per-server upstream credential for a passthrough-mode server, or None.

    Sourced from ``x-mcp-{alias}-authorization`` (string or per-header dict form) or the deprecated
    global ``x-mcp-auth`` fallback. Per-server headers are the multi-server shape: they bind one
    token to one server, so an aggregate scope with several passthrough-mode servers never replays
    a single credential across upstreams. The value is forwarded verbatim, so it must be the full
    header value (e.g. ``Bearer <upstream-token>``)."""
    if isinstance(mcp_auth_header, str):
        return mcp_auth_header or None
    if isinstance(mcp_auth_header, dict):
        return next((v for k, v in mcp_auth_header.items() if k.lower() == "authorization"), None)
    return None


def to_server_spec_fail_closed(server: MCPServer) -> ServerSpec | None:
    """`to_server_spec`, except a half-configured `oauth2_id_jag` server refuses instead of deferring.

    ID-JAG has no v1 arm, so deferring to v1 would let `resolve_mcp_auth` honor a caller x-mcp-*
    override or fall through to the static `authentication_token`, both of which bypass the per-user
    identity assertion the mode promises. That is an operator misconfiguration, not a fallback.
    """
    spec: Final = to_server_spec(server)
    if spec is None and server.auth_type == MCPAuth.oauth2_id_jag:
        raise_public(
            CredError.of_misconfigured(
                "oauth2_id_jag requires token_exchange_endpoint, id_jag_resource_token_endpoint, "
                "client_id, and a client_secret or client_private_key; refusing to fall back to "
                "a static credential."
            )
        )
    return spec


async def resolve_upstream_auth(
    *,
    server: MCPServer,
    spec: ServerSpec,
    root_path: str,
    provider: UpstreamCredentialProvider,
    subject_token: str | None,
    user_api_key_auth: UserAPIKeyAuth | None,
    extra_headers: dict[str, str] | None,
) -> tuple[httpx2.Auth | None, dict[str, str] | None]:
    """Resolve a v2-owned server's upstream credential into ``(resolved_auth, extra_headers)``.

    On a missing/rejected per-user credential this raises the mode's discovery challenge
    (authorization_code's browser-OAuth 401, token_exchange's RFC 9728 challenge) or maps any
    other ``CredError`` onto its public HTTP status; it never returns an error as a value.
    """
    match await resolve_credentials_with_source(provider, to_subject(user_api_key_auth, subject_token), spec):
        case Ok(credential):
            auth: Final = credential.auth
            # NoOpAuth has no header_name and so never conflicts.
            header_name: Final[str | None] = getattr(auth, "header_name", None)
            if header_name is None or not extra_headers:
                source: Final = (
                    AuthResolution.extra_headers
                    if credential.source == AuthResolution.no_auth and extra_headers
                    else credential.source
                )
                record_auth_resolution(server.server_id, source)
                return auth, extra_headers
            if not has_header(extra_headers, header_name):
                record_auth_resolution(server.server_id, credential.source)
                return auth, extra_headers
            if isinstance(
                spec.config,
                (TokenExchangeConfig, AuthorizationCodeConfig, IdJagConfig, ClientCredentialsConfig),
            ):
                # The resolver owns the credential here (token_exchange's exchanged token,
                # authorization_code's stored token, id_jag's minted assertion,
                # client_credentials' gateway-minted M2M token). It is authoritative: a
                # guardrail such as MCPJWTSigner, static_headers, or any other injected
                # Authorization must NOT shadow it (otherwise the upstream gets e.g. the
                # signer's JWT instead of the minted token and rejects it, and for M2M the
                # one-shot 401 refetch is lost with it). Drop only the header the resolved
                # credential is about to occupy, so a static credential the operator aimed at a
                # DIFFERENT header still reaches upstream.
                record_auth_resolution(server.server_id, credential.source)
                return auth, without_header(extra_headers, header_name)
            # Other modes: an Authorization already supplied via extra_headers (a forwarded caller
            # header or static_headers) is intentional and wins; v1 applies those last.
            record_auth_resolution(server.server_id, AuthResolution.extra_headers)
            return None, extra_headers
        case Error(err):
            record_auth_resolution(server.server_id, AuthResolution.failed)
            if err.tag == "unauthorized" and isinstance(spec.config, AuthorizationCodeConfig):
                # authorization_code's missing per-user token -> the per-server browser-OAuth
                # challenge, built here where the full MCPServer is in hand.
                raise_user_oauth_challenge(server, root_path=root_path)
            if err.tag == "unauthorized" and isinstance(spec.config, TokenExchangeConfig):
                # token_exchange (OBO): a missing/rejected subject token -> the RFC 9728 challenge
                # pointing at the IdP the client must SSO with to obtain one, rather than an opaque
                # 401. No gateway-side browser flow. An IdP step-up rejection (Entra Conditional
                # Access) threads its claims blob into the challenge for the client to satisfy.
                raise_token_exchange_challenge(
                    server,
                    root_path=root_path,
                    claims=err.unauthorized.claims,
                )
            raise_public(err)


def _stdio_config(server: MCPServer, stdio_env: dict[str, str] | None) -> MCPStdioConfig | None:
    if not is_mcp_stdio_enabled():
        raise HTTPException(status_code=403, detail=MCP_STDIO_DISABLED_MESSAGE)
    if server.command:
        command: Final = os.path.basename(server.command)
        lowercase: Final = command.lower()
        normalized: Final = next(
            (
                lowercase.removesuffix(suffix)
                for suffix in (".exe", ".cmd", ".bat", ".com")
                if lowercase.endswith(suffix)
            ),
            lowercase,
        )
        if command not in MCP_STDIO_ALLOWED_COMMANDS and normalized not in MCP_STDIO_ALLOWED_COMMANDS:
            raise HTTPException(
                status_code=403,
                detail=f"MCP stdio command '{server.command}' is not in the allowlist ({sorted(MCP_STDIO_ALLOWED_COMMANDS)}). "
                "Add it to LITELLM_MCP_STDIO_EXTRA_COMMANDS to allow this command.",
            )
    if not server.command or server.args is None:
        return None
    environment: Final = stdio_env if stdio_env is not None else server.env
    return MCPStdioConfig(
        command=server.command,
        args=server.args,
        env={"NPM_CONFIG_CACHE": MCP_NPM_CACHE_DIR, **environment} if environment is not None else None,
    )


async def prepare_upstream_client(
    server: MCPServer,
    *,
    provider: UpstreamCredentialProvider,
    root_path: str,
    mcp_auth_header: str | dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
    stdio_env: dict[str, str] | None = None,
    subject_token: str | None = None,
    user_api_key_auth: UserAPIKeyAuth | None = None,
    protocol_version: MCPUpstreamProtocol,
    sampling_callback: SamplingCallback | None = None,
    elicitation_callback: ElicitationCallback | None = None,
) -> MCPClient:
    transport: Final[MCPTransportType] = server.transport or MCPTransport.sse
    server_spec: Final = None if transport == MCPTransport.stdio else to_server_spec_fail_closed(server)
    spec: Final = (
        None
        if server_spec is not None
        and mcp_auth_header
        and not isinstance(
            server_spec.config, (AuthorizationCodeConfig, IdJagConfig, PassthroughConfig, TokenExchangeConfig)
        )
        else server_spec
    )
    auth_value: Final = await resolve_mcp_auth(server, mcp_auth_header) if spec is None else None
    auth_header_name: Final = resolved_token_header(server, mcp_auth_header) if spec is None else None
    timeout: Final = server.timeout if server.timeout is not None else MCP_CLIENT_TIMEOUT
    if transport == MCPTransport.stdio:
        config: Final = _stdio_config(server, stdio_env)
        record_auth_resolution(server.server_id, AuthResolution.not_applicable)
        return MCPClient(
            server_url="",
            transport_type=transport,
            protocol_version=protocol_version,
            auth_type=server.auth_type,
            auth_value=auth_value,
            timeout=timeout,
            stdio_config=config,
            extra_headers=extra_headers,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
        )
    if spec is not None:
        inbound_token, forwarded_headers = (
            take_forwarded_authorization(extra_headers)
            if isinstance(spec.config, PassthroughConfig)
            else (subject_token, extra_headers)
        )
        per_server_token: Final = (
            passthrough_token_from_mcp_auth_header(mcp_auth_header)
            if isinstance(spec.config, PassthroughConfig)
            else None
        )
        resolved_auth, resolved_headers = await resolve_upstream_auth(
            server=server,
            spec=spec,
            provider=provider,
            root_path=root_path,
            subject_token=per_server_token if per_server_token is not None else inbound_token,
            user_api_key_auth=user_api_key_auth,
            extra_headers=forwarded_headers,
        )
        return await prepare_mcp_client(
            server,
            MCPClient(
                server_url=server.url or "",
                transport_type=transport,
                protocol_version=protocol_version,
                auth_type=server.auth_type,
                timeout=timeout,
                extra_headers=resolved_headers,
                resolved_auth=resolved_auth,
                sampling_callback=sampling_callback,
                elicitation_callback=elicitation_callback,
            ),
        )
    aws_auth: Final = (
        MCPSigV4Auth(
            aws_access_key_id=server.aws_access_key_id,
            aws_secret_access_key=server.aws_secret_access_key,
            aws_session_token=server.aws_session_token,
            aws_region_name=server.aws_region_name,
            aws_service_name=server.aws_service_name,
            aws_role_name=server.aws_role_name,
            aws_session_name=server.aws_session_name,
        )
        if server.auth_type == MCPAuth.aws_sigv4
        else None
    )
    source: Final = (
        AuthResolution.aws_sigv4
        if aws_auth is not None
        else AuthResolution.extra_headers
        if extra_headers and has_header(extra_headers, auth_header_name or "Authorization")
        else AuthResolution.per_request_header
        if mcp_auth_header
        else AuthResolution.static_token
        if auth_value
        else AuthResolution.extra_headers
        if extra_headers
        else AuthResolution.no_auth
    )
    record_auth_resolution(server.server_id, source)
    return await prepare_mcp_client(
        server,
        MCPClient(
            server_url=server.url or "",
            transport_type=transport,
            protocol_version=protocol_version,
            auth_type=server.auth_type,
            auth_value=auth_value,
            auth_header_name=auth_header_name,
            timeout=timeout,
            extra_headers=extra_headers,
            aws_auth=aws_auth,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
        ),
    )

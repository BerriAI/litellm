"""The v1 <-> v2 bridge for the credential resolver.

These edge functions translate v1's request objects into the resolver's typed inputs and map
its typed errors onto the proxy's public exception contract. They import v1 and live outside the
package's public surface so the resolver core (``resolver.py`` / ``types.py``) stays v1-free.
``_create_mcp_client`` wires them in via ``_resolve_v2_auth`` for every migrated mode.

``to_server_spec`` maps only the modes the resolver has gone live for, returning ``None`` for
every other mode so the caller defers to v1 (parity-safe); it grows one branch per migrated mode.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Literal, NoReturn, Optional

from fastapi import HTTPException
from pydantic import SecretStr
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)
from litellm.types.mcp import DEFAULT_SUBJECT_TOKEN_TYPE, MCPAuth

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


def to_subject(user_api_key_auth: Optional[UserAPIKeyAuth], subject_token: Optional[str]) -> Subject:
    """Map v1's authenticated principal onto the resolver's Subject.

    tenant_id / subject_id are empty for an unauthenticated caller; the per-user arms must reject
    an empty subject rather than share one credential slot across callers. A validated delegation
    assertion (UserAPIKeyAuth.delegated_user_id, stamped at MCP admission after the consent check)
    replaces the credential subject so per-user upstream credentials resolve as the delegated
    user; admission, permissions, and attribution stay on the calling key.
    """
    inbound = SecretStr(subject_token) if subject_token else None
    if user_api_key_auth is None:
        return Subject(tenant_id="", subject_id="", inbound_token=inbound)
    return Subject(
        tenant_id=user_api_key_auth.org_id or user_api_key_auth.team_id or "",
        subject_id=user_api_key_auth.delegated_user_id or user_api_key_auth.user_id or "",
        inbound_token=inbound,
    )


def to_server_spec(server: MCPServer) -> Optional[ServerSpec]:
    """Map a v1 server onto a ServerSpec for a migrated mode, or None to defer to v1.

    BYOK is the per-user source of the ``api_key`` mode; its scheme rides on ``auth_type`` just
    like a shared key, but the value is per-user and not migrated yet, so a BYOK server defers
    to v1 regardless of ``auth_type`` (this guard is the seam the BYOK arm replaces later).

    Dispatches on the declared ``auth_type``. The match is exhaustive over ``MCPAuthType`` with
    an ``assert_never`` tail, so a newly added auth mode fails the type gate here until it is
    explicitly mapped or explicitly deferred, rather than silently falling through to v1. Live
    modes: ``none``, the static-header family (``api_key`` plus the Authorization schemes,
    all shared-key), ``oauth2`` per-user tokens (``authorization_code``), ``oauth2_token_exchange``
    (OBO), and the client-forwarded token modes ``true_passthrough`` / ``oauth_delegate``
    (``PassthroughConfig``); client_credentials (M2M), delegated/passthrough oauth2, and SigV4
    return None and stay on v1.
    """
    if server.is_byok:
        return None  # per-user BYOK source not migrated yet -> defer to v1 (any auth_type)
    resource = server.url or server.server_id
    auth_type = server.auth_type
    match auth_type:
        case None | MCPAuth.none:
            if server.is_oauth_passthrough:
                return None  # passthrough is not migrated yet -> defer to v1
            return ServerSpec(server_id=server.server_id, resource=resource, config=NoneConfig())
        case MCPAuth.api_key:
            return _shared_key_spec(server, resource, "X-API-Key", "")
        case MCPAuth.bearer_token:
            return _shared_key_spec(server, resource, "Authorization", "Bearer")
        case MCPAuth.token:
            return _shared_key_spec(server, resource, "Authorization", "token")
        case MCPAuth.authorization:
            return _shared_key_spec(server, resource, "Authorization", "")
        case MCPAuth.basic:
            return _shared_key_spec(server, resource, "Authorization", "Basic", encode=True)
        case MCPAuth.oauth2:
            if server.needs_user_oauth_token and not server.delegate_auth_to_upstream:
                return ServerSpec(
                    server_id=server.server_id,
                    resource=resource,
                    config=AuthorizationCodeConfig(),
                )
            # client_credentials (M2M) and delegate/passthrough oauth2 stay on v1
            return None
        case MCPAuth.true_passthrough | MCPAuth.oauth_delegate:
            return ServerSpec(server_id=server.server_id, resource=resource, config=PassthroughConfig())
        case MCPAuth.oauth2_token_exchange:
            return _token_exchange_spec(server, resource)
        case MCPAuth.aws_sigv4:
            return None  # SigV4 is not migrated yet -> defer to v1
    assert_never(auth_type)


def _token_exchange_spec(server: MCPServer, resource: str) -> Optional[ServerSpec]:
    """Build a token_exchange (OBO) spec, or defer (None) when it is not OBO-configured.

    An OBO server with ``client_id``/``client_secret`` is owned by the v2 arm even if the
    ``token_exchange_endpoint``/``token_url`` is absent: a missing endpoint then fails closed (412) at
    the exchanger rather than silently deferring to v1 and connecting unauthenticated, since the
    gateway must not guess the IdP or fall back to a weaker source. Without client credentials there is
    nothing to own, so the server stays on v1 (parity-safe). ``profile`` selects the wire dialect
    (``rfc8693`` default, ``entra_obo`` for Microsoft Entra On-Behalf-Of); an unrecognized value
    normalizes to ``rfc8693`` so a bad config value cannot crash spec-building. ``audience`` is
    forwarded only when the operator set it; a missing one is omitted, not derived.
    """
    endpoint = server.token_exchange_endpoint or server.token_url
    if not server.client_id or not server.client_secret:
        return None
    profile: Literal["rfc8693", "entra_obo"] = (
        "entra_obo" if server.token_exchange_profile == "entra_obo" else "rfc8693"
    )
    return ServerSpec(
        server_id=server.server_id,
        resource=resource,
        config=TokenExchangeConfig(
            profile=profile,
            subject_token_type=server.subject_token_type or DEFAULT_SUBJECT_TOKEN_TYPE,
            token_exchange_endpoint=endpoint,
            audience=server.audience,
            client_id=server.client_id,
            client_secret=SecretStr(server.client_secret),
            token_endpoint_auth_method=server.token_endpoint_auth_method,
            scopes=tuple(server.scopes or ()),
        ),
    )


def _shared_key_spec(
    server: MCPServer,
    resource: str,
    header_name: str,
    value_prefix: str,
    *,
    encode: bool = False,
) -> Optional[ServerSpec]:
    """Build an api_key spec from the server's static token, or defer (None) if it is absent.

    Covers the whole shared-key static-header family: ``api_key`` on ``X-API-Key`` and the
    Authorization schemes (bearer / token / authorization sent verbatim, basic base64-encoded).
    """
    token = server.authentication_token
    if not token:
        return None  # no key configured -> defer to v1 (parity-safe)
    value = base64.b64encode(token.encode("utf-8")).decode() if encode else token
    return ServerSpec(
        server_id=server.server_id,
        resource=resource,
        config=ApiKeyConfig(
            header_name=header_name,
            value_prefix=value_prefix,
            key_source=SharedKey(value=SecretStr(value)),
        ),
    )


def raise_public(error: CredError) -> NoReturn:
    """Map a resolver CredError onto the proxy's public HTTP contract. The one edge that raises."""
    match error.tag:
        case "unauthorized":
            challenge = error.unauthorized
            raise HTTPException(
                status_code=401,
                detail=challenge.body if challenge.body is not None else error.summary,
                headers=({"WWW-Authenticate": challenge.www_authenticate} if challenge.www_authenticate else None),
            )
        case "misconfigured":
            raise HTTPException(status_code=500, detail=error.summary)
        case "upstream_unavailable":
            raise HTTPException(status_code=503, detail=error.summary)
        case "unsupported_mode":
            raise HTTPException(status_code=500, detail=error.summary)
        case "precondition_required":
            raise HTTPException(status_code=412, detail=error.summary)
        case "not_implemented":
            raise HTTPException(status_code=501, detail=error.summary)
    assert_never(error.tag)


def oauth_protected_resource_path(root_path: str, server: MCPServer) -> str:
    """The server's RFC 9728 Protected Resource Metadata path, the shared anchor of both challenges.

    ``root_path`` is the proxy's ``SERVER_ROOT_PATH``, resolved by the caller (the imperative shell)
    so this stays a pure function of its inputs; ``"/"`` and ``""`` both mean no prefix. The path is
    relative, so it resolves against the caller's own host (correct even behind a reverse proxy).
    """
    prefix = "" if root_path == "/" else root_path
    name = server.alias or server.server_name or server.name or server.server_id
    return f"/.well-known/oauth-protected-resource{prefix}/mcp/{name}"


def raise_user_oauth_challenge(server: MCPServer, *, root_path: str) -> NoReturn:
    """Raise the 401 an ``authorization_code`` server returns at egress when the user has no token.

    Points at the server's RFC 9728 Protected Resource Metadata, which names the upstream
    authorization server the client must complete OAuth with. The listing-phase 401 still emits the
    RFC 8414 ``authorization_uri`` form pending the format unification; both target the same server,
    so the difference is cosmetic.
    """
    resource_metadata = oauth_protected_resource_path(root_path, server)
    raise HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": f'Bearer resource_metadata="{resource_metadata}"'},
    )


def raise_token_exchange_challenge(
    server: MCPServer,
    *,
    root_path: str,
    claims: str | None = None,
) -> NoReturn:
    """Raise the RFC 9728 / RFC 6750 challenge an OBO (``token_exchange``) server returns when the
    caller's subject token is missing or the IdP rejected it.

    Points at the server's Protected Resource Metadata, whose ``authorization_servers`` names the IdP
    the client must SSO with to obtain a subject token; ``error="invalid_token"`` tells a
    spec-compliant MCP client to discover that AS and retry with a fresh bearer. Mirrors
    ``raise_user_oauth_challenge`` but for the exchange flow: there is no gateway-side browser OAuth —
    the client re-authenticates directly with the IdP, and LiteLLM then exchanges the resulting token.

    An IdP step-up rejection (Entra Conditional Access / CAE) passes its ``claims`` blob. Per the
    Microsoft claims-challenge format the challenge then uses ``error="insufficient_claims"`` (the
    value MSAL-family clients key on) and carries the claims base64-encoded in a ``claims`` parameter
    the client replays to the IdP to satisfy the step-up. Without a claims blob the challenge keeps
    ``error="invalid_token"`` and is byte-identical to the static one. Both the error value (one of
    two literals) and the base64 claims draw from a fixed alphabet, so nothing from the IdP body
    reaches the header unescaped.
    """
    resource_metadata = oauth_protected_resource_path(root_path, server)
    encoded_claims = base64.b64encode(claims.encode()).decode() if claims else None
    error = "insufficient_claims" if encoded_claims else "invalid_token"
    error_description = (
        "Step-up authentication required; satisfy the returned claims challenge with the IdP and retry"
        if encoded_claims
        else "Missing or invalid subject token; authenticate with the IdP and retry"
    )
    www_authenticate = ", ".join(
        (
            f'Bearer resource_metadata="{resource_metadata}"',
            f'error="{error}"',
            f'error_description="{error_description}"',
            *((f'claims="{encoded_claims}"',) if encoded_claims else ()),
        )
    )
    raise HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": www_authenticate},
    )

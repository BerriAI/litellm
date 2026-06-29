"""The v1 <-> v2 bridge for the credential resolver.

These edge functions translate v1's request objects into the resolver's typed inputs and map
its typed errors onto the proxy's public exception contract. They import v1 and live outside the
package's public surface so the resolver core (``resolver.py`` / ``types.py``) stays v1-free.
Nothing wires them into ``_create_mcp_client`` yet.

``to_server_spec`` maps only the modes the resolver has gone live for, returning ``None`` for
every other mode so the caller defers to v1 (parity-safe); it grows one branch per migrated mode.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, NoReturn, Optional

from fastapi import HTTPException
from pydantic import SecretStr
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    CredError,
    NoneConfig,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)
from litellm.types.mcp import MCPAuth

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


def to_subject(user_api_key_auth: Optional[UserAPIKeyAuth], subject_token: Optional[str]) -> Subject:
    """Map v1's authenticated principal onto the resolver's Subject.

    tenant_id / subject_id are empty for an unauthenticated caller; the per-user arms must reject
    an empty subject rather than share one credential slot across callers.
    """
    inbound = SecretStr(subject_token) if subject_token else None
    if user_api_key_auth is None:
        return Subject(tenant_id="", subject_id="", inbound_token=inbound)
    return Subject(
        tenant_id=user_api_key_auth.org_id or user_api_key_auth.team_id or "",
        subject_id=user_api_key_auth.user_id or "",
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
    all shared-key), ``oauth2`` per-user tokens (``authorization_code``), and
    ``oauth2_token_exchange`` (RFC 8693 OBO); client_credentials (M2M), delegated/passthrough
    oauth2, and SigV4 return None and stay on v1.
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
        case MCPAuth.oauth2_token_exchange:
            return _token_exchange_spec(server, resource)
        case MCPAuth.aws_sigv4:
            return None  # SigV4 is not migrated yet -> defer to v1
    assert_never(auth_type)


def _token_exchange_spec(server: MCPServer, resource: str) -> Optional[ServerSpec]:
    """Build a token_exchange (RFC 8693 OBO) spec, or defer (None) if the exchange config is absent.

    Mirrors v1's ``has_token_exchange_config`` precondition: an endpoint (``token_exchange_endpoint``
    or ``token_url``) plus ``client_id``/``client_secret`` must all be present, else there is nothing
    to exchange against and the server stays on v1 (parity-safe). ``audience`` is forwarded only when
    the operator set it; a missing one is omitted, not derived.
    """
    endpoint = server.token_exchange_endpoint or server.token_url
    if not endpoint or not server.client_id or not server.client_secret:
        return None
    return ServerSpec(
        server_id=server.server_id,
        resource=resource,
        config=TokenExchangeConfig(
            subject_token_type=server.subject_token_type or "urn:ietf:params:oauth:token-type:access_token",
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


def raise_user_oauth_challenge(server: MCPServer) -> NoReturn:
    """Raise the 401 an ``authorization_code`` server returns at egress when the user has no token.

    Points at the server's RFC 9728 Protected Resource Metadata (``resource_metadata``), which names
    the upstream authorization server the client must complete OAuth with. The URL is per-server and
    relative, so it resolves against the caller's own host (correct even behind a reverse proxy)
    without needing request context. The listing-phase 401 still emits the RFC 8414 ``authorization_uri``
    form pending the format unification; both target the same server, so the difference is cosmetic.
    """
    from litellm.proxy.utils import get_server_root_path  # noqa: PLC0415

    root = get_server_root_path()
    prefix = "" if root == "/" else root
    name = server.alias or server.server_name or server.name or server.server_id
    resource_metadata = f"/.well-known/oauth-protected-resource{prefix}/mcp/{name}"
    raise HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": f'Bearer resource_metadata="{resource_metadata}"'},
    )

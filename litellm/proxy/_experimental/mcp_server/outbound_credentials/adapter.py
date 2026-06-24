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
    CredError,
    NoneConfig,
    ServerSpec,
    SharedKey,
    Subject,
)
from litellm.types.mcp import MCPAuth

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


def to_subject(
    user_api_key_auth: Optional[UserAPIKeyAuth], subject_token: Optional[str]
) -> Subject:
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
    modes: ``none`` and the static-header family (``api_key`` plus the Authorization schemes),
    all shared-key; every other mode returns None and stays on v1.
    """
    if server.is_byok:
        return (
            None  # per-user BYOK source not migrated yet -> defer to v1 (any auth_type)
        )
    resource = server.url or server.server_id
    auth_type = server.auth_type
    match auth_type:
        case None | MCPAuth.none:
            if server.is_oauth_passthrough:
                return None  # passthrough is not migrated yet -> defer to v1
            return ServerSpec(
                server_id=server.server_id, resource=resource, config=NoneConfig()
            )
        case MCPAuth.api_key:
            return _shared_key_spec(server, resource, "X-API-Key", "")
        case MCPAuth.bearer_token:
            return _shared_key_spec(server, resource, "Authorization", "Bearer")
        case MCPAuth.token:
            return _shared_key_spec(server, resource, "Authorization", "token")
        case MCPAuth.authorization:
            return _shared_key_spec(server, resource, "Authorization", "")
        case MCPAuth.basic:
            return _shared_key_spec(
                server, resource, "Authorization", "Basic", encode=True
            )
        case MCPAuth.oauth2 | MCPAuth.oauth2_token_exchange | MCPAuth.aws_sigv4:
            return None  # OAuth grants and SigV4 are not migrated yet -> defer to v1
    assert_never(auth_type)


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
            raise HTTPException(status_code=401, detail=error.summary)
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

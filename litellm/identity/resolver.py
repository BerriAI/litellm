"""Compose extractors + DB load into a single ``IdentityContext`` per request.

Two call shapes:

- ``resolve_identity_for_principal`` — given pre-extracted credentials, decide
  the principal kind and resolve it. Used by the proxy auth chain after it
  already pulled the api-key out of the request.

- ``resolve_identity`` — request-scoped composition for new entrypoints. Not
  yet wired into ``user_api_key_auth.py``; lives here so callers without a
  hashed-token-in-hand (CLI, MCP, background jobs) can still build a
  ``IdentityContext``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm.identity.context import AuditInfo, ClientInfo, IdentityContext
from litellm.identity.extractors.api_key import (
    extract_api_key_principal,
    hash_principal_token,
)
from litellm.identity.extractors.client import extract_client_info
from litellm.identity.extractors.end_user import extract_end_user_id
from litellm.identity.extractors.header import extract_audit_changed_by
from litellm.identity.extractors.jwt import extract_jwt_principal
from litellm.identity.principal import (
    AnonymousPrincipal,
    Principal,
    ServiceAccountPrincipal,
)
from litellm.identity.service_accounts import SERVICE_ACCOUNT_NAMES
from litellm.integrations.otel.model.spans import SpanRole
from litellm.integrations.otel.runtime import traced

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    from litellm.identity.cache import IdentityCache


def _principal_from_raw_key(api_key: Optional[str]) -> Principal:
    if api_key and api_key in SERVICE_ACCOUNT_NAMES:
        return ServiceAccountPrincipal(name=api_key)

    jwt_principal = extract_jwt_principal(api_key)
    if jwt_principal is not None:
        return jwt_principal

    api_key_principal = extract_api_key_principal(api_key)
    if api_key_principal is not None:
        return api_key_principal

    return AnonymousPrincipal()


@traced(
    "identity.resolve",
    role=SpanRole.SERVICE,
    attrs=lambda result: {
        "identity.principal.kind": result.principal.kind,
        "identity.end_user.present": result.end_user_id is not None,
    },
)
async def resolve_identity(
    *,
    api_key: Optional[str] = None,
    request: Any = None,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    general_settings: Optional[Dict[str, Any]] = None,
) -> IdentityContext:
    """Build an ``IdentityContext`` purely from request-side signals.

    Does not touch the database. The hydrated-row variant lives in
    ``store.load_identity`` and is composed by callers that have a
    ``PrismaClient`` + ``IdentityCache`` in hand.
    """
    principal = _principal_from_raw_key(api_key)
    end_user_id = extract_end_user_id(body=body, headers=headers)
    audit = AuditInfo(changed_by=extract_audit_changed_by(headers))
    client: ClientInfo
    if request is not None:
        client = extract_client_info(request=request, general_settings=general_settings)
    else:
        client = ClientInfo()

    return IdentityContext(
        principal=principal,
        end_user_id=end_user_id,
        audit=audit,
        client=client,
    )


async def resolve_user_api_key_auth(
    *,
    api_key: str,
    prisma_client: "PrismaClient",
    identity_cache: "IdentityCache",
    user_api_key_cache: "UserApiKeyCache",
    parent_otel_span=None,
    proxy_logging_obj: Optional["ProxyLogging"] = None,
) -> "UserAPIKeyAuth":
    """Cache-or-DB resolve a hashed virtual key into ``UserAPIKeyAuth``.

    This is the surface ``_user_api_key_auth_builder`` calls in place of
    the legacy ``get_key_object``. The hashed-token computation is
    centralized via ``hash_principal_token`` so the auth chain doesn't
    re-implement the JWT-vs-key hashing rules.
    """
    from litellm.identity.store import load_identity

    hashed_token = hash_principal_token(api_key)
    return await load_identity(
        hashed_token=hashed_token,
        prisma_client=prisma_client,
        cache=identity_cache,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )

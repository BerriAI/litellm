"""Caller-identity module.

Domain types, the identity cache/store, and a bidirectional adapter to
``UserAPIKeyAuth``. The proxy still drives identity through
``litellm/proxy/auth/`` today; this module is the new home those flows
migrate to.

The public surface is small on purpose; downstream code should depend on
``IdentityContext`` and the ``Principal`` union, not on individual
internals.
"""

from litellm.identity.cache import IdentityCache, get_identity_cache
from litellm.identity.jwt import build_user_api_key_auth_from_jwt_result
from litellm.identity.oauth2 import build_user_api_key_auth_from_oauth2_response
from litellm.identity.context import (
    AuditInfo,
    ClientInfo,
    IdentityContext,
    RequestIds,
)
from litellm.identity.principal import (
    AnonymousPrincipal,
    ApiKeyPrincipal,
    JWTPrincipal,
    Principal,
    SSOPrincipal,
    ServiceAccountPrincipal,
)
from litellm.identity.resolver import resolve_identity, resolve_user_api_key_auth
from litellm.identity.store import load_identity

__all__ = [
    "AnonymousPrincipal",
    "ApiKeyPrincipal",
    "AuditInfo",
    "ClientInfo",
    "IdentityCache",
    "IdentityContext",
    "JWTPrincipal",
    "Principal",
    "RequestIds",
    "SSOPrincipal",
    "ServiceAccountPrincipal",
    "build_user_api_key_auth_from_jwt_result",
    "build_user_api_key_auth_from_oauth2_response",
    "get_identity_cache",
    "load_identity",
    "resolve_identity",
    "resolve_user_api_key_auth",
]

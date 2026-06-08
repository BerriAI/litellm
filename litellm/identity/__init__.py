"""Caller-identity module.

Phase 1: domain types + extractors + bidirectional adapter to
``UserAPIKeyAuth``. The proxy still drives identity through
``litellm/proxy/auth/`` today; this module is the new home those flows
will migrate to.

The public surface is small on purpose; downstream code should depend on
``IdentityContext`` and the ``Principal`` union, not on individual
extractor internals.
"""

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

__all__ = [
    "AnonymousPrincipal",
    "ApiKeyPrincipal",
    "AuditInfo",
    "ClientInfo",
    "IdentityContext",
    "JWTPrincipal",
    "Principal",
    "RequestIds",
    "SSOPrincipal",
    "ServiceAccountPrincipal",
]

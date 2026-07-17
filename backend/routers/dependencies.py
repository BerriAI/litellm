from __future__ import annotations

from typing import Callable, cast

from fastapi import Request
from fastapi.security import SecurityScopes

from litellm.proxy.auth_v2 import Principal, Role
from litellm.proxy.auth_v2.errors import forbidden_role
from litellm.proxy.auth_v2.resolvers import ProvisioningStore
from litellm.proxy.auth_v2.security import AuthSecurity


def get_auth(request: Request) -> AuthSecurity:
    return cast(AuthSecurity, request.app.state.auth_v2)


def team_store(request: Request) -> ProvisioningStore:
    return cast(ProvisioningStore, get_auth(request).resolver)


def require_roles(*allowed: Role) -> Callable[[Request], object]:
    """Request-scoped role gate built on the ``auth_v2`` Security layer.

    Mirrors ``AuthSecurity.require_roles`` but reaches the per-app ``AuthSecurity``
    via ``request.app.state`` at request time, since these routers are wired into
    the app after import rather than closing over an instance.
    """

    async def dependency(request: Request) -> Principal:
        auth = get_auth(request)
        principal = await auth.principal(SecurityScopes(scopes=[]), request)
        if not auth.authorizer.has_any_role(principal, allowed):
            raise forbidden_role()
        return principal

    return dependency

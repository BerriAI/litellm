from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, List

from fastapi import FastAPI, Request, Security
from fastapi.security import SecurityScopes

from . import errors
from .authenticators import Authenticator, build_authenticators
from .config import AuthConfig
from .models import Principal
from .network import resolve_network_context
from .rbac import Role, has_any_role, has_required_scopes
from .resolver import IdentityResolver


@dataclass
class AuthContext:
    config: AuthConfig
    authenticators: List[Authenticator]
    resolver: IdentityResolver


def install_auth(
    app: FastAPI,
    config: AuthConfig,
    resolver: IdentityResolver,
    *,
    mount_scim: bool = True,
    mount_oidc: bool = True,
    mount_saml: bool = True,
) -> AuthContext:
    ctx = AuthContext(config, build_authenticators(config), resolver)
    app.state.auth_v2 = ctx
    if mount_scim:
        from .scim import build_scim_router

        app.include_router(build_scim_router())
    if mount_oidc and config.oidc_providers:
        from .oidc import build_oidc_router

        app.include_router(build_oidc_router(config))
    if mount_saml and config.saml is not None and config.saml.enabled:
        from .saml import SamlAuthenticator, SamlSessionStore, build_saml_router

        session_store = SamlSessionStore()
        ctx.authenticators.append(SamlAuthenticator(config.saml, session_store))
        app.include_router(build_saml_router(config.saml, session_store))
    return ctx


def _ctx(request: Request) -> AuthContext:
    return request.app.state.auth_v2


def _combined_challenge(authenticators: List[Authenticator]) -> str:
    seen: List[str] = []
    for authenticator in authenticators:
        challenge = authenticator.challenge()
        if challenge and challenge not in seen:
            seen.append(challenge)
    return ", ".join(seen)


async def get_current_principal(
    security_scopes: SecurityScopes, request: Request
) -> Principal:
    ctx = _ctx(request)
    credential = None
    for authenticator in ctx.authenticators:
        credential = await authenticator.authenticate(request)
        if credential is not None:
            break
    if credential is None:
        raise errors.unauthenticated(_combined_challenge(ctx.authenticators))

    resolved = await ctx.resolver.resolve(credential)
    principal = resolved.model_copy(
        update={"network": resolve_network_context(request, ctx.config.network)}
    )

    if not has_required_scopes(security_scopes, principal):
        raise errors.insufficient_scope()
    return principal


def require_roles(*allowed: Role) -> Callable[..., object]:
    async def dependency(
        principal: Annotated[Principal, Security(get_current_principal)],
    ) -> Principal:
        if not has_any_role(principal, allowed):
            raise errors.forbidden_role()
        return principal

    return dependency

from typing import Annotated, Callable, List, Optional

from fastapi import Request, Security
from fastapi.security import SecurityScopes

from . import errors
from .authenticators import (
    Authenticator,
    BasicAuthVerifier,
    build_authenticators,
)
from .config import AuthConfig
from .models import Principal
from .network import resolve_network_context
from .rbac import RBACEngine, Role, has_required_scopes
from .resolver import IdentityResolver
from .session import SessionAuthenticator, SessionStore


def _combined_challenge(authenticators: List[Authenticator]) -> str:
    seen: List[str] = []
    for authenticator in authenticators:
        challenge = authenticator.challenge()
        if challenge and challenge not in seen:
            seen.append(challenge)
    return ", ".join(seen)


class AuthSecurity:
    """Enforcement layer consumed purely through FastAPI ``Security()``.

    Construct once at the composition root and pass the bound methods
    (``principal``, ``require_roles``, ``require_permission``) to ``Security()``;
    routers receive the instance explicitly via ``build_*_router(auth)``. There is
    no app mutation and no ``app.state``.

    Deployment note for trusted-proxy IP resolution: uvicorn's ``--proxy-headers``
    (on by default) overwrites ``request.client`` from ``X-Forwarded-For`` before
    this module's ``trusted_proxy_cidrs`` check runs, silently bypassing it. Run
    uvicorn with ``--no-proxy-headers`` and let this module resolve the client IP,
    or leave ``trusted_proxy_cidrs`` empty and rely on uvicorn's
    ``--forwarded-allow-ips``. Do not enable both.
    """

    def __init__(
        self,
        config: AuthConfig,
        resolver: IdentityResolver,
        rbac: Optional[RBACEngine] = None,
        authenticators: Optional[List[Authenticator]] = None,
        basic_verifier: Optional[BasicAuthVerifier] = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.rbac = rbac or RBACEngine(config.casbin_policy_path)
        self.session_store = SessionStore(
            config.session.ttl_seconds, config.session.max_size
        )
        self.oauth_txn_store = SessionStore(
            config.session.login_state_ttl, config.session.max_size
        )
        chain = (
            list(authenticators)
            if authenticators is not None
            else build_authenticators(config, basic_verifier=basic_verifier)
        )
        chain.append(SessionAuthenticator(config.session.cookie, self.session_store))
        self.authenticators = chain

    async def principal(
        self, security_scopes: SecurityScopes, request: Request
    ) -> Principal:
        credential = None
        for authenticator in self.authenticators:
            credential = await authenticator.authenticate(request)
            if credential is not None:
                break
        if credential is None:
            raise errors.unauthenticated(_combined_challenge(self.authenticators))

        resolved = await self.resolver.resolve(credential)
        principal = resolved.model_copy(
            update={"network": resolve_network_context(request, self.config.network)}
        )
        if not has_required_scopes(security_scopes, principal):
            raise errors.insufficient_scope()
        return principal

    def require_roles(self, *allowed: Role) -> Callable[..., object]:
        async def dependency(
            principal: Annotated[Principal, Security(self.principal)],
        ) -> Principal:
            if not self.rbac.has_any_role(principal, allowed):
                raise errors.forbidden_role()
            return principal

        return dependency

    def require_permission(self, obj: str, act: str) -> Callable[..., object]:
        async def dependency(
            principal: Annotated[Principal, Security(self.principal)],
        ) -> Principal:
            if not self.rbac.enforce(principal, obj, act):
                raise errors.forbidden_permission()
            return principal

        return dependency

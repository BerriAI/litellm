from typing import Annotated, Callable, List, Optional

from fastapi import Request, Security
from fastapi.security import SecurityScopes

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.authenticators import (
    Authenticator,
    BasicAuthVerifier,
    build_authenticators,
)
from litellm.proxy.auth_v2.config import AuthConfig
from litellm.proxy.auth_v2.models import Principal
from litellm.proxy.auth_v2.network import resolve_network_context
from litellm.proxy.auth_v2.authorization import (
    Authorizer,
    RBACEngine,
    Role,
    has_required_scopes,
)
from litellm.proxy.auth_v2.authenticators.session import SessionAuthenticator
from litellm.proxy.auth_v2.resolvers import IdentityResolver
from litellm.proxy.auth_v2.sessions import StateBackend, StateStore
from litellm.proxy.auth_v2.sessions.schemas import OAuthTransaction, SessionState


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
    (``principal``, ``require_roles``, ``require_permission``) to ``Security()``.
    Routers reach this instance at request time via ``request.app.state.auth_v2``
    (see ``routers/dependencies.py``), so assign it there when wiring the app.

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
        authorizer: Optional[Authorizer] = None,
        authenticators: Optional[List[Authenticator]] = None,
        basic_verifier: Optional[BasicAuthVerifier] = None,
        state_backend: Optional[StateBackend] = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.authorizer = authorizer or RBACEngine(config.casbin_policy_path)
        self._state = state_backend or StateBackend(None)
        self.session_store: StateStore[SessionState] = self._state.store(
            "sessions", default_ttl=config.session.ttl_seconds
        )
        self.oauth_txn_store: StateStore[OAuthTransaction] = self._state.store(
            "oauth_txn", default_ttl=config.session.login_state_ttl
        )
        chain = (
            list(authenticators)
            if authenticators is not None
            else build_authenticators(config, basic_verifier=basic_verifier)
        )
        chain.append(SessionAuthenticator(config.session.cookie, self.session_store))
        self.authenticators = chain

    async def principal(self, security_scopes: SecurityScopes, request: Request) -> Principal:
        """Resolve the caller to a Principal, enforcing scheme OR and required scopes."""
        credential = None
        for authenticator in self.authenticators:
            credential = await authenticator.authenticate(request)
            if credential is not None:
                break
        if credential is None:
            raise errors.unauthenticated(_combined_challenge(self.authenticators))

        resolved = await self.resolver.resolve(credential)
        principal = resolved.model_copy(update={"network": resolve_network_context(request, self.config.network)})
        if not has_required_scopes(security_scopes, principal):
            raise errors.insufficient_scope()
        return principal

    def require_roles(self, *allowed: Role) -> Callable[..., object]:
        """Security() dependency that admits a principal holding any allowed role (hierarchy-aware)."""

        async def dependency(
            principal: Annotated[Principal, Security(self.principal)],
        ) -> Principal:
            if not self.authorizer.has_any_role(principal, allowed):
                raise errors.forbidden_role()
            return principal

        return dependency

    def require_permission(self, obj: str, act: str) -> Callable[..., object]:
        """Security() dependency that admits a principal whose roles permit obj/act via Casbin."""

        async def dependency(
            principal: Annotated[Principal, Security(self.principal)],
        ) -> Principal:
            if not self.authorizer.enforce(principal, obj, act):
                raise errors.forbidden_permission()
            return principal

        return dependency

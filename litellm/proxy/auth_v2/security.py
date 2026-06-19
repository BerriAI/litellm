import os
from typing import TYPE_CHECKING, Annotated, Callable, Dict, List, Optional, cast

from fastapi import Request, Security
from fastapi.security import SecurityScopes

if TYPE_CHECKING:
    from redis.asyncio import Redis

from litellm._redis import get_redis_async_client
from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.authenticators import (
    Authenticator,
    BasicAuthVerifier,
    Carrier,
    build_authenticators,
)
from litellm.proxy.auth_v2.config import AuthConfig
from litellm.proxy.auth_v2.models import Principal
from litellm.proxy.auth_v2.network import resolve_network_context
from litellm.proxy.auth_v2.authorization import (
    Authorizer,
    RBACEngine,
    Role,
)
from litellm.proxy.auth_v2.authenticators.session import SessionAuthenticator
from litellm.proxy.auth_v2.resolvers import Resolver
from litellm.proxy.auth_v2.sessions import (
    InMemorySessionStore,
    RedisSessionStore,
    SessionStore,
    SessionValue,
)
from litellm.proxy.auth_v2.sessions.types import OAuthTransaction, SessionState

_REDIS_ENV_SIGNALS = (
    "REDIS_URL",
    "REDIS_HOST",
    "REDIS_CLUSTER_NODES",
    "REDIS_SENTINEL_NODES",
)


def _open_session_store(
    namespace: str, *, default_ttl: int
) -> SessionStore[SessionValue]:
    """Build the session/login-state store for ``namespace``.

    Uses Redis when configured via the environment (required so state is shared
    across pods in a multi-pod deployment); otherwise a process-local in-memory
    store for single-process/dev. Chosen from the environment, not by probing
    Redis, so a configured-but-unreachable Redis fails loudly on use rather than
    silently stranding state on one pod.
    """
    if any(os.getenv(signal) for signal in _REDIS_ENV_SIGNALS):
        return RedisSessionStore(
            cast("Redis", get_redis_async_client()), namespace, default_ttl
        )
    return InMemorySessionStore(namespace, default_ttl)


def _combined_challenge(authenticators: List[Authenticator]) -> str:
    challenges = (authenticator.challenge() for authenticator in authenticators)
    return ", ".join(dict.fromkeys(c for c in challenges if c))


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
        resolver: Resolver,
        authorizer: Optional[Authorizer] = None,
        authenticators: Optional[List[Authenticator]] = None,
        basic_verifier: Optional[BasicAuthVerifier] = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.authorizer = authorizer or RBACEngine(config.casbin_policy_path)
        self.session_store: SessionStore[SessionState] = _open_session_store(
            "sessions", default_ttl=config.session.ttl_seconds
        )
        self.oauth_txn_store: SessionStore[OAuthTransaction] = _open_session_store(
            "oauth_txn", default_ttl=config.session.login_state_ttl
        )
        chain = (
            list(authenticators)
            if authenticators is not None
            else build_authenticators(config, basic_verifier=basic_verifier)
        )
        chain.append(SessionAuthenticator(config.session.cookie, self.session_store))
        self.authenticators = chain
        self._by_carrier: Dict[Carrier, Authenticator] = {}
        for authenticator in chain:
            for carrier in authenticator.carriers():
                self._by_carrier.setdefault(carrier, authenticator)

    def _authenticator_for(self, request: Request) -> Optional[Authenticator]:
        """The single authenticator whose credential the request carries, by scheme_order."""
        return next(
            (a for carrier, a in self._by_carrier.items() if carrier.present(request)),
            None,
        )

    async def principal(
        self, security_scopes: SecurityScopes, request: Request
    ) -> Principal:
        """Resolve the caller to a Principal, enforcing scheme OR and required scopes."""
        authenticator = self._authenticator_for(request)
        if authenticator is None:
            raise errors.unauthenticated(_combined_challenge(self.authenticators))
        credential = await authenticator.authenticate(request)
        if credential is None:
            raise errors.unauthenticated(_combined_challenge(self.authenticators))

        principal = await self.resolver.resolve(credential)
        principal.network = resolve_network_context(request, self.config.network)
        if not principal.has_required_scopes(security_scopes):
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

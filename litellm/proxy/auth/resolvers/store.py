from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _copy_user_api_key_auth_for_cache,
    _load_key_object_on_cache_miss,
)
from litellm.proxy.auth.auth_method import AuthMethod
from litellm.proxy.auth.network import NetworkContext
from litellm.proxy.auth.resolvers.exceptions import (
    KeyNotFoundError,
    KeyNotInCacheError,
    NoDatabaseConnectionError,
    PrincipalMissingSourceKeyError,
)
from litellm.proxy.auth.resolvers.models import (
    CredentialRef,
    EndUserIdentity,
    OrganizationIdentity,
    Principal,
    PrincipalType,
    ProjectIdentity,
    TeamIdentity,
    UserIdentity,
)
from litellm.proxy.auth.roles import TeamRole, map_role, team_role

if TYPE_CHECKING:
    from litellm.integrations.opentelemetry import Span
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging


class IdentityStore:
    """The auth flow's resolver: one combined_view lookup, projected into a Principal.

    ``resolve`` does the lookup (cache, then DB via the shared lower-level helpers,
    then write-back) and returns the per-caller Principal. The Principal carries the
    source key object so ``key_from_principal`` can hand it back to the parts of the
    request flow that still consume ``UserAPIKeyAuth`` (budget, rate limits, policy);
    that carrier is a stopgap until those consumers read identity off the Principal.
    The Prisma client, key cache, the request's tracing span / logging sink, and
    whether this store may only read the cache are injected so the composition root
    can build the store once the proxy DB is connected; the span and logging sink
    are infra the DB call is instrumented with and ``check_cache_only`` is a store
    mode, none of them inputs to resolving identity. ``auth_checks.get_key_object``
    stays as the legacy entrypoint for its other callers until they migrate onto
    this store.
    """

    def __init__(
        self,
        prisma_client: PrismaClient | None,
        cache: UserApiKeyCache,
        *,
        parent_otel_span: Span | None = None,
        proxy_logging_obj: ProxyLogging | None = None,
        check_cache_only: bool = False,
    ) -> None:
        self._prisma = prisma_client
        self._cache = cache
        self._parent_otel_span = parent_otel_span
        self._proxy_logging_obj = proxy_logging_obj
        self._check_cache_only = check_cache_only

    async def resolve(
        self,
        hashed_token: str,
        *,
        auth_method: AuthMethod = AuthMethod.API_KEY,
        network: NetworkContext | None = None,
    ) -> Principal:
        key: Final = await self._resolve_key(hashed_token)
        return self._principal_from_key(
            key,
            auth_method=auth_method,
            network=network,
            subject_fallback=key.token,
            credential_ref=CredentialRef(token_id=key.token),
        )

    @staticmethod
    def key_from_principal(principal: Principal) -> UserAPIKeyAuth:
        """Hand back the resolved key object carried on the Principal.

        Stopgap for the request flow that still consumes ``UserAPIKeyAuth`` for
        budget, rate-limit, and policy state. Only Principals produced by
        ``resolve`` carry a source key.
        """
        if principal.source_key is None:
            raise PrincipalMissingSourceKeyError()
        return principal.source_key

    async def _resolve_key(self, hashed_token: str) -> UserAPIKeyAuth:
        if self._prisma is None:
            raise NoDatabaseConnectionError()

        cached: Final = await self._cache.async_get_cache(key=hashed_token, model_type=UserAPIKeyAuth)
        if cached is not None:
            return _copy_user_api_key_auth_for_cache(user_api_key_obj=cached)

        if self._check_cache_only:
            raise KeyNotInCacheError(hashed_token)

        key: Final = await _load_key_object_on_cache_miss(
            hashed_token=hashed_token,
            prisma_client=self._prisma,
            user_api_key_cache=self._cache,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
        )
        if key is None:
            raise KeyNotFoundError(hashed_token)
        return key

    @staticmethod
    def _principal_from_key(
        key: UserAPIKeyAuth,
        *,
        auth_method: AuthMethod,
        issuer: str | None = None,
        subject_fallback: str | None = None,
        scopes: Sequence[str] = (),
        credential_ref: CredentialRef | None = None,
        network: NetworkContext | None = None,
    ) -> Principal:
        """Project the identity slice off an already-resolved key object and carry
        the key on the Principal so ``key_from_principal`` can recover it.

        Pure: issues no lookup. Both ``resolve`` and the auth seam call this so
        identity is projected once off whichever key object they already hold.
        """
        teams: Final[list[TeamIdentity]] = []
        if key.team_id is not None:
            role: Final = team_role(key.team_member.role) if key.team_member else TeamRole.MEMBER
            teams.append(TeamIdentity(id=key.team_id, name=key.team_alias, role=role))
        organization: Final = (
            OrganizationIdentity(id=key.org_id, name=key.organization_alias) if key.org_id is not None else None
        )
        user: Final = UserIdentity(id=key.user_id, email=key.user_email) if key.user_id is not None else None
        project = ProjectIdentity(id=key.project_id, name=key.project_alias) if key.project_id is not None else None
        end_user: Final = EndUserIdentity(id=key.end_user_id) if key.end_user_id is not None else None
        mapped: Final = map_role(key.user_role)
        return Principal(
            principal_type=(PrincipalType.HUMAN if key.user_id else PrincipalType.SERVICE_ACCOUNT),
            subject=key.user_id or key.key_alias or subject_fallback or "",
            issuer=issuer,
            user=user,
            organization=organization,
            teams=teams,
            project=project,
            end_user=end_user,
            roles=[mapped] if mapped else [],
            scopes=list(scopes),
            auth_method=auth_method,
            credential_ref=credential_ref or CredentialRef(),
            network=network or NetworkContext(),
            source_key=key,
        )

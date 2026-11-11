"""Load a caller's user row and team membership from the database.

The virtual-key path reads these off the combined-view SQL join. Every other credential (an IdP JWT, a
``lite login`` session token) carries only identifiers, so it reads the live rows on each request through
``GrantResolver``. gateway-auth-core decides what the rows mean.
"""

from __future__ import annotations

from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import ValidationError
from pydantic.main import IncEx

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_TeamMembership, LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import forget_missing_user, get_team_membership, get_user_object
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.proxy._types import Span
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging


class UserLoader(Protocol):
    def __call__(
        self,
        *,
        user_id: str | None,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache,
        user_id_upsert: bool,
        parent_otel_span: Span | None,
        proxy_logging_obj: ProxyLogging | None,
        sso_user_id: str | None,
        user_email: str | None,
    ) -> Coroutine[object, object, LiteLLM_UserTable | None]: ...


class MembershipLoader(Protocol):
    def __call__(
        self,
        *,
        user_id: str,
        team_id: str,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache,
        parent_otel_span: Span | None,
        proxy_logging_obj: ProxyLogging | None,
    ) -> Coroutine[object, object, LiteLLM_TeamMembership | None]: ...


@dataclass(frozen=True, slots=True)
class UserLookup:
    """The user a credential names, plus the hints ``get_user_object`` may fall back to when the id alone
    matches no row."""

    user_id: str | None
    user_email: str | None = None
    sso_user_id: str | None = None
    upsert: bool = False


_MODELS_COLUMN: Final[Mapping[str, IncEx | bool]] = MappingProxyType({"models": True})


class _UserModelColumn(LiteLLMBaseModel):
    """``LiteLLM_UserTable.models`` is a bare ``list``; re-read it with the shape a token's ``models`` takes."""

    models: tuple[str, ...] = ()


def user_models(user_object: LiteLLM_UserTable) -> tuple[str, ...]:
    try:
        return _UserModelColumn.model_validate(user_object.model_dump(include=_MODELS_COLUMN)).models
    except ValidationError:
        return ()


def canonical_user_id(user_id: str | None, user_object: LiteLLM_UserTable | None) -> str | None:
    if user_object is not None and user_object.user_id:
        return user_object.user_id
    return user_id


class GrantResolver:
    """Reads the user and membership rows for a credential through injected loaders, which default to the
    shared ``auth_checks`` readers. Loader errors surface as they are."""

    def __init__(
        self,
        prisma_client: PrismaClient | None,
        cache: UserApiKeyCache,
        *,
        parent_otel_span: Span | None = None,
        proxy_logging_obj: ProxyLogging | None = None,
        load_user: UserLoader = get_user_object,
        load_membership: MembershipLoader = get_team_membership,
    ) -> None:
        self._prisma = prisma_client
        self._cache = cache
        self._parent_otel_span = parent_otel_span
        self._proxy_logging_obj = proxy_logging_obj
        self._load_user = load_user
        self._load_membership = load_membership

    async def resolve_identity(
        self, lookup: UserLookup, team_id: str | None
    ) -> tuple[LiteLLM_UserTable | None, LiteLLM_TeamMembership | None, str | None]:
        user_object: Final = await self._user(lookup) if lookup.user_id else None
        effective_user_id: Final = canonical_user_id(lookup.user_id, user_object)
        if effective_user_id != lookup.user_id:
            verbose_proxy_logger.debug(
                "Auth: rebinding user_id %r -> DB user_id %r (email/sso match)",
                lookup.user_id,
                effective_user_id,
            )
        membership: Final = (
            await self._membership(user_id=effective_user_id, team_id=team_id)
            if effective_user_id and team_id
            else None
        )
        return user_object, membership, effective_user_id

    async def _user(self, lookup: UserLookup) -> LiteLLM_UserTable | None:
        if lookup.user_id is not None:
            forget_missing_user(lookup.user_id)
        return await self._load_user(
            user_id=lookup.user_id,
            prisma_client=self._prisma,
            user_api_key_cache=self._cache,
            user_id_upsert=lookup.upsert,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
            user_email=lookup.user_email,
            sso_user_id=lookup.sso_user_id,
        )

    async def _membership(self, user_id: str, team_id: str) -> LiteLLM_TeamMembership | None:
        return await self._load_membership(
            user_id=user_id,
            team_id=team_id,
            prisma_client=self._prisma,
            user_api_key_cache=self._cache,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
        )

"""Load a caller's user row, team row, and team membership from the database and validate them together.

The virtual-key path reads these off the combined-view SQL join. Every other credential (an IdP JWT, a
``lite login`` session token) carries only identifiers, or a snapshot of grants taken when it was minted, so
it has to read the live rows on each request. Both of those paths resolve the same rows with the same
membership rule, and ``GrantResolver`` is the one place that rule lives.
"""

from __future__ import annotations

from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, TypeAlias

from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError
from pydantic.main import IncEx
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.proxy.auth.auth_checks import (
    TeamNotFoundError,
    UserNotFoundError,
    get_team_membership,
    get_team_object,
    get_user_object,
)

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


class TeamLoader(Protocol):
    def __call__(
        self,
        *,
        team_id: str,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache,
        parent_otel_span: Span | None,
        proxy_logging_obj: ProxyLogging | None,
    ) -> Coroutine[object, object, LiteLLM_TeamTableCachedObj]: ...


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


@dataclass(frozen=True, slots=True)
class ResolvedGrants:
    """The live rows behind a credential. ``effective_user_id`` is the DB row's id when a fuzzy match found a
    legacy row under a different id (GH #26789), otherwise the id the credential named."""

    user_object: LiteLLM_UserTable | None
    team_object: LiteLLM_TeamTableCachedObj | None
    team_membership: LiteLLM_TeamMembership | None
    effective_user_id: str | None


@dataclass(frozen=True, slots=True)
class UserGone:
    user_id: str


@dataclass(frozen=True, slots=True)
class TeamGone:
    team_id: str


@dataclass(frozen=True, slots=True)
class NotAMember:
    user_id: str
    team_id: str


@dataclass(frozen=True, slots=True)
class LookupDegraded:
    """A row could not be read for a reason that says nothing about the caller: the database is down or a
    loader failed. The caller decides whether a grant it already holds may stand in."""

    error: Exception


GrantDenial: TypeAlias = UserGone | TeamGone | NotAMember
GrantOutcome: TypeAlias = ResolvedGrants | GrantDenial | LookupDegraded


_MODELS_COLUMN: Final[Mapping[str, IncEx | bool]] = MappingProxyType({"models": True})


class _UserModelColumn(BaseModel):
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


def raise_public(denial: GrantDenial) -> NoReturn:
    match denial:
        case UserGone(user_id=user_id):
            raise ProxyException(
                message=f"Authentication Error, user '{user_id}' no longer exists.",
                type=ProxyErrorTypes.auth_error,
                param="user_id",
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case TeamGone(team_id=team_id):
            raise TeamNotFoundError(team_id=team_id)
        case NotAMember(team_id=team_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Team '{team_id}' is not in your team memberships.",
            )
        case _:
            assert_never(denial)


class GrantResolver:
    """Reads the user, membership, and team rows for a credential through injected loaders.

    The loaders default to the shared ``auth_checks`` readers. A caller passes its own module's names for them
    so the reads stay interceptable where that module's callers already intercept them. ``resolve_identity``
    is the JWT half: user and membership only, since the JWT builder selects the team itself and lets loader
    errors surface as they are. ``resolve`` also reads the team row and applies the membership rule, which is
    what a credential carrying a grant snapshot needs to refresh it.
    """

    def __init__(
        self,
        prisma_client: PrismaClient | None,
        cache: UserApiKeyCache,
        *,
        parent_otel_span: Span | None = None,
        proxy_logging_obj: ProxyLogging | None = None,
        load_user: UserLoader = get_user_object,
        load_team: TeamLoader = get_team_object,
        load_membership: MembershipLoader = get_team_membership,
    ) -> None:
        self._prisma = prisma_client
        self._cache = cache
        self._parent_otel_span = parent_otel_span
        self._proxy_logging_obj = proxy_logging_obj
        self._load_user = load_user
        self._load_team = load_team
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

    async def resolve(self, lookup: UserLookup, team_id: str | None) -> GrantOutcome:
        try:
            user_object, membership, effective_user_id = await self.resolve_identity(lookup, team_id)
        except UserNotFoundError:
            return UserGone(user_id=lookup.user_id or "")
        except Exception as error:
            return LookupDegraded(error=error)
        if team_id is None:
            return ResolvedGrants(user_object, None, membership, effective_user_id)
        if user_object is not None and team_id not in user_object.teams:
            return NotAMember(user_id=user_object.user_id, team_id=team_id)
        try:
            team_object: Final = await self._load_team(
                team_id=team_id,
                prisma_client=self._prisma,
                user_api_key_cache=self._cache,
                parent_otel_span=self._parent_otel_span,
                proxy_logging_obj=self._proxy_logging_obj,
            )
        except TeamNotFoundError:
            return TeamGone(team_id=team_id)
        except Exception as error:
            return LookupDegraded(error=error)
        return ResolvedGrants(user_object, team_object, membership, effective_user_id)

    async def _user(self, lookup: UserLookup) -> LiteLLM_UserTable | None:
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

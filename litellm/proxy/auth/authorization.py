from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Generic, TypeAlias, TypeVar

from litellm.proxy._types import KeyManagementRoutes, LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth

GrantT = TypeVar("GrantT", covariant=True)


@dataclass(frozen=True, slots=True)
class AnyOf(Generic[GrantT]):
    grants: tuple[GrantT, ...]


def any_of(*grants: GrantT) -> AnyOf[GrantT]:
    return AnyOf(grants)


@dataclass(frozen=True, slots=True)
class AllLogs:
    """Unrestricted reads, granted by the consuming endpoint's role checks."""


@dataclass(frozen=True, slots=True)
class UserLogs:
    user_id: str | None


@dataclass(frozen=True, slots=True)
class TeamLogs:
    team_id: str


LogGrant: TypeAlias = UserLogs | TeamLogs


LogReadScope: TypeAlias = AllLogs | LogGrant | AnyOf[LogGrant]


async def resolve_log_read_scope(
    user_id: str | None,
    permitted_team_lookup: Callable[[], Awaitable[Sequence[str]]],
) -> UserLogs | AnyOf[LogGrant]:
    """Resolve own-user and permitted-team reads, falling back to own-user on lookup failure."""
    try:
        team_ids: Final = tuple(await permitted_team_lookup())
    except Exception:  # noqa: BLE001  # preserve spend-log own-user fallback for every permission lookup failure
        return UserLogs(user_id)
    return any_of(UserLogs(user_id), *(TeamLogs(team_id) for team_id in team_ids)) if team_ids else UserLogs(user_id)


def can_read_team_logs(auth: UserAPIKeyAuth, team: LiteLLM_TeamTable) -> bool:
    from litellm.proxy.management.teams.access import is_team_admin
    from litellm.proxy.management_endpoints.common_utils import (
        _team_member_has_permission,  # pyright: ignore[reportPrivateUsage]  # reuse existing team permission policy
    )

    return is_team_admin(user_api_key_dict=auth, team_obj=team) or _team_member_has_permission(
        user_api_key_dict=auth,
        team_obj=team,
        permission=KeyManagementRoutes.SPEND_LOGS.value,
    )


def permitted_log_team_ids(auth: UserAPIKeyAuth, teams: Iterable[LiteLLM_TeamTable]) -> tuple[str, ...]:
    return tuple(team.team_id for team in teams if can_read_team_logs(auth, team))


async def can_read_log_owner(
    user_id: str | None,
    owner_user: str | None,
    owner_team_id: str | None,
    team_permission_lookup: Callable[[str], Awaitable[bool]],
) -> bool:
    """Authorize stored ownership without swallowing direct team-lookup failures."""
    if owner_user is not None and owner_user == user_id:
        return True
    if owner_team_id:
        return await team_permission_lookup(owner_team_id)
    return False


@dataclass(frozen=True, slots=True)
class ApiKeyLogs:
    api_key_hash: str


TraceLogGrant: TypeAlias = LogGrant | ApiKeyLogs
TraceReadScope: TypeAlias = AllLogs | TraceLogGrant | AnyOf[TraceLogGrant]


async def resolve_trace_read_scope(
    auth: UserAPIKeyAuth,
    permitted_team_lookup: Callable[[], Awaitable[Sequence[str]]],
) -> TraceReadScope | None:
    if auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        return AllLogs()
    if not auth.user_id:
        return ApiKeyLogs(auth.token) if auth.token else None
    user_scope: Final = await resolve_log_read_scope(auth.user_id, permitted_team_lookup)
    user_grants: Final = user_scope.grants if isinstance(user_scope, AnyOf) else (user_scope,)
    key_grants: Final = (ApiKeyLogs(auth.token),) if auth.token else ()
    return any_of(*user_grants, *key_grants)
